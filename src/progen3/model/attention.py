"""
attention.py  ── Core multi-head / GQA attention layer for ProGen3
───────────────────────────────────────────────────────────────────
Research-oriented overview
──────────────────────────
Role in ProGen3
    • Implements the flash-SDPA attention used in every transformer block.
    • Supports grouped-query attention (GQA) via num_key_value_heads < num_heads.
    • Adds RoPE positional encoding with a user-configurable θ (rope_theta).
    • Handles cache update for KV in autoregressive decoding.

Why this design?
    1. Flash-Attention 2 (torch.sdpa with backend=FLASH_ATTENTION) is the
       current sweet spot for long-context (8 k tokens) training and fp16/bf16
       inference.  Staying inside PyTorch keeps the codebase dependency-light
       compared with xformers or triton kernels.
    2. GQA (a la Mistral) reduces KV memory/latency ~4× while preserving
       quality.  Important when the sequence length is huge and MoE already
       bloats the model.
    3. RoPE is chosen over ALiBi because ProGen3 uses infilling (GLM).  RoPE
       allows absolute and relative positions, and the custom θ lets us trade
       off attention-range extrapolation (Sec 2.1 of paper).
    4. Key/value cache handling is copied from HF Llama utils so that external
       PEFT / LoRA or HF-Serve tooling "just works".

Extension hooks
    • Structure adapters (as in proseLM) could inject bias terms into
      `self._sdpa_attn` via `torch.nn.attention.attn_bias`.  You'd pass a dense
      bias tensor shaped (batch, heads, q_len, k_len).
    • LongRope / DynamicRoPE: change `RotaryPositionalEmbedding` to compute
      angles on the fly or with extrapolation (Yu et al. 2023).
    • Multi-query instead of GQA: set num_key_value_heads=1 and simplify
      repeat_kv path.
    • For very long contexts >16 k, swap Flash-Attention with xformers' memory-
      efficient attention, but be aware of datatype / causal-mask API skew.
"""

# ------------------------------------------------------------------------------------------
# Imports
# --------
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel           # Flash-SDPA entry points
from torch.nn.attention.bias import causal_lower_right          # For asymmetric causal masks
from transformers.cache_utils import Cache                     # HF cache wrapper
from transformers.utils import logging

from ..config import ProGen3Config

logger = logging.get_logger(__name__)

# ==========================================================================================
# Helper utilities (verbatim copies from HF Llama code with minimal tweaks)
# ------------------------------------------------------------------------------------------

# NOTE repeat_kv is required for GQA.  If num_key_value_heads == num_heads it is a no-op.


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Repeat key/value heads `n_rep` times to match the query-head count.

    hidden_states : (batch, num_kv_heads, seq_len, head_dim)
    returns       : (batch, num_kv_heads * n_rep, seq_len, head_dim)

    • For multi-query attention n_rep = num_heads.
    • For GQA n_rep = num_query_groups (= num_heads // num_kv_heads).
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states  # Fast path
    # Expand introduces a view (no new memory); reshape flattens the group dim.
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads,
                                                           n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


# --- rotary helpers -----------------------------------------------------------------------


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Standard RoPE helper: (x1,x2)->(-x2,x1)."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


# ==========================================================================================
# Rotary embedding module
# ------------------------------------------------------------------------------------------
class RotaryPositionalEmbedding(nn.Module):
    """
    Implements RoPE as in Su et al. 2021 with cosine caching.

    Differences from HF Llama:
        • `base` (θ) is user-settable via config.rope_theta (ProGen3 uses 10⁵).
        • Handles >max_position_embeddings by enlarging cache on-the-fly.
        • Returns q′, k′ tensors already rotated.
    """

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 2048,
        base: float = 10000,
        device: torch.device | None = None,
    ):
        super().__init__()

        self.dim = dim
        self.base = base
        self.max_position_embeddings = max_position_embeddings

        # Precompute 1/θ^{2i/d} terms.
        inv_freq = base ** -(torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build initial sin/cos cache
        self._set_sin_cos_cache(seq_len=max_position_embeddings, device=self.inv_freq.device)

    # ------------------------------------------------------------------
    def _set_sin_cos_cache(self, seq_len: int, device: torch.device) -> None:
        """(Re)compute sin/cos tables up to `seq_len`."""
        self.max_seq_len_cached = seq_len
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        angles = torch.outer(t, self.inv_freq.to(device))        # (seq, dim/2)
        angles = torch.cat((angles, angles), dim=1)              # duplicate for full dim
        self.register_buffer("cos_cached", angles.cos(), persistent=False)
        self.register_buffer("sin_cached", angles.sin(), persistent=False)

    # ------------------------------------------------------------------
    def forward(
        self,
        q: torch.Tensor,            # (bsz, seq, n_heads, head_dim)
        k: torch.Tensor,
        position_ids: torch.LongTensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply rotary transform to q,k given position_ids.

        • position_ids can be non-monotonic (GLM infilling).
        • Works with fp16/bf16 because cached tables are promoted to q.dtype.
        """
        device, dtype = q.device, q.dtype
        seq_len = position_ids.max().item() + 1
        if seq_len > self.max_seq_len_cached:                    # enlarge cache if needed
            self._set_sin_cos_cache(seq_len=seq_len, device=device)

        idxs = position_ids.to(device)                           # (bsz, seq)
        cos = self.cos_cached.to(device=device, dtype=dtype).unsqueeze(-2)[idxs]
        sin = self.sin_cached.to(device=device, dtype=dtype).unsqueeze(-2)[idxs]

        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)
        return q_embed, k_embed


# ==========================================================================================
# Attention module
# ------------------------------------------------------------------------------------------
class Attention(nn.Module):
    """
    Flash-Attention-2 backed multi-head / GQA module used in ProGen3 blocks.

    Key implementation notes
    ------------------------
    • q_proj / k_proj / v_proj are *bias-less* linear layers (standard in
      large-scale LMs for memory and slight speed gain).
    • num_heads may be 4× num_kv_heads (default: 16 vs 4) implementing GQA.
    • clip_qkv (optional) clamps activations to stabilize fp16 training under
      very deep MoE models (cf. Mistral & DeepSeek-V2).
    • _sdpa_attn(): wraps torch.scaled_dot_product_attention with Flash kernel;
      if you need masking for infilling you may inject `causal_mask`
      trickiness here (current code supports right-padding asymmetry only).
    """

    def __init__(self, config: ProGen3Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx          # needed for cache update order

        # ---------- basic dims ----------
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_kv_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_kv_heads

        # ---------- rope / dropout / clip ----------
        self.max_position_embeddings = config.max_position_embeddings
        self.max_num_seqs = config.max_num_sequences        # used in block­level cache
        self.rope_theta = config.rope_theta
        self.attention_dropout = config.attention_dropout
        self.clip_qkv = config.clip_qkv

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got hidden_size:"
                f" {self.hidden_size}  num_heads: {self.num_heads})."
            )

        # ---------- projections ----------
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        # ---------- rotary embedding ----------
        self.rotary_emb = RotaryPositionalEmbedding(
            dim=self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

    # ======================================================================================
    # QKV preparation (projection, clipping, RoPE, cache update)
    # --------------------------------------------------------------------------------------
    def prepare_qkv(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.LongTensor,
        past_key_value: Cache | None = None,
        use_cache: bool | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns per-head q,k,v with shape
            q : (bsz, seq, n_heads, head_dim)
            k/v : (bsz, seq, n_kv_heads, head_dim)
        After RoPE rotation and (optional) cache concat.
        """
        bsz, q_len, _ = hidden_states.size()
        query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim)
        key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_kv_heads, self.head_dim)
        val_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_kv_heads, self.head_dim)

        # --- optional activation clipping (helps fp16 stability) ---
        if self.clip_qkv is not None:
            query_states = query_states.clamp(-self.clip_qkv, self.clip_qkv)
            key_states = key_states.clamp(-self.clip_qkv, self.clip_qkv)
            val_states = val_states.clamp(-self.clip_qkv, self.clip_qkv)

        # --- apply RoPE ---
        query_states, key_states = self.rotary_emb(query_states, key_states, position_ids)

        # --- KV cache update for autoregressive decoding ---
        if use_cache and past_key_value is not None:
            # HF Cache expects (bsz, n_kv_heads, seq, dim)
            key_states, val_states = key_states.transpose(1, 2), val_states.transpose(1, 2)
            key_states, val_states = past_key_value.update(key_states, val_states, self.layer_idx)
            key_states, val_states = key_states.transpose(1, 2), val_states.transpose(1, 2)

        # --- dtype sanity (handles PEFT up-casting & quantization) ---
        input_dtype = query_states.dtype
        if torch.is_autocast_enabled():
            target_dtype = torch.get_autocast_gpu_dtype()
        elif hasattr(self.config, "_pre_quantization_dtype"):
            target_dtype = self.config._pre_quantization_dtype
        else:
            target_dtype = self.q_proj.weight.dtype

        if input_dtype != target_dtype:
            logger.warning_once(
                f"Input hidden states were cast to {input_dtype}; "
                f"casting back to {target_dtype} for attention matmul."
            )
            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            val_states = val_states.to(target_dtype)

        return query_states, key_states, val_states

    # ======================================================================================
    # Public forward
    # --------------------------------------------------------------------------------------
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.LongTensor,
        past_key_value: Cache | None = None,
        output_attentions: bool | None = None,
        use_cache: bool | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, Cache | None]:
        """
        Compute attention and return:
            attn_output  : (bsz, seq, hidden)
            attn_weights : None (weights disabled to save memory; see _attn)
            past_key_value (possibly updated)
        """
        query_states, key_states, val_states = self.prepare_qkv(
            hidden_states, position_ids, past_key_value, use_cache
        )

        attn_output, attn_weights = self._attn(
            query_states, key_states, val_states, output_attentions
        )
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights, past_key_value

    # ======================================================================================
    # Attention computation wrappers
    # --------------------------------------------------------------------------------------
    def _attn(  # noqa: D401
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        val_states: torch.Tensor,
        output_attentions: bool | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Delegates to SDPA / Flash.  output_attentions is disabled for speed.
        """
        assert not output_attentions, "output_attentions not supported"
        return self._sdpa_attn(query_states, key_states, val_states)

    # ------------------------------------------------------------------
    def _sdpa_attn(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        val_states: torch.Tensor,
    ) -> tuple[torch.Tensor, None]:
        """
        FlashAttention2 via torch.scaled_dot_product_attention.
        Handles GQA by repeating KV heads.
        """

        # PyTorch SDPA expects (bsz, heads, seq, dim)
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        val_states = val_states.transpose(1, 2)

        # --- repeat KV to match query head count (GQA) ---
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        val_states = repeat_kv(val_states, self.num_key_value_groups)

        bsz, q_len = query_states.shape[0], query_states.shape[2]
        k_len = key_states.shape[2]

        # --- asymmetric causal mask for prefix-padding (GLM decode) ---
        causal_mask = None
        if k_len > q_len:
            causal_mask = causal_lower_right(q_len, k_len)
        elif k_len < q_len:
            raise ValueError("k_len must be >= q_len")

        with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
            attn_output = F.scaled_dot_product_attention(
                query_states,
                key_states,
                val_states,
                is_causal=causal_mask is None,
                attn_mask=causal_mask,
            )

        # reshape back to (bsz, seq, hidden)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        return attn_output, None        # weights intentionally dropped


# End of file
# ==========================================================================================
