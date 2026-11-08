"""FlashAttention v1 forward pass, using (batch, heads, sequence, head_dim) tensors.

Start with equal query/key sequence lengths, one dtype, and contiguous CUDA inputs.
Add other layouts, causal masking, and backward support after the forward path works.
"""

import torch
import triton
import triton.language as tl

@triton.jit
def _flash_attention_forward_kernel(
    Q, K, V, O,
    N: tl.constexpr,
    d: tl.constexpr,
    B_r: tl.constexpr,
    B_c: tl.constexpr,
    sm_scale,
):
    row_block_index = tl.program_id(0)
    row_range = tl.arange(0, B_r)[:, None] * d + tl.arange(0, d)[None, :] + row_block_index * B_r * d
    col_range = tl.arange(0, B_c)[:, None] * d + tl.arange(0, d)[None, :]

    m_i = tl.full((B_r,), float("-inf"), tl.float32)
    l_i = tl.zeros((B_r,), tl.float32)
    o_i = tl.zeros((B_r, d), tl.float32)

    # load the q
    q_i = tl.load(Q + row_range, (row_block_index * B_r + tl.arange(0, B_r))[:, None] < N, 0.0)

    # iterate over cols (K and V)
    for j in range(0, N, B_c):
        # j gives us the column title to look at
        # col_range gives us block dimensions
        # offset col_range with current tile index
        curr_col_range = col_range + j * d
        kt = tl.load(K + curr_col_range, (j + tl.arange(0, B_c))[:, None] < N, 0.0)
        vt = tl.load(V + curr_col_range, (j + tl.arange(0, B_c))[:, None] < N, 0.0)

        s_ij = tl.dot(q_i, kt.T) * sm_scale
        s_ij = tl.where((j+tl.arange(0, B_c))[None, :] < N, s_ij, float("-inf"))

        m_ij = tl.max(s_ij, axis = 1)
        p_ij = tl.exp(s_ij - m_ij[:, None])
        l_ij = tl.sum(p_ij, axis = 1)

        m_i_new = tl.maximum(m_i, m_ij)
        l_i_new = tl.exp(m_i-m_i_new) * l_i + tl.exp(m_ij - m_i_new) * l_ij
        o_i_new = ((l_i * tl.exp(m_i-m_i_new))[:, None]*o_i + tl.exp(m_ij - m_i_new)[:, None] * tl.dot(p_ij.to(vt.dtype), vt))/l_i_new[:, None]

        m_i = m_i_new
        l_i = l_i_new
        o_i = o_i_new

    tl.store(O + row_range, o_i, (row_block_index * B_r + tl.arange(0, B_r))[:, None] < N)


def flash_attention_forward(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    *, M: int = 32768,
) -> torch.Tensor:
    """Return attention output with the same shape as q.

    Expected initial contract: q, k, v have shape (B, H, N, D), share a dtype,
    and are contiguous CUDA tensors. M is a chosen on-chip memory budget in
    elements, following the paper's abstract memory model; its default is
    illustrative, not a measured hardware limit.
    """
    if any(x.ndim != 4 for x in (q, k, v)):
        raise ValueError("q, k, v must have shape (B, H, N, d)")
    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError("q, k, v must have identical shapes")
    if not all(x.is_cuda for x in (q, k, v)):
        raise ValueError("q, k, v must be CUDA tensors")
    if q.device != k.device or q.device != v.device:
        raise ValueError("q, k, v must be on the same device")
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise ValueError("q, k, v must have the same dtype")
    if q.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("This kernel supports float16 and bfloat16 inputs")
    if not all(x.is_contiguous() for x in (q, k, v)):
        raise ValueError("q, k, v must be contiguous")
    B, H, N, d = q.shape
    if min(B, H, N) <= 0:
        raise ValueError("Batch size, head count, and sequence length must be positive")
    if d < 16 or d & (d - 1):
        raise ValueError("This kernel requires a power-of-two d >= 16")
    if not isinstance(M, int) or M <= 0:
        raise ValueError("M must be a positive integer budget in elements")

    # Algorithm 1: B_c = ceil(M / (4d)), B_r = min(B_c, d).
    B_c = triton.cdiv(M, 4 * d)
    B_r = min(B_c, d)

    # Adapt the paper's sizes to power-of-two tl.arange dimensions.
    B_c = 1 << (B_c.bit_length() - 1)
    B_r = 1 << (B_r.bit_length() - 1)
    if min(B_r, B_c) < 16:
        raise ValueError("M is too small for this kernel's dot-product tiles")

    sm_scale = d ** -0.5

    with torch.cuda.device(q.device):
        # Running softmax state is local to each program.
        O = torch.zeros(q.shape, device=q.device, dtype=torch.float32)

        # Each program owns one Q tile and iterates over all K/V tiles.
        grid = (triton.cdiv(N, B_r),)
        for b in range(B):
            for h in range(H):
                _flash_attention_forward_kernel[grid](
                    q[b, h], k[b, h], v[b, h],
                    O[b, h],
                    N=N, d=d, B_r=B_r, B_c=B_c, sm_scale=sm_scale,
                )

        return O.to(q.dtype)
