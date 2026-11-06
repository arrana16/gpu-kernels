"""FlashAttention v1 forward pass, using (batch, heads, sequence, head_dim) tensors.

Start with equal query/key sequence lengths, one dtype, and contiguous CUDA inputs.
Add other layouts, causal masking, and backward support after the forward path works.
"""

import torch
import triton
import triton.language as tl

@triton.jit
def _flash_attention_forward_kernel(
    Q, K, V, O, l, m,
    N: tl.constexpr,
    d: tl.constexpr,
    B_r: tl.constexpr,
    B_c: tl.constexpr,
    sm_scale,
):
    row_range = tl.arange(0, B_r)[:, None] * d + tl.arange(0, d)[None, :]
    col_range = tl.arange(0, B_c)[:, None] * d + tl.arange(0, d)[None, :]

    # iterate over cols (K and V)
    for j in range(0, N, B_c):
        # j gives us the column title to look at
        # col_range gives us block dimensions
        # offset col_range with current tile index
        curr_col_range = col_range + j * d
        kt = tl.load(K + curr_col_range, (j + tl.arange(0, B_c))[:, None] < N, 0.0)
        vt = tl.load(V + curr_col_range, (j + tl.arange(0, B_c))[:, None] < N, 0.0)

        # loop over Q tiles
        for i in range(0, N, B_r):
            curr_row_range = row_range + i * d
            qt = tl.load(Q + curr_row_range, (i + tl.arange(0, B_r))[:, None] < N, 0.0)
            o_i = tl.load(O + curr_row_range, (i + tl.arange(0, B_r))[:, None] < N, 0.0)

            rows = i + tl.arange(0, B_r)
            l_i = tl.load(l + rows, (i + tl.arange(0, B_r)) < N)
            m_i = tl.load(m + rows, (i + tl.arange(0, B_r)) < N)

            s_ij = tl.dot(qt, kt.T) * sm_scale
            m_ij = tl.max(s_ij, axis = 1)
            p_ij = tl.exp(s_ij - m_ij[:, None])
            l_ij = tl.sum(p_ij, axis = 1)

            m_i_new = tl.maximum(m_i, m_ij)
            l_i_new = tl.exp(m_i-m_i_new) * l_i + tl.exp(m_ij - m_i_new) * l_ij
            o_i_n = (l_i * tl.exp(m_i-m_i_new)*o_i + tl.exp(m_ij - m_i_new) * p_ij * vt)/l_i_new

            tl.store(O + curr_row_range, o_i_n, (i + tl.arange(0, B_r))[:, None] < N)
            tl.store(l + rows, l_i_new, (i + tl.arange(0, B_r)) < N)
            tl.store(m + rows, m_i_new, (i + tl.arange(0, B_r)) < N)


def flash_attention_forward(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    """Return attention output with the same shape as q.

    Expected initial contract: q, k, v have shape (B, H, N, D), share a dtype,
    and are contiguous CUDA tensors. Implement input checks, allocation, and
    the Triton launch here once the kernel's interface is decided.
    """
    raise NotImplementedError("FlashAttention forward kernel is not implemented yet")
