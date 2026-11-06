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

    # paper says to initialize full tensors, but we'll assume those get passed in
    # these will be used to process within the tiles
    O_i = tl.full((B_r, d), 0.0, tl.float32)
    l_i = tl.full((B_r,), 0.0, tl.float32)
    m_i = tl.full((B_r,), float("-inf"), tl.float32)

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

            s_ij = tl.dot(qt, kt.T) * sm_scale
            m_ij = tl.max(s_ij, axis = 1)
            p_ij = tl.exp(s_ij - m_ij[:, None])
            l_ij = tl.sum(p_ij, axis = 1)

def flash_attention_forward(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    """Return attention output with the same shape as q.

    Expected initial contract: q, k, v have shape (B, H, N, D), share a dtype,
    and are contiguous CUDA tensors. Implement input checks, allocation, and
    the Triton launch here once the kernel's interface is decided.
    """
    raise NotImplementedError("FlashAttention forward kernel is not implemented yet")
