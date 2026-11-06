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
    row_range = tl.arange(0, B_r)[:, None] * d + tl.arange(0, d)[None, :]
    col_range = tl.arange(0, B_c)[:, None] * d + tl.arange(0, d)[None, :]

    # iterate over cols (K and V)
    for j in range(0, N, B_c):
        pass

def flash_attention_forward(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    """Return attention output with the same shape as q.

    Expected initial contract: q, k, v have shape (B, H, N, D), share a dtype,
    and are contiguous CUDA tensors. Implement input checks, allocation, and
    the Triton launch here once the kernel's interface is decided.
    """
    raise NotImplementedError("FlashAttention forward kernel is not implemented yet")
