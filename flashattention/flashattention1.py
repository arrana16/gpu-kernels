"""FlashAttention v1 forward pass, using (batch, heads, sequence, head_dim) tensors.

Start with equal query/key sequence lengths, one dtype, and contiguous CUDA inputs.
Add other layouts, causal masking, and backward support after the forward path works.
"""

import torch
import triton
import triton.language as tl

@triton.jit


def flash_attention_forward(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    """Return attention output with the same shape as q.

    Expected initial contract: q, k, v have shape (B, H, N, D), share a dtype,
    and are contiguous CUDA tensors. Implement input checks, allocation, and
    the Triton launch here once the kernel's interface is decided.
    """
    raise NotImplementedError("FlashAttention forward kernel is not implemented yet")
