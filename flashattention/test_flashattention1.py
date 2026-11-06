"""Run from the repository root: python -m unittest flashattention.test_flashattention1 -v."""

import os
import unittest

import torch

from flashattention.flashattention1 import flash_attention_forward


def naive_attention(q, k, v):
    """Explicitly materialize scores and probabilities, using float32 math."""
    scores = (q.float() @ k.float().transpose(-2, -1)) * q.shape[-1] ** -0.5
    probabilities = torch.softmax(scores, dim=-1)
    return (probabilities @ v.float()).to(q.dtype)


@unittest.skipUnless(torch.cuda.is_available(), "Requires a CUDA GPU")
class FlashAttentionForwardTest(unittest.TestCase):
    def check_attention(self, q, k, v):
        with torch.no_grad():
            expected = naive_attention(q, k, v)
            actual = flash_attention_forward(q, k, v)

        self.assertEqual(actual.shape, q.shape)
        self.assertEqual(actual.dtype, q.dtype)
        self.assertEqual(actual.device, q.device)
        self.assertTrue(torch.isfinite(actual).all().item())
        # Softmax weights are cast to the input dtype before the kernel's
        # P @ V product; allow the resulting FP16/BF16 rounding error.
        tolerance = 2e-2 if q.dtype == torch.bfloat16 else 2e-3
        torch.testing.assert_close(
            actual, expected, atol=tolerance, rtol=tolerance,
        )

    def test_random_inputs(self):
        torch.manual_seed(0)
        for dtype in (torch.float16, torch.bfloat16):
            if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
                continue
            # Include partial Q/K tiles, multiple K/V iterations, and
            # multiple batch/head pairs to check independent output state.
            for shape in ((1, 1, 17, 32), (1, 1, 128, 64),
                          (2, 3, 257, 64), (1, 2, 129, 128)):
                with self.subTest(dtype=dtype, shape=shape):
                    q, k, v = [
                        torch.randn(shape, device="cuda", dtype=dtype)
                        for _ in range(3)
                    ]
                    self.check_attention(q, k, v)

    def test_uniform_attention(self):
        # Zero queries give equal weights over all valid keys. In particular,
        # padding in the last K/V tile must not enter the denominator.
        torch.manual_seed(1)
        shape = (1, 2, 257, 64)
        q = torch.zeros(shape, device="cuda", dtype=torch.float16)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        self.check_attention(q, k, v)

    def test_input_magnitudes(self):
        torch.manual_seed(2)
        for dtype in (torch.float16, torch.bfloat16):
            if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
                continue
            # Scale Q/K to stress the stable softmax without also scaling
            # the output tolerance through very large V values.
            for scale in (1e-3, 1.0, 10.0, 100.0):
                with self.subTest(dtype=dtype, scale=scale):
                    shape = (1, 2, 257, 64)
                    q = torch.randn(shape, device="cuda", dtype=dtype) * scale
                    k = torch.randn(shape, device="cuda", dtype=dtype) * scale
                    v = torch.randn(shape, device="cuda", dtype=dtype)
                    self.check_attention(q, k, v)

    @unittest.skipUnless(
        os.environ.get("FLASH_ATTENTION_STRESS") == "1",
        "Set FLASH_ATTENTION_STRESS=1 to enable the large-tensor sweep",
    )
    def test_large_tensors(self):
        torch.manual_seed(3)
        max_n = int(os.environ.get("FLASH_ATTENTION_MAX_N", "8192"))
        if max_n < 512:
            raise ValueError("FLASH_ATTENTION_MAX_N must be at least 512")
        N = 512
        while N <= max_n:
            shape = (1, 8, N, 64)
            free_bytes, _ = torch.cuda.mem_get_info()
            # Conservative estimate: three FP32 attention matrices plus
            # inputs, outputs, running state and reference intermediates.
            estimated_bytes = 3 * 4 * 8 * N * N + 64 * 8 * N * 64
            if estimated_bytes > 0.7 * free_bytes:
                print(f"\nStopping before N={N}: estimated memory exceeds "
                      "70% of currently free VRAM", flush=True)
                break
            with self.subTest(shape=shape):
                print(f"\nComparing shape={shape}", flush=True)
                torch.cuda.reset_peak_memory_stats()
                q, k, v = [
                    torch.randn(shape, device="cuda", dtype=torch.float16)
                    for _ in range(3)
                ]
                try:
                    self.check_attention(q, k, v)
                    torch.cuda.synchronize()
                    peak_mib = torch.cuda.max_memory_allocated() / 2**20
                    print(f"Passed; peak PyTorch allocation: {peak_mib:.1f} MiB",
                          flush=True)
                finally:
                    del q, k, v
            N *= 2


if __name__ == "__main__":
    unittest.main()
