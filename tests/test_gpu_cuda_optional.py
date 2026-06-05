import importlib.util
import unittest


def _torch_or_skip():
    if importlib.util.find_spec("torch") is None:
        raise unittest.SkipTest("torch is not installed")
    import torch

    return torch


class OptionalCudaTest(unittest.TestCase):
    def setUp(self) -> None:
        self.torch = _torch_or_skip()
        if not self.torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is not available")

    def test_cuda_device_metadata_is_readable(self) -> None:
        major, minor = self.torch.cuda.get_device_capability()
        name = self.torch.cuda.get_device_name(0)
        props = self.torch.cuda.get_device_properties(0)
        self.assertGreaterEqual(major, 1)
        self.assertGreaterEqual(minor, 0)
        self.assertTrue(name)
        self.assertGreater(props.total_memory, 0)

    def test_small_cuda_matmul_matches_cpu(self) -> None:
        torch = self.torch
        torch.manual_seed(0)
        a_cpu = torch.randn(32, 32, dtype=torch.float32)
        b_cpu = torch.randn(32, 32, dtype=torch.float32)
        expected = a_cpu @ b_cpu
        got = (a_cpu.cuda() @ b_cpu.cuda()).cpu()
        self.assertTrue(torch.allclose(got, expected, atol=1e-4, rtol=1e-4))

    def test_cuda_memory_counters_increase_after_allocation(self) -> None:
        torch = self.torch
        torch.cuda.empty_cache()
        before = torch.cuda.memory_allocated()
        tensor = torch.empty((1024, 1024), device="cuda", dtype=torch.float32)
        after = torch.cuda.memory_allocated()
        self.assertGreater(after, before)
        del tensor
        torch.cuda.empty_cache()

    def test_bfloat16_tensor_roundtrip_when_supported(self) -> None:
        torch = self.torch
        if not torch.cuda.is_bf16_supported():
            raise unittest.SkipTest("CUDA device does not report BF16 support")
        x = torch.ones((16, 16), device="cuda", dtype=torch.bfloat16)
        y = (x + x).float().cpu()
        self.assertTrue(torch.all(y == 2.0))


class OptionalSGLangImportTest(unittest.TestCase):
    def test_sglang_import_when_installed(self) -> None:
        if importlib.util.find_spec("sglang") is None:
            raise unittest.SkipTest("sglang is not installed")
        import sglang

        self.assertTrue(getattr(sglang, "__file__", None))

    def test_flashinfer_import_when_installed(self) -> None:
        if importlib.util.find_spec("flashinfer") is None:
            raise unittest.SkipTest("flashinfer is not installed")
        import flashinfer

        self.assertTrue(getattr(flashinfer, "__file__", None))


if __name__ == "__main__":
    unittest.main()
