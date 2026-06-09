import ast
import atexit
import os
import queue
import tempfile
import threading
import unittest
from pathlib import Path
from typing import Optional

import torch


def _load_async_kv_dump_writer():
    source_path = (
        Path(__file__).resolve().parents[1]
        / "third_party/OSCAR/sglang-dump-qkv/python/sglang/srt/layers/attention/triton_backend.py"
    )
    if not source_path.is_file():
        raise unittest.SkipTest("dump-qkv triton_backend.py is not available")

    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(source_path))
    class_node = next(
        (
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "_AsyncKvDumpWriter"
        ),
        None,
    )
    if class_node is None:
        raise unittest.SkipTest("_AsyncKvDumpWriter is not available")

    namespace = {
        "os": os,
        "queue": queue,
        "threading": threading,
        "atexit": atexit,
        "torch": torch,
        "get_int_env_var": lambda name, default: int(os.environ.get(name, default)),
        "Optional": Optional,
    }
    module = ast.Module(body=[class_node], type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, str(source_path), "exec"), namespace)
    return namespace["_AsyncKvDumpWriter"]


class AsyncKvDumpWriterTest(unittest.TestCase):
    def test_writer_persists_chunk_payloads(self) -> None:
        writer_cls = _load_async_kv_dump_writer()
        writer = writer_cls()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                save_dir = Path(tmp)
                torch.manual_seed(0)
                q = torch.randn(2, 3, 4)
                k = torch.randn(2, 1, 4)
                v = torch.randn(2, 1, 4)
                seq = torch.tensor([2], dtype=torch.int32)

                writer.submit(
                    save_dir=str(save_dir),
                    layer_id=7,
                    chunk_idx=0,
                    q_dump=q,
                    k_dump=k,
                    v_dump=v,
                    chunk_seq_lens_t=seq,
                    event=None,
                )
                writer.close()

                self.assertTrue((save_dir / "layer_7" / "q" / "0.pt").is_file())
                self.assertTrue((save_dir / "layer_7" / "k" / "0.pt").is_file())
                self.assertTrue((save_dir / "layer_7" / "v" / "0.pt").is_file())
                self.assertTrue(
                    (save_dir / "layer_7" / "seq_lens" / "0.pt").is_file()
                )

                loaded_q = torch.load(
                    save_dir / "layer_7" / "q" / "0.pt",
                    weights_only=True,
                    map_location="cpu",
                )
                loaded_seq = torch.load(
                    save_dir / "layer_7" / "seq_lens" / "0.pt",
                    weights_only=True,
                    map_location="cpu",
                )
                self.assertTrue(torch.equal(loaded_q, q))
                self.assertTrue(torch.equal(loaded_seq, seq))
        finally:
            writer.close()


if __name__ == "__main__":
    unittest.main()
