#!/usr/bin/env python3
"""Backport vLLM PR #47644 onto glm53-flash-cu129 @ 487ecf187.

The day-0 GLM image already uses a blocking CUDA event under async scheduling.
Preserve that behavior and only widen event creation to every configuration
with overlapping engine batches (notably PP), i.e. max_concurrent_batches > 1.
"""
from __future__ import annotations

import argparse
from pathlib import Path


def patch(text: str) -> str:
    marker = "if self.vllm_config.max_concurrent_batches > 1:"
    if marker in text:
        return text

    old = '''        # Separate cuda stream for overlapping transfer of sampled token ids from
        # GPU to CPU when async scheduling is enabled.
        self.async_output_copy_stream: torch.cuda.Stream | None = None
        # cuda event to synchronize use of reused CPU tensors between steps
        # when async scheduling is enabled.
        self.prepare_inputs_event: torch.Event | None = None
        if self.use_async_scheduling:
            self.async_output_copy_stream = torch.cuda.Stream()
            # Blocking (sleep) event to avoid busy-polling the CUDA driver lock;
            # under TP contention that spin can balloon and make the rank a straggler.
            self.prepare_inputs_event = torch.cuda.Event(blocking=True)
'''
    new = '''        # Separate cuda stream for overlapping transfer of sampled token ids from
        # GPU to CPU when async scheduling is enabled.
        self.async_output_copy_stream: torch.cuda.Stream | None = None
        if self.use_async_scheduling:
            self.async_output_copy_stream = torch.cuda.Stream()

        # Synchronize reused pinned CPU input tensors whenever engine steps can
        # overlap. 487ecf187 already used a blocking event for async scheduling;
        # retain that behavior while extending the guard to the PP batch queue.
        self.prepare_inputs_event: torch.Event | None = None
        if self.vllm_config.max_concurrent_batches > 1:
            self.prepare_inputs_event = torch.cuda.Event(blocking=True)
'''
    n = text.count(old)
    if n != 1:
        raise RuntimeError(
            f"#47644 glm53-cu129 anchor mismatch: expected one block, found {n}"
        )
    return text.replace(old, new, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vllm-root", required=True)
    args = ap.parse_args()
    root = Path(args.vllm_root).resolve()
    p = root / "v1/worker/gpu_model_runner.py"
    src = p.read_text(encoding="utf-8")
    out = patch(src)
    p.write_text(out, encoding="utf-8")
    print(f"[#47644] patched {p}")


if __name__ == "__main__":
    main()
