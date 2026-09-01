#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


MARKER = "Fence draft input-buffer writes before the next step consumes them"
FENCE = (
    "        # Fence draft input-buffer writes before the next step consumes them.\n"
    "        # wtdcode/vllm-backport@0439b310 (vllm#40756) traced sporadic\n"
    "        # MTP illegal-memory-access failures on SM86-SM121 to this race.\n"
    "        # Synchronization is illegal during CUDA graph capture and is not\n"
    "        # needed inside a captured graph, where replay ordering is fixed.\n"
    "        if not torch.cuda.is_current_stream_capturing():\n"
    "            torch.accelerator.current_stream().synchronize()\n"
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vllm-root", required=True)
    args = ap.parse_args()

    path = (
        Path(args.vllm_root).resolve()
        / "v1/worker/gpu/spec_decode/autoregressive/speculator.py"
    )
    if not path.exists():
        raise RuntimeError(f"missing autoregressive speculator: {path}")

    text = path.read_text(encoding="utf-8")
    if MARKER in text:
        if text.count("torch.accelerator.current_stream().synchronize()") != 2:
            raise RuntimeError("MTP fence marker present but expected two stream fences")
        print(f"[mtp-stream-fence] already patched: {path}")
        return

    prefill_anchor = (
        "        self.input_buffers.positions[:num_reqs] = positions\n"
        "\n"
        "    def _multi_step_decode(\n"
    )
    if text.count(prefill_anchor) != 1:
        raise RuntimeError(
            f"MTP prefill fence anchor expected once, found {text.count(prefill_anchor)}"
        )
    text = text.replace(
        prefill_anchor,
        "        self.input_buffers.positions[:num_reqs] = positions\n"
        + FENCE
        + "\n"
        + "    def _multi_step_decode(\n",
        1,
    )

    decode_anchor = (
        "            advance_draft_positions=self.advance_draft_positions,\n"
        "        )\n"
        "\n"
        "\n"
        "@triton.jit\n"
    )
    if text.count(decode_anchor) != 1:
        raise RuntimeError(
            f"MTP decode fence anchor expected once, found {text.count(decode_anchor)}"
        )
    decode_fence = (
        "            advance_draft_positions=self.advance_draft_positions,\n"
        "        )\n"
        "        # Same fence after update_draft_inputs for non-captured MTP steps.\n"
        "        if not torch.cuda.is_current_stream_capturing():\n"
        "            torch.accelerator.current_stream().synchronize()\n"
        "\n"
        "\n"
        "@triton.jit\n"
    )
    text = text.replace(decode_anchor, decode_fence, 1)

    if text.count("torch.accelerator.current_stream().synchronize()") != 2:
        raise RuntimeError("expected exactly two MTP stream fences after patch")
    if text.count("if not torch.cuda.is_current_stream_capturing():") < 2:
        raise RuntimeError("MTP stream fences must be CUDA-graph-capture safe")

    path.write_text(text, encoding="utf-8")
    print(f"[mtp-stream-fence] patched and verified: {path}")


if __name__ == "__main__":
    main()
