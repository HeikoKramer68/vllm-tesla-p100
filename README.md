# vllm-tesla-p100

**vLLM for NVIDIA Tesla P100 (SM 6.0).** Extends the Pascal/P40 fork by
[uaysk](https://github.com/uaysk/vllm-pascal) to support the Tesla P100
(GP100), with a complete benchmark matrix covering 2×/6× P100 and 4× RTX
3090.

> ⚠️ **Draft / work in progress.** This README is an initial draft and
> will be revised further. Verify values, commands, and notes before
> production use.

------------------------------------------------------------------------

## What is this?

vLLM officially runs on Volta (SM 7.0) and newer. There is no official
Pascal support. uaysk ported vLLM to the **Tesla P40** (GP102, SM 6.1);
this fork additionally brings it to the **Tesla P100** (GP100, SM 6.0)
and documents its behavior with reproducible benchmarks.

The core modification: Pascal has no native bf16/FP8 units. At the
affected locations (`cast_to_float2`, `silu2_v2`, `warp_max`, `clip`),
`#if __CUDA_ARCH__ >= 800` guards introduce fallback paths that use
conversion through `float` on Pascal instead of unavailable intrinsics.

------------------------------------------------------------------------

## Origin & Attribution

This fork builds on two layers of prior work:

-   **vLLM** (base, Apache 2.0) -- commit `36e14f2c1` (2026-07-05, CUDA
    12.1)
-   **uaysk/vllm-pascal** -- Pascal/P40 support (SM 6.1), including bf16
    packing fix
-   **This fork** -- Tesla P100 compatibility (SM 6.0) + benchmarks

See [`NOTICE`](./NOTICE) for details. Modified CUDA files contain the
corresponding headers. License: **Apache 2.0** (unchanged from vLLM).

------------------------------------------------------------------------

## Benchmark Matrix

All values use the same prompt, `temperature=0.0`, `max_tokens=512`.
tok/s = decode throughput (generation), unless otherwise stated.

  Model   Quant   Hardware   Configuration      tok/s
  ------- ------- ---------- ------------------ --------------
  14B     INT8    2× P100    TP2, eager         12.4--13.4 tks/
  14B     FP16    4× P100    TP4, eager         13.6 tks/
  14B     FP16    2× P100    torch, Pipeline    7.8 tks/
  14B     INT8    4× 3090    TP4, CUDA Graphs   68.1 tks/
  14B     FP16    4× 3090    TP4, CUDA Graphs   57tks/
  32B     FP16    6× P100    TP2/PP3, eager     6.7 tks/
  32B     FP16    4× 3090    TP4, CUDA Graphs   32.7 tks/
  32B     FP16    4× 3090    TP4, eager         12.1 tks/
  32B     INT8    4× 3090    TP4, CUDA Graphs   47.1 tks/
  72B     INT8    4× 3090    TP4, CUDA Graphs   26 tks
  72B     INT8    6× P100    TP2/PP3, eager     7.13 tks

Models: Qwen2.5-Instruct (14B/32B/72B), INT8 using GPTQ.

**Note:** 14B FP16 does **not** run under vLLM on 2× P100 (the KV cache
does not fit in memory). Use INT8 on this hardware, or FP16 with at
least 4 cards.

------------------------------------------------------------------------


## Installation

Tested on **Ubuntu 24.04, CUDA 12.1, PyTorch 2.5.1**.

> **Driver note:** Install the CUDA toolkit, but **deselect the driver
> bundled with the toolkit**. Keep the distribution-provided driver
> instead (for example, 535). Newer drivers may restrict Pascal support.
> The toolkit provides the compiler and libraries; the distro driver
> provides the runtime.

``` bash
# 1. Clone the repository and enter it
git clone https://github.com/HeikoKramer68/vllm-tesla-p100.git
cd vllm-tesla-p100

# 2. Create and activate a virtual environment
python3.12 -m venv venv
source venv/bin/activate

# 3. System build tools
sudo apt update
sudo apt install -y cmake ninja-build python3.12-dev gcc-12 g++-12

# 4. Set the build environment (CUDA path, compiler, target architecture SM 6.0)
source build-env.sh

# 5. Python build tools
python -m pip install --upgrade pip setuptools wheel setuptools_scm

# 6. Use the existing PyTorch installation for the build
python use_existing_torch.py

# 7. Remove old build artifacts (if present)
rm -rf build .deps

# 8. Compile and install the fork in editable mode
python -m pip install -v -e . --no-build-isolation
```

The central Pascal setting is defined in
[`build-env.sh`](./build-env.sh): `TORCH_CUDA_ARCH_LIST="6.0"`. Without
it, the build targets the wrong architecture.

> **TODO (author verification):** Confirm whether `VLLM_USE_V1=0` must
> be set at runtime (old engine), and document it here.

------------------------------------------------------------------------

## Configuration Notes

When running on multiple Pascal cards, keep the following in mind:

-   **Tensor Parallel must divide the number of attention heads.**
    Qwen2.5 has 40 heads → not divisible by 6. For 6 cards, use
    `TP=2 / PP=3` or `TP=5`, not `TP=6`.
-   **Pipeline Parallel works only through the API server**, not
    directly through the `LLM()` object.
-   **KV-cache overflow ≠ hallucination.** If output degrades into
    repetition or word salad after a certain length, `max_model_len` is
    too small; this is not a model failure. Set `max_model_len`
    significantly larger than (prompt + max_tokens), with additional
    headroom.
-   **Mount RAID via fstab/UUID**, not through `/media` automount, to
    avoid path inconsistencies.

------------------------------------------------------------------------

## Startup Messages That Can Be Ignored

These messages may appear at startup and can be **ignored**; the server
runs normally:

``` text
ERROR ... Failed to import Triton kernels ... No module named 'triton.language.target_info'
```

→ Affects MoE kernels (for example GPT-OSS). Irrelevant for dense models
such as Qwen2.5.

``` text
SymmMemCommunicator: Device capability 6.0 not supported
```

→ Pascal does not support this communication feature; vLLM uses a
fallback.

``` text
Custom allreduce is disabled ... PCIe-only GPUs
```

→ No NVLink; the normal communication path is used. Expected behavior.

------------------------------------------------------------------------

## Hardware

Tested on Dell R720 with:

-   2× / 6× Tesla P100-PCIE-16GB (passively cooled, with attached fans)
-   4× RTX 3090 (comparison system)

Build notes:

-   Risers with x16 connectivity (with 4 or more cards, slots drop to
    x8)
-   Symmetrical distribution across both CPU sockets
-   Passive P100 cards require active airflow (attached fans are
    sufficient)

*(Photos to follow: the rig and a run under load with nvidia-smi.)*

------------------------------------------------------------------------

## Why This Exists

Local AI inference does not have to start with the most expensive
hardware. A used Tesla P100 costs a fraction of a current-generation
GPU. For anyone looking for a local entry point for a few hundred euros,
this project provides a way to run vLLM at all. It is not intended as a
replacement for modern hardware, but as an open door for people who
would otherwise be left out.

------------------------------------------------------------------------

## Contributing

The biggest open issue is [Issue #1](../../issues/1) (eager mode /
Triton). If you have Pascal hardware, reproduce the build, or make
progress on the Triton issue, PRs and feedback are welcome.

Contact: `pascal.lives26@gmail.com`

------------------------------------------------------------------------

*Built on the work of [uaysk](https://github.com/uaysk/vllm-pascal) and
the [vLLM project](https://github.com/vllm-project/vllm).*

---

## Example Server: Qwen2.5-72B GPTQ INT8 on 6× Tesla P100

This example starts the OpenAI-compatible vLLM API server on six Tesla P100
cards using `TP=2 / PP=3`. It is the same basic configuration used for the
72B GPTQ INT8 tests. Only change the model path to match your local setup.

Save as `server.py`:

```python
import os
import subprocess
import sys

os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ.pop("VLLM_USE_V1", None)

cmd = [
    sys.executable,
    "-m",
    "vllm.entrypoints.openai.api_server",

    "--model", "/path/to/Qwen2.5-72B-Instruct-GPTQ-Int8",
    "--served-model-name", "qwen72b",

    "--quantization", "gptq",
    "--dtype", "half",

    "--tensor-parallel-size", "2",
    "--pipeline-parallel-size", "3",

    "--max-model-len", "4096",
    "--gpu-memory-utilization", "0.95",

    "--enforce-eager",
    "--disable-custom-all-reduce",

    "--host", "0.0.0.0",
    "--port", "8000",
]

subprocess.run(cmd, check=True)
```

Start it with:

```bash
python server.py
```

---

## Simple Streaming Test Client

This client connects to the OpenAI-compatible API endpoint, streams the model
response directly to the terminal, and measures basic performance data. It
reports time to first token (TTFT), decode time, total time, and decode
tokens per second. This makes it useful both as a minimal chat client and as a
quick sanity/performance test after installation.

Install the only additional dependency:

```bash
python -m pip install requests
```

Save the following as `client.py`:

```python
import json
import time
from typing import Optional

import requests


URL = "http://127.0.0.1:8000/v1/chat/completions"
MODEL = "qwen72b"

MAX_TOKENS = 512
TEMPERATURE = 0.0
TOP_P = 0.9

REQUEST_TIMEOUT = 600


def ask_model(prompt: str) -> None:
    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "stream": True,
    }

    start_time = time.perf_counter()
    first_token_time: Optional[float] = None

    text_parts: list[str] = []
    stream_chunks = 0
    usage_completion_tokens: Optional[int] = None
    finish_reason: Optional[str] = None

    try:
        with requests.post(
            URL,
            json=payload,
            stream=True,
            timeout=REQUEST_TIMEOUT,
        ) as response:

            if not response.ok:
                print("\nServer error:")
                print(f"HTTP {response.status_code}")
                print(response.text)
                return

            print("\nResponse:\n")

            for raw_line in response.iter_lines(decode_unicode=True):
                if not raw_line:
                    continue

                line = raw_line.strip()

                if line.startswith("data:"):
                    line = line[5:].strip()

                if line == "[DONE]":
                    break

                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if "error" in chunk:
                    print("\nAPI error:")
                    print(json.dumps(chunk["error"], indent=2, ensure_ascii=False))
                    return

                choices = chunk.get("choices", [])

                if choices:
                    choice = choices[0]
                    delta = choice.get("delta", {})

                    token_text = delta.get("content", "")

                    if token_text:
                        now = time.perf_counter()

                        if first_token_time is None:
                            first_token_time = now

                        stream_chunks += 1
                        text_parts.append(token_text)

                        print(token_text, end="", flush=True)

                    if choice.get("finish_reason") is not None:
                        finish_reason = choice["finish_reason"]

                usage = chunk.get("usage")

                if usage:
                    usage_completion_tokens = usage.get("completion_tokens")

    except requests.exceptions.ConnectionError:
        print("\n\nCould not connect to the server.")
        print(f"Checked address: {URL}")
        return

    except requests.exceptions.Timeout:
        print("\n\nTimed out while waiting for the server.")
        return

    except requests.exceptions.RequestException as exc:
        print(f"\n\nHTTP error: {exc}")
        return

    end_time = time.perf_counter()

    total_elapsed = end_time - start_time

    if first_token_time is not None:
        prefill_time = first_token_time - start_time
        decode_time = end_time - first_token_time
    else:
        prefill_time = 0.0
        decode_time = 0.0

    completion_tokens = (
        usage_completion_tokens
        if usage_completion_tokens is not None
        else stream_chunks
    )

    decode_tps = (
        completion_tokens / decode_time
        if decode_time > 0
        else 0.0
    )

    print("\n\n--- Statistics ---")
    print(f"Completion Tokens:  {completion_tokens}")

    if usage_completion_tokens is None:
        print(f"Stream Chunks:       {stream_chunks}")
        print("Note: Without usage data, stream chunks are not exact token counts.")

    print(f"Prefill (TTFT):      {prefill_time:.2f} s")
    print(f"Decode Time:         {decode_time:.2f} s")
    print(f"Total Time:          {total_elapsed:.2f} s")
    print(f"Decode Tokens/s:     {decode_tps:.2f}")

    if finish_reason is not None:
        print(f"Finish Reason:       {finish_reason}")

    print()


def main() -> None:
    print("Qwen Chat Client")
    print(f"Server: {URL}")
    print(f"Model: {MODEL}")
    print("Exit with: exit, quit, or q")

    while True:
        try:
            prompt = input("\nPrompt: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nClient stopped.")
            break

        if prompt.lower() in {"exit", "quit", "q"}:
            print("Client stopped.")
            break

        if not prompt:
            continue

        ask_model(prompt)


if __name__ == "__main__":
    main()
```

Run it with:

```bash
python client.py
```
