# Qwen3.8-27B decode on one RTX 5090: NInfer fork and recipe

This fork of [Neroued/ninfer](https://github.com/Neroued/ninfer) targets single-user serving on a GeForce RTX 5090.
Branch `codex/cache-independent-prefill` extends `native-windows` with Windows compiler fixes and cache-independent
prefill scheduling, on top of the earlier `rtx5090-round2` kernel and runtime work. Apache 2.0, unchanged.

Start with the [native Windows instructions](#native-windows-build-and-run) below. The server runs directly on
Windows; WSL is only needed if you choose to run the existing Linux-only offline conversion helpers locally.
The [Linux/WSL reproduction steps](#reproduce) remain available farther below.

## Native Windows build and run

### Prerequisites

- Windows x64 and a GeForce RTX 5090 with 32 GB VRAM. This branch builds only for `sm_120a`.
- A public CUDA Toolkit supporting `sm_120a`, with a compatible NVIDIA driver installed separately. Use an MSVC
  version supported by that toolkit's [Windows installation guide](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html).
- Visual Studio 2022 or Build Tools with Desktop development with C++, the x64 MSVC toolset, and a Windows SDK.
- Git, CMake 3.28 or newer, Ninja, and Python 3.11 for the artifact download and optional tests.
- Space for the source and converted model files, and enough host RAM for model loading, compilation and the
  recipe's explicit 8 GiB pinned host cache. These are additional to GPU memory.

Open **x64 Native Tools Command Prompt for VS 2022**, selecting a CUDA-compatible toolset, then run
`powershell -NoProfile`. All Windows command blocks below use PowerShell from the repository root.
`CUDA_PATH` should identify the selected public toolkit installation; do not mix compilers from different toolkits.
If your Python installation has no `py -3.11` launcher alias, substitute the path to its Python 3.11 executable.

### Build the native server

```powershell
git clone --branch codex/cache-independent-prefill --single-branch https://github.com/kedarpotdar-nv/ninfer.git ninfer-5090
Set-Location ninfer-5090

py -3.11 -m venv .venv
$python = (Resolve-Path .\.venv\Scripts\python.exe).Path
$nvcc = Join-Path $env:CUDA_PATH 'bin\nvcc.exe'
$env:Path = "$(Join-Path $env:CUDA_PATH 'bin');$env:Path"
Get-Command cl.exe, cmake.exe, ninja.exe
& $nvcc --list-gpu-code | Select-String 'sm_120a'
nvidia-smi

cmake -S . -B build-win -G Ninja `
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=cl -DCMAKE_CXX_COMPILER=cl `
  "-DCMAKE_CUDA_COMPILER=$nvcc" -DCMAKE_CUDA_HOST_COMPILER=cl `
  -DCMAKE_CUDA_ARCHITECTURES=120a -DNINFER_BUILD_APPS=ON `
  -DBUILD_TESTING=ON "-DPython3_EXECUTABLE=$python" -DNINFER_BUILD_BENCHMARKS=OFF `
  -DNINFER_ENABLE_FFMPEG=OFF -DNINFER_ENABLE_CURL=OFF
cmake --build build-win --target ninfer-serve -j
```

The executable is `build-win\apps\ninfer-serve.exe`. This text-serving build needs neither FFmpeg nor libcurl;
HTTP serving, reasoning and tool calling remain enabled. Image/video decoding and remote media fetching are
disabled. The C++ dependencies are included in the repository. Stop if configure or build fails; use a fresh build
directory when changing compilers. Reduce build parallelism, for example `-j 2`, if compilation exhausts host RAM.

Optional focused checks for this patch:

```powershell
cmake --build build-win --target ninfer_qwen3_5_frontend_test ninfer_qwen3_5_runtime_mechanisms_test -j
ctest --test-dir build-win -R 'ninfer_qwen3_5_(frontend|runtime_mechanisms)_test' --output-on-failure
```

These are focused tests; the complete suite includes tests requiring additional model artifacts.

### Prepare the model

Download the public [Qwen3.8-27B NVFP4 artifact](https://huggingface.co/neroued/Qwen3.8-27B-nvfp4-NInfer)
using the [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/guides/cli):

```powershell
& .\.venv\Scripts\python.exe -m pip install huggingface_hub
& .\.venv\Scripts\hf.exe download neroued/Qwen3.8-27B-nvfp4-NInfer qwen3_8_27b_nvfp4.ninfer --local-dir models
$sourceHash = (Get-FileHash .\models\qwen3_8_27b_nvfp4.ninfer -Algorithm SHA256).Hash
if ($sourceHash -ne '74d2c57145e6ff11d1d2faa79594477f9bc903a611af1fb20218189fbbb77d82') {
  throw 'The downloaded artifact differs from the recipe input. Check the model card before proceeding.'
}
```

Our fastest tested artifact for this recipe is `qwen3_8_27b_gdn4_a4_draftq4.ninfer`: NVFP4/A4 GDN projections and
Q4 draft FFN gate/up weights, derived from the public artifact above. Follow [Artifact conversion](#2-artifact)
once on Linux or WSL, then place the final file under `models` in this Windows checkout. The conversion helpers use
POSIX file APIs and cannot currently run under native Windows Python. The resulting `.ninfer` file is portable;
the Windows server does not need WSL or Python for inference.

To skip conversion entirely, use `models\qwen3_8_27b_nvfp4.ninfer` in the serve command. It already includes DFlash2,
but its weight representation and performance differ from the compacted recipe. Neither file is GGUF/Q4_K_M.

Verify the compacted file before selecting it:

```powershell
$artifact = (Resolve-Path .\models\qwen3_8_27b_gdn4_a4_draftq4.ninfer).Path
if ((Get-FileHash $artifact -Algorithm SHA256).Hash -ne 'f54becd56fb5a3481de60c46b857671025a43741c50ea25c2e8c18663fc552a8') {
  throw 'Compacted artifact checksum mismatch.'
}
```

### Serve: preferred 5090 recipe for one active conversation

Use this 32K configuration as the starting point for single-user text and tool-call workloads. It keeps both the
logical context and resident KV pool bounded while retaining BF16 KV and DFlash2. It is the preferred tested
configuration for this setup, not a guarantee of maximum throughput on every prompt or driver.

```powershell
$env:Path = "$(Join-Path $env:CUDA_PATH 'bin');$env:Path"
$artifact = (Resolve-Path .\models\qwen3_8_27b_gdn4_a4_draftq4.ninfer).Path
& .\build-win\apps\ninfer-serve.exe $artifact `
  --host 127.0.0.1 --port 9932 --model-id qwen3.8-27b `
  --max-context 32768 --kv-capacity 32768 --max-concurrency 1 `
  --kv-dtype bf16 --prefill-chunk 2048 `
  --device-state-slots 4 --host-context-mib 8192 `
  --preserve-thinking --agent-prompt-cache `
  --spec dflash2 --draft-tokens 7 --lm-head-draft
```

| Setting | Purpose |
| --- | --- |
| 32,768 context and KV capacity | Bounds prompt plus generated history; leaves more memory headroom than a large context reservation. |
| BF16 KV, prefill chunk 2,048 | Selected cache precision and prefill batch limit. |
| One active conversation, four extra device states, 8 GiB host cache | Retains reusable conversation state while keeping generation concurrency at one. |
| DFlash2, seven draft tokens, optimized proposal head | Speculative decode with the artifact's included draft model. |
| Preserved thinking and agent prompt cache | Retains reasoning and enables prefix reuse for agent conversations. |

Leave `NINFER_PDL` unset to use the default optimized route; `NINFER_PDL=0` is an experimental control, not this
recipe. Existing `serve-windows.ps1` and `serve.sh` wrappers default to the earlier 64K recipe; use the explicit
command above for 32K. Sampling is set on each API request, not by this launch command.

### Send a request

Keep the server running and open another PowerShell window:

```powershell
Invoke-RestMethod http://127.0.0.1:9932/health
Invoke-RestMethod http://127.0.0.1:9932/v1/models
$request = @{
  model = 'qwen3.8-27b'
  messages = @(@{ role = 'user'; content = 'Explain prefill and decode in two sentences.' })
  temperature = 1.0; top_p = 0.95; top_k = 20; min_p = 0.0
  presence_penalty = 0.0; frequency_penalty = 0.0
  max_tokens = 8192; stream = $false
  reasoning_effort = 'xhigh'
  chat_template_kwargs = @{ enable_thinking = $true; preserve_thinking = $true; reasoning_effort = 'xhigh' }
}
$response = Invoke-RestMethod http://127.0.0.1:9932/v1/chat/completions `
  -Method Post -ContentType 'application/json' -TimeoutSec 600 `
  -Body ($request | ConvertTo-Json -Depth 10)
$response.choices[0].message.content
$response.usage
$response.timings
```

`max_tokens` includes reasoning and is limited by remaining context. The 8,192-token value above is an example
request budget; choose it for the application. For multi-turn tools, send prior `tool_calls`, tool results and
`reasoning_content` back unchanged. See [HTTP serving](docs/serving.md) for the supported protocol.
`usage.prompt_tokens` includes the cached subset; do not add cached tokens again. `timings.predicted_per_second`
is server decode throughput, not end-to-end request throughput or time to first token.

### Limits and troubleshooting

- 32K is a total context limit, not 32K input plus 32K output. Longer conversations can return a context-limit
  error. Increase both context and KV capacity only after checking GPU and host-memory headroom; that changes
  the performance configuration. Choose an explicit application history policy rather than silently truncating.
- If pinned-host allocation fails, free host memory or reduce `--host-context-mib`; this reduces retained cache.
  A larger context and more retained device states also use more GPU memory. Keep one server resident when comparing recipes.
- If CUDA DLLs are missing, ensure the selected toolkit's `bin` directory is on `PATH`. If `cl` is missing or NVCC
  rejects the host compiler, use the x64 developer shell with a supported MSVC toolset; do not bypass compiler checks.
- This branch makes prompt chunk boundaries independent of cache admission and rejects incompatible prompt
  checkpoints. Fixed-token tests matched on the tested paths. Growing messages can require more recomputation;
  this does not establish an overall accuracy gain or universal bitwise equality for generated histories.

## Historical measurements: round-2 branch

The following tables describe the earlier `rtx5090-round2` revision and their stated workloads. They are retained
for provenance; they are not new measurements of the cache/prefill patch or the 32K native Windows recipe above.

RTX 5090 (GB202, 32 GB), Ubuntu WSL2 on Windows 11, `sm_120a`. One request at a time,
DFlash2 with seven draft tokens, BF16 KV, greedy, 256 output tokens, thinking enabled. Decode rates are the server's
`timings.predicted_per_second`, arithmetic mean over requests, from alternating paired runs.

Provenance of the numbers: all measurements below were taken with the branch as of the `--agent-prompt-cache`
commit, before `ops(linear_add): keep upstream's K=6144 NVFP4 A4 threshold`. That later commit restores upstream's
route selection for NVFP4 `linear_add` weights with K=6144; the selected artifact (and the public one) contains no
such weight (its 64 K=6144 output projections are FP8, `AllowA8`; NVFP4 tensors are at K=5120 and K=17408 only), so
the code path never executed during these runs and the round-2 head reproduced them unchanged. Checked directly:
the binary used for the GSM8K, BFCL and tool-calling runs (built before the revert) and the round-2 head, both on
the selected artifact and the 15 `throughput_8k` requests, produced 15 of 15 byte-identical responses with identical
draft acceptance (2,753 of 7,370) and identical decode rate (249.4 vs 249.4 tok/s;
`20261006T194637Z-engine-sweep-paired-analysis.json`). The accuracy results therefore apply to that round-2 revision.

**Confirmation on the pushed head, and the engine-only share** (`throughput_8k`, same harness, two alternating
repeats, run the next day; `20261006T193732Z-engine-sweep-paired-analysis.json`, `speed-bench-category-decode.json`
under `head_confirmation`):

| Arm (same session) | Decode tok/s | Paired vs public NInfer |
| --- | ---: | --- |
| Public NInfer `68c5435` binary, public artifact | 224.4 | reference |
| **Fork head binary, public artifact (engine only)** | **238.2** | **x1.062** (+13.8 tok/s, 95% CI +6.5 to +21.2); all 15 responses identical to the public binary's |
| Fork head binary, re-quantized artifact | **251.9** | x1.123 (+27.5 tok/s, CI +12.2 to +42.8) |

So on the code as pushed, the engine alone is worth +6.2% with bit-identical outputs, and the two artifact
conversions add a further +5.7 points on top. Absolute rates in this session ran 1 to 4% above the previous day's
(224.4 vs 221.9 for the unchanged public arm; 251.9 vs 241.9 for the fork), which is day-to-day drift of the machine
(fresh WSL VM, clocks); compare ratios within a session rather than absolute numbers across sessions. The headline
table keeps the original session because it is the one with the llama.cpp arm.

| Workload | llama.cpp b11425, Q4_K_M weights + Q4_K_M DFlash2 draft | Public NInfer `68c5435`, public artifact (NVFP4 MLP, FP8 elsewhere) | This fork, re-quantized artifact (public + GDN projections NVFP4/A4 + draft FFN Q4) |
| --- | ---: | ---: | ---: |
| SPEED-Bench `throughput_8k`, 15 requests (~8K-token prompts) | 156.4 tok/s | 221.9 | **241.9** (+9.0% vs NInfer, +54.6% vs llama.cpp) |
| SPEED-Bench qualitative, first 3 turns of each of 11 categories (33 requests) | | 227.9 | **271.5** (+19.1%) |
| AgentPerf local 0.3.3 `agentperf-default-v1`, total normalized replay time, recorded output policy | | | **174 s (2.9 min)**; round-1 engine 307 s |

The three columns are not the same weights: llama.cpp runs Q4_K_M GGUF, NInfer runs a mixed NVFP4/FP8 engine
artifact, and the fork column adds two more conversions on top of the public artifact. The GSM8K and BFCL section at
the end is what makes the speed comparison fair. Within the NInfer-to-fork gain, the artifact conversions carry part
of it: on `throughput_8k` the draft-FFN Q4 conversion alone is worth about 7 of the 20 tok/s (234.4 to 241.3 in the
`20261005T220134Z` paired analysis), and the GDN NVFP4/A4 conversion carried the round-1 gain. The engine-only
share was measured directly afterwards (next table): the fork binary on the public artifact is +6.2% over the
public binary with identical outputs.

Paired statistics are in `recipes/rtx5090-qwen38-27b/*-paired-analysis.json`; the AgentPerf summary in
`agentperf-default-v1-summary.json`. The AgentPerf run needs `--host-context-mib 8192 --device-state-slots 4`.
NInfer does not honor `ignore_eos`, so the AgentPerf figure is from the recorded output policy, not exact mode.

### Decode rate by SPEED-Bench category

Mean of the per-request decode rate within each category, then mean over the alternating repeats
(`speed-bench-category-decode.json`). Same requests to every server; 256 output tokens; thinking on.

`throughput_8k`, 5 requests per category, 2 repeats per server (weights per column as in the table above):

| Category | llama.cpp b11425, Q4_K_M | Public NInfer `68c5435`, public artifact | This fork, re-quantized artifact | Fork vs llama.cpp |
| --- | ---: | ---: | ---: | ---: |
| high_entropy | 141.3 | 194.3 | **207.3** | +46.8% |
| mixed | 178.7 | 259.5 | **282.4** | +58.0% |
| low_entropy | 149.3 | 211.8 | **236.0** | +58.1% |
| overall (15 requests) | 156.4 | 221.9 | **241.9** | +54.7% |

Qualitative, first 3 turns of each category, 3 repeats per server (no llama.cpp arm on this workload):

| Category | Public NInfer `68c5435`, public artifact | This fork, `NINFER_PDL=0` | This fork, re-quantized artifact | Fork vs public NInfer |
| --- | ---: | ---: | ---: | ---: |
| coding | 226.1 | 281.4 | **286.3** | +26.6% |
| humanities | 193.0 | 235.6 | **240.1** | +24.4% |
| math | 297.4 | 320.3 | **326.4** | +9.7% |
| multilingual | 172.7 | 255.4 | **259.4** | +50.2% |
| qa | 193.5 | 199.1 | **202.3** | +4.5% |
| rag | 293.6 | 308.7 | **314.1** | +7.0% |
| reasoning | 225.0 | 269.9 | **274.8** | +22.1% |
| roleplay | 210.5 | 254.1 | **258.4** | +22.8% |
| stem | 257.7 | 275.7 | **280.4** | +8.8% |
| summarization | 246.7 | 259.4 | **263.7** | +6.9% |
| writing | 190.8 | 276.4 | **281.2** | +47.4% |
| overall (33 requests) | 227.9 | 266.9 | **271.5** | +19.1% |

The fork wins every category on both workloads. The spread against public NInfer (+4.5% on `qa` to +50% on
`multilingual` and `writing`) is mostly the draft-acceptance rate: categories where DFlash2 accepts more draft tokens
per round gain more from the per-round kernel savings. The `NINFER_PDL=0` column isolates programmatic dependent
launch at +1.7% overall; the rest is the fused GDN record route, the post-fold wait removal and the draft-Q4 artifact.

## Changes carried by this branch

| Commit slice | Effect |
| --- | --- |
| round-1 kernels: recurrent record input staging, FP8 GDN projection/convolution fusion, NVFP4 down-projection 16x32 / K512 / 3-stage tile | about +3% serving over `abb7f14` |
| Fused NVFP4 GDN Record route (B=1, T=8): convolution, SiLU and record publication inside the A4 GEMM epilogue | +2.1%, outputs bit-identical |
| Post-fold host wait only when host memory is still read asynchronously | the next graph launch overlaps the fold |
| Programmatic dependent launch for NVFP4/FP8/Q8 GEMMs and quantizers; weights are prefetched before the grid-dependency join (`NINFER_PDL=0` disables) | +1.6 to +2.1% |
| Q4 `linear_add` K=17408 and a Q4 projection route in `linear_dynamic_grouped_conv_add` (needed by the draft-FFN Q4 artifact); the NVFP4 `linear_add` A4 threshold is upstream's | enabling changes for the artifact |
| `--agent-prompt-cache` (opt-in, default off): a second automatic prompt-cache candidate at the boundary before the final message, OpenAI chat and Responses | agent loops that rewrite their last message keep the shared prefix: AgentPerf prompt-cache hits 44% to 88% |
| Tests: valid prefixes 1..8 and T=8 graph replay for the fused record route; Q4 K=17408 and fused-op Q4 profiles; schema test for the new flag | see Validation |
| Native Windows source fixes: constexpr kernel-pointer locals and explicit lambda captures | MSVC and device-link compatibility |
| Cache-independent prompt prefill/replay scheduling and compatible-checkpoint checks | consistent execution on the tested cache-on/off paths; may recompute more tokens for growing messages |

## Reproduce

### 1. Build

For Linux or WSL2, use a public CUDA toolkit supporting `sm_120a`; install curl and FFmpeg development packages as
in [upstream's build notes](docs/UPSTREAM-README.md#quick-start). These commands build the current branch. To
reproduce a historical table, use the specific revision and settings recorded with that table instead.

```bash
git clone --branch codex/cache-independent-prefill --single-branch https://github.com/kedarpotdar-nv/ninfer.git
cd ninfer
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=120a \
      -DNINFER_BUILD_APPS=ON -DBUILD_TESTING=ON -DNINFER_BUILD_BENCHMARKS=ON
cmake --build build --target ninfer-serve -j
```

### 2. Artifact

Start from the public v3 artifact `qwen3_8_27b_nvfp4.ninfer` (23,719,715,844 bytes;
[model card](https://huggingface.co/neroued/Qwen3.8-27B-nvfp4-NInfer)). These are **offline Linux/WSL conversion**
commands; they use POSIX file APIs. Open a Linux/WSL shell in this checkout and use Python 3.11. For a checkout on
a Windows drive, navigate to its `/mnt/<drive>/...` path in WSL. The output can then be served by the native Windows
executable without running WSL. If your input already exists under `models`, skip the download but verify its hash.

```bash
python3.11 -m venv .local/convert-linux
source .local/convert-linux/bin/activate
python -m pip install numpy huggingface_hub
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
hf download neroued/Qwen3.8-27B-nvfp4-NInfer qwen3_8_27b_nvfp4.ninfer --local-dir models
printf '%s  %s\n' \
  '74d2c57145e6ff11d1d2faa79594477f9bc903a611af1fb20218189fbbb77d82' \
  'models/qwen3_8_27b_nvfp4.ninfer' | sha256sum --check
```

Run the two conversions with this environment; they refuse to overwrite an existing output artifact. The default
draft conversion changes only gate/up weights; do not add `--include-down` or `--include-attn-output` for this recipe.

```bash
# 48 GDN Q/K/V/z parents FP8 -> NVFP4 with AllowA4 (perplexity +0.41% on the 1M-token quick corpus,
# GSM8K paired checks below); ~9 minutes
python recipes/rtx5090-qwen38-27b/compact_native_weights.py --self-test
python recipes/rtx5090-qwen38-27b/compact_native_weights.py \
  --source models/qwen3_8_27b_nvfp4.ninfer --output models/qwen3_8_27b_gdn4_a4.ninfer --report models/gdn4-conversion.json
# five DFlash2 draft gate/up parents Q8 -> Q4 G64 (target distribution unchanged); ~1 minute
python recipes/rtx5090-qwen38-27b/draft_ffn_q4_weights.py \
  --source models/qwen3_8_27b_gdn4_a4.ninfer --output models/qwen3_8_27b_gdn4_a4_draftq4.ninfer --report models/draftq4-conversion.json
```

Expected result: 21,483,090,948 bytes, SHA-256 `f54becd56fb5a3481de60c46b857671025a43741c50ea25c2e8c18663fc552a8`
(`ninfer-draft-ffn-q4-conversion.json` records every object hash). Perplexity of this artifact equals the GDN-compacted
one exactly (4.333745 on the quick corpus, `perplexity-compact-comparison.json`).

### 3. Serve

```bash
recipes/rtx5090-qwen38-27b/serve.sh models/qwen3_8_27b_gdn4_a4_draftq4.ninfer 9932
```

This wrapper retains the historical 65,536-context flags (BF16 KV, concurrency 1, prefill chunk 2048,
`--spec dflash2 --draft-tokens 7 --lm-head-draft --preserve-thinking`, `--host-context-mib 8192
--device-state-slots 4 --agent-prompt-cache`). `NINFER_PDL=0` in the environment gives the ordinary-launch control
for A/B runs. For the current 32K recipe, use the [explicit Windows command](#serve-preferred-5090-recipe-for-one-active-conversation),
or the same arguments with `./build/apps/ninfer-serve` on Linux. Cache-independent chunk scheduling remains active
regardless of `--agent-prompt-cache`.

### 4. Measure

```bash
pip install datasets pandas requests
python recipes/rtx5090-qwen38-27b/prepare_speed_throughput.py throughput_8k     # resolves the workload (pinned)
python recipes/rtx5090-qwen38-27b/prepare_speed_throughput.py qualitative
python recipes/rtx5090-qwen38-27b/run_resolved_bench.py --url http://127.0.0.1:9932 --model qwen3.8-27b \
  --dataset throughput_8k --per-category 5 --output speed-bench.json
```

`run_resolved_bench.py` sends the same requests to any OpenAI-compatible server (llama.cpp, public NInfer, this
fork) and reports per-category and overall decode rates from the server's `timings`; alternate
the servers and repeat. For AgentPerf local, point `agentperf_local run --output-token-policy recorded
--sampling standard --tool-choice none --tool-mode none` at the server.

## Validation

Every kernel change ships with its oracle test (independent FP64 reference, unchanged tolerances, graph replay) and a
cold-cache op benchmark; serving changes were accepted only on alternating paired sweeps with response-identity
checks. Historical CTest on the round-2 Linux build (`NINFER_TEST_ARTIFACT` set to the selected artifact): 127 of 131 suites
pass, one is skipped (`loading_real`), and the three remaining failures need other artifacts: the prefix suite's `all`
scenario carries a Qwen3.5 chat-template token golden (the public Qwen3.8-27B artifact fails it identically), the
MoE suite needs the 35B-A3B artifact, and the DFlash suite needs a DFlash-v1 draft component. Every other
`NINFER_PREFIX_REAL_SCENARIO` of the prefix suite (concurrent, anthropic-prefix-regression, nested-tool-markers,
shared-rewrite-materialization, late-instructions, agent-continuation, explicit-prefix, explicit-anchor,
rewrite-checkpoint, stream-observations, attention) passes when run on its own.

Accuracy checks (GSM8K, BFCL) are in the last section.

Tool calling through `/v1/chat/completions` with the selected build and `--agent-prompt-cache`
(`smoke-tool-calls.json`): `tool_choice: auto` produces a well-formed call with valid JSON arguments, the follow-up
turn with the tool result answers from it and reuses 400 of 439 prompt tokens from the cache, `tool_choice: none`
suppresses calls, and a two-tool prompt yields two parallel calls. Forcing a named function or `tool_choice:
required` returns HTTP 400 `tool_choice_not_supported`; that is upstream NInfer's policy and is unchanged here. The optional output-projection NVFP4 lane (+1.3%, perplexity +0.55%, `perplexity-oproj16-report.json`) is not
part of the selected artifact.

## Accuracy: GSM8K and BFCL

These historical checks used the round-2 build, before the cache/prefill patch. Both ran through the serving API
with the selected artifact, and against llama.cpp b11425
(Q4_K_M target + Q4_K_M DFlash2 draft) on the same machine with identical requests. Server defaults otherwise, so
thinking is on for both.

### GSM8K

Full test split, 1,319 questions, zero-shot chat, greedy, 8,192 max tokens, exact match of the final `#### <answer>`
line (`gsm8k-protocol.json`, `gsm8k-validated-analysis-vs-round1.json`). Four requests in flight on the NInfer arm.

| Arm | Accuracy | Wilson 95% CI | Paired vs this build |
| --- | ---: | ---: | --- |
| **This build, draft-Q4 artifact** | **97.12%** (1,281) | 96.07 to 97.89 | |
| llama.cpp b11425 Q4_K_M | 96.97% (1,279) | 95.90 to 97.77 | this build +0.15 pp; bootstrap 95% CI -0.53 to +0.83; McNemar p = 0.82 |
| Round-1 NInfer engine, GDN-compacted artifact | 97.57% (1,287) | 96.60 to 98.28 | this build -0.45 pp; CI -0.99 to +0.08; p = 0.18; 4 wins / 10 losses / 1,305 ties |

Both paired differences sit inside the 2 pp engineering noninferiority screen. The two NInfer arms share target
weights (draft Q4 only changes the exactly verified DFlash2 draft), so the 14 flipped answers are kernel-numerics
trajectory changes; the visible shift is more 8,192-token truncations (4 to 15) and fewer wrong answers (28 to 22).

### BFCL v4, single-turn function calling

Berkeley Function Calling Leaderboard, `bfcl-eval` 2026.3.23, all 13 single-turn categories (3,641 entries), native
function-calling mode: the request carries `tools`, the model answers with `tool_calls`, and BFCL's AST and relevance
checkers grade them. Temperature 0.001 (the harness default), four requests in flight on both servers, both servers
exposing the model id `qwen3.8-27b`. Scores per category and arm: `bfcl-single-turn-scores.json`.

| Category | This build | llama.cpp b11425 Q4_K_M | Delta (pp) | Public NInfer `68c5435`, FP8 artifact |
| --- | ---: | ---: | ---: | ---: |
| simple_python | 93.75% (375/400) | 93.75% (375/400) | 0.00 | 93.00% (372/400) |
| simple_java | 61.00% (61/100) | 61.00% (61/100) | 0.00 | 61.00% (61/100) |
| simple_javascript | 72.00% (36/50) | 68.00% (34/50) | +4.00 | 68.00% (34/50) |
| multiple | 94.00% (188/200) | 95.50% (191/200) | -1.50 | 93.00% (186/200) |
| parallel | 93.00% (186/200) | 93.50% (187/200) | -0.50 | 94.50% (189/200) |
| parallel_multiple | 85.50% (171/200) | 86.00% (172/200) | -0.50 | 84.00% (168/200) |
| irrelevance | 80.83% (194/240) | 79.17% (190/240) | +1.67 | 80.00% (192/240) |
| live_simple | 88.76% (229/258) | 84.88% (219/258) | +3.88 | 86.82% (224/258) |
| live_multiple | 78.92% (831/1053) | 78.44% (826/1053) | +0.47 | 79.11% (833/1053) |
| live_parallel | 81.25% (13/16) | 81.25% (13/16) | 0.00 | 81.25% (13/16) |
| live_parallel_multiple | 79.17% (19/24) | 75.00% (18/24) | +4.17 | 75.00% (18/24) |
| live_irrelevance | 75.34% (666/884) | 74.89% (662/884) | +0.45 | 75.57% (668/884) |
| live_relevance | 75.00% (12/16) | 75.00% (12/16) | 0.00 | 81.25% (13/16) |
| **All single-turn entries, micro-average** | **81.87%** (2,981/3,641) | **81.30%** (2,960/3,641) | **+0.58** | 81.60% (2,971/3,641) |
| BFCL Non-Live AST accuracy (group mean) | 87.02% | 87.31% | -0.29 | 86.38% |
| BFCL Live accuracy (group mean) | 80.83% | 79.64% | +1.19 | 80.53% |

Function-calling quality is the same within noise across all three servers: this build and llama.cpp agree exactly
on five categories, and no category moves by more than three entries except `live_simple` (+10 for this build).
Against public NInfer with the FP8 artifact the micro-average is +0.27 pp with ten categories within two entries.
Every one of the 3,641 requests returned a response on all three servers. These historical scores cover single-turn categories only. Forced tool choice (a named function
or `tool_choice: required`) is not part of this suite and is rejected by NInfer by upstream policy.
