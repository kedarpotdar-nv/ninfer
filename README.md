# Qwen3.8-27B decode on one RTX 5090: NInfer fork and recipe

This fork of [Neroued/ninfer](https://github.com/Neroued/ninfer) (branch `rtx5090-round2`, based on upstream
`68c5435`) carries the kernel, runtime and prompt-cache changes from two optimization rounds on a GeForce RTX 5090,
plus everything needed to rebuild the measured artifact and repeat the measurements. Upstream NInfer requires an
Issue before any pull request; this fork exists to reproduce the numbers while that happens. Apache 2.0, unchanged.

## Measured

RTX 5090 (GB202, 32 GB), driver 591.86, Ubuntu WSL2 on Windows 11, CUDA 13.4.2, `sm_120a`. One request at a time,
DFlash2 with seven draft tokens, BF16 KV, greedy, 256 output tokens, thinking enabled. Decode rates are the server's
`timings.predicted_per_second`, arithmetic mean over requests, from alternating paired runs.

| Workload | llama.cpp b11425 (Q4_K_M + DFlash2 Q4_K_M) | Public NInfer `68c5435`, FP8 artifact | This fork, draft-Q4 artifact |
| --- | ---: | ---: | ---: |
| SPEED-Bench `throughput_8k`, 15 requests (~8K-token prompts) | 156.4 tok/s | 221.9 | **241.9** (+9.0% vs NInfer, +54.6% vs llama.cpp) |
| SPEED-Bench qualitative, first 3 turns of each of 11 categories (33 requests) | | 227.9 | **271.5** (+19.1%) |
| AgentPerf local 0.3.3 `agentperf-default-v1`, total normalized replay time, recorded output policy | | | **174 s (2.9 min)**; round-1 engine 307 s |

Paired statistics are in `recipes/rtx5090-qwen38-27b/*-paired-analysis.json`; the AgentPerf summary in
`agentperf-default-v1-summary.json`. The AgentPerf run needs `--host-context-mib 8192 --device-state-slots 4`.
NInfer does not honor `ignore_eos`, so the AgentPerf figure is from the recorded output policy, not exact mode.

### Decode rate by SPEED-Bench category

Mean of the per-request decode rate within each category, then mean over the alternating repeats
(`speed-bench-category-decode.json`). Same requests to every server; 256 output tokens; thinking on.

`throughput_8k`, 5 requests per category, 2 repeats per server:

| Category | llama.cpp b11425 (Q4_K_M + DFlash2 Q4_K_M) | Public NInfer `68c5435`, FP8 artifact | This fork, draft-Q4 artifact | Fork vs llama.cpp |
| --- | ---: | ---: | ---: | ---: |
| high_entropy | 141.3 | 194.3 | **207.3** | +46.8% |
| mixed | 178.7 | 259.5 | **282.4** | +58.0% |
| low_entropy | 149.3 | 211.8 | **236.0** | +58.1% |
| overall (15 requests) | 156.4 | 221.9 | **241.9** | +54.7% |

Qualitative, first 3 turns of each category, 3 repeats per server (no llama.cpp arm on this workload):

| Category | Public NInfer `68c5435`, FP8 artifact | This fork, `NINFER_PDL=0` | This fork, draft-Q4 artifact | Fork vs public NInfer |
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

## What this branch changes

| Commit slice | Effect |
| --- | --- |
| round-1 kernels: recurrent record input staging, FP8 GDN projection/convolution fusion, NVFP4 down-projection 16x32 / K512 / 3-stage tile | about +3% serving over `abb7f14` |
| Fused NVFP4 GDN Record route (B=1, T=8): convolution, SiLU and record publication inside the A4 GEMM epilogue | +2.1%, outputs bit-identical |
| Post-fold host wait only when host memory is still read asynchronously | the next graph launch overlaps the fold |
| Programmatic dependent launch for NVFP4/FP8/Q8 GEMMs and quantizers; weights are prefetched before the grid-dependency join (`NINFER_PDL=0` disables) | +1.6 to +2.1% |
| Q4 `linear_add` K=17408, Q4 route in `linear_dynamic_grouped_conv_add`, NVFP4 `linear_add` A4 route from T>=8 for K=6144 | enabling changes for the artifact experiments |
| `--agent-prompt-cache` (opt-in, default off): a second automatic prompt-cache candidate at the boundary before the final message, OpenAI chat and Responses | agent loops that rewrite their last message keep the shared prefix: AgentPerf prompt-cache hits 44% to 88% |
| Tests: valid prefixes 1..8 and T=8 graph replay for the fused record route; Q4 K=17408 and fused-op Q4 profiles; schema test for the new flag | see Validation |

## Reproduce

### 1. Build

Linux or WSL2 with CUDA 13.4 and an `sm_120a` GPU; project-local curl and FFmpeg as in upstream's build notes.

```bash
git clone https://github.com/kedarpotdar-nv/ninfer.git && cd ninfer && git checkout rtx5090-round2
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=120a \
      -DNINFER_BUILD_APPS=ON -DBUILD_TESTING=ON -DNINFER_BUILD_BENCHMARKS=ON
cmake --build build --target ninfer-serve --parallel 4
```

### 2. Artifact

Start from the public v3 artifact `qwen3_8_27b_nvfp4.ninfer` (23,719,715,844 bytes; see upstream's
`docs/performance/qwen3.8-27b.md`). Two offline, reproducible conversions, run with the repository's Python
environment (NumPy; the second also needs PyTorch, CPU is fine):

```bash
# 48 GDN Q/K/V/z parents FP8 -> NVFP4 with AllowA4 (perplexity +0.41% on the 1M-token quick corpus,
# GSM8K paired checks below); ~9 minutes
python recipes/rtx5090-qwen38-27b/compact_native_weights.py --self-test
python recipes/rtx5090-qwen38-27b/compact_native_weights.py \
  --source qwen3_8_27b_nvfp4.ninfer --output qwen3_8_27b_gdn4_a4.ninfer --report gdn4-conversion.json
# five DFlash2 draft gate/up parents Q8 -> Q4 G64 (target distribution unchanged); ~1 minute
python recipes/rtx5090-qwen38-27b/draft_ffn_q4_weights.py \
  --source qwen3_8_27b_gdn4_a4.ninfer --output qwen3_8_27b_gdn4_a4_draftq4.ninfer --report draftq4-conversion.json
```

Expected result: 21,483,090,948 bytes, SHA-256 `f54becd56fb5a3481de60c46b857671025a43741c50ea25c2e8c18663fc552a8`
(`ninfer-draft-ffn-q4-conversion.json` records every object hash). Perplexity of this artifact equals the GDN-compacted
one exactly (4.333745 on the quick corpus, `perplexity-compact-comparison.json`).

### 3. Serve

```bash
recipes/rtx5090-qwen38-27b/serve.sh qwen3_8_27b_gdn4_a4_draftq4.ninfer 9932
```

The script runs `ninfer-serve` with the measured flags (65,536 context, BF16 KV, concurrency 1, prefill chunk 2048,
`--spec dflash2 --draft-tokens 7 --lm-head-draft --preserve-thinking`, `--host-context-mib 8192
--device-state-slots 4 --agent-prompt-cache`). `NINFER_PDL=0` in the environment gives the ordinary-launch control
for A/B runs. Without `--agent-prompt-cache` the server's prompt-cache behaviour is identical to upstream.

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
checks. CTest on this branch (rebased build, `NINFER_TEST_ARTIFACT` set to the selected artifact): 127 of 131 suites
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

Both checks run through the serving API against this build with the selected artifact, and against llama.cpp b11425
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

| Category | This build | llama.cpp b11425 Q4_K_M | Delta (pp) |
| --- | ---: | ---: | ---: |
| simple_python | 93.75% (375/400) | 93.75% (375/400) | 0.00 |
| simple_java | 61.00% (61/100) | 61.00% (61/100) | 0.00 |
| simple_javascript | 72.00% (36/50) | 68.00% (34/50) | +4.00 |
| multiple | 94.00% (188/200) | 95.50% (191/200) | -1.50 |
| parallel | 93.00% (186/200) | 93.50% (187/200) | -0.50 |
| parallel_multiple | 85.50% (171/200) | 86.00% (172/200) | -0.50 |
| irrelevance | 80.83% (194/240) | 79.17% (190/240) | +1.67 |
| live_simple | 88.76% (229/258) | 84.88% (219/258) | +3.88 |
| live_multiple | 78.92% (831/1053) | 78.44% (826/1053) | +0.47 |
| live_parallel | 81.25% (13/16) | 81.25% (13/16) | 0.00 |
| live_parallel_multiple | 79.17% (19/24) | 75.00% (18/24) | +4.17 |
| live_irrelevance | 75.34% (666/884) | 74.89% (662/884) | +0.45 |
| live_relevance | 75.00% (12/16) | 75.00% (12/16) | 0.00 |
| **All single-turn entries, micro-average** | **81.87%** (2,981/3,641) | **81.30%** (2,960/3,641) | **+0.58** |
| BFCL Non-Live AST accuracy (group mean) | 87.02% | 87.31% | -0.29 |
| BFCL Live accuracy (group mean) | 80.83% | 79.64% | +1.19 |

Function-calling quality is the same within noise: the two arms agree exactly on five categories, and no category
moves by more than three entries except `live_simple` (+10 for this build). Every one of the 3,641 requests returned
a response on both servers. Multi-turn and agentic BFCL categories were not run. Forced tool choice (a named function
or `tool_choice: required`) is not part of this suite and is rejected by NInfer by upstream policy.
