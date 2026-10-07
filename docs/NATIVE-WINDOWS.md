# Native Windows build

Branch `native-windows` builds NInfer with MSVC 2022 and the Windows CUDA 13.4 toolkit, no WSL. The engine, the
OpenAI/Anthropic server and the test suite compile; media decoding (FFmpeg) and remote media URLs (libcurl) are
compiled out by default on Windows and fail with an explicit unsupported-media error.

## Requirements

- Windows 11 x64, GeForce RTX 5090 (`sm_120a`), driver 591.86 or newer.
- Visual Studio 2022 with the MSVC v143 14.41 toolset (`-vcvars_ver=14.41`; newer toolsets may not be accepted by
  nvcc).
- CUDA Toolkit 13.4 for Windows at `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4` (nvcc, cudart, CCCL,
  NVTX; the display driver is not required from the toolkit installer).
- CMake 3.28+, Ninja, Python 3 (tests only).

## Build, serve, test

```bat
build-windows.bat configure   :: Ninja + MSVC 14.41 + CUDA 13.4, FFmpeg/curl off, tests on
build-windows.bat build       :: ninfer-serve.exe -> build-win\apps\
build-windows.bat test        :: all targets + ctest
```

```powershell
.\serve-windows.ps1 -Artifact C:\models\qwen3_8_27b_gdn4_a4_draftq4.ninfer -Port 9932
```

`serve-windows.ps1` passes the same flags as `recipes/rtx5090-qwen38-27b/serve.sh`. The artifact must be on a local
NTFS volume (the direct-read path uses `FILE_FLAG_NO_BUFFERING`); a `\\wsl$` path is not supported.

## Measured

Same machine as the Linux recipe (RTX 5090, driver 591.86), the selected draft-Q4 artifact copied byte-exact to NTFS,
SPEED-Bench `throughput_8k` (15 requests, 256 output tokens, one request at a time), two alternating rounds against
the same source built in WSL (CUDA 13.4.2, GCC):

| Round | WSL build (tok/s) | Native Windows build (tok/s) |
| --- | ---: | ---: |
| 1 | 250.7 | 247.1 |
| 2 | 250.9 | 248.5 |

Outputs are identical on all 15 requests in every run (2,736 accepted of 7,486 drafted tokens, 3,840 completion
tokens). Native Windows decodes about 1% slower than the WSL build of the same source in this session; the WDDM driver
path does not remove the per-round host launch cost that WSL pays. Engine load from NTFS: 7.9 s for 19.3 GiB of
weights with the file in the OS cache.

CTest on Windows: see "Test results" at the end of this file. Two Windows-only findings while getting there:

- CTest 4.3 on Windows leaks an `ENVIRONMENT "VAR="` (empty value) setting into every later test in the same
  run, because Windows cannot hold an empty-valued environment variable and CTest's restore leaves a stale entry
  in its own process. The device-sync "empty" test is therefore Linux-only.
- A Windows clone with `core.autocrlf=true` checked out the `.jinja` template fixtures with CRLF, which the frontend
  test's renderer rejects. `.gitattributes` now pins LF for all text files (CRLF only for `.bat`/`.ps1`).

## What the port changes

| Area | Change |
| --- | --- |
| 128-bit integers | `unsigned __int128` (4 sites: prefill work, resource ranking, context cost, option parsing) replaced by `src/core/wide_math.h`, a two-limb helper used on every platform |
| Artifact reads | `src/artifact/file_io_windows.cpp`: `ReadFile` with `OVERLAPPED` offsets, `FILE_FLAG_NO_BUFFERING` for the aligned direct path |
| TMA kernel parameters | CUDA declares `CUtensorMap` `alignas(128)`; MSVC refuses by-value parameters above 64-byte alignment (C2719), and nvcc passes `__grid_constant__` parameters by value in the host stub. `src/ops/common/tma_param.cuh` carries each map as a 64-byte-aligned byte copy (PTX only needs 64) |
| Media libraries | `NINFER_ENABLE_FFMPEG` / `NINFER_ENABLE_CURL` options (default off on Windows); `decode_stub.cpp` and a curl-free `acquire.cpp` path |
| MSVC flags | `/bigobj`, `/utf-8`, `/Zc:preprocessor` (required by CCCL), `/Zc:__cplusplus`, `NOMINMAX`, `WIN32_LEAN_AND_MEAN` |
| POSIX call sites | `getpid`, `isatty`, `localtime_r`, `gmtime_r`, `ioctl(TIOCGWINSZ)` and the llama-jinja `strftime_now` helper behind `#ifdef _WIN32` |
| C++20 strictness | `shared_ptr::unique()` (removed in C++20) replaced by `use_count() == 1`; `<array>` includes made explicit; `static constexpr` kernel locals so MSVC lambdas need no capture; `std::sqrt` not used in `constexpr` initialisers (tests) |
| Tests | `pipe`/`dup2` capture through the CRT equivalents, portable `mkdtemp`, `_aligned_malloc`; the GNU ld `--wrap` fault-injection test and the BSD-socket transport test are Linux-only; a helper named `near` renamed (windows.h macro) |

## Test results

Native Windows, `ctest --test-dir build-win -j1` (MSVC 14.41, CUDA 13.4.59, no FFmpeg/curl): **126 of 126 passed**
in 945 s; the 11 real-artifact suites report "skipped" until `NINFER_TEST_ARTIFACT` points at an artifact on NTFS.
Compared with the Linux suite list, four tests are absent on Windows by design: the artifact materialization
fault-injection test and its writer-interop companion (GNU ld `--wrap`), the BSD-socket HTTP transport test, and the
empty-`NINFER_CUDA_SYNC` device test; the media decode test and the frontend test's media sub-tests only run when
FFmpeg is enabled.

Real-artifact suites natively (`run-real-tests.ps1 -Artifact <NTFS path>`): native transactions, preemption, score,
vision workspace, DFlash2, DFlash prefill and agent continuation all pass; `loading_real` skips as it does on Linux;
all eleven `NINFER_PREFIX_REAL_SCENARIO` runs of the prefix suite pass (the `stream-observations` scenario crashed
once while a Linux test run was sharing the GPU and then passed four consecutive reruns). The MoE and DFlash-v1
suites need other artifacts, as on Linux.

Linux check of the same branch (WSL2, CUDA 13.4.2, GCC, full rebuild, suite run with the selected artifact):
**127 of 128 pass**, including the real-artifact suites and all prefix-reuse scenarios. The one miss is the
softmax-attention oracle test, which passed in 400 s on the pre-port tree the same morning and then timed out at the
int8 variant on the port tree. The WSL toolchain was unhealthy at that point: nvcc and nvlink segfaulted repeatedly
on the int8 translation units and the device link, the kernel reported a bad-page taint, and two forced WSL restarts
left corrupted journals, so the int8 object is suspect rather than the port (the only source change in that path is
a `static constexpr` kernel-pointer local, and the pre-port server runs `--kv-dtype int8` correctly on the same
machine). A clean rebuild of the ops library is the pending confirmation; this line will be updated with its result.
