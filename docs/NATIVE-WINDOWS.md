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

CTest on Windows: see the line recorded at the end of this file.

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

Linux behaviour is unchanged: the same source passes the Linux suites that cover the touched code (serve options,
context cost, resource manager, request log, logging, artifact reader/materialization, media, prompt input,
OpenAI/Anthropic schemas).
