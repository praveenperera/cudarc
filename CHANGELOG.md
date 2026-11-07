# Changelog

This file covers `cudakit-cudarc`, a fork of
[cudarc](https://github.com/chelsea0x3b/cudarc). Each version builds on the upstream
release with the same number and changes only dynamic library loading. Upstream's own
changes are in its repository and releases.

## 0.19.10

Based on upstream cudarc 0.19.10.

### Changed

- Linux: versioned sonames come before the unversioned dev symlink. The first is the
  soname the bindings' CUDA version ships. That is usually its major, but some
  libraries number their own: CUDA 12's cuFFT and cuSOLVER are `.so.11`, cuRAND is
  `.so.10`, cuDNN `.so.9`, NCCL `.so.2`, cuFile `.so.0`, and the driver and NVTX
  `.so.1`. Then come 13, 12, 11, 10, 9, 2 and 1, so `libcublas.so.13` is tried before
  `libcublas.so`. This stops a process from holding two cuBLAS copies, or a cuBLAS and
  cuBLASLt from different installs. The CUDA 11 sonames (`libcudart.so.11.0`,
  `libnvrtc.so.11.2`, `libcupti.so.11.<minor>`, `libcufft.so.10`) are now candidates too.
- Windows: CUDA 13 DLL names (`cublas64_13.dll`, `cublasLt64_13.dll`,
  `nvrtc64_130_0.dll`) are candidates for every build, not just CUDA 13 builds, and the
  build's own major is tried first. `%CUDA_PATH%\bin\x64` and `%CUDA_PATH%\bin` are
  searched after `PATH`.
- Windows: the CUDA driver loads only from System32. This builds on upstream PR #608,
  and loads the driver by its full path from `GetSystemDirectoryW`, because a bare
  name, even with `LOAD_LIBRARY_SEARCH_SYSTEM32`, returns a module of that name already
  loaded from anywhere.
- Windows: no other library loads from the current directory or a relative `PATH`
  entry. A name is opened from a module already loaded under it, then from the
  executable's directory, System32 or `AddDllDirectory` directories
  (`LOAD_LIBRARY_SEARCH_DEFAULT_DIRS`), then by full path from each absolute `PATH`
  entry and `CUDA_PATH`, with `LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR` so its imports come
  from its own directory. A file that fails to load moves on to the next directory or
  name, never to Windows' default search. A DLL whose imports are in another
  directory that only `PATH` reaches, such as one of NVIDIA's pip wheel `bin`
  directories, needs those imports loaded first or their directory added with
  `AddDllDirectory`.
- `cublas::result::hgemm` and `hgemm_strided_batched` use the bindings' cuBLAS handle.
  Before, they opened `cublas.dll` or `libcublas.so` themselves with the default search,
  which on Windows always reached the current directory, since no install ships
  `cublas.dll`.
- These Windows rules cover the loads cudarc makes. A library can load more DLLs at
  runtime with its own search. cuDNN 9's `cudnn64_9.dll` loads `cudnn_graph64_9.dll`
  and its other sublibraries by bare name, which cudarc doesn't control.
- `is_culib_present()` and `culib()` share one handle per library. A library that
  `is_culib_present()` finds stays loaded, and neither function searches again after a
  library has loaded. A failed search is retried on the next call.

### Tested

- CI, without a GPU: Linux uses stub cuBLAS and cuBLASLt libraries for a toolkit and a
  pip wheel (`tests/linux_sonames.rs`), and these tests fail on upstream 0.19.10.
  Windows uses copies of `version.dll` for the current directory, relative and absolute
  `PATH` entries, a broken DLL, `CUDA_PATH`, already-loaded modules and the
  System32-only driver. The driver cases cover a `cuda.dll` already loaded from elsewhere,
  and a System32 DLL that must load from System32 while a same-named copy from another
  directory is loaded (`src/loading.rs`). On Linux, a stub whose constructor
  leaves a marker file shows that a library that has loaded is not searched for again.
- `cargo clippy` for `x86_64-unknown-linux-gnu` and `x86_64-pc-windows-msvc`.
- GPU: cudarc's own lib tests (`cargo test --lib -F <version>,f16`), run on Linux with
  an RTX 5090, driver 595.99.02 and the CUDA 13.2 toolkit, for both this fork and
  upstream 0.19.10. Both fail the same tests:
  - `cuda-12040`: 4 tests. Grouped GEMM's deliberate CUDA 12.5+ panic, a pinned-copy
    timing check, and CUDA 13's NVRTC rejecting `compute_53`.
  - `cuda-13040`: 51 tests, failing in handle creation. The cause was not
    investigated, since upstream fails the same tests on the same machine.
  The fork passes every other test, plus its own loading tests.
- Not run: anything on Windows with a GPU or a CUDA install, cuDNN on a GPU, and the
  consumers (cudakit, sam3rs, scriptrs, speakrs) against this fork.
