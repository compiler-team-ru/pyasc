# Troubleshooting: PyAsc environment and build

Open this when a phase of the skill fails or a check reports something unexpected. Match the symptom, apply the fix. Every fix that installs or changes anything still requires asking the user first. `$VENV` is the chosen virtual environment, `$BUILD_ROOT` the repository root, `$P` the LLVM toolchain path. Do not hardcode versions — check the pins in `pyproject.toml` / `requirements-build.txt` and the build docs.

## LLVM

| Symptom | Cause | Fix |
|---|---|---|
| No candidate found anywhere (Phase 1 search) | Toolchain not installed | Report, then offer to download the pre-built LLVM+MLIR tarball (version and x86_64/aarch64 links in `experimental/asctile/docs/installation/build-from-source.rst`); ask before downloading |
| Candidate fails the checklist: `MLIRConfig.cmake` / headers / `mlir-tblgen` missing | Incomplete installation, or a distro package (`/usr/lib/llvm-*`) picked up by mistake | Report exactly which paths are missing; keep searching for the pre-built tarball — distro packages are not documented or tested |
| `llvm-config --version` differs from the documented toolchain version | Wrong LLVM version; the build does not constrain it at configure time and fails later | Stop and ask; use the documented version |
| `Permission denied` running `$LLVM_INSTALL_PREFIX/bin/*` | Lost execute bits | Ask the user, then `chmod +x $LLVM_INSTALL_PREFIX/bin/*` |
| `which FileCheck` is empty after a successful build | `$P/bin` not on PATH | `export PATH="$LLVM_INSTALL_PREFIX/bin:$PATH"` — lit/pytest from the repo root and the `pyasc-generate-pass-test` skill need FileCheck on PATH |

## Build tools and build

| Symptom | Cause | Fix |
|---|---|---|
| `RuntimeError: cmake is required` during the install | CMake/Ninja are not on PATH for a `--no-build-isolation` install (including editable) — often because they exist only in an unactivated venv | `source "$VENV/bin/activate"`, then re-run the install |
| System CMake picked up, version outside the project pin | The project pins cmake in `pyproject.toml` / `requirements-build.txt`; a too-new major can break the build | Use `$VENV/bin/cmake` matching the pin; do not approve an out-of-pin system CMake |
| C++ compilation errors during the build | Compiler/LLVM skew, or an unsupported Python/dependency combination | Check the interpreter against the `setup.py` classifiers and the dependency pins (Phase 3.2); do not treat 3.13+ as supported without user confirmation. For a gcc fallback, unset `PYASC_SETUP_CLANG_LLD` and clang-specific compiler/linker overrides; handle a cached toolchain as in SKILL.md Phase 4 |
| Build recompiles everything from scratch | The toolchain/configuration changed vs the cache (compiler, linker, Debug↔Release) — a full recompile is expected; ccache only makes it cheaper | Match the cached compiler/linker (see SKILL.md Phase 4) or accept the recompile with `PYASC_SETUP_CCACHE=1` |
| Experimental code still built after switching to Standard without asctile | `CANN_ASC_USE_EXPERIMENTAL=ON` persists in `CMakeCache.txt` — unsetting `PYASC_SETUP_EXPERIMENTAL` does not pass OFF | Ask the user, then either delete the cache or override explicitly: `export PYASC_SETUP_CMAKE_APPEND="-DCANN_ASC_USE_EXPERIMENTAL=OFF"` |
| Dependency has no wheel for the interpreter (e.g. numpy on a newer Python) | The pin predates the interpreter; a locally cached/source-built wheel masks this on the current machine only | Stop and ask: a source build or a pin update may be needed. Probe with `--no-cache-dir`, otherwise it can return the local wheel and pass |
| `python3 -m venv` fails (ensurepip missing) | `python3-venv` not installed (Ubuntu splits it out) | Ask, then `sudo apt install python3-venv` |
| CMake configure fails: `Python.h` / `Development.Module` not found | Python development headers missing | Ask, then install the dev package (`python3-dev` on Ubuntu, `python3-devel` on dnf systems) |
| Linking fails on `libzstd` | System library missing (the quick start checks `libz` and `libzstd`) | Check `test -f /usr/lib/$(uname -m)-linux-gnu/libzstd.so`; if missing, ask, then `sudo apt install libzstd-dev` |
| Build fails with MLIR not found / `cmake is required` although everything is installed | Env vars exported in a previous shell call did not survive — each tool call may be a new process | Re-export the Phase 2 variables with the build command, or run the flow in one persistent shell |

## Virtual environment

| Symptom | Cause | Fix |
|---|---|---|
| `source "$VENV/bin/activate"` fails | Broken venv | Do not delete it yourself — ask the user to remove it or point to a working one |
| pip installs into system Python despite the venv | Wrong interpreter invoked | Use the venv's binaries explicitly: `"$VENV/bin/python" -m pip` |

## Runtime (post-build — beyond this skill's scope, for quick triage only)

| Symptom | Cause | Fix |
|---|---|---|
| `from asc.experimental import asctile` fails after a rebuild | Experimental modules may have been skipped, or the install/import used a different environment | Check the traceback and `"$VENV/bin/pip" list`; if experimental modules are absent because the build lacked `PYASC_SETUP_EXPERIMENTAL=1`, rebuild with it exported |
| `ascir-opt` not on PATH after an editable install | Editable installs do not guarantee devtools on PATH | Check `$VENV/bin`; rebuild with the Developer (non-editable) install if needed, or accept the limitation |
| `ASCEND_HOME_PATH is not set` | CANN toolkit not installed/activated | Only needed for simulator/NPU runs; see `experimental/asctile/docs/installation/setup-runtime-env.rst` (`source /usr/local/Ascend/cann/set_env.sh`) |
| Simulator tests fail to load libraries | `LD_LIBRARY_PATH` missing the simulator libs for the target SoC | `export LD_LIBRARY_PATH="$ASCEND_HOME_PATH/tools/simulator/$SOC_VERSION/lib:$LD_LIBRARY_PATH"` — `SOC_VERSION` is the target SoC (Model backend defaults to `Ascend910B1`; the docs example is `Ascend950PR_9599`) |

## Harmless observations

- A stale `build/` directory with the **same** configuration: pip reuses/overwrites it — no action needed. A configuration or toolchain **switch** triggers a reconfigure/recompile (see SKILL.md Phase 4); ask before deleting the cache.
