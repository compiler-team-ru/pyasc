---
name: pyasc-check-env-and-build
description: Use when the user asks to build or install PyAsc or set up its build environment, or when a PyAsc build fails - missing LLVM/MLIR toolchain, MLIRConfig.cmake not found, "cmake is required", compiler/CMake/venv/Python version issues, or `from asc.experimental import asctile` failing after a rebuild.
---

# PyAsc: Check Environment and Build

## Core Principles

- **Build root**: the directory containing `pyproject.toml` (the repo root). Locate it first — whatever directory you were started in — and run every command from there: `BUILD_ROOT=$(git rev-parse --show-toplevel)` (verify `pyproject.toml` is there). Below: `$BUILD_ROOT` — repo root, `$VENV` — the venv (Phase 3.1), `$P` — the LLVM toolchain path (Phase 1). `<...>` in bash is a redirection, never copy it literally.
- Each shell call may be a fresh process: run the flow in one persistent shell, or re-export the Phase 2 variables with every build command. Exports do not survive across separate tool calls by themselves.
- A pre-built LLVM+MLIR toolchain is the hard prerequisite; `LLVM_INSTALL_PREFIX` is not pip/apt-installable. Compilers, CMake, Ninja and Python dependencies are.
- **The build configuration is the user's call.** Only an explicit statement in this conversation ("build Debug") is an answer. An existing venv, a source checkout, a prior `build/` cache, delegation ("up to you"), and silence are context, not answers. Never infer — ask (Phase 2) and wait; no reply is not consent, never fall back to a default.
- Never install packages, run `sudo`, download files, or create/overwrite a venv without asking the user first. Never install into the system Python: run the venv's interpreter explicitly — `"$VENV/bin/python" -m pip …` — or an activated venv; bare `pip`/`python3` resolve to the system interpreter.
- Do not modify source files (`setup.py`, `CMakeLists.txt`, requirements, ...) to "fix" a build — report the problem and ask the user.
- Do not hardcode tool versions from memory: sources of truth are `pyproject.toml`, `requirements-build.txt`, the classifiers in `setup.py`, and `experimental/asctile/docs/installation/build-from-source.rst`. Check against them; they change.
- If a check fails or reports something unexpected, open `references/troubleshooting.md` (same directory as this skill) before improvising.

## Phase 1: LLVM Toolchain

1. If `LLVM_INSTALL_PREFIX` is set, verify it directly; if it fails the checklist, continue the search. Otherwise search, in order: the build root itself (`$BUILD_ROOT/llvm*` — the quick start unpacks the tarball there), the parent of the build root (the most common location), `/usr/local/llvm-*`, `/usr/local/llvm`, `/opt/LLVM-*`, `/opt/llvm-*` (the Docker image uses `/opt`), `~/llvm-*`, `~/llvm`, `~/.local/llvm-*`. Distro packages (`/usr/lib/llvm-*`) are not documented or tested — verify a candidate with the checklist, but prefer the tarball.
2. A candidate is complete if (set `P` to its path):
   ```bash
   test -f "$P/lib/cmake/mlir/MLIRConfig.cmake" && test -f "$P/lib/cmake/llvm/LLVMConfig.cmake" \
     && test -d "$P/include/mlir" && test -d "$P/include/llvm" \
     && test -x "$P/bin/mlir-tblgen" && test -x "$P/bin/FileCheck" \
     && "$P/bin/llvm-config" --version
   ```
   The version must match the one documented in `build-from-source.rst` (it appears in the tarball file name). The build does not constrain the LLVM version at configure time — a wrong version passes configure and fails later. On a mismatch, stop and ask.
3. Found → export for this session:
   ```bash
   export LLVM_INSTALL_PREFIX="$P"
   export PATH="$P/bin:$PATH"   # FileCheck lives here; lit/pytest from the repo root and the pyasc-generate-pass-test skill need it on PATH (may shadow a system clang)
   ```
4. Not found → report, then offer to download the pre-built LLVM+MLIR tarball (version and x86_64/aarch64 links in `build-from-source.rst`). Ask before downloading.

## Phase 2: Build Configuration

Ask exactly this (skip only if the user already stated the configuration explicitly in this conversation; if ambiguous, clarify). State the question **before** any recommendation — "I'll do X unless you object" is choosing, not asking. Ask once, then proceed without re-asking (ask-before-install still applies to every install):

> Which build configuration do you need?
> - **Developer build** (Debug): includes devtools (ascir-opt, ascir-lsp, ascir-translate), ccache, clang/lld.
> - **Standard build** (Release): minimal build for running kernels. Debug tools will not be available.
> Need a different combination (e.g. Release with devtools, RelWithDebInfo)? State it explicitly.

`setup.py` applies a default only when a variable is **unset** — exports left over from a previous build in the same session silently change the configuration. Each block below therefore sets its variables explicitly; the Standard block also unsets the options it turns off.

**Standard build**:

```bash
export LLVM_INSTALL_PREFIX="$P"
export PYASC_SETUP_CONFIG=Release
unset PYASC_SETUP_DEVTOOLS PYASC_SETUP_CCACHE PYASC_SETUP_CLANG_LLD PYASC_SETUP_JOBS PYASC_SETUP_COMPILER PYASC_SETUP_LINKER PYASC_SETUP_EXPERIMENTAL
```

**Developer build**:

```bash
export LLVM_INSTALL_PREFIX="$P"
export PYASC_SETUP_CONFIG=Debug
export PYASC_SETUP_DEVTOOLS=1
export PYASC_SETUP_CCACHE=1
export PYASC_SETUP_CLANG_LLD=1   # only if clang++ and ld.lld exist (Phase 3.3); else unset it and build with gcc
export PYASC_SETUP_JOBS=$(nproc)
```

**For BOTH configurations** — if the work involves asctile (anything under `experimental/asctile`, including its tests), also `export PYASC_SETUP_EXPERIMENTAL=1`. Without it the experimental modules are silently skipped from the build and `from asc.experimental import asctile` fails afterwards. Standard + asctile = the Standard block above + this export, nothing more. (`PYASC_DUMP_PATH` is a runtime JIT setting, not a build option.)

## Phase 3: Dependencies

### 3.1 Virtual environment (find it first — the checks below use it)

Check in order: `$VIRTUAL_ENV` (activated venv — use it); then the build root (`.venv` is the most common name there, hidden — glob `*venv*` misses it):
```bash
find "$BUILD_ROOT" -maxdepth 1 -type d \( -name '*venv*' -o -name '*pythonenv*' \) 2>/dev/null
```
then `$HOME/.pyasc_venv` (recommended by `build-from-source.rst`).

One found → use it (`VENV=<path>`), no question. Several → ask which. None → check `python3 --version` against the `setup.py` classifiers first (Phase 3.2), then ask whether to create one (verify the target does not exist, then `python3 -m venv .venv --prompt pyasc`, `VENV=$BUILD_ROOT/.venv`). User declines → stop and ask for a path to an existing venv; never proceed with `$VENV` undefined; never fall back to system installs. Never create or overwrite a venv that already exists.

### 3.2 Python

Check `"$VENV/bin/python" -V` against the classifiers in `setup.py`, and that the dependency pins have wheels for that interpreter — both `requirements-build.txt` (the `--no-build-isolation` path) and `install_requires` in `setup.py` (the isolated install / runtime path). The probe downloads from the network — ask first. `--no-cache-dir` is mandatory (a locally cached wheel is not a PyPI wheel and masks exactly the failure a fresh machine hits):

```bash
"$VENV/bin/python" -m pip download --no-cache-dir --only-binary=:all: --no-deps \
  -d "$(mktemp -d)" '<pin from requirements-build.txt / setup.py>'
```

Interpreter unsupported by the classifiers or `setup.py`, or a pin with no wheel → **stop and ask** (a source build or pin update may be needed); proceed only on the user's explicit confirmation and state it in the summary. A network error is not "no wheel": distinguish offline from a missing distribution.

### 3.3 Compiler

`gcc` works; `clang++`+`lld` are faster (the Developer build sets `PYASC_SETUP_CLANG_LLD=1`; `setup.py` invokes `clang++`, so check `clang++` and `ld.lld`, not just `clang`). Missing → report and ask before installing. If the user declines clang/lld → unset `PYASC_SETUP_CLANG_LLD` and clang-specific compiler/linker overrides; handle the CMake cache as in Phase 4, and tell the user. System package prerequisites are listed in `build-from-source.rst` (Ubuntu/dnf).

### 3.4 CMake and Ninja

Check the **virtual environment first** — both are pip packages of this project (`build-system.requires` in `pyproject.toml`):
```bash
"$VENV/bin/cmake" --version
"$VENV/bin/ninja" --version
```
Versions must satisfy the pins in `pyproject.toml` / `requirements-build.txt` (a CMake outside the pinned range can break the build). Then the system PATH, against the same pins. For isolated installs, pip supplies these tools itself — no venv install is needed. For `--no-build-isolation`, missing in both → install into the venv (ask first): `"$VENV/bin/python" -m pip install -r requirements-build.txt`. No venv → stop (Phase 3.1). CMake/Ninja missing from the system PATH is normal on this project — not an error.

### 3.5 Python build dependencies

`requirements-build.txt` is the project's build-dependency list (a superset of `build-system.requires`). For `--no-build-isolation`, check it against `"$VENV/bin/pip" list`. Missing → ask before installing: `"$VENV/bin/python" -m pip install -r requirements-build.txt`. Isolated installs do not need these packages preinstalled in the venv.

### 3.6 CANN toolkit (conditional)

Only if the user needs to run kernels on a simulator or NPU — skip for build-only workflows. Check `/usr/local/Ascend/cann/set_env.sh` (or `~/Ascend/cann/set_env.sh`), activate it, and for the simulator also:
```bash
export LD_LIBRARY_PATH="$ASCEND_HOME_PATH/tools/simulator/$SOC_VERSION/lib:$LD_LIBRARY_PATH"
```
`SOC_VERSION` is the target SoC (Model backend default `Ascend910B1`, see `python/asc/runtime/config.py`; the asctile docs use `Ascend950PR_9599`). Not installed → report; installation is in `experimental/asctile/docs/installation/setup-runtime-env.rst`.

## Phase 4: Build

Run from the build root with the Phase 2 environment exported (re-export if this is a fresh shell). Always use the venv's interpreter — the install destination is whichever pip you run:

- **`--no-build-isolation` installs (including editable with `--no-build-isolation`)**: activate the venv (`source "$VENV/bin/activate"`). `setup.py` locates CMake and Ninja on PATH (`shutil.which`); system installations also work, but if they exist only in the venv, without activation the build fails with `RuntimeError: cmake is required`.
- **Isolated installs (`pip install .`, plain `pip install -e .`)**: pip installs the build dependencies into an isolated environment itself (needs network), so activation is not needed for cmake/ninja — but still run `"$VENV/bin/python" -m pip`, never bare `python3 -m pip`.

Pick by purpose:

| Target | Commands | Notes |
|---|---|---|
| Standard install | `"$VENV/bin/python" -m pip install .` | Simplest; Release defaults |
| Developer install | `"$VENV/bin/python" -m pip install -r requirements-build.txt` then `"$VENV/bin/python" -m pip install --no-build-isolation . -v` | No build isolation; puts ascir-opt, ascir-lsp, ascir-translate on PATH (plus asctile-opt/asctile-lsp with `PYASC_SETUP_EXPERIMENTAL=1`) |
| Editable install | `"$VENV/bin/python" -m pip install -e .` | Modify Python files without reinstalling; devtools on PATH are **not guaranteed** — check `$VENV/bin` |
| Wheel | `"$VENV/bin/python" -m pip wheel --wheel-dir=wheels --no-deps .` | For installing elsewhere; build only — install the wheel before verifying the installation |

**Reusing an existing `build/` cache**: unsetting a `PYASC_SETUP_*` variable does **not** reset options already stored in the cache — e.g. `CANN_ASC_USE_EXPERIMENTAL=ON` persists in `build/cmake*/CMakeCache.txt` (default location; `PYASC_SETUP_BUILD_DIR` moves it) when switching to Standard without asctile. On a configuration switch, ask before deleting the cache, or override the cached option explicitly via `PYASC_SETUP_CMAKE_APPEND`. Also check the cached compiler **and** linker in `CMakeCache.txt`: if the previous build used different ones, set `PYASC_SETUP_COMPILER`/`PYASC_SETUP_LINKER` to match (the linker value is passed to `-fuse-ld=`: use `lld`, not `ld.lld`) or CMake recompiles everything. The cache tells you *how* the last build ran; it never answers *which* configuration to build now (Phase 2).

**Verify after installing:**

```bash
"$VENV/bin/python" -c "import asc; print('asc OK')"
"$VENV/bin/pip" list | grep pyasc        # default distribution name; with PYASC_SETUP_NAME check that name
"$VENV/bin/python" -c "from asc.experimental import asctile; print('asctile OK')"   # only meaningful with PYASC_SETUP_EXPERIMENTAL=1
for t in ascir-opt ascir-lsp ascir-translate; do test -x "$VENV/bin/$t" || echo "MISSING: $VENV/bin/$t"; done   # only after the Developer (non-editable) install
for t in asctile-opt asctile-lsp; do test -x "$VENV/bin/$t" || echo "MISSING: $VENV/bin/$t"; done               # only after the Developer install with PYASC_SETUP_EXPERIMENTAL=1
```

`import asctile` at top level never works — the package is `asc.experimental.asctile`. Don't use it as a check.

This skill ends at the verified build. Running tests and debugging failures is out of scope (see the Testing section of `experimental/asctile/AGENTS.md`).

## Rationalizations

| Excuse | Reality |
|---|---|
| "The user is obviously a developer — venv and checkout say so" / "They delegated the choice" / "Asking would bother them" | Directory contents, delegation ("up to you", "as you see fit"), and deadline pressure are context, not an answer. Only an explicit user statement counts. Ask the Phase 2 question. |
| "No answer came — I'll keep the existing / a default configuration" | Silence is not consent. Stop and wait. |
| "The previous configuration is recoverable from CMakeCache.txt" | The cache records a past build, not the user's current choice. Ask. |
| "The venv exists — bare `pip`/`python3` will use it" | Bare pip/python resolve to the system interpreter unless the venv is activated. Use `"$VENV/bin/python" -m pip`. |
| "Standard uses defaults — nothing to reset" | Defaults apply only to unset variables; stale exports from a previous build silently change the configuration. Unset them (Phase 2). |
| "The wheel probe passed — the interpreter is fine" | Without `--no-cache-dir` the probe can return a locally built wheel that does not exist on PyPI; a fresh machine will still fail. |
| "cmake is not on the system PATH — install it with apt" | CMake and Ninja are pip packages here; check the venv first. Missing from PATH is normal. |
| "`pip install -e .` is the developer build with devtools" | Editable installs do not guarantee ascir-* on PATH (check `$VENV/bin`). Use the Developer (non-editable) install when devtools are needed. |
| "EXPERIMENTAL is exotic, skip it" / "I'll verify with `import asctile`" | Without `PYASC_SETUP_EXPERIMENTAL=1` asctile is silently skipped and its import fails after every rebuild; the check is `from asc.experimental import asctile`. |
| "The build fails — I'll patch setup.py/CMakeLists.txt to fix it" | Source files are not yours to modify. Report the problem and ask. |
| "This numpy/LLVM/CMake version worked before" / "The previous build on this Python succeeded" | Check the current pins and docs, not memory; a past success (possibly on a locally built wheel) is not project support — the classifiers/pins gate still applies (Phase 3.2). |

## Red Flags — STOP

- About to run `sudo`, `apt`, `pip install`, or download anything without asking — or run the wheel probe without asking / without `--no-cache-dir`.
- About to run bare `pip`/`python3` for an install or verification instead of `"$VENV/bin/python"`.
- About to pick the configuration from context clues, state your pick before asking, treat "up to you" as an answer, or proceed on a default because the user didn't reply.
- About to create a venv without having checked `$VIRTUAL_ENV`, the build-root `find`, and `$HOME/.pyasc_venv` — or before checking the interpreter against the classifiers.
- About to build Standard without unsetting the previous configuration's variables, or set `PYASC_SETUP_CLANG_LLD=1` without checking clang++/ld.lld.
- About to verify the build with `import asctile`, or edit `setup.py`, `CMakeLists.txt`, or a requirements file to make the build pass.
- About to delete `build/` (or any cache) without asking, or build on an interpreter newer than the classifiers without explicit confirmation.
