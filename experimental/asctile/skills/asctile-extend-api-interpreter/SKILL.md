---
name: asctile-extend-api-interpreter
description: Use when adding a public asc.experimental.asctile API or changing its signature, behavior, return type, or LocalTensor binding, including interpreter-mode failures, missing NumPy overrides, and tensor-method/operator mismatches.
---

# AscTile: Extend API Interpreter Support

Run this after the language-side work on a new or changed public asctile API. This workflow requires explicit interpreter support for public kernel-body APIs: a missing override must not be hidden by unrelated green tests or a `no_interpret` skip. The developer/API owner must decide whether an API is host-only; record that decision in the report and do not give the API a meaningless kernel-body override.

Understand the dispatch mechanism before diagnosing failures:

- `experimental/asctile/python/asctile/interpreter/patch.py::make_module` copies only exports that have no `__wrapped__`, then installs registered overrides. `@require_jit` produces `__wrapped__` through `functools.wraps`, but any other decorator that sets `__wrapped__` has the same effect.
- `interpret_asctile` patches the `asctile`, `range`, and sometimes `asc` globals of the current JIT function. Direct imports such as `from asc.experimental.asctile import add` are not replaced. A JIT-decorated device function patches its own globals when invoked, but a plain helper defined in another module is not covered by its caller's namespace patch.
- Module-level operations normally belong in `experimental/asctile/python/asctile/interpreter/operations.py`. Tensor methods are attached to the interpreter `LocalTensor` by the `bind_tensor_method` decorator in `experimental/asctile/python/asctile/interpreter/operations.py`; object-only methods, properties, and class APIs may instead require work in `experimental/asctile/python/asctile/interpreter/types.py`.

File references in the prose are repository-root-relative. Run commands from `experimental/asctile/`; command arguments use paths relative to that directory.

## Required Regression Test

Existing tests alone do not establish support for a new or changed API. Before implementing or fixing its override, add a focused test that exercises the changed contract and watch it fail for the expected reason under `ASCTILE_INTERPRET=1`. After the change, watch the same test pass in a fresh process. If the developer/API owner classifies an API as host-only, record the owner, rationale, and scope in the report and test the applicable host-side contract instead; no kernel-body interpreter test is required for that API.

Cover every applicable public call surface and changed behavior:

- module function, tensor method, renamed method, operator, and reflected operator forms;
- positional, keyword, and default arguments affected by the change;
- scalar/tensor inputs and their return types;
- relevant dtypes, shapes, broadcasting, validation, and exceptions;
- in-place behavior, mask, and location semantics when part of the API contract.

A focused test that is skipped or xfailed does not satisfy this gate. Use `no_interpret` only when the developer/API owner approves an unsupported API with an explicit stub or an interpreter-mode limitation; record the owner, rationale, and affected tests in the report. Here a limitation means behavior inherent to interpreter mode, not a missing override, binding, export, or another implementation gap. Never use the marker by itself to make a new API's verification green. For every approved skip path, add a separate unmarked focused contract test that executes and passes: verify the explicit stub's expected `NotImplementedError` or the approved limitation's documented behavior in the report. Confirm from the output that this contract test itself was executed, not skipped by a module-level marker or autouse fixture. The skipped end-to-end test alone never satisfies the gate.

## Verification Cascade

Set `ASCTILE_INTERPRET=1` before each pytest process starts. `interpreter_enabled()` is `lru_cache`d: setting `os.environ` inside a running session has no effect. Each stage is a fresh process.

1. `ASCTILE_INTERPRET=1 pytest python/test/operations/<actual_test_file>.py` — the real file containing the focused operation tests, such as `test_unary_ops.py` or `test_softmax.py`.
2. `ASCTILE_INTERPRET=1 pytest python/test/operations/` — catches indirect breakage in shared helpers and types.
3. Consumers: search `python/test/kernels/`, `python/test/target/`, and `python/tutorials/` for the public API name, method names (including renamed methods such as `reduce_sum` → `.sum()`), and applicable operator forms. Run every matching pytest file the same way. For a matching tutorial, use its documented invocation when the current environment supports it; otherwise report it as unrun rather than claiming coverage.

Record `passed`, `skipped`, and `xfailed` counts. A green exit status with the focused tests skipped is not verification.

This workflow verifies only the API under change. If a broader cascade stage exposes a failure outside that API, do not investigate, classify, register, or fix it as part of this skill; report that the broader stage was not clean. Such a failure does not become a work item in this pipeline.

Completion requires an executed, passing focused test from Required Regression Test (including the applicable host-only, stub, or limitation contract test), passing tests that cover the changed API, and all applicable Validation steps below. A failure that prevents those tests from running blocks completion.

## Triage the Current API

Diagnose only failures that exercise the API under change. Do not infer a root cause from exception text alone; inspect the relevant registry, binding, or call path.

| Symptom in the changed API | Required diagnosis |
|---|---|
| `AttributeError: module 'asctile' has no attribute '<op>'` | Inspect `__all__`, `__wrapped__`, and the override registry. A missing module-level override leads to the complexity fork. |
| `AttributeError: 'LocalTensor' object has no attribute '<op>'` | Inspect the language call surface, module-level override registry, and tensor bindings. The override may be absent, present without a binding, or irrelevant because this is an object-only API. Do not assume an override exists. |
| `RuntimeError: '<op>' cannot be called without initialization of global builder` | Check direct imports, cross-module plain helpers, and the override registry. This symptom does not prove that an override is absent. For an incidental direct import in the current JIT body, use the module-attribute call there. If direct-import support itself is the contract under test, do not rewrite the test to hide it; report the limitation and use the approved limitation path. For a cross-module plain helper, when the contract permits, use an `@asctile.jit` helper and module-attribute calls inside it; otherwise report the limitation and its completion status. An existing registry entry means no new override is needed. |
| Override exists but fails or returns wrong values | Compare the implementation with the current language contract and API diff, then repair the override or return to the complexity fork if the intended semantics are unclear. |
| Failure unrelated to the interpreter (environment, infra) | Stop and report to the developer. |

## Complexity Fork

A feature is **simple** only when its intended contract is clear and you can name the emulation: a direct analogue in the existing asctile API or a specific NumPy operation (or combination). A NumPy analogue alone is insufficient when signature, validation, return-type, dtype, side-effect, mask, or location semantics remain open. Such a feature is **complex** and requires research.

- Simple → implement after the focused test has demonstrated the missing or broken behavior.
- Complex → research and propose. Continue on this path even if research later makes the emulation look trivial.

## Implement: Simple Feature

Check the override registry before adding a definition: `@override` rejects duplicate names. Add module-level operations to `experimental/asctile/python/asctile/interpreter/operations.py`, next to overrides of the same category. Object-only APIs may belong in `experimental/asctile/python/asctile/interpreter/types.py`; do not force them into a module-level function.

Reuse existing helpers where their semantics fit: `unary_op`, `binary_op`, `apply_mask`, and `bind_tensor_method` from `experimental/asctile/python/asctile/interpreter/operations.py`; `LocalTensor`, `dtype_to_numpy`, and `cast_data` from `experimental/asctile/python/asctile/interpreter/types.py`.

Mirror the language-side **call surface**, not decorator syntax. The two `bind_tensor_method` helpers have different interfaces:

| Language side | Interpreter side |
|---|---|
| `@bind_tensor_method` | `@bind_tensor_method` |
| `@bind_tensor_method(name="__neg__")` | `@bind_tensor_method("__neg__")` |
| `binary_op=...` / `unary_op=...` | Do not copy these arguments. Select the required dunder binding from existing interpreter precedents. |
| An operator with reflected operands | Use `binary=True` only when the interpreter must also install the reflected `__r...` method. |

Some APIs require multiple bindings; see the interpreter `abs` override for both `.abs()` and `__abs__`. Adding a missing binding to an existing override may be the entire implementation change, but regression tests are still required.

Existing unary-op pattern with a tensor method:

```python
@bind_tensor_method
@override
def sqrt(input: Union[LocalTensor, Number]) -> LocalTensor:
    return unary_op(np.sqrt, input)
```

Binary op with a reflected operator:

```python
@bind_tensor_method("__add__", binary=True)
@override
def add(input: Union[LocalTensor, Number], other: Union[LocalTensor, Number]) -> LocalTensor:
    return binary_op(np.add, input, other)
```

For every simple implementation, check signature/defaults, accepted inputs, return type, relevant dtypes/shapes, validation/exceptions, and applicable side effects and contexts against the language contract.

## Research and Propose: Complex Feature

No clear contract or direct analogue → do not implement and do not guess. Research first, then bring the developer a proposal. The interpreter is a debugging aid whose value is fidelity; a subtly wrong override makes kernels debug correctly on CPU and silently diverge on hardware.

Research checklist:

1. **Compiled path — the source of truth.** Follow the op from `experimental/asctile/python/asctile/language/<module>.py` (signature, validation, docstring) through codegen (`python/asc/codegen/`, `FunctionVisitor`) to its MLIR operation, the `experimental/asctile/lib/Conversion/LowerToAsc/` lowering, and the emitted Ascend C. Extract the *observable* semantics: values, error behavior, and side effects.
2. **Precedent.** Existing overrides of similar ops — e.g. `copy` emulates `DistribMode` as logical slices via `block_context` (value semantics, not physical bank layout).
3. **Analogues.** How NumPy and torch express the parts of the semantics.
4. **Edge cases.** dtypes, shapes and broadcasting, scalar and return-type behavior, negative offsets, overflow, and error behavior.

Proposal format — bring the developer:

- 2–3 implementation variants with code sketches;
- a table of how the variants differ on the edge cases;
- trade-offs: fidelity to hardware behavior vs interpretation speed vs maintenance;
- your recommendation and the open questions.

**Approval gate:** no interpreter code until the developer picks a variant. If research concludes the feature is observationally trivial, that is variant 1 of the proposal — not a license to skip the gate.

If the feature cannot be interpreted at all (low-level hardware access, like `inline`/`from_asc`/`to_asc`), the proposal's variant is an explicit stub at the end of `experimental/asctile/python/asctile/interpreter/operations.py` next to the existing stub block, paired with `@pytest.mark.no_interpret` on the affected tests. Apply the owner-approval and separate executed contract-test requirements from Required Regression Test. The following uses an illustrative placeholder name:

```python
@override
def low_level_op(*args: Any, **kwargs: Any) -> None:
    raise NotImplementedError("'low_level_op' is not supported by the interpreter")
```

## Validation

After implementing or fixing the override:

1. Re-run the full verification cascade with `ASCTILE_INTERPRET=1`, including the focused test. Report executed/skip/xfail counts and recorded owner approvals for every approved skip path. If a broader stage is not clean because of failures outside the changed API, report that fact without adding those failures to this workflow.
2. Run affected operation and consumer tests **without** the env var so the compiled path does not regress. Use the configured project runtime; for a standard CANN installation, `source /usr/local/Ascend/cann/set_env.sh` (or `~/Ascend/cann/set_env.sh` for a user installation) puts `bisheng` on `PATH`. For the Model backend, add the simulator directories for the installed CANN/platform to `LD_LIBRARY_PATH`. Missing `bisheng` or simulator libraries are environment failures, not API regressions.
3. `ruff check python/` and `yapf -d <touched files>` (picks up `pyproject.toml` automatically; exit 0 with no diff = clean). The `git diff | yapf-diff -i` form is fine if the `yapf-diff` wrapper is installed.

## Rationalizations

| Excuse | Reality |
|---|---|
| "All existing tests are green" | They may never call the new API or its method/operator form; an executed focused regression test is required. |
| "This tensor AttributeError means only the binding is missing" | The override may be absent too, or the API may be object-only; inspect the registry and language surface. |
| "Compiled tests pass, the work is done" | A public kernel-body API still requires its executed interpreter-path test and override or approved explicit stub; host-only APIs follow their approved host-side test path. |
| "I'll set `ASCTILE_INTERPRET` inside the session" | `interpreter_enabled()` is `lru_cache`d; launch a fresh pytest process with the variable set. |
| "Research showed a trivial emulation, no need to bother the developer" | A feature that entered the complex path still requires variant approval. |
| "NumPy can emulate anything, I'll write my best interpretation" | A guessed override makes the interpreter a false oracle; unclear semantics require research. |
| "`no_interpret` makes the cascade green" | The skip requires developer/API-owner approval and a separate executed contract test for the stub or limitation; by itself it hides missing support. |
| "The broader suite exposed another failure, so I should absorb it" | This skill covers the API under change; unrelated failures are not work items in this pipeline. |

## Red Flags — STOP

- You are about to finish an API change without the executed focused contract test required for its kernel-body, host-only, stub, or limitation path.
- You are writing an `@override` for a feature whose contract is still unclear.
- You are implementing interpreter code for a complex feature before the developer picked a variant.
- You are adding `no_interpret` without developer/API-owner approval for an explicit stub or mode limitation.
- You are relying on a skipped `no_interpret` test without a separate executed contract test.
- You are self-classifying an API as host-only or a mode limitation to bypass the kernel-body regression test.
- You are expanding into unrelated interpreter failures instead of completing the current API workflow.
- You are about to set `ASCTILE_INTERPRET` from inside a running process.
