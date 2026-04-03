# Project overview

## 1. Goals & Scope

**PyAsc2** is a tile-based programming model for writing Ascend NPU kernels in Python. It sits at a higher abstraction level than the original PyAsc (`asc`) model, which mapped Python APIs 1:1 to Ascend C intrinsics.

**Core goals:**
- Let developers express kernels in terms of *tensors* (ND-arrays in global memory) and *tiles* (fixed-shape chunks in on-chip memory), without managing buffer addresses, `TPipe`/`TQue` lifecycles, or synchronization barriers directly.
- Provide a NumPy-like operation set: arithmetic, reductions, shape manipulation, masking, atomics.
- Automate synchronization insertion, UB memory allocation, and loop unrolling through compiler passes.
- Support Ascend910B, Ascend910_93, and Ascend910_95 hardware families.

**Relationship to PyAsc1 (`asc`):**
PyAsc2 is implemented *on top of* the existing PyAsc infrastructure. The `asc2.jit` decorator re-uses `JITFunction`, the same AST-to-IR codegen, the same MLIR pass manager, and the same Bisheng compilation step. What is new is the `asctile` MLIR dialect and a dedicated lowering pipeline that converts tile-level IR down to the `ascendc` dialect before the existing backend takes over.

**Out of scope:**
- Direct access to Ascend C intrinsics (use `asc` for that).
- Dynamic shapes at IR level — tile shapes must be statically known at JIT time; tensor shapes may be dynamic (runtime values).

---

## 2. Key Challenges

> **NOTE: This section is not finished.**

### 2.1 Programming Model

Defining a tile-based Python API that is expressive enough for real kernels while remaining statically compilable. The API must hide `TPipe`/`TQue` management and synchronization from the user, yet map cleanly to Ascend C intrinsics after lowering. Key open questions: ergonomics of multi-dimensional tiling, handling of mixed `UB`/`L0` tile locations, and interoperability with `asc` (PyAsc1) primitives.

### 2.2 Sync Insertion

Automatic insertion of `set_flag`/`wait_flag` barriers between pipeline stages at the `ascendc` level. The challenge is correctly inferring producer/consumer relationships across loop iterations after tile-level unrolling, without inserting redundant or missing syncs. The `InsertBufIdSyncV2` algorithm addresses this for 910_95, but correctness across all hardware variants and unroll patterns is still being validated.

### 2.3 Ping-Pong Optimization

Enabling double-buffering (ping-pong) at the tile level so that loads for the next iteration overlap with computation on the current tile. This requires the compiler to automatically split the loop body, hoist loads outside the dependency chain, and insert the correct sync barriers — without the user annotating it explicitly.

### 2.4 UB Memory Allocation and Reuse

Mapping tile SSA values (which may be numerous after unrolling) to a finite on-chip UB region. Challenges include: computing liveness across unrolled iterations, reusing freed regions for new tiles of compatible size (`ReuseUBAllocation`), and ensuring the total live footprint does not exceed hardware UB limits (enforced by `ComputeMemoryConsumption`).

---

## 3. Architecture Overview

```
  ┌──────────────────────────────────────────────────────────┐
  │                 User Python Code                         │
  │             @asc2.jit                                    │
  └──────────────────────┬───────────────────────────────────┘
                         │
  ┌──────────────────────▼───────────────────────────────────┐
  │              PyAsc2 Frontend  (python/asc2/)             │
  │                                                          │
  │  Tensor (GM)   Tile (UB/L0/…)   asc2.range   asc2.mask  │
  │  load / store  arithmetic  reductions  shape ops         │
  │  atomics  matmul  unary  creation  indexing              │
  └──────────────────────┬───────────────────────────────────┘
                         │  FunctionVisitor (AST → IR)
  ┌──────────────────────▼───────────────────────────────────┐
  │           asctile MLIR Dialect                           │
  │   TensorType / TileType    LoadOp / StoreOp              │
  │   BinaryOps  UnaryOps  ReductionOps  ShapeOps            │
  │   AtomicRMWOp  MatmulOp  SoftmaxOp  SelectOp            │
  │   CountMaskOp  BitwiseMaskOp                             │
  └──────────────────────┬───────────────────────────────────┘
                         │  AscTile passes (lib/Dialect/AscTile/)
                         │  + AscLower passes (lib/Dialect/AscLower/)
  ┌──────────────────────▼───────────────────────────────────┐
  │           ascendc MLIR Dialect  (existing backend)       │
  │   LocalTensor / GlobalTensor   TPipe / TQue              │
  │   set_flag / wait_flag   Ascend C API ops                │
  └──────────────────────┬───────────────────────────────────┘
                         │  lib/Target/AscendC/ (CodeEmitter)
  ┌──────────────────────▼───────────────────────────────────┐
  │           Ascend C source (.cce)                         │
  └──────────────────────┬───────────────────────────────────┘
                         │  Bisheng compiler
                         ▼
                    NPU kernel binary (.o)
```

### End-to-end compilation flow

```
@asc2.jit kernel invocation
  → argument type specialization  (same as asc1)
  → two-level cache lookup         (same as asc1)
      → [miss]
          AST walk → asctile MLIR (FunctionVisitor + asctile builder APIs)
          ↓
          AscTile passes  (unrolling, loop transforms, math specialization)
          ↓
          AscLower passes (lower asctile → ascendc dialect)
          ↓
          ascendc passes  (UB allocation, sync insertion, boilerplate gen)
          ↓
          CodeEmitter     (ascendc → Ascend C source)
          ↓
          Bisheng         (Ascend C → .o binary)
          ↓
          cache store
  → kernel launch via runtime rt library calls  (same as asc1)
```

---

## 4. Programming Model

### 4.1 The two types: `Tensor` and `Tile`

| | `Tensor` | `Tile` |
|--|----------|--------|
| **Memory** | Global Memory (HBM, GM) | On-chip: UB, L0A, L0B, L0C, L1 |
| **Shape** | ND, may be dynamic (RuntimeInt dims) | ND, must be static (known at JIT time) |
| **Creation** | `asc2.tensor(ptr, shape)` | Returned by `asc2.load(...)` or creation ops |
| **Mutability** | Passed to/from kernel; never computed in-kernel | Value semantics; produced by ops, consumed once |
| **IR type** | `asctile.tensor<shape x dtype>` | `asctile.tile<shape x dtype, location>` |

`Tensor` is a *descriptor* — it holds a pointer and a shape. `Tile` carries `ValueSemantics` in MLIR, meaning instances are value types — self-contained and safe to CSE, copy, and move. Each op produces a new tile SSA value. The compiler is free to map multiple logical tiles to the same physical UB region.

### 4.2 Memory hierarchy (`TileLocation`)

| Enum value | Hardware unit | Typical use |
|------------|--------------|-------------|
| `UB` | Unified Buffer | Vector operations (default) |
| `L0A` / `L0B` | L0 matrix input buffers | Matmul inputs |
| `L0C` | L0 matrix output buffer | Matmul accumulator |
| `L1` | L1 cache | Intermediate staging |

`load` defaults to `TileLocation.UB`. Matmul implicitly creates `L0A`/`L0B`/`L0C` tiles; the LowerToL0 pass handles the layout conversion.

### 4.3 Load and store

```python
# Load a tile from a 2-D tensor — two addressing modes:
tile = asc2.load(tensor, shape=[128], offsets=[base])       # explicit byte offsets
tile = asc2.load(tensor, shape=[128], tile_id=[block_idx])  # tile_id * shape = offset

# Load a scalar (no shape → scalar load)
scalar = asc2.load(tensor, offsets=[i])

# Store tile or scalar back
asc2.store(tile, tensor, offsets=[base])
asc2.store(scalar_value, tensor, offsets=[i])
```

The last dimension of every tile shape must be aligned to `ub_block_size` (32 bytes). This is enforced at JIT time via `check_data_alignment`.

### 4.4 Operations

All operations produce new `Tile` or scalar values; no mutation.

**Arithmetic / comparison** (tile ⊕ tile, tile ⊕ scalar, scalar ⊕ tile):
`add`, `sub`, `mul`, `div`, `maximum`, `minimum`, `left_shift`, `right_shift`,
`equal`, `not_equal`, `greater`, `greater_equal`, `less`, `less_equal`

Operator overloads on `Tile` (`+`, `-`, `*`, `/`, `>`, `==`, …) call the same functions. Dtypes are promoted automatically via `infer_common_dtype`.

**Unary**:
`abs`, `ceil`, `floor`, `negative`, `relu`,
`sqrt`, `rsqrt`, `exp`, `exp2`, `log`, `log2`,
`sin`, `cos`, `tan`, `sinh`, `cosh`, `tanh`, `erf`, `softmax`

**Reductions** (one or more axes, optional `keep_dims`):
`reduce_sum`, `reduce_max`, `reduce_min`, `reduce_prod`
— with no axes: returns a scalar `PlainValue`.
— also bound as methods: `tile.sum()`, `tile.max()`, etc.

**Shape manipulation**:
`reshape`, `broadcast_to`, `expand_dims`, `squeeze`

**Creation**:
`full(shape, value)`, `zeros(shape)`, `full_like`, `zeros_like`

**Matrix multiply**:
`matmul(a, b)` — 2-D float tiles; result is always `float32`.

**Conditional / masking**:
- `where(mask, src0, src1)` — element-wise select.
- `mask(count=…)` / `mask(bits=(hi, lo), other=…)` — context manager that wraps the enclosed operations in a conditional region (`CountMaskOp` / `BitwiseMaskOp`).

**Atomics**:
`atomic_add`, `atomic_max`, `atomic_min` — write back to a global `Tensor`.

### 4.5 Programming model operations

```python
i = asc2.block_idx()    # current NPU block index (PlainValue)
n = asc2.block_num()    # total number of blocks (PlainValue)
k = asc2.num_tiles(tensor, axis=0, shape=[128])  # how many tiles fit along an axis
```

### 4.6 `asc2.range`

`asc2.range` extends the base range with two attributes that control compiler behaviour:

```python
for i in asc2.range(start, stop, step, unroll_factor=4, parallel=False):
    ...
```

- `unroll_factor` — how many iterations to unroll. Tagged on the MLIR `for` op and processed by `TagUnrollGroups` + `UnrollLoop` passes.
- `parallel=True` — marks the loop body so `ParallelLoadStore` pass can emit stores and loads in parallel.

---

## 5. IR Design

### 5.1 `asctile` dialect

Defined in `include/ascir/Dialect/AscTile/IR/` using TableGen.

**Types:**

| Type | MLIR syntax | Python proxy |
|------|------------|--------------|
| `asctile.tensor<[N,M] x f32>` | ND global descriptor | `Tensor` |
| `asctile.tile<[128] x f16, UB>` | Fixed-shape local chunk | `Tile` |

`Tile` carries `ValueSemantics`, meaning instances are value types — self-contained and safe to CSE, copy, and move. Both `Tile` and `Tensor` implement `ShapedTypeInterface`.

**Key operations (from `Ops.td`):**

| Op | Description |
|----|-------------|
| `TensorOp` | Create a `Tensor` from a pointer and shape list |
| `DimOp` | Read a dynamic dimension from a `Tensor` at runtime |
| `SplatOp` | Fill a tile with a scalar constant |
| `LoadOp` | Load tile from `Tensor` with offsets → `Tile` |
| `StoreOp` | Store `Tile` to `Tensor` with offsets |
| `GetValueOp` | Scalar load from `Tensor` |
| `SetValueOp` | Scalar store to `Tensor` |
| `CastOp` | Type conversion between tile dtypes |
| `ReshapeOp` | Reshape a tile |
| `BroadcastOp` | Broadcast tile to a wider shape |
| `SelectOp` | Element-wise conditional select (`where`) |
| `ReluOp` | Fused ReLU (single op, not composed) |
| `SoftmaxOp` | Full softmax on a tile |
| `MatmulOp` | 2-D matrix multiply |
| `AtomicRMWOp` | Atomic read-modify-write (add / max / min) |
| `CountMaskOp` | Region: conditional on element count |
| `BitwiseMaskOp` | Region: conditional on bitmask |
| `YieldOp` | Region terminator |

**Standard dialect ops** (`arith`, `scf`, `math`) are emitted directly by the AST walker for scalar arithmetic and control flow; the AscLower passes later handle them.

### 5.2 IR generation

`FunctionVisitor` (Python AST walker, `python/asc/codegen/function_visitor.py`) calls pybind11-exposed builder methods from `python/src/OpBuilder.cpp` to create `asctile.*` ops. Every `asc2.*` Python API call produces exactly one (or a small fixed number of) `asctile` operations. No analysis is needed at this stage.

### 5.3 Lowering chain

```
asctile dialect
    ↓  AscTile passes  (loop unrolling, math specialization, masking)
    ↓  AscLower passes (lower each asctile op → ascendc ops + scf + arith)
ascendc dialect
    ↓  ascendc passes  (UB allocation, sync insertion, boilerplate)
Ascend C source
```

---

## 6. Optimization Passes

The full pass pipeline is scheduled by `Compiler._schedule_passes()` inside `python/asc/runtime/compiler.py`. The `run_asc2_passes=True` flag (set automatically by `asc2.jit`) activates the AscTile and AscLower phases.

### 6.1 AscTile passes  (`lib/Dialect/AscTile/Transforms/`)

These operate on the `asctile` dialect before lowering begins.

| Pass | Purpose |
|------|---------|
| `TagUnrollGroups` | Scan `asc2.range` loops annotated with `unroll_factor > 1` and tag contiguous groups of ops that should be unrolled together |
| `PromotePureOps` | Hoist pure (side-effect-free) ops — e.g., shape computations — out of loop bodies |
| `DensifyUnrollGroups` *(densify_load_store)* | Cluster load/store ops within an unroll group so they appear adjacent, enabling more efficient pipeline scheduling |
| `TransformMathOps` | Specialise generic `math.*` ops into tile-aware equivalents that the AscLower pass knows how to handle |
| `UnrollLoop` | Physically unroll loops tagged by `TagUnrollGroups` by `unroll_factor` |
| `Canonicalizer`, `CSE` | Standard MLIR cleanup between stages |

### 6.2 AscLower passes  (`lib/Dialect/AscLower/`)

These lower `asctile.*` operations into `ascendc.*` + standard MLIR ops.

| Pass | Purpose |
|------|---------|
| `ExpandMath` | Replace `math.*` ops with sequences of `ascendc` vector intrinsics |
| `RedressI1Tile` | Widen boolean (i1) tiles to `i8`/`ui8` as required by Ascend C |
| `LowerArith` | Lower `arith.*` scalar ops to `ascendc` equivalents |
| `LowerArithBinary` | Lower tile-scalar and scalar-tile arithmetic |
| `LowerArithI1` | Handle boolean arithmetic special cases |
| `LowerAtomic` | Lower `AtomicRMWOp` → `ascendc.atomic_*` |
| `LowerAscTile` | Main lowering: `LoadOp`/`StoreOp`/`CastOp`/`ReshapeOp`/`ReductionOps`/… → `ascendc` data-copy + vector ops |
| `LowerMath` | Lower remaining `math.*` ops |
| `LowerScf` | Lower `scf.for` loops to `ascendc`-compatible form |
| `RealizeConversionCast` | Materialise any pending unrealized casts from the lowering |
| `ExpandMask` | Lower `CountMaskOp`/`BitwiseMaskOp` regions → `ascendc.set_mask`/`wait_mask` sequences |
| `FillAscOperands` | Fill in default optional operands required by `ascendc` ops |

### 6.3 ascendc passes  (after lowering, `lib/Dialect/Asc/Transforms/`)

Once all `asctile` ops are gone, the existing ascendc pipeline takes over:

| Phase | Key passes |
|-------|-----------|
| **Memory allocation** | `InputOutputTensor`, `ReuseUBAllocation`, `HoistUBAllocation`, `MaterializeTensor` / `AllocateTensor`, `UnifyPipe` |
| **Synchronization** | `EraseSync`, `HoistQueBind`, `InsertSync` (910B/93) or `InsertBufIdSync(V2)` + `FuseBufIdSync` + `ParallelLoadStore` (910_95) |
| **Optimization** | `LICM`, `SCCP`, `Canonicalizer` |
| **Postprocessing** | `DeclarePyStruct`, `GenerateBoilerplate`, `LegalizeKernelArgs`, `DetectKernelType`, `ComputeMemoryConsumption` |

`ComputeMemoryConsumption` is **only active in asc2 mode** and records per-`TPosition` UB usage as a module attribute; it raises a compile-time error on overflow.

---

## 7. Memory & Resource Model

### 7.1 Tile-level view

From the user's perspective, memory management is invisible. The user:
1. Creates `Tensor` descriptors pointing to HBM buffers.
2. Calls `load(tensor, shape, offsets=…)` to bring data into a `Tile`.
3. Applies operations to produce new tiles.
4. Calls `store(tile, tensor, offsets=…)` to write results back.

The compiler is responsible for mapping each `Tile` SSA value to a physical UB region.

### 7.2 Compiler-side UB allocation

Two strategies, selected per compilation:

| Strategy | Option | How it works |
|----------|--------|-------------|
| **TPipe-managed** (default) | `static_alloc=False` | `MaterializeTensor` emits `TPipe` + `AllocTensor`/`FreeTensor` in the generated Ascend C. Flexible; incurs scalar overhead per tile. |
| **Static allocation** | `static_alloc=True` | `AllocateTensor` computes a fixed layout at compile time and emits direct UB address arithmetic. Zero scalar overhead; requires all tile sizes to be statically known. |

**UB pressure reduction (both modes):**
- `HoistUBAllocation` — move allocations above loops so one allocation covers all iterations.
- `ReuseUBAllocation` (`reuse_ub=True`) — when a tile's lifetime ends before a new one is needed, reuse its UB region. `reuse_ub_in_out=True` extends this to input/output tiles (experimental).
- `ComputeMemoryConsumption` — sums all live tile sizes per `TPosition` and fails the compilation if UB limits are exceeded.

### 7.3 Synchronization

Because asc2 users don't write `set_flag`/`wait_flag`, all synchronization is inserted by the compiler:

- `insert_sync=True` is set automatically by `asc2.jit`.
- On **910B / 910_93**: `InsertSync` uses the classic queue-position–based algorithm.
- On **910_95**: `InsertBufIdSync` uses a newer algorithm that reduces sync overhead by tracking buffer IDs rather than queue positions; `FuseBufIdSync` merges adjacent sync ops.

---

## 8. Usage Examples

### Vector addition

```python
import asc2

@asc2.jit
def vadd(x_ptr, y_ptr, out_ptr, size: int, TILE: asc.ConstExpr[int]):
    x_gm   = asc2.tensor(x_ptr,   [size])
    y_gm   = asc2.tensor(y_ptr,   [size])
    out_gm = asc2.tensor(out_ptr, [size])

    tiles_per_block = asc2.num_tiles(x_gm, 0, [TILE])
    base = asc2.block_idx() * tiles_per_block * TILE

    for i in asc2.range(tiles_per_block):
        off = base + i * TILE
        x   = asc2.load(x_gm,   [TILE], offsets=[off])
        y   = asc2.load(y_gm,   [TILE], offsets=[off])
        asc2.store(x + y, out_gm, offsets=[off])

vadd[8](x, y, out, n, TILE=256)
```

### Softmax (row-wise)

```python
@asc2.jit
def softmax(x_ptr, out_ptr, rows: int, cols: int, TILE: asc.ConstExpr[int]):
    x_gm   = asc2.tensor(x_ptr,   [rows, cols])
    out_gm = asc2.tensor(out_ptr, [rows, cols])

    for row in asc2.range(asc2.block_idx(), rows, asc2.block_num()):
        x   = asc2.load(x_gm, [1, TILE], offsets=[row, 0])
        exp_x = asc2.exp(x - x.max())
        asc2.store(exp_x / exp_x.sum(), out_gm, offsets=[row, 0])
```

---

## Appendix: Key `CompileOptions` for PyAsc2

| Option | Default in `asc2.jit` | Effect |
|--------|-----------------------|--------|
| `run_asc2_passes` | `True` | Enable AscTile + AscLower pipeline |
| `insert_sync` | `True` | Auto-insert sync barriers |
| `static_alloc` | `False` | Static vs TPipe-managed UB allocation |
| `reuse_ub` | `False` | Reuse freed UB regions |
| `reuse_ub_in_out` | `False` | Extend reuse to I/O tiles (experimental) |
| `densify_load_store` | `False` | Densify load/store groups (experimental) |
| `opt_level` | `3` | Bisheng `-O` level (1–3) |
| `always_compile` | `False` | Bypass cache, recompile every call |
