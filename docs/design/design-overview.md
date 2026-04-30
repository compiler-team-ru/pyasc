# Design overview

```{contents} Table of Contents
:local:
```

## 1. Goals & Scope

**PyAsc2** is a tile-based programming model for writing Ascend NPU kernels in Python. It sits at a higher abstraction level than the original PyAsc (`asc`) model, which mapped Python APIs 1:1 to Ascend C intrinsics.

**Core goals** (in priority order):
- Let developers express kernels in terms of *tensors* (ND-arrays in global memory) and *tiles* (fixed-shape chunks in on-chip memory). Buffer addresses, `TPipe`/`TQue` lifecycles, and synchronization barriers are not exposed to the user.
- Reach **≈90% of the performance of hand-optimized Ascend C operators** for representative workloads.
- Provide a NumPy-like operation set: arithmetic, reductions, shape manipulation, masking, atomics.
- **[Main engineering challenge]** Automate synchronization insertion, UB memory allocation, and ping-pong optimization through compiler passes and compiler hints.
- Support Ascend **A2** (910B), **A3** (910C — two 910B dies in one package), and **A5** (950) hardware families.

**Relationship to PyAsc1 (`asc`):**
PyAsc2 is implemented *on top of* the existing PyAsc infrastructure. The `asc2.jit` decorator re-uses `JITFunction`, the same AST-to-IR codegen, the same MLIR pass manager, and the same Bisheng compilation step. What is new is the `asctile` MLIR dialect and a dedicated lowering pipeline that converts tile-level IR down to the `ascendc` dialect before the existing backend takes over.

**Out of scope:**
- Direct access to Ascend C intrinsics (use `asc` for that).
- Dynamic shapes at IR level — tile shapes must be statically known at JIT time; tensor shapes may be dynamic (runtime values).

---

## 2. Key Challenges for Ascend NPU

Ascend A2 AI Core has two types of compute cores:
- **AIC** (Cube) — matrix / tensor operations.
- **AIV** (Vector) — vector operations; 1 AIC : 2 AIV ratio on A2.

Each core has its own MTE (DMA), Scalar Unit, and Compute Unit. AIC and AIV have no direct interconnect — data exchange goes through L2 / Global Memory.

### 2.1 Synchronization Insertion

MTE and Compute Unit within each core run in parallel. Explicit `set_flag` / `wait_flag` barriers must be inserted between them. Missing barriers cause silent data hazards; redundant barriers cause stalls. Must be fully automated by the compiler.

### 2.2 Ping-Pong (Double Buffering)

MTE loads for tile N+1 must overlap with Compute on tile N. Requires the compiler to split the loop body, hoist loads, and insert correct barriers — without user annotation.

### 2.3 UB Memory Allocation and Reuse

On-chip UB (Unified Buffer) is ~256 KB per Da Vinci core (A2). After loop unrolling, many tile SSA values can be live simultaneously. The compiler must compute liveness and reuse freed UB regions. Tile shapes are statically known at JIT time; total live footprint must be validated at compile time.

### 2.4 Hardware Differences: A2/A3 vs A5

**A3** (910C) is two A2 (910B) dies in one package — same Da Vinci architecture, same AIC / AIV model, same challenges as A2. This section therefore focuses on A5.

A5 is not an incremental refresh of A2. It introduces new hardware capabilities (register-addressable SIMD, kernel-side printf) and two new Ascend C programming levels (covered in §9 and the companion *Programming Models Insights*). Findings below are derived from inspection of the CANN 9 SDK preview. <sup>[[58]](#ref-58)</sup>

#### 2.4.1 Hardware Capability Deltas

A5 adds the following hardware capabilities relative to A2: <sup>[[58]](#ref-58)</sup>

- **Register file is first-class.** The vector register file is now exposed as a planned address space for tile computation.
- **Low-precision matmul first-class.** Bit-mode matmul is now exposed as a dedicated hardware path, consistent with the new MX-family data formats (below).
- **New data formats.** MXFP4, MXFP8, HiF8 (in addition to FP16 / BF16 / INT8); the type-conversion matrix grew roughly 3× to cover the new dtypes, including saturating casts.
- **AIC↔AIV communication moves on-chip.** On A2/A3 there is no dedicated hardware channel between Cube and Vector cores — they exchange data through Global Memory (DMA in and out). On A5, a dedicated tag-based dual-channel FIFO lives in the consumer's on-chip SRAM to enable **zero-copy**. The PyPTO framework's TPUSH / TPOP protocol is the software layer built on top of this hardware. <sup>[[4]](#ref-4)</sup>
- **Cross-core address resolution.** On-chip ring-buffer placement means the producer must know the consumer's SRAM base. Resolved via per-function constants plus an allocator-reserved region in the consumer's SRAM.
- **Pipeline sync unchanged at the hardware level.** Handshake between memory engines and vector cores remains explicit — no new hardware barriers.

#### 2.4.2 Implications for the Three DSL Challenges

- **Sync insertion.** Three levels now coexist with different sync models: basic_api keeps explicit `set_flag` / `wait_flag` and TPipe events; MicroAPI adds `MaskReg` predication at the register-tile granularity (masks are not barriers — barriers remain basic_api's responsibility); SIMT-API adds warp-level primitives for fine-grained sync. A DSL must decide which model to expose (or hide) and stay coherent across levels.
- **Ping-pong.** The on-chip TPUSH / TPOP ring buffer on A5 removes the GM round-trip for Cube ↔ Vector handoff — the cost model of a pipelined stage shifts substantially. On MicroAPI, pipelining is a concern at register-tile granularity, not UB-tile granularity, because loops iterate over `RegTensor` chunks.
- **UB memory allocation.** Two new address-spaces to plan: the register file (first-class in MicroAPI) and the reserved consumer SRAM segment for the TPUSH / TPOP ring buffer (on A5, a fixed exclusion zone inside UB or L1). The DSL allocator must model both.
- **Portability.** Targeting only basic_api is the conservative choice but forfeits A5's register-file throughput. Targeting MicroAPI reaches the register file but requires c310 (A5-only builds). A DSL that claims to target A2 + A5 must lower to basic_api only, or implement per-target lowering.

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
                         │  AscTile passes (lib/Dialect/AscTile/Transforms/)
                         │  + AscLower passes (lib/Conversion/LowerToAsc/)
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
tile = asc2.load(tensor, shape=[128], offsets=[base])       # explicit element offsets
tile = asc2.load(tensor, shape=[128], tile_id=[block_idx])  # tile_id * shape = offset

# Load a scalar (no shape → scalar load)
scalar = asc2.load(tensor, offsets=[i])

# Optional padding for partial tiles
tile = asc2.load(tensor, shape=[128], offsets=[base], pad_value=0.0)

# Copy a tile (or sub-region) into a fresh tile, optionally on another location
clone = asc2.copy(tile)
sub   = asc2.copy(tile, shape=[64], offsets=[0])

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
`sin`, `cos`, `tan`, `sinh`, `cosh`, `tanh`, `erf`, `softmax`, `rms_norm`

**Reductions** (one or more axes, optional `keep_dims`):
`reduce_sum`, `reduce_max`, `reduce_min`, `reduce_prod`
— with no axes: returns a scalar `PlainValue`.
— also bound as methods: `tile.sum()`, `tile.max()`, etc.

**Shape manipulation**:
`reshape`, `ravel`, `broadcast_to`, `expand_dims`, `squeeze`

**Creation**:
`full(shape, value)`, `zeros(shape)`, `full_like`, `zeros_like`,
`zeros_acc(shape, dtype)` — zeros tile in L0C (matmul accumulator),
`concat(*tiles)` — concatenate along the first dimension.

**Matrix multiply**:
- `matmul(a, b)` — 2-D float tiles; result is always `float32`.
- `matmul_acc(a, b, acc)` — fused matmul that accumulates into an existing `acc` tile (typically created with `zeros_acc`).

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

| Category | Op | Description |
|----------|----|-------------|
| Tensor / tile creation | `TensorOp` | Create a `Tensor` from a pointer and shape list |
| | `DimOp` | Read a dynamic dimension from a `Tensor` at runtime |
| | `SplatOp` | Fill a tile with a scalar constant |
| | `AccumulatorOp` | Create a zeros tile in L0C (matmul accumulator) |
| Memory transfer | `LoadOp` | Load tile from `Tensor` with offsets → `Tile` |
| | `StoreOp` | Store `Tile` to `Tensor` with offsets |
| | `StoreFixpipeOp` | Store an L0C tile to a `Tensor` via the fixpipe path |
| | `CopyOp` | Copy a tile (or sub-region) into a new tile |
| | `CopyFixpipeOp` | Copy an L0C tile into another tile via fixpipe |
| | `GetValueOp` | Scalar load from `Tensor` |
| | `SetValueOp` | Scalar store to `Tensor` |
| Element-wise compute | `CastOp` | Type conversion between tile dtypes |
| | `ReluOp` | Fused ReLU |
| | `SelectOp` | Element-wise conditional select (`where`) |
| | `CmpOp` / `CmpSOp` | Element-wise compare (tile/tile and tile/scalar) |
| | `AddSOp` / `SubSOp` / `MulSOp` / `DivSOp` / `MinSOp` / `MaxSOp` / `ShLSOp` / `ShRSOp` | Vector-scalar binary ops |
| Reduction / norm | `ReduceOp` | Reduce tile along one or more axes |
| | `ReduceAs1dOp` | Reduce a tile to a 1-element tile or a scalar |
| | `SoftmaxOp` | Row-wise softmax on a 1D / 2D tile |
| | `RmsNormOp` | RMS normalisation |
| Matmul | `MatmulOp` | 2-D matrix multiply |
| | `MatmulAccOp` | 2-D matrix multiply accumulating into an L0C tile |
| Shape | `ReshapeOp` | Reshape a tile |
| | `BroadcastOp` | Broadcast tile to a wider shape |
| | `ConcatOp` | Concatenate tiles along the first dimension |
| Atomic | `AtomicRMWOp` | Atomic read-modify-write (add / max / min) |
| Region / control | `CountMaskOp` | Region: conditional on element count |
| | `BitwiseMaskOp` | Region: conditional on bitmask |
| | `YieldOp` | Region terminator |

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
| `SplitCubeLoad` | Split loads destined for L0A/L0B into the legal Cube data path |
| `LegalizeMatmul` | Rewrite `MatmulOp`/`MatmulAccOp` into the form expected by the AscLower passes |
| `TransformMathOps` | Specialise generic `math.*` ops into tile-aware equivalents that the AscLower pass knows how to handle |
| `TransformStoreFixpipe` | Convert eligible `StoreOp`s of L0C tiles into `StoreFixpipeOp` |
| `UnscalarizeReduction` (910_95 only) | Replace scalar-tail reductions with vector forms |
| `UnrollLoop` | Physically unroll loops tagged by `TagUnrollGroups` by `unroll_factor` |
| `Canonicalizer`, `CSE` | Standard MLIR cleanup between stages |

### 6.2 AscLower passes  (`lib/Conversion/LowerToAsc/`)

These lower `asctile.*` operations into `ascendc.*` + standard MLIR ops.

| Pass | Purpose |
|------|---------|
| `ExpandMath` | Replace `math.*` ops with sequences of `ascendc` vector intrinsics |
| `RedressI1Tile` | Widen boolean (i1) tiles to `i8`/`ui8` as required by Ascend C |
| `LowerArith` | Lower `arith.*` scalar ops to `ascendc` equivalents |
| `LowerArithBinary` | Lower tile-scalar and scalar-tile arithmetic |
| `LowerAtomic` | Lower `AtomicRMWOp` → `ascendc.atomic_*` |
| `LowerAscTileDataTransfer` | Lower `LoadOp`/`StoreOp`/`CopyOp`/fixpipe variants to `ascendc` data-copy ops |
| `LowerAscTile` | Main lowering: `CastOp`/`ReshapeOp`/reduction ops/… → `ascendc` vector ops |
| `LowerAscTileI1` | Handle boolean tile lowering special cases |
| `LowerMath` | Lower remaining `math.*` ops |
| `LowerSCF` | Lower `scf.for` loops to `ascendc`-compatible form |
| `DisplaceConcat` | Replace `ConcatOp` with adjacent allocations / copies |
| `RealizeConversionCast` | Materialise any pending unrealized casts from the lowering |
| `ExpandMask` | Lower `CountMaskOp`/`BitwiseMaskOp` regions → `ascendc.set_mask`/`wait_mask` sequences |

### 6.3 ascendc passes  (after lowering, `lib/Dialect/Asc/Transforms/`)

Once all `asctile` ops are gone, the existing ascendc pipeline takes over:

| Phase | Key passes |
|-------|-----------|
| **Lowering glue** | `PrivatizeFunc`, `FillAscOperands`, `FixupMmadAccParams`, `LowerToL0` |
| **Memory allocation** | `InputOutputTensor`, `ReuseUBAllocation`, `HoistUBAllocation`, `FuseVFBlock` (when `vf_fusion=True`), `AllocateTensor` (when `static_alloc=True`) / `MaterializeTensor` (otherwise), `UnifyPipe` |
| **Synchronization** | `EraseSync`, `HoistQueBind`, then either `InsertSync` (910B / 910_93) or `InsertBufIdSync` + `FuseBufIdSync` (910_95); finally `UnifyPipe`. `VerifySync` runs when `verify_sync=True` |
| **Optimization** | `LICM`, `SCCP`, `Canonicalizer` |
| **Postprocessing** | `DeclarePyStruct`, `GenerateBoilerplate`, `DefineCubeOnly` (when `matmul_cube_only=True`), `LegalizeKernelArgs`, `DetectKernelType`, `DetectEnableDebug`, `ComputeMemoryConsumption` |

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
| **TPipe-managed** | `static_alloc=False` | `MaterializeTensor` emits `TPipe` + `AllocTensor`/`FreeTensor` in the generated Ascend C. Flexible; incurs scalar overhead per tile. Default on 910B / 910_93. |
| **Static allocation** | `static_alloc=True` | `AllocateTensor` computes a fixed layout at compile time and emits direct UB address arithmetic. Zero scalar overhead; requires all tile sizes to be statically known. Default on 910_95. |

`static_alloc` is `Optional[bool]`; when left at `None`, `Compiler.__init__` resolves it to `True` on 910_95 and `False` otherwise.

**UB pressure reduction (both modes):**
- `HoistUBAllocation` — move allocations above loops so one allocation covers all iterations.
- `ReuseUBAllocation` (`reuse_ub=True`) — when a tile's lifetime ends before a new one is needed, reuse its UB region. `reuse_ub_in_out=True` extends this to input/output tiles (experimental).
- `ComputeMemoryConsumption` — sums all live tile sizes per `TPosition` and fails the compilation if UB limits are exceeded.

### 7.3 Synchronization

Because asc2 users don't write `set_flag`/`wait_flag`, all synchronization is inserted by the compiler:

- `insert_sync=True` is set automatically by `asc2.jit`.
- On **910B / 910_93**: `InsertSync` uses the classic queue-position–based algorithm.
- On **910_95**: `InsertBufIdSync` uses a newer algorithm that reduces sync overhead by tracking buffer IDs rather than queue positions; `FuseBufIdSync` merges adjacent sync ops.
- `verify_sync=True` runs an additional `VerifySync` pass after sync insertion as a sanity check.

`asc2.range(parallel=True)` annotates a loop with an MLIR attribute that is intended to relax sync insertion for parallel-safe load/store groups. Today the attribute is only set; the consuming pass is not yet wired up.

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

## 9. Insights from Existing Programming Models

A comparative analysis of existing tile-based and kernel DSLs relevant to Ascend NPU (AscendC, Triton, cuTile, TileLang-Ascend, Triton-Ascend, PyPTO, Pallas, Mojo) is maintained in a companion document:

→ [Programming Models Insights](programming-models-insights.md)

The observations from that document inform the design decisions in §10.

---

## 10. Key Design Decisions

_TODO: to be filled in._

---

## 11. References

<a id="ref-4"></a>**[4]** PTO ISA — TPUSH/TPOP protocol.
<https://github.com/hw-native-sys/pypto/blob/main/docs/en/reference/pto-isa/01-tpush_tpop.md>

<a id="ref-58"></a>**[58]** CANN 9 SDK preview — 950 / A5 programming model findings (internal notes from SDK source inspection).

---

## Appendix: Key `CompileOptions` for PyAsc2

| Option | Default in `asc2.jit` | Effect |
|--------|-----------------------|--------|
| `run_asc2_passes` | `True` | Enable AscTile + AscLower pipeline |
| `insert_sync` | `True` | Auto-insert sync barriers |
| `static_alloc` | `None` → arch-dependent (`True` on 910_95, `False` on 910B / 910_93) | Static vs TPipe-managed UB allocation |
| `reuse_ub` | `False` | Reuse freed UB regions |
| `reuse_ub_in_out` | `False` | Extend reuse to I/O tiles (experimental) |
| `densify_load_store` | `False` | Densify load/store groups (experimental) |
| `vf_fusion` | `False` | Fuse consecutive vector ops into Ascend C MicroAPI VF blocks |
| `verify_sync` | `False` | Run `VerifySync` pass after sync insertion |
| `matmul_cube_only` | `False` | Emit cube-only kernels (drives `DefineCubeOnly`) |
| `auto_sync` | `True` | Pass `--cce-auto-sync` to Bisheng (separate from MLIR `insert_sync`) |
| `opt_level` | `3` | Bisheng `-O` level (1–3) |
| `always_compile` | `False` | Bypass cache, recompile every call |
