/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASCTILE_PASSES_TD
#define ASCTILE_PASSES_TD

include "mlir/Pass/PassBase.td"

def DensifyUnrollGroups : Pass<"asctile-densify-unroll-groups", "func::FuncOp"> {
  let summary = "Group operations tagged with the same unroll_group attribute together after loop unrolling";
  let constructor = "mlir::asctile::createDensifyUnrollGroupsPass()";
  let description = [{
    This pass collects operations tagged with `asctile.unroll_group` attribute (added by TagUnrollGroups pass)
    and moves them together within their blocks to improve locality and enable better optimization opportunities.

    For each unroll group, operations are densified according to their type:
    - LoadOps: moved toward the top (earliest valid position respecting dominance)
    - StoreOps: moved toward the bottom (latest valid position respecting dominance)

    The movement respects dominance constraints and memory effects to preserve program semantics.
    After densification, the `unroll_group` attribute is removed from all operations.

    Example transformation:
    ```mlir
    // Before: load, relu, load, store, relu, store (with groups 0 and 1)
    %0 = load {unroll_group=0}
    %1 = relu %0
    %2 = load {unroll_group=0}
    store %0 {unroll_group=1}
    %3 = relu %2
    store %2 {unroll_group=1}

    // After: loads grouped, stores grouped
    %0 = load
    %2 = load
    %1 = relu %0
    %3 = relu %2
    store %0
    store %2
    ```
  }];
}

def LegalizeMatmul : Pass<"asctile-legalize-matmul", "func::FuncOp"> {
  let summary = "Verify that matmul accumulator operations use valid accumulator operands";
  let constructor = "mlir::asctile::createLegalizeMatmulPass()";
  let description = [{
    This pass validates that all `asctile.matmul_acc` operations use an accumulator operand that is
    defined by an `asctile.accumulator` operation. This ensures the matmul accumulator pattern is correct
    and the hardware constraints are satisfied.

    If any `matmul_acc` operation violates this constraint, the pass emits an error and signals failure.

    This is a verification-only pass that does not transform the IR.
  }];
}

def PromotePureOps : Pass<"asctile-promote-pure-ops", "func::FuncOp"> {
  let summary = "Hoist pure operations upward within their blocks to enable code motion optimizations";
  let constructor = "mlir::asctile::createPromotePureOpsPass()";
  let description = [{
    This pass moves pure (side-effect-free) operations as early as possible within their containing block,
    respecting dominance constraints. This enables loop-invariant code motion and improves scheduling.

    Pure operations include arithmetic ops, constants, and other ops without side effects.
    Operations with side effects (e.g., memory ops, barriers) remain in their original positions.

    Example transformation:
    ```mlir
    // Before: pure ops interleaved with barriers inside a loop
    scf.for {
      %0 = addf %a, %b          // pure
      %1 = addf %a, %0          // pure, depends on %0
      gpu.barrier               // side effect
      %2 = mulf %a, %b          // pure, independent
      %3 = constant 7.0         // pure, independent
      %4 = addf %a, %3          // pure, depends on %3
      %5 = subf %4, %2          // pure
      gpu.barrier               // side effect
    }

    // After: pure ops hoisted as early as possible
    scf.for {
      %3 = constant 7.0
      %2 = mulf %a, %b
      %4 = addf %a, %3
      %0 = addf %a, %b
      %1 = addf %a, %0
      %5 = subf %4, %2
      gpu.barrier
      gpu.barrier
    }
    ```
  }];
}

def SplitCubeLoad : Pass<"asctile-split-cube-load", "func::FuncOp"> {
  let summary = "Split direct GM-to-L0 loads into GM-to-L1 followed by L1-to-L0 copies for cube operations";
  let constructor = "mlir::asctile::createSplitCubeLoadPass()";
  let dependentDialects = ["arith::ArithDialect", "asctile::AscTileDialect"];
  let description = [{
    This pass transforms direct loads from global memory (GM) to L0A/L0B buffer locations into a two-step
    sequence: load to L1 first, then copy from L1 to L0A/L0B. This is required for cube (matrix multiplication)
    operations where operands must reside in specific L0 buffers.

    The pass applies two patterns:
    1. **ConvertLoadGMToL0**: Splits `load` producing L0A/L0B tile into:
       - `load` producing L1 tile (from GM)
       - `copy` from L1 to L0A/L0B with zero offsets

    2. **MarkTileOperandInMmad**: Marks L1 tiles used exclusively for matrix multiplication:
       - If an L1 tile is copied only to L0A, marks it with `isMatrixA` attribute
       - Validates consistency: same L1 tile should not be copied to both L0A and L0B
       - Emits error if L1 tile is used for non-copy operations

    Example transformation:
    ```mlir
    // Before: direct load to L0A
    %0 = load %tensor[%offset] : tensor, tile<16x16xf32, L0A>

    // After: split into L1 load + L1-to-L0A copy
    %0 = load %tensor[%offset] : tensor, tile<16x16xf32, L1>
    %zero = constant 0 : i32
    %1 = copy %0[%zero, %zero] : tile<16x16xf32, L1> to tile<16x16xf32, L0A>
    ```
  }];
}

def TagUnrollGroups : Pass<"asctile-tag-unroll-groups", "func::FuncOp"> {
  let summary = "Tag Load and Store operations inside unrollable loops with unroll_group attributes";
  let constructor = "mlir::asctile::createTagUnrollGroupsPass()";
  let options = [
    Option<"smallGroups", "small-groups", "bool", "false",
           "Tag each operation with unique group index; otherwise all loads share one group and stores another">,
  ];
  let description = [{
    This pass tags `asctile.load` and `asctile.store` operations inside `scf.for` loops that have
    the `asctile.unroll_factor` attribute. The tags are used by DensifyUnrollGroups pass after
    loop unrolling to group related operations together.

    Tagging behavior depends on the `small-groups` option:
    - **smallGroups=false (default)**: All LoadOps in a loop get the same group index,
      all StoreOps get the next group index. This enables densification to bring all loads
      together and all stores together.
    - **smallGroups=true**: Each LoadOp/StoreOp gets a unique group index. This preserves
      the original ordering during densification.

    Only loops with `unroll_factor` attribute are processed; other loops are skipped.

    Example transformation:
    ```mlir
    // Before: loop with unroll_factor
    scf.for {unroll_factor = 4} {
      %0 = load ...
      %1 = load ...
      store %0, ...
    }

    // After (smallGroups=false): loads share group 0, stores share group 1
    scf.for {unroll_factor = 4} {
      %0 = load {unroll_group = 0}
      %1 = load {unroll_group = 0}
      store %0, ... {unroll_group = 1}
    }

    // After (smallGroups=true): each op gets unique group
    scf.for {unroll_factor = 4} {
      %0 = load {unroll_group = 0}
      %1 = load {unroll_group = 1}
      store %0, ... {unroll_group = 2}
    }
    ```
  }];
}

def TransformMathOps : Pass<"asctile-transform-math-ops", "func::FuncOp"> {
  let summary = "Convert tile arithmetic operations with splat operands to scalar-splat variants for hardware efficiency";
  let constructor = "mlir::asctile::createTransformMathOpsPass()";
  let dependentDialects = ["asctile::AscTileDialect"];
  let description = [{
    This pass specializes arithmetic and comparison operations on tiles when one operand is a splat value
    (uniform constant across all elements). The scalar-splat variants (`adds`, `muls`, `cmps`, etc.) are
    more efficient on the hardware than tile-tile operations.

    Transformation patterns (applied greedily with pattern benefits):
    1. **MaxWithZeroToRelu** (benefit 2): `max(x, 0)` or `max(0, x)` → `relu(x)` for `maximumf`, `maxnumf`, `maxsi`

    2. **ScalarizeArithOp** (benefit 1, for commutative ops: add, mul, max, min):
       - If one operand is splat constant → converts to scalar-splat variant
       - Can handle splat on either lhs or rhs (swaps operands if lhs is splat)
       - `addf(tile, splat)` → `adds(tile, scalar)`
       - `mulf(splat, tile)` → `muls(tile, scalar)` (swapped)

    3. **ScalarizeArithRhsOp** (benefit 1, for non-commutative ops: sub, div):
       - Only rhs splat is converted (cannot swap operands)
       - `subf(tile, splat)` → `subs(tile, scalar)`
       - `subf(splat, tile)` remains unchanged

    4. **ScalarizeShL/ScalarizeShR**: `shli/shrsi(tile, splat)` → `shls/shrs(tile, scalar)`

    5. **ScalarizeCompare**: `cmp(mode, tile, splat)` → `cmps(mode, tile, scalar)`
       - If lhs is splat, comparison mode is inverted (LT→GT, LE→GE, etc.)

    Example transformations:
    ```mlir
    // Splat constant converted to scalar
    %cst = constant dense<7.0> : tile<32xf32>
    %0 = addf %tile, %cst : tile<32xf32>
    // →
    %scalar = constant 7.0 : f32
    %0 = adds %tile, %scalar : tile<32xf32>

    // Max with zero becomes relu
    %zero = constant dense<0.0> : tile<32xf32>
    %0 = maximumf %zero, %tile : tile<32xf32>
    // →
    %0 = relu %tile : tile<32xf32>

    // Comparison with splat, lhs case (mode inverted)
    %cst = constant dense<1.0> : tile<32xf32>
    %0 = cmp LT %cst, %tile : tile<32xf32>
    // →
    %scalar = constant 1.0 : f32
    %0 = cmps GT %tile, %scalar : tile<32xf32>  // LT → GT
    ```
  }];
}

def TransformStoreFixpipe : Pass<"asctile-transform-store-fixpipe", "func::FuncOp"> {
  let summary = "Convert store/copy operations from L0C to fixpipe variants and fold relu/cast into fixpipe flags";
  let constructor = "mlir::asctile::createTransformStoreFixpipePass()";
  let description = [{
    This pass handles data movement from L0C (matrix multiplication output buffer) by converting regular
    store/copy operations to fixpipe variants, which support fused relu and quantization operations.

    The "fixpipe" operation is a hardware-specific pipeline that can combine:
    - Data transfer from L0C to UB or GM
    - Optional relu activation
    - Optional quantization (cast operation)

    Transformation patterns:
    1. **TransformCopyOp**: `copy` with L0C source → `copy_fixpipe`
    2. **TransformStoreOp**: `store` with L0C value → `store_fixpipe`
    3. **TransformFixpipeReluOp/CopyFixpipeReluOp**: If fixpipe input is `relu` on L0C,
       use the relu operand directly and set `relu=true` flag on fixpipe
    4. **TransformFixpipeCastOp/CopyFixpipeCastOp**: If fixpipe input is `cast` on L0C,
       use the cast input directly and set `quantize=true` flag on fixpipe

    Example transformations:
    ```mlir
    // Before: separate store and relu
    %result = matmul_acc %a, %b, %acc : tile<16x16xf32, L0C>
    %activated = relu %result : tile<16x16xf32, L0C>
    store %activated, %tensor[%offset] : tile<16x16xf32, L0C>, tensor

    // After: fused store_fixpipe with relu flag
    %result = matmul_acc %a, %b, %acc : tile<16x16xf32, L0C>
    store_fixpipe relu=true %result, %tensor[%offset] : tile<16x16xf32, L0C>, tensor

    // Before: separate copy and cast
    %result = matmul_acc ... : tile<16x16xf32, L0C>
    %casted = cast %result : tile<16x16xf32, L0C> to tile<16x16xi8, UB>
    copy %casted, [%zero] : tile<16x16xf32, L0C> to tile<16x16xi8, UB>

    // After: fused copy_fixpipe with quantize flag
    %result = matmul_acc ... : tile<16x16xf32, L0C>
    copy_fixpipe quantize=true %result, [%zero] : tile<16x16xf32, L0C> to tile<16x16xi8, UB>
    ```
  }];
}

def UnrollLoop : Pass<"asctile-unroll-loop", "func::FuncOp"> {
  let summary = "Unroll scf.for loops by their unroll_factor attribute value";
  let constructor = "mlir::asctile::createUnrollLoopPass()";
  let description = [{
    This pass unrolls `scf.for` loops by the factor specified in their `asctile.unroll_factor` attribute.
    After unrolling, the attribute is removed from the loop.

    For loops where the iteration count is statically known and evenly divisible by the unroll factor,
    the loop step is multiplied by the factor and the body is replicated.

    For dynamic loops (unknown upper bound) or when iteration count is not evenly divisible, an epilogue
    loop is created to handle the remaining iterations. The `unroll_group` attributes are removed from
    operations in the epilogue loop since they don't correspond to the original unrolled iterations.

    Example transformations:
    ```mlir
    // Static loop: 32 iterations, unroll factor 2
    scf.for %i = 0 to 32 step 1 {unroll_factor = 2} {
      %val = index_cast %i
      set_value %val, %tensor[0]
    }
    // After: 16 iterations with step 2, body replicated twice
    scf.for %i = 0 to 32 step 2 {
      %val = index_cast %i
      set_value %val, %tensor[0]
      %i1 = addi %i, 1
      %val1 = index_cast %i1
      set_value %val1, %tensor[0]
    }

    // Dynamic loop: unknown upper bound, unroll factor 3
    scf.for %i = 0 to %n step 1 {unroll_factor = 3} { ... }
    // After: main loop with step 3 + epilogue loop for remainder
    %remainder = remsi %n, 3
    %aligned = subi %n, %remainder
    scf.for %i = 0 to %aligned step 3 { /* body replicated 3 times */ }
    scf.for %i = %aligned to %n step 1 { /* original body, no unroll_group */ }
    ```
  }];
}

def UnscalarizeReduction : Pass<"asctile-unscalarize-reduction", "func::FuncOp"> {
  let summary = "Convert scalar reduction results to 1-element tiles when used in tile operations";
  let constructor = "mlir::asctile::createUnscalarizeReductionPass()";
  let dependentDialects = ["arith::ArithDialect", "asctile::AscTileDialect"];
  let description = [{
    This pass avoids scalar results from `reduce_as_1d` operations when all users are scalar-tile operations
    (adds, subs, muls, divs, mins, maxs, splat). Instead, the result is kept as a 1-element tile, which is
    more efficient for the hardware as it avoids scalar-to-tile conversions.

    The pass uses dialect conversion to transform:
    1. `reduce_as_1d` with scalar result → `reduce_as_1d` with 1-element tile result
    2. `splat` (scalar to tile) → `broadcast` (1-element tile to larger tile)
    3. Scalar-tile ops (adds, subs, etc.) → tile-tile ops (addf, subf, etc.)

    The transformation only applies when ALL users of the scalar reduction result are unscalarizable.
    If any user operates on the scalar directly (e.g., scalar arithmetic), the reduction is left unchanged.

    Example transformations:
    ```mlir
    // Before: scalar reduction used in scalar-tile ops
    %scalar = reduce_as_1d <sum> %tile : tile<16xf32>, f32
    %result1 = adds %tile, %scalar : tile<16xf32>
    %result2 = subs %tile, %scalar : tile<16xf32>

    // After: 1-element tile reduction used in tile-tile ops
    %reduced = reduce_as_1d <sum> %tile : tile<16xf32>, tile<1xf32>
    %broadcast1 = broadcast %reduced : tile<1xf32> to tile<16xf32>
    %result1 = addf %tile, %broadcast1 : tile<16xf32>
    %broadcast2 = broadcast %reduced : tile<1xf32> to tile<16xf32>
    %result2 = subf %tile, %broadcast2 : tile<16xf32>

    // Case skipped: scalar has non-unscalarizable user
    %scalar = reduce_as_1d <sum> %tile : tile<16xf32>, f32
    %result = adds %tile, %scalar : tile<16xf32>  // unscalarizable
    %scalar_op = addf %scalar, %scalar : f32      // scalar user -> NOT transformed
    ```
  }];
}

#endif // ASCTILE_PASSES_TD
