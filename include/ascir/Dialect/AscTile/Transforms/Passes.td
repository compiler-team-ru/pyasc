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

def CubeTransposeToLoad : Pass<"asctile-cube-transpose-to-load", "func::FuncOp"> {
  let summary = "Remove transpose operations on cube operands and add transpose attributes to load/copy operations";
  let constructor = "mlir::asctile::createCubeTransposeToLoadPass()";
  let description = [{
    This pass eliminates `asctile.transpose` operations on cube (matrix multiplication) operands in L0A/L0B
    buffers by adding `transposeA` or `transposeB` attributes to the underlying load and copy operations.

    The transformation applies when:
    1. The transpose result is in L0A or L0B buffer (cube operands)
    2. The transpose input is a `copy` operation with exactly one use
    3. The copy's source is a `load` operation with exactly one use

    The pass:
    - Removes the transpose operation
    - Adds `transposeA` attribute to load and copy if result is in L0A
    - Adds `transposeB` attribute to load and copy if result is in L0B
    - Swaps the dimensions in the copy result type (transposed shape)

    Example transformation:
    ```mlir
    // Before: separate load, copy, and transpose
    %0 = load %tensor[%i, %j], %pad : tensor, tensor<32x128xf16, L1>
    %1 = copy %0[%x, %y] : tensor<32x128xf16, L1>, tensor<32x128xf16, L0A>
    %2 = transpose %1 : tensor<32x128xf16, L0A> to tensor<128x32xf16, L0A>

    // After: transpose attribute propagated, transpose op removed
    %0 = load %tensor[%i, %j], %pad {transposeA} : tensor, tensor<32x128xf16, L1>
    %1 = copy %0[%x, %y] {transposeA} : tensor<128x32xf16, L1>, tensor<128x32xf16, L0A>
    ```
  }];
}

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

def DetectBiasLoad : Pass<"asctile-detect-bias-load", "func::FuncOp"> {
  let summary = "Detect load operation from GM to L1 for bias";
  let constructor = "mlir::asctile::createDetectBiasLoadPass()";
  let description = [{
    This pass identifies load operations that feed bias data for matrix multiplication and marks them
    with the `isBias` attribute. This enables subsequent passes to optimize bias handling in matmul operations.

    The pass identifies bias loads by checking if the loaded tensor is copied to `TileLocation::BT` (bias table),
    which is the dedicated memory location for bias values in matmul operations.

    Example transformation:
    ```mlir
    // Before: load without bias marking
    %0 = load %tensor[%offset] : tensor, tensor<128xf16, L1>
    %1 = copy %0[%base] : tensor<128xf16, L1> to tensor<128xf16, BT>

    // After: load marked as bias
    %0 = load %tensor[%offset] {isBias} : tensor, tensor<128xf16, L1>
    %1 = copy %0[%base] : tensor<128xf16, L1> to tensor<128xf16, BT>
    ```
  }];
}

def FoldCast : Pass<"asctile-fold-cast", "func::FuncOp"> {
  let summary = "Fold chains of cast operations and verify supported type conversions";
  let constructor = "mlir::asctile::createFoldCastPass()";
  let description = [{
    This pass folds chains of consecutive `asctile.cast` operations and validates that all remaining
    cast operations perform type conversions that are supported by the hardware.

    **Folding**: When two cast operations are chained (e.g., `cast(i8→i16)` followed by `cast(i16→i32)`),
    they are folded into a single direct cast (e.g., `cast(i8→i32)`), if the resulting conversion is
    supported.

    **Validation**: After folding, the pass checks that all cast operations use supported type conversions:
    - Integer to integer: i8→i16, i8→i32, i16→i32, i32→i16, i32→i64, i64→i32
    - Integer to float: i8/i16/i32→f16, i16/i32/i64→f32
    - Float to integer: bf16→i32, f16→i8/i16/i32, f32→i16/i32/i64
    - Float to float: bf16→f16/f32, f16→bf16/f32, f32→bf16/f16/f32

    If any cast operation violates these constraints, the pass emits an error and signals failure.
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
    1. **ConvertLoadGMToL0**: Splits `load` producing L0A/L0B tensor into:
       - `load` producing L1 tensor (from GM)
       - `copy` from L1 to L0A/L0B with zero offsets

    2. **MarkTileOperandInMmad**: Marks L1 tiles used exclusively for matrix multiplication:
       - If an L1 tensor is copied only to L0A, marks it with `isMatrixA` attribute
       - Validates consistency: same L1 tensor should not be copied to both L0A and L0B
       - Emits error if L1 tensor is used for non-copy operations

    Example transformation:
    ```mlir
    // Before: direct load to L0A
    %0 = load %tensor[%offset] : tensor, tensor<16x16xf32, L0A>

    // After: split into L1 load + L1-to-L0A copy
    %0 = load %tensor[%offset] : tensor, tensor<16x16xf32, L1>
    %zero = constant 0 : i32
    %1 = copy %0[%zero, %zero] : tensor<16x16xf32, L1> to tensor<16x16xf32, L0A>
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
  let summary = "Convert tensor arithmetic operations with splat operands to scalar-splat variants for HW efficiency";
  let constructor = "mlir::asctile::createTransformMathOpsPass()";
  let dependentDialects = ["asctile::AscTileDialect"];
  let description = [{
    This pass specializes arithmetic and comparison operations on tiles when one operand is a splat value
    (uniform constant across all elements). The scalar-splat variants (`adds`, `muls`, `cmps`, etc.) are
    more efficient on the hardware than tensor-tensor operations.

    Transformation patterns (applied greedily with pattern benefits):
    1. **MaxWithZeroToRelu** (benefit 2): `max(x, 0)` or `max(0, x)` → `relu(x)` for `maximumf`, `maxnumf`, `maxsi`

    2. **ScalarizeArithOp** (benefit 1, for commutative ops: add, mul, max, min):
       - If one operand is splat constant → converts to scalar-splat variant
       - Can handle splat on either lhs or rhs (swaps operands if lhs is splat)
       - `addf(tensor, splat)` → `adds(tensor, scalar)`
       - `mulf(splat, tensor)` → `muls(tensor, scalar)` (swapped)

    3. **ScalarizeArithRhsOp** (benefit 1, for non-commutative ops: sub, div):
       - Only rhs splat is converted (cannot swap operands)
       - `subf(tensor, splat)` → `subs(tensor, scalar)`
       - `subf(splat, tensor)` remains unchanged

    4. **ScalarizeShL/ScalarizeShR**: `shli/shrsi(tensor, splat)` → `shls/shrs(tensor, scalar)`

    5. **ScalarizeCompare**: `cmp(mode, tensor, splat)` → `cmps(mode, tensor, scalar)`
       - If lhs is splat, comparison mode is inverted (LT→GT, LE→GE, etc.)

    Example transformations:
    ```mlir
    // Splat constant converted to scalar
    %cst = constant dense<7.0> : tensor<32xf32>
    %0 = addf %tensor, %cst : tensor<32xf32>
    // →
    %scalar = constant 7.0 : f32
    %0 = adds %tensor, %scalar : tensor<32xf32>

    // Max with zero becomes relu
    %zero = constant dense<0.0> : tensor<32xf32>
    %0 = maximumf %zero, %tensor : tensor<32xf32>
    // →
    %0 = relu %tensor : tensor<32xf32>

    // Comparison with splat, lhs case (mode inverted)
    %cst = constant dense<1.0> : tensor<32xf32>
    %0 = cmp LT %cst, %tensor : tensor<32xf32>
    // →
    %scalar = constant 1.0 : f32
    %0 = cmps GT %tensor, %scalar : tensor<32xf32>  // LT → GT
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
    %result = matmul_acc %acc, %a, %b : tensor<16x16xf32, L0C>
    %activated = relu %result : tensor<16x16xf32, L0C>
    store %activated, %tensor[%offset] : tensor<16x16xf32, L0C>, tensor

    // After: fused store_fixpipe with relu flag
    %result = matmul_acc %acc, %a, %b : tensor<16x16xf32, L0C>
    store_fixpipe relu=true %result, %tensor[%offset] : tensor<16x16xf32, L0C>, tensor

    // Before: separate copy and cast
    %result = matmul_acc ... : tensor<16x16xf32, L0C>
    %casted = cast %result : tensor<16x16xf32, L0C> to tensor<16x16xi8, UB>
    copy %casted, [%zero] : tensor<16x16xf32, L0C> to tensor<16x16xi8, UB>

    // After: fused copy_fixpipe with quantize flag
    %result = matmul_acc ... : tensor<16x16xf32, L0C>
    copy_fixpipe quantize=true %result, [%zero] : tensor<16x16xf32, L0C> to tensor<16x16xi8, UB>
    ```
  }];
}

def UnrollLoop : Pass<"asctile-unroll-loop", "func::FuncOp"> {
  let summary = "Unroll scf.for loops by their unroll_factor attribute value";
  let constructor = "mlir::asctile::createUnrollLoopPass()";
  let options = [Option<"annotate", "annotate", "bool", "false", "Add annotation attribute to each inner operation">];
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
  let summary = "Convert scalar reduction results to 1-element tiles when used in tensor operations";
  let constructor = "mlir::asctile::createUnscalarizeReductionPass()";
  let dependentDialects = ["arith::ArithDialect", "asctile::AscTileDialect"];
  let description = [{
    This pass avoids scalar results from `reduce_as_1d` operations when all users are scalar-tensor operations
    (adds, subs, muls, divs, mins, maxs, splat). Instead, the result is kept as a 1-element tensor, which is
    more efficient for the hardware as it avoids scalar-to-tensor conversions.

    The pass uses dialect conversion to transform:
    1. `reduce_as_1d` with scalar result → `reduce_as_1d` with 1-element tensor result
    2. `splat` (scalar to tensor) → `broadcast` (1-element tensor to larger tensor)
    3. Scalar-tensor ops (adds, subs, etc.) → tensor-tensor ops (addf, subf, etc.)

    The transformation only applies when ALL users of the scalar reduction result are unscalarizable.
    If any user operates on the scalar directly (e.g., scalar arithmetic), the reduction is left unchanged.

    Example transformations:
    ```mlir
    // Before: scalar reduction used in scalar-tensor ops
    %scalar = reduce_as_1d <sum> %tensor : tensor<16xf32>, f32
    %result1 = adds %tensor, %scalar : tensor<16xf32>
    %result2 = subs %tensor, %scalar : tensor<16xf32>

    // After: 1-element tensor reduction used in tensor-tensor ops
    %reduced = reduce_as_1d <sum> %tensor : tensor<16xf32>, tensor<1xf32>
    %broadcast1 = broadcast %reduced : tensor<1xf32> to tensor<16xf32>
    %result1 = addf %tensor, %broadcast1 : tensor<16xf32>
    %broadcast2 = broadcast %reduced : tensor<1xf32> to tensor<16xf32>
    %result2 = subf %tensor, %broadcast2 : tensor<16xf32>

    // Case skipped: scalar has non-unscalarizable user
    %scalar = reduce_as_1d <sum> %tensor : tensor<16xf32>, f32
    %result = adds %tensor, %scalar : tensor<16xf32>  // unscalarizable
    %scalar_op = addf %scalar, %scalar : f32      // scalar user -> NOT transformed
    ```
  }];
}

def VectorTransposeToLoad : Pass<"asctile-vector-transpose-to-load", "func::FuncOp"> {
  let summary = "Replace transpose and load to single operation when possible";
  let constructor = "mlir::asctile::createVectorTransposeToLoadPass()";
  let dependentDialects = ["asctile::AscTileDialect"];
  let description = [{
    Replace `asctile.load` and `asctile.transpose` into single `asctile.load` with `asctile.transpose_dims` attribute.
    This pass only handles GM->UB transfers.

    Example:
    ```mlir
    // Before:
    %1 = asctile.load : tensor<16x32xf32, UB>
    %2 = asctile.transpose %1 : tensor<16x32xf32, UB> to tensor<32x16xf32, UB>

    // After:
    %2 = asctile.load {asctile.transpose_dims = [1, 0]} : tensor<32x16xf32, UB>
    ```
  }];
}

#endif // ASCTILE_PASSES_TD
