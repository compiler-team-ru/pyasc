/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASCTILE_OPS_TD
#define ASCTILE_OPS_TD

include "Attributes.td"
include "Dialect.td"
include "Types.td"

include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ViewLikeInterface.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"

class AscTile_Op<string mnemonic, list<Trait> traits = []>
    : Op<AscTile_Dialect, mnemonic, traits> {
  let cppNamespace = "::mlir::asctile";
}

def TensorOp : AscTile_Op<"tensor", [Pure]> {
  let summary = "Create tensor";
  let arguments = (ins AnyRankedOrUnrankedMemRef:$base, Variadic<I32>:$sizes);
  let results = (outs Tensor:$result);
  let assemblyFormat = [{
    $base `(` $sizes `)` attr-dict `:` qualified(type($base)) `,`
    qualified(type($result))
  }];
  let hasVerifier = 1;
}

def DimOp : AscTile_Op<"dim", [Pure]> {
  let summary = "Obtain dimension of given tensor";
  let arguments = (ins Tensor:$base, I32Attr:$index);
  let results = (outs I32:$result);
  let assemblyFormat = "$base `,` $index attr-dict `:` qualified(type($base))";
  let hasFolder = 1;
  let hasVerifier = 1;
}

def SplatOp
    : AscTile_Op<"splat", [Pure, ShapedMatchElemType<"result", "value">]> {
  let summary = "Create a tile filled with splat value";
  let arguments = (ins AscTile_Numeric:$value);
  let results = (outs NumericTile:$result);
  let assemblyFormat = "$value attr-dict `:` qualified(type($result))";
}

def LoadOp : AscTile_Op<"load", [
  AttrSizedOperandSegments,
  AllElementTypesMatch<["base", "result"]>,
  OptionalShapedMatchElemType<"result", "padValue">,
  MemoryEffectsOpInterface,
]> {
  let summary = "Load from the tensor into a new tile";
  let arguments = (ins Arg<Tensor, "tensor to load from", [MemRead]>:$base,
                   Variadic<I32>:$offsets, Optional<AscTile_Numeric>:$padValue);
  let results = (outs NumericTile:$result);
  let assemblyFormat = [{
    $base `[` $offsets `]` (`,` $padValue^)? attr-dict `:`
    qualified(type($base)) `,` qualified(type($result))
  }];
  let hasCanonicalizeMethod = 1;
}

def StoreOp : AscTile_Op<"store", [
  AllElementTypesMatch<["base", "value"]>, MemoryEffectsOpInterface
]> {
  let summary = "Store a tile into a tensor";
  let arguments = (ins NumericTile:$value,
                   Arg<Tensor, "tensor to store to", [MemWrite]>:$base,
                   Variadic<I32>:$offsets);
  let assemblyFormat = [{
    $value `,` $base `[` $offsets `]` attr-dict `:` qualified(type($value)) `,`
    qualified(type($base))
  }];
}

def GetValueOp : AscTile_Op<"get_value", [
  ShapedMatchElemType<"base", "result">, MemoryEffectsOpInterface
]> {
  let summary = "Load value from tensor";
  let arguments = (ins Arg<Tensor, "tensor to read from", [MemRead]>:$base,
                   Variadic<I32>:$offsets);
  let results = (outs AscTile_Numeric:$result);
  let assemblyFormat = [{
    $base `[` $offsets `]` attr-dict `:` qualified(type($base))
  }];
}

def SetValueOp : AscTile_Op<"set_value", [
  ShapedMatchElemType<"base", "value">, MemoryEffectsOpInterface
]> {
  let summary = "Store value into a tensor";
  let arguments = (ins AscTile_Numeric:$value,
                   Arg<Tensor, "tensor to be updated", [MemWrite]>:$base,
                   Variadic<I32>:$offsets);
  let assemblyFormat = [{
    $value `,` $base `[` $offsets `]` attr-dict `:` qualified(type($base))
  }];
}

def YieldOp : AscTile_Op<"yield", [Terminator]> {
  let summary = "Terminator for operations with region";
  let assemblyFormat = "attr-dict";
}

def CountMaskOp
    : AscTile_Op<"count_mask", [SingleBlockImplicitTerminator<"YieldOp">]> {
  let summary = "Apply mask (count mode) to nested operations";
  let arguments = (ins I64:$count, Optional<AscTile_Numeric>:$other);
  let regions = (region SizedRegion<1>:$region);
  let assemblyFormat = [{
    $count (`,` $other^ `:` type($other))? $region attr-dict
  }];
}

def BitwiseMaskOp
    : AscTile_Op<"bitwise_mask", [SingleBlockImplicitTerminator<"YieldOp">]> {
  let summary = "Apply mask (bit mode) to nested operations";
  let arguments = (ins I64:$highBits, I64:$lowBits,
                   Optional<AscTile_Numeric>:$other);
  let regions = (region SizedRegion<1>:$region);
  let assemblyFormat = [{
    $highBits `,` $lowBits (`,` $other^ `:` type($other))? $region attr-dict
  }];
}

class AscTile_VecScalarOp<string mnemonic, string kind, list<Trait> traits = []>
    : AscTile_Op<mnemonic, traits # [
  Pure, AllTypesMatch<["base", "result"]>, ShapedMatchElemType<"base", "value">
]> {
  let summary = "Compute element-wise " # kind # " of tile and scalar";
  let arguments = (ins NumericTile:$base, AscTile_Numeric:$value);
  let results = (outs NumericTile:$result);
  let assemblyFormat = "$base `,` $value attr-dict `:` qualified(type($base))";
}

def AddSOp : AscTile_VecScalarOp<"adds", "addition">;
def SubSOp : AscTile_VecScalarOp<"subs", "subtraction">;
def MulSOp : AscTile_VecScalarOp<"muls", "multiplication">;
def ShLSOp : AscTile_VecScalarOp<"shls", "bitwise left shift">;
def ShRSOp : AscTile_VecScalarOp<"shrs", "bitwise right shift">;
def DivSOp : AscTile_VecScalarOp<"divs", "division">;
def MinSOp : AscTile_VecScalarOp<"mins", "minimum value">;
def MaxSOp : AscTile_VecScalarOp<"maxs", "maximum value">;

def CmpSOp : AscTile_Op<"cmps", [
  Pure,
  ShapedMatchElemType<"base", "value">,
  TypesMatchWith<
    "result type has i1 element type and same shape as operands",
    "base", "result", "::getI1SameShape($_self)">
]> {
  let summary = "Compute element-wise comparison of tile and scalar";
  let arguments = (ins NumericTile:$base, AscTile_Numeric:$value,
                   AscTile_CompareModeAttr:$cmpMode);
  let results = (outs TileOf<[I1]>:$result);
  let assemblyFormat = [{
    $cmpMode $base `,` $value attr-dict `:` qualified(type($base))
  }];
}

def ReluOp : AscTile_Op<"relu", [Pure, AllTypesMatch<["operand", "result"]>]> {
  let summary = "Compute element-wise ReLU value of tile";
  let arguments = (ins NumericTile:$operand);
  let results = (outs NumericTile:$result);
  let assemblyFormat = "$operand attr-dict `:` qualified(type($result))";
}

def CastOp : AscTile_Op<"cast", [
  Pure, SameOperandsAndResultShape, DeclareOpInterfaceMethods<CastOpInterface>
]> {
  let summary = "Cast tile elements to other data type";
  let arguments = (ins NumericTile:$in);
  let results = (outs NumericTile:$out);
  let assemblyFormat = [{
    $in attr-dict `:` qualified(type($in)) `to` qualified(type($out))
  }];
  let hasFolder = 1;
}

def ReshapeOp : AscTile_Op<"reshape", [
  Pure, SameOperandsAndResultElementType, DeclareOpInterfaceMethods<CastOpInterface>
]> {
  let summary = "Create tile with same data but different shape";
  let arguments = (ins NumericTile:$in);
  let results = (outs NumericTile:$out);
  let assemblyFormat = [{
    $in attr-dict `:` qualified(type($in)) `to` qualified(type($out))
  }];
  let hasFolder = 1;
}

def SelectOp : AscTile_Op<"select", [
  Pure,
  AllTypesMatch<["src0", "src1", "result"]>,
  TypesMatchWith<
    "mask type has i1 element type and same shape as operands and result",
    "result", "selMask", "::getI1SameShape($_self)">
]> {
  let summary = "Select elements from two tiles depending on a mask";
  let arguments = (ins TileOf<[I1]>:$selMask, NumericTile:$src0,
                   NumericTile:$src1);
  let results = (outs NumericTile:$result);
  let assemblyFormat = [{
    $selMask `,` $src0 `,` $src1 attr-dict `:` qualified(type($result))
  }];
}

class AscTile_ReduceAs1dOp<string mnemonic>
    : AscTile_Op<mnemonic, [Pure, ShapedMatchElemType<"operand", "result">]> {
  let summary = "Reduce tile to a single value";
  let arguments = (ins NumericTile:$operand);
  let results = (outs AscTile_Numeric:$result);
  let assemblyFormat = "$operand attr-dict `:` qualified(type($operand))";
}

def ReduceSumAs1dOp : AscTile_ReduceAs1dOp<"reduce_sum_as_1d">;
def ReduceMaxAs1dOp : AscTile_ReduceAs1dOp<"reduce_max_as_1d">;
def ReduceMinAs1dOp : AscTile_ReduceAs1dOp<"reduce_min_as_1d">;

def AtomicRMWOp : AscTile_Op<"atomic_rmw", [
  AllElementTypesMatch<["base", "value"]>, MemoryEffectsOpInterface
]> {
  let summary = "Atomic accumulation the tile data into a tensor";
  let arguments = (ins NumericTile:$value,
                   Arg<Tensor, "tensor to store to", [MemWrite]>:$base,
                   Variadic<I32>:$offsets, AscTile_AtomicKindAttr:$kind);
  let assemblyFormat = [{
    `<` $kind `>` $value `,` $base `[` $offsets `]` attr-dict `:`
    qualified(type($value)) `,` qualified(type($base))
  }];
}

def SoftmaxOp
    : AscTile_Op<"softmax", [Pure, AllTypesMatch<["operand", "result"]>]> {
  let summary = "Compute row-wise softmax of 1D or 2D tile";
  Type allowedTileType = TileOfRankAndType<[1, 2], [AscTile_Numeric]>;
  let arguments = (ins allowedTileType:$operand);
  let results = (outs allowedTileType:$result);
  let assemblyFormat = "$operand attr-dict `:` qualified(type($result))";
}

def MatmulOp : AscTile_Op<"matmul", [
  Pure, AllElementTypesMatch<["matrixA", "matrixB"]>
]> {
  let summary = "Compute matrix multiplication on 2D tiles";
  Type inputTileType = TileOfRankAndType<[2], [AscTile_Float]>;
  let arguments = (ins inputTileType:$matrixA, inputTileType:$matrixB);
  let results = (outs TileOfRankAndType<[2], [F32]>:$result);
  let assemblyFormat = [{
    $matrixA `,` $matrixB attr-dict `:` type($matrixA) `,` type($matrixB) `->`
    type($result)
  }];
}

def BroadcastOp : AscTile_Op<"broadcast", [
  Pure, AllElementTypesMatch<["operand", "result"]>
]> {
  let summary = "Broadcast tile to the provided shape";
  let arguments = (ins NumericTile:$operand);
  let results = (outs NumericTile:$result);
  let assemblyFormat = [{
    $operand attr-dict `:` qualified(type($operand)) `to` qualified(type($result))
  }];
}

class AscTile_ReduceOp<string mnemonic> : AscTile_Op<mnemonic, [
  Pure, AllElementTypesMatch<["operand", "result"]>
]> {
  let summary = "Reduce tile to the provided shape";
  let arguments = (ins NumericTile:$operand, I32ArrayAttr:$dims);
  let results = (outs NumericTile:$result);
  let assemblyFormat = [{
    $operand attr-dict `:` qualified(type($operand)) `,` qualified(type($result))
  }];
}

def AscTile_ReduceSumOp: AscTile_ReduceOp<"reduce_sum">;
def AscTile_ReduceMinOp: AscTile_ReduceOp<"reduce_min">;
def AscTile_ReduceMaxOp: AscTile_ReduceOp<"reduce_max">;
def AscTile_ReduceProdOp: AscTile_ReduceOp<"reduce_prod">;

#endif // ASCTILE_OPS_TD
