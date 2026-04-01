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

include "Dialect.td"
include "Types.td"

include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ViewLikeInterface.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"

include "Attributes.td"

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

def SplatOp : AscTile_Op<"splat", [
  Pure,
  TypesMatchWith<"type of 'value' matches element type of 'result'",
                 "result", "value",
                 "::llvm::cast<ShapedType>($_self).getElementType()">,
]> {
  let summary = "Create a tile filled with splat value";
  let arguments = (ins AnyType:$value);
  let results = (outs Tile:$result);
  let assemblyFormat = [{
    $value attr-dict `:` qualified(type($result))
  }];
}

def LoadOp : AscTile_Op<"load", [
  AttrSizedOperandSegments,
  AllElementTypesMatch<["base", "result"]>,
  OptionalTypesMatchWith<"type of 'padValue' matches element type of 'result'",
                         "result", "padValue",
                         "::llvm::cast<ShapedType>($_self).getElementType()">,
  MemoryEffectsOpInterface,
]> {
  let summary = "Load from the tensor into a new tile";
  let arguments = (ins Arg<Tensor, "tensor to load from", [MemRead]>:$base,
                   Variadic<I32>:$offsets, Optional<AnyType>:$padValue);
  let results = (outs Tile:$result);
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
  let arguments = (ins Tile:$value,
                   Arg<Tensor, "tensor to store to", [MemWrite]>:$base,
                   Variadic<I32>:$offsets);
  let assemblyFormat = [{
    $value `,` $base `[` $offsets `]` attr-dict `:` qualified(type($value)) `,`
    qualified(type($base))
  }];
}

def GetValueOp : AscTile_Op<"get_value", [
  AllElementTypesMatch<["base", "result"]>,
  MemoryEffectsOpInterface,
]> {
  let summary = "Load value from tensor";
  let arguments = (ins Arg<Tensor, "tensor to read from", [MemRead]>:$base,
                   Variadic<I32>:$offsets);
  let results = (outs AnyType:$result);
  let assemblyFormat = [{
    $base `[` $offsets `]` attr-dict `:` qualified(type($base)) `,` qualified(type($result))
  }];
}

def SetValueOp : AscTile_Op<"set_value", [
  AllElementTypesMatch<["base", "value"]>,
  MemoryEffectsOpInterface,
]> {
  let summary = "Store value into a tensor";
  let arguments = (ins AnyType:$value,
                   Arg<Tensor, "tensor to be updated", [MemWrite]>:$base,
                   Variadic<I32>:$offsets);
  let assemblyFormat = [{
    $value `,` $base `[` $offsets `]` attr-dict `:` qualified(type($value)) `,`
    qualified(type($base))
  }];
}

def YieldOp : AscTile_Op<"yield", [Terminator]> {
  let assemblyFormat = "attr-dict";
}

def CountMaskOp
    : AscTile_Op<"count_mask", [SingleBlockImplicitTerminator<"YieldOp">]> {
  let arguments = (ins I64:$count, Optional<AnyType>:$other);
  let regions = (region SizedRegion<1>:$region);
  let assemblyFormat = [{
    $count (`,` $other^ `:` type($other))? $region attr-dict
  }];
}

def BitwiseMaskOp
    : AscTile_Op<"bitwise_mask", [SingleBlockImplicitTerminator<"YieldOp">]> {
  let arguments = (ins I64:$highBits, I64:$lowBits, Optional<AnyType>:$other);
  let regions = (region SizedRegion<1>:$region);
  let assemblyFormat = [{
    $highBits `,` $lowBits (`,` $other^ `:` type($other))? $region attr-dict
  }];
}

class AscTile_VecScalarOp<string mnemonic, list<Trait> traits = []>
    : AscTile_Op<mnemonic, traits # [
  Pure,
  AllTypesMatch<["base", "result"]>,
  TypesMatchWith<"type of 'value' matches element type of 'base'",
                 "base", "value",
                 "::llvm::cast<ShapedType>($_self).getElementType()">,
]> {
  let arguments = (ins Tile:$base, AnyType:$value);
  let results = (outs Tile:$result);
  let assemblyFormat = "$base `,` $value attr-dict `:` qualified(type($base))";
}

def AddSOp : AscTile_VecScalarOp<"adds">;
def SubSOp : AscTile_VecScalarOp<"subs">;
def MulSOp : AscTile_VecScalarOp<"muls">;
def ShLSOp : AscTile_VecScalarOp<"shls">;
def ShRSOp : AscTile_VecScalarOp<"shrs">;
def DivSOp : AscTile_VecScalarOp<"divs">;
def MinSOp : AscTile_VecScalarOp<"mins">;
def MaxSOp : AscTile_VecScalarOp<"maxs">;

def CmpSOp : AscTile_Op<"cmps", [
  Pure,
  TypesMatchWith<"type of 'value' matches element type of 'base'",
                 "base", "value",
                 "::llvm::cast<ShapedType>($_self).getElementType()">,
]> {
  let arguments = (ins Tile:$base, AnyType:$value,
                   AscTile_CompareModeAttr:$cmpMode);
  let results = (outs Tile:$result);
  let assemblyFormat = [{
    $cmpMode $base `,` $value attr-dict `:` qualified(type($base)) `,`
    qualified(type($result))
  }];
}

def ReluOp : AscTile_Op<"relu", [Pure, AllTypesMatch<["operand", "result"]>]> {
  let arguments = (ins Tile:$operand);
  let results = (outs Tile:$result);
  let assemblyFormat = "$operand attr-dict `:` qualified(type($result))";
}

def CastOp : AscTile_Op<"cast", [
  Pure, SameOperandsAndResultShape, DeclareOpInterfaceMethods<CastOpInterface>
]> {
  let arguments = (ins Tile:$in);
  let results = (outs Tile:$out);
  let assemblyFormat = [{
    $in attr-dict `:` qualified(type($in)) `to` qualified(type($out))
  }];
  let hasFolder = 1;
}

def ReshapeOp : AscTile_Op<"reshape", [
  Pure, SameOperandsAndResultElementType, DeclareOpInterfaceMethods<CastOpInterface>
]> {
  let arguments = (ins Tile:$in);
  let results = (outs Tile:$out);
  let assemblyFormat = [{
    $in attr-dict `:` qualified(type($in)) `to` qualified(type($out))
  }];
  let hasFolder = 1;
}

def SelectOp : AscTile_Op<"select", [Pure, AllTypesMatch<["src0", "src1", "result"]>]> {
  let arguments = (ins Tile:$selMask, Tile:$src0, Tile:$src1);
  let results = (outs Tile:$result);
  let assemblyFormat = [{
    $selMask `,` $src0 `,` $src1 `,` attr-dict `:` qualified(type($selMask)) `,`
    qualified(type($result))
  }];
}

class AscTile_ReduceAs1dOp<string mnemonic> : AscTile_Op<mnemonic, [
  Pure,
  TypesMatchWith<"type of 'value' matches element type of 'base'",
                 "operand", "result",
                 "::llvm::cast<ShapedType>($_self).getElementType()">,
]> {
  let arguments = (ins Tile:$operand);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$operand attr-dict `:` qualified(type($operand))";
}

def ReduceSumAs1dOp : AscTile_ReduceAs1dOp<"reduce_sum_as_1d">;
def ReduceMaxAs1dOp : AscTile_ReduceAs1dOp<"reduce_max_as_1d">;
def ReduceMinAs1dOp : AscTile_ReduceAs1dOp<"reduce_min_as_1d">;

def AtomicRMWOp : AscTile_Op<"atomic_rmw", [
  AllElementTypesMatch<["base", "value"]>, MemoryEffectsOpInterface
]> {
  let summary = "Atomic accumulation the tile data into a tensor";
  let arguments = (ins Tile:$value,
                   Arg<Tensor, "tensor to store to", [MemWrite]>:$base,
                   Variadic<I32>:$offsets, AscTile_AtomicKindAttr:$kind);
  let assemblyFormat = [{
    `<` $kind `>` $value `,` $base `[` $offsets `]` attr-dict `:` qualified(type($value)) `,`
    qualified(type($base))
  }];
}

def SoftmaxOp : AscTile_Op<"softmax", [
  AllTypesMatch<["operand", "result"]>,
]> {
  let summary = "high-level softmax";
  let arguments = (ins Tile:$operand);
  let results = (outs Tile:$result);
  let assemblyFormat = "$operand attr-dict `:` qualified(type($result))";
}

def MatmulOp : AscTile_Op<"matmul", [
  Pure, AllElementTypesMatch<["matrixA", "matrixB"]>, MemoryEffectsOpInterface
]> {
  let summary = "Low-level matrix multiplication the tile data";
  let arguments = (ins Tile:$matrixA, Tile:$matrixB);
  let results = (outs Tile:$result);
  let assemblyFormat = [{
    $matrixA `,` $matrixB attr-dict `:` type($matrixA) `,` type($matrixB) `->`
    type($result)
  }];
}

def BroadcastOp : AscTile_Op<"broadcast", [
  Pure, AllElementTypesMatch<["input", "result"]>
]> {
  let summary = "Broadcast tile to the provided shape";
  let arguments = (ins Tile:$input);
  let results = (outs Tile:$result);
  let assemblyFormat = [{
    $input attr-dict `:` qualified(type($input)) `to` qualified(type($result))
  }];
}

class AscTile_ReduceOp<string mnemonic> : AscTile_Op<mnemonic, [
  Pure,
  AllElementTypesMatch<["operand", "result"]>
]> {
  let arguments = (ins Tile:$operand, I32ArrayAttr:$dims);
  let results = (outs Tile:$result);
  let assemblyFormat = [{
    $operand attr-dict `:` qualified(type($operand)) `,` qualified(type($result))
  }];
}

def AscTile_ReduceSumOp: AscTile_ReduceOp<"reduce_sum">;
def AscTile_ReduceMinOp: AscTile_ReduceOp<"reduce_min">;
def AscTile_ReduceMaxOp: AscTile_ReduceOp<"reduce_max">;
def AscTile_ReduceProdOp: AscTile_ReduceOp<"reduce_prod">;

#endif // ASCTILE_OPS_TD
