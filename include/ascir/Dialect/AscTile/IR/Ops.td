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

class AscTile_Op<string mnemonic, list<Trait> traits = []>
    : Op<AscTile_Dialect, mnemonic, traits> {
  let cppNamespace = "::mlir::asctile";
}

def TensorOp : AscTile_Op<"tensor", [Pure]> {
  let summary = "Create tensor";
  let arguments = (ins AnyRankedOrUnrankedMemRef:$base);
  let results = (outs Tensor:$result);
  let assemblyFormat = [{
    $base attr-dict `:` qualified(type($base)) `,` qualified(type($result))
  }];
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
]> {
  let summary = "Load from the tensor into a new tile";
  let arguments = (ins Tensor:$base, Variadic<I32>:$offsets,
                   Optional<AnyType>:$padValue);
  let results = (outs Tile:$result);
  let assemblyFormat = [{
    $base `[` $offsets `]` (`,` $padValue^)? attr-dict `:`
    qualified(type($base)) `,` qualified(type($result))
  }];
  let hasCanonicalizeMethod = 1;
}

def StoreOp : AscTile_Op<"store", [AllElementTypesMatch<["base", "value"]>]> {
  let summary = "Store a tile into a tensor";
  let arguments = (ins Tile:$value, Tensor:$base, Variadic<I32>:$offsets);
  let assemblyFormat = [{
    $value `,` $base `[` $offsets `]` attr-dict `:` qualified(type($value)) `,`
    qualified(type($base))
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

def AddsOp : AscTile_VecScalarOp<"adds">;
def SubsOp : AscTile_VecScalarOp<"subs">;
def MulsOp : AscTile_VecScalarOp<"muls">;

def ReluOp : AscTile_Op<"relu", [Pure, AllTypesMatch<["operand", "result"]>]> {
  let arguments = (ins Tile:$operand);
  let results = (outs Tile:$result);
  let assemblyFormat = "$operand attr-dict `:` qualified(type($result))";
}

#endif // ASCTILE_OPS_TD
