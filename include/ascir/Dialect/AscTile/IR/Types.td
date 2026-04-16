/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASCTILE_TYPES_TD
#define ASCTILE_TYPES_TD

include "Dialect.td"
include "mlir/IR/BuiltinTypeInterfaces.td"
include "mlir/IR/BuiltinTypes.td"

class AscTile_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<AscTile_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def Tensor : AscTile_Type<"Tensor", "tensor",
                       [DeclareTypeInterfaceMethods<ShapedTypeInterface>]> {
  let parameters = (ins ArrayRefParameter<"int64_t">:$shape,
                        "Type":$elementType);
  let builders = [
    TypeBuilderWithInferredContext<(ins
      "ArrayRef<int64_t>":$shape, "Type":$elementType)>,
  ];
  let assemblyFormat = "`<` custom<ShapeXType>($shape, $elementType) `>`";
}

def Tile : AscTile_Type<"Tile", "tile", [
  DeclareTypeInterfaceMethods<ShapedTypeInterface>, ValueSemantics
]> {
  let parameters = (ins ArrayRefParameter<"int64_t">:$shape,
                        "Type":$elementType, "TileLocationAttr":$locAttr);
  let builders = [
    TypeBuilderWithInferredContext<(ins
      "ArrayRef<int64_t>":$shape, "Type":$elementType,
      CArg<"TileLocation", "TileLocation::UB">:$loc)>,
  ];
  let assemblyFormat = [{
    `<` custom<ShapeXType>($shape, $elementType) `,`
    custom<TileLocation>($locAttr) `>`
  }];
  let extraClassDeclaration = [{
    TileLocation getLoc() const;
  }];
}

def IsTileTypePred : And<[
  CPred<"::llvm::isa<::mlir::asctile::TileType>($_self)">,
  CPred<"::llvm::cast<::mlir::asctile::TileType>($_self).getRank() > 0">,
]>;

class IsTileOfRankPred<list<int> allowedRanks> : And<[
  IsTileTypePred,
  Or<!foreach(allowedlength, allowedRanks,
     CPred<[{::llvm::cast<::mlir::asctile::TileType>($_self).getRank()
           == }] # allowedlength>)>,
]>;

class IsTileOfLengthPred<list<int> allowedLengths> : And<[
  IsTileTypePred,
  Or<!foreach(allowedlength, allowedLengths,
     CPred<[{::llvm::cast<::mlir::asctile::TileType>($_self).getNumElements()
           == }] # allowedlength>)>,
]>;

class TileOf<list<Type> allowedTypes>
    : ShapedContainerType<allowedTypes, IsTileTypePred, "tile",
                          "::mlir::asctile::TileType">;

class TileOfRank<list<int> allowedRanks> : Type<
  IsTileOfRankPred<allowedRanks>,
  " of ranks " # !interleave(allowedRanks, "/"), "::mlir::asctile::TileType">;

class TileOfRankAndType<list<int> allowedRanks, list<Type> allowedTypes>
    : AllOfType<
  [TileOf<allowedTypes>, TileOfRank<allowedRanks>],
  TileOf<allowedTypes>.summary # TileOfRank<allowedRanks>.summary,
  "::mlir::asctile::TileType">;

class TileOfLength<list<int> allowedLengths>
    : Type<IsTileOfLengthPred<allowedLengths>,
           " of length " # !interleave(allowedLengths, "/"),
           "::mlir::asctile::TileType">;

class TileOfLengthAndType<list<int> allowedLengths, list<Type> allowedTypes>
    : AllOfType<
  [TileOf<allowedTypes>, TileOfLength<allowedLengths>],
  TileOf<allowedTypes>.summary # TileOfLength<allowedLengths>.summary,
  "::mlir::asctile::TileType">;

def AscTile_Float : AnyTypeOf<[F16, BF16, F32]>;
def AscTile_Int : AnyTypeOf<[I8, I16, I32]>;
def AscTile_Numeric : AnyTypeOf<[AscTile_Float, AscTile_Int]>;

def FloatTile : TileOf<[AscTile_Float]>;
def IntTile : TileOf<[AscTile_Int]>;
def NumericTile : TileOf<[AscTile_Numeric]>;

class ShapedMatchElemType<string shapedType, string type> : TypesMatchWith<
  "type of '" # type # "' matches element type of '" # shapedType # "'",
  shapedType, type, "::llvm::cast<ShapedType>($_self).getElementType()">;

class OptionalShapedMatchElemType<string shapedType, string type>
    : OptionalTypesMatchWith<
  "type of '" # type # "' matches element type of '" # shapedType # "'",
  shapedType, type, "::llvm::cast<ShapedType>($_self).getElementType()">;

#endif // ASCTILE_TYPES_TD
