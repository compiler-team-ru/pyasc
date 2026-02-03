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

#endif // ASCTILE_TYPES_TD
