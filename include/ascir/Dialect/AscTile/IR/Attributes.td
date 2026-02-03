/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASCTILE_ATTRIBUTES_TD
#define ASCTILE_ATTRIBUTES_TD

include "Dialect.td"

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"

class AscTile_Attr<string name, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<AscTile_Dialect, name, traits> {
  let cppNamespace = "::mlir::asctile";
  let mnemonic = attrMnemonic;
}

def AscTile_TileLocationAttr : I32EnumAttr<"TileLocation", "tile location", [
  I32EnumAttrCase<"L0A", 23>,
  I32EnumAttrCase<"L0B", 24>,
  I32EnumAttrCase<"L0C", 25>,
  I32EnumAttrCase<"UB", 26>,
  I32EnumAttrCase<"L1", 27>,
  I32EnumAttrCase<"FIX", 28>,
]> {
  let cppNamespace = "::mlir::asctile";
  let description = "Tile memory location";
}

#endif // ASCTILE_ATTRIBUTES_TD
