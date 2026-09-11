/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "asctile/Dialect/AscTile/Utils/Utils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/OpDefinition.h"

namespace mlir {
namespace asctile {

TypedAttr getSplatAttr(arith::ConstantOp cstOp)
{
    if (!cstOp)
        return {};
    auto splat = dyn_cast<SplatElementsAttr>(cstOp.getValue());
    if (!splat)
        return {};
    return splat.getSplatValue<TypedAttr>();
}

OpFoldResult getSplatValue(Value cstTile)
{
    if (auto splat = getSplatAttr(cstTile.getDefiningOp<arith::ConstantOp>()))
        return splat;
    if (auto splat = cstTile.getDefiningOp<tensor::SplatOp>())
        return splat.getInput();
    return {};
}

Value materializeSplatValue(OpBuilder& builder, Value cstTile)
{
    auto splat = getSplatValue(cstTile);
    if (!splat)
        return {};
    if (auto value = splat.dyn_cast<Value>())
        return value;
    return builder.create<arith::ConstantOp>(cstTile.getLoc(), cast<TypedAttr>(cast<Attribute>(splat)));
}

} // namespace asctile
} // namespace mlir
