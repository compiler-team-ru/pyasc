/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ascir/Dialect/AscTile/IR/AscTile.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"

#define GET_OP_CLASSES
#include "ascir/Dialect/AscTile/IR/AscTileOps.cpp.inc"

using namespace mlir;
using namespace mlir::asctile;

//===----------------------------------------------------------------------===//
// TensorOp
//===----------------------------------------------------------------------===//

LogicalResult TensorOp::verify()
{
    if (getType().getNumDynamicDims() != getSizes().size())
        return emitOpError("must have value in 'sizes' for each dynamic dimension");
    return success();
}

//===----------------------------------------------------------------------===//
// DimOp
//===----------------------------------------------------------------------===//

OpFoldResult DimOp::fold([[maybe_unused]] FoldAdaptor adaptor)
{
    auto index = getIndex();
    auto type = getBase().getType();
    auto dim = type.getDimSize(index);
    if (!ShapedType::isDynamic(dim))
        return IntegerAttr::get(IntegerType::get(getContext(), 32), dim);
    auto tensorOp = getBase().getDefiningOp<TensorOp>();
    if (!tensorOp)
        return OpFoldResult {};
    auto dynamicIndex = type.getDynamicDimIndex(index);
    assert(dynamicIndex < tensorOp.getSizes().size() && "dim index must be less than number of dynamic sizes");
    return tensorOp.getSizes()[dynamicIndex];
}

LogicalResult DimOp::verify()
{
    if (getIndex() >= getBase().getType().getRank())
        return emitOpError("'index' must not exceed the tensor rank");
    return success();
}

//===----------------------------------------------------------------------===//
// LoadOp
//===----------------------------------------------------------------------===//

LogicalResult LoadOp::canonicalize(LoadOp op, PatternRewriter &rewriter)
{
    if (op->getUses().empty()) {
        rewriter.eraseOp(op);
        return success();
    }
    return failure();
}

//===----------------------------------------------------------------------===//
// CastOp
//===----------------------------------------------------------------------===//

bool CastOp::areCastCompatible(TypeRange inputs, TypeRange outputs)
{
    if (inputs.size() != 1 || outputs.size() != 1)
        return false;
    auto inType = dyn_cast_if_present<TileType>(inputs.front());
    auto outType = dyn_cast_if_present<TileType>(outputs.front());
    if (!inType || !outType || inType.getLoc() != outType.getLoc() || inType.getShape() != outType.getShape())
        return false;
    auto inElType = inType.getElementType();
    auto outElType = outType.getElementType();
    if (!inElType.isIntOrFloat() || !outElType.isIntOrFloat())
        return false;
    return true;
}

//===----------------------------------------------------------------------===//
// AscTileDialect
//===----------------------------------------------------------------------===//

void AscTileDialect::registerOps()
{
    addOperations<
#define GET_OP_LIST
#include "ascir/Dialect/AscTile/IR/AscTileOps.cpp.inc"
        >();
}
