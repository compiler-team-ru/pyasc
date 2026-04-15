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

using namespace mlir;
using namespace mlir::asctile;

namespace {

template <typename OpT>
OpFoldResult foldCastLike(OpT op)
{
    Value in = op.getIn();
    Type resultType = op.getResult().getType();
    if (in.getType() == resultType)
        return in;
    if (auto defOp = in.getDefiningOp<OpT>()) {
        Value defIn = defOp.getIn();
        if (resultType == defIn.getType())
            return defIn;
        op.setOperand(defIn);
        return op.getResult();
    }
    return {};
}

Type getI1SameShape(Type type)
{
    auto i1Type = IntegerType::get(type.getContext(), 1);
    if (auto shapedType = llvm::dyn_cast<ShapedType>(type))
        return shapedType.cloneWith(std::nullopt, i1Type);
    return i1Type;
}

} // namespace

//===----------------------------------------------------------------------===//
// AccumulatorOp
//===----------------------------------------------------------------------===//

LogicalResult AccumulatorOp::canonicalize(AccumulatorOp op, PatternRewriter& rewriter)
{
    if (op->getUses().empty()) {
        rewriter.eraseOp(op);
        return success();
    }
    return failure();
}

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
        return OpFoldResult{};
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
// CopyOp
//===----------------------------------------------------------------------===//

LogicalResult CopyOp::canonicalize(CopyOp op, PatternRewriter& rewriter)
{
    if (op->getUses().empty()) {
        rewriter.eraseOp(op);
        return success();
    }
    return failure();
}

//===----------------------------------------------------------------------===//
// LoadOp
//===----------------------------------------------------------------------===//

LogicalResult LoadOp::canonicalize(LoadOp op, PatternRewriter& rewriter)
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
    auto inType = dyn_cast<TileType>(inputs.front());
    auto outType = dyn_cast<TileType>(outputs.front());
    return inType && outType && inType.getLoc() == outType.getLoc() && inType.getShape() == outType.getShape() &&
           inType.getElementType().isIntOrFloat() && outType.getElementType().isIntOrFloat();
}

OpFoldResult CastOp::fold(FoldAdaptor) { return foldCastLike(*this); }

//===----------------------------------------------------------------------===//
// ReshapeOp
//===----------------------------------------------------------------------===//

bool ReshapeOp::areCastCompatible(TypeRange inputs, TypeRange outputs)
{
    if (inputs.size() != 1 || outputs.size() != 1)
        return false;
    auto inType = dyn_cast<TileType>(inputs.front());
    auto outType = dyn_cast<TileType>(outputs.front());
    return inType && outType && inType.getLoc() == outType.getLoc() &&
           inType.getElementType() == outType.getElementType() && inType.getNumElements() == outType.getNumElements();
}

OpFoldResult ReshapeOp::fold(FoldAdaptor) { return foldCastLike(*this); }

//===----------------------------------------------------------------------===//
// StoreFixpipeOp
//===----------------------------------------------------------------------===//

LogicalResult StoreFixpipeOp::verify()
{
    if (!getQuantize() && getElementTypeOrSelf(getBase()) != getElementTypeOrSelf(getValue())) {
        return emitOpError("failed to verify that all of {base, value} have same element type");
    }
    return success();
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

#define GET_OP_CLASSES
#include "ascir/Dialect/AscTile/IR/AscTileOps.cpp.inc"
