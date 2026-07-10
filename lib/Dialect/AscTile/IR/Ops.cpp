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
#include "ascir/Dialect/AscTile/Utils/Attributes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "llvm/ADT/STLExtras.h"

#include <numeric>

using namespace mlir;
using namespace mlir::asctile;

namespace {

template <typename OpT>
OpFoldResult foldCastLike(OpT op, bool allowTransitCast = true)
{
    Value in = op.getIn();
    Type resultType = op.getResult().getType();
    if (in.getType() == resultType)
        return in;
    if (auto defOp = in.getDefiningOp<OpT>()) {
        Value defIn = defOp.getIn();
        if (resultType == defIn.getType())
            return defIn;
        if (!allowTransitCast)
            return {};
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
// BroadcastOp
//===----------------------------------------------------------------------===//

OpFoldResult BroadcastOp::fold([[maybe_unused]] FoldAdaptor adaptor)
{
    SplatElementsAttr attr;
    if (!matchPattern(getOperand(), m_Constant(&attr)))
        return {};
    return SplatElementsAttr::get(getType(), attr.getSplatValue<Attribute>());
}

//===----------------------------------------------------------------------===//
// ConcatOp
//===----------------------------------------------------------------------===//

OpFoldResult ConcatOp::fold([[maybe_unused]] FoldAdaptor adaptor)
{
    if (getNumOperands() == 1)
        return getOperand(0);
    return nullptr;
}

LogicalResult ConcatOp::verify()
{
    if (getNumOperands() < 1)
        return emitOpError("must have at least one operand");
    if (!llvm::all_of(getOperands(), [](Value opnd) {
            return cast<LocalTensorType>(opnd.getType()).getLoc() == TileLocation::UB;
        }))
        return emitOpError("tensor operands must have UB tile location");
    if (!llvm::all_equal(llvm::map_range(getOperands(), [](Value opnd) {
            return cast<LocalTensorType>(opnd.getType()).getShape().drop_front();
        })))
        return emitOpError("tensor operands must have the same shape except their first dimension");
    SmallVector<int64_t> firstDims(llvm::map_range(getOperands(), [](Value opnd) {
        return cast<LocalTensorType>(opnd.getType()).getShape().front();
    }));
    SmallVector<int64_t> resultShape(cast<LocalTensorType>(getOperand(0).getType()).getShape());
    resultShape.front() = std::accumulate(firstDims.begin(), firstDims.end(), 0, std::plus<int64_t>());
    if (resultShape != getType().getShape())
        return emitOpError() << "result tensor shape must be [" << resultShape << "]";
    return success();
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

LogicalResult LoadOp::verify()
{
    SmallVector<Value> realShape = getRealShape();
    if (realShape.empty())
        return success();
    auto tileShape = getType().getShape();
    if (tileShape.size() != realShape.size())
        return emitOpError() << "real_shape must have same size as tensor shape";

    if (auto attr = getOperation()->getAttrOfType<DenseI32ArrayAttr>(asctile::attr::transposeDims)) {
        ArrayRef<int32_t> transposeDims = attr;
        if (transposeDims.size() != tileShape.size())
            return emitOpError() << "transpose_dims must have same rank as tensor";
        SmallVector<Value> tmp;
        for (size_t i = 0; i < tileShape.size(); ++i) {
            auto dim = transposeDims[i];
            tmp.push_back(realShape[dim]);
        }
        tmp.swap(realShape);
    }
    for (auto [realDimValue, tileDim] : llvm::zip_equal(realShape, tileShape)) {
        APInt realDim;
        if (!matchPattern(realDimValue, m_ConstantInt(&realDim)))
            continue;
        if (realDim.getSExtValue() > tileDim)
            return emitOpError() << "real_shape exceeds tensor shape";
    }
    return success();
}

//===----------------------------------------------------------------------===//
// SplatOp
//===----------------------------------------------------------------------===//

OpFoldResult SplatOp::fold([[maybe_unused]] FoldAdaptor adaptor)
{
    Attribute attr;
    if (matchPattern(getOperand(), m_Constant(&attr)))
        return SplatElementsAttr::get(getType(), attr);
    return {};
}

//===----------------------------------------------------------------------===//
// StoreOp
//===----------------------------------------------------------------------===//

LogicalResult StoreOp::verify()
{
    auto realShape = getRealShape();
    if (realShape.empty())
        return success();
    auto tileShape = getValue().getType().getShape();
    if (tileShape.size() != realShape.size())
        return emitOpError() << "real_shape must have same size as tensor shape";

    for (auto [realDimValue, tileDim] : llvm::zip_equal(realShape, tileShape)) {
        APInt realDim;
        if (!matchPattern(realDimValue, m_ConstantInt(&realDim)))
            continue;
        if (realDim.getSExtValue() > tileDim)
            return emitOpError() << "real_shape exceeds tensor shape";
    }
    return success();
}

//===----------------------------------------------------------------------===//
// CastOp
//===----------------------------------------------------------------------===//

bool CastOp::areCastCompatible(TypeRange inputs, TypeRange outputs)
{
    if (inputs.size() != 1 || outputs.size() != 1)
        return false;
    auto inType = dyn_cast<LocalTensorType>(inputs.front());
    auto outType = dyn_cast<LocalTensorType>(outputs.front());
    return inType && outType && inType.getLoc() == outType.getLoc() && inType.getShape() == outType.getShape() &&
           inType.getElementType().isIntOrFloat() && outType.getElementType().isIntOrFloat();
}

OpFoldResult CastOp::fold(FoldAdaptor) { return foldCastLike(*this, false); }

//===----------------------------------------------------------------------===//
// ReshapeOp
//===----------------------------------------------------------------------===//

bool ReshapeOp::areCastCompatible(TypeRange inputs, TypeRange outputs)
{
    if (inputs.size() != 1 || outputs.size() != 1)
        return false;
    auto inType = dyn_cast<LocalTensorType>(inputs.front());
    auto outType = dyn_cast<LocalTensorType>(outputs.front());
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
    if (auto realShape = getRealShape(); !realShape.empty()) {
        auto tileShape = getValue().getType().getShape();
        if (tileShape.size() != realShape.size())
            return emitOpError() << "real_shape must have same size as tensor shape";
        for (auto [realDimValue, tileDim] : llvm::zip_equal(realShape, tileShape)) {
            APInt realDim;
            if (!matchPattern(realDimValue, m_ConstantInt(&realDim)))
                continue;
            if (realDim.getSExtValue() > tileDim)
                return emitOpError() << "real_shape exceeds tensor shape";
        }
    }
    return success();
}

//===----------------------------------------------------------------------===//
// CopyFixpipeOp
//===----------------------------------------------------------------------===//

LogicalResult CopyFixpipeOp::canonicalize(CopyFixpipeOp op, PatternRewriter& rewriter)
{
    if (op->getUses().empty()) {
        rewriter.eraseOp(op);
        return success();
    }
    return failure();
}

LogicalResult CopyFixpipeOp::verify()
{
    if (!getQuantize() && getElementTypeOrSelf(getBase()) != getElementTypeOrSelf(getResult())) {
        return emitOpError("failed to verify that all of {base, result} have same element type");
    }
    return success();
}

//===----------------------------------------------------------------------===//
// AccumulatorOp
//===----------------------------------------------------------------------===//

LogicalResult AccumulatorOp::verify()
{
    auto bias = getBias();
    if (!bias)
        return success();
    auto biasType = bias.getType();
    if (biasType.getLoc() != TileLocation::BT)
        return emitOpError("bias must have BT tile location");
    auto result = getResult();
    auto resultShape = result.getType().getShape();
    auto biasShape = biasType.getShape();
    if (resultShape.size() != 2)
        return emitOpError("result must be a 2D tensor");
    if (biasShape[0] != resultShape[1])
        return emitOpError("bias shape must match result's second dimension");
    return success();
}

//===----------------------------------------------------------------------===//
// MatmulOp
//===----------------------------------------------------------------------===//

LogicalResult MatmulOp::verify()
{
    if (getMatrixA().getType().getLoc() != TileLocation::L0A) {
        return emitOpError("matrixA must have L0A tile location");
    }
    if (getMatrixB().getType().getLoc() != TileLocation::L0B) {
        return emitOpError("matrixB must have L0B tile location");
    }
    if (getResult().getType().getLoc() != TileLocation::L0C) {
        return emitOpError("result must have L0C tile location");
    }
    if (getBias()) {
        if (getBias().getType().getLoc() != TileLocation::BT) {
            return emitOpError("bias must have BT tile location");
        }
    }
    return success();
}

//===----------------------------------------------------------------------===//
// MatmulAccOp
//===----------------------------------------------------------------------===//

LogicalResult MatmulAccOp::verify()
{
    if (getMatrixA().getType().getLoc() != TileLocation::L0A) {
        return emitOpError("matrixA must have L0A tile location");
    }
    if (getMatrixB().getType().getLoc() != TileLocation::L0B) {
        return emitOpError("matrixB must have L0B tile location");
    }
    if (getAcc().getType().getLoc() != TileLocation::L0C) {
        return emitOpError("acc must have L0C tile location");
    }
    return success();
}

//===----------------------------------------------------------------------===//
// AscTileDialect
//===----------------------------------------------------------------------===//

Operation* AscTileDialect::materializeConstant(OpBuilder& builder, Attribute value, Type type, Location loc)
{
    return arith::ConstantOp::materialize(builder, value, type, loc);
}

void AscTileDialect::registerOps()
{
    addOperations<
#define GET_OP_LIST
#include "ascir/Dialect/AscTile/IR/AscTileOps.cpp.inc"
        >();
}

#define GET_OP_CLASSES
#include "ascir/Dialect/AscTile/IR/AscTileOps.cpp.inc"
