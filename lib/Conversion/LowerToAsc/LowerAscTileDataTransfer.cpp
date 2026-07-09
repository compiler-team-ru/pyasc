/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ascir/Conversion/LowerToAsc/Passes.h"
#include "ascir/Dialect/Asc/IR/Asc.h"
#include "ascir/Dialect/Asc/Utils/Attributes.h"
#include "ascir/Dialect/Asc/Utils/Utils.h"
#include "ascir/Dialect/AscTile/IR/AscTile.h"
#include "ascir/Dialect/AscTile/Utils/Attributes.h"
#include "ascir/Dialect/EmitAsc/IR/EmitAsc.h"
#include "ascir/Dialect/EmitAsc/Utils/InitStructBuilder.h"
#include "ascir/Dialect/Utils/ConstantOpBuilder.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"

#include "Common.h"

namespace mlir {
namespace asclower {
#define GEN_PASS_DEF_LOWERASCTILEDATATRANSFER
#include "ascir/Conversion/LowerToAsc/Passes.h.inc"
} // namespace asclower
} // namespace mlir

using namespace mlir;
using namespace mlir::asclower;

namespace {

constexpr int64_t cubeKBlockBytes = ascendc::ubBlockSize;
constexpr int64_t fractalNum = 2;

SmallVector<Value> getTensorShape(OpBuilder& builder, asctile::TensorOp tensorOp)
{
    ascir::ConstantOpBuilder consts(builder);
    auto type = tensorOp.getType();
    auto dynamicSizes = tensorOp.getSizes();
    size_t dynamicSizeIndex = 0;
    SmallVector<Value> tensorShape;
    for (auto dim : type.getShape()) {
        if (ShapedType::isDynamic(dim))
            tensorShape.push_back(dynamicSizes[dynamicSizeIndex++]);
        else
            tensorShape.push_back(consts.i32(dim));
    }
    return tensorShape;
}

Value linearizeOffset(OpBuilder& builder, Location loc, ArrayRef<Value> tensorShape, ValueRange offsets)
{
    ascir::ConstantOpBuilder consts(builder);
    assert(offsets.size() == tensorShape.size() && "must be one offset for each dimension");
    Value linearOffset = consts.i32(0);
    Value stride = consts.i32(1);
    for (size_t i = tensorShape.size(); i-- > 0;) {
        Value next = builder.create<arith::MulIOp>(loc, offsets[i], stride);
        linearOffset = builder.create<arith::AddIOp>(loc, linearOffset, next);
        stride = builder.create<arith::MulIOp>(loc, tensorShape[i], stride);
    }
    return linearOffset;
}

SmallVector<Value> getStaticShape(OpBuilder& builder, ShapedType type)
{
    ascir::ConstantOpBuilder consts(builder);
    SmallVector<Value> shape;
    for (auto dim : type.getShape())
        shape.push_back(consts.i32(dim));
    return shape;
}

struct TensorInfo {
    SmallVector<Value> shape;
    Value tensor;
    ascendc::BaseTensorType type;
};

TensorInfo prepareTensorInfo(ConvertRewriter& rewriter, Location loc, Value base, ValueRange offsets = {})
{
    auto tensorOp = base.getDefiningOp<asctile::TensorOp>();
    assert(tensorOp && "tensor must be created by asctile.tensor op");
    SmallVector<Value> shape = getTensorShape(rewriter, tensorOp);
    Value tensor = rewriter.getRemappedValue(base);
    auto type = cast<ascendc::BaseTensorType>(tensor.getType());
    if (!offsets.empty()) {
        Value linearOffset = linearizeOffset(rewriter, loc, shape, offsets);
        tensor = rewriter.create<ascendc::GlobalTensorSubIndexOp>(loc, type, tensor, linearOffset);
    }
    return {shape, tensor, type};
}

std::optional<ascendc::QuantMode> getQuantizeMode(
    ascendc::BaseTensorType srcType, ascendc::BaseTensorType dstType, ConvertRewriter& rewriter)
{
    auto srcElType = srcType.getElementType();
    auto dstElType = dstType.getElementType();
    auto floatType = rewriter.getF32Type();
    auto halfType = rewriter.getF16Type();
    auto int32Type = rewriter.getIntegerType(32);
    auto int8Type = rewriter.getIntegerType(8);
    auto uint8Type = rewriter.getIntegerType(8, false);
    if (srcElType == floatType && dstElType == floatType) {
        return ascendc::QuantMode::NoQuant;
    }
    if (srcElType == floatType && dstElType == halfType) {
        return ascendc::QuantMode::F322F16;
    }
    if (srcElType == floatType && dstElType == rewriter.getBF16Type()) {
        return ascendc::QuantMode::F322BF16;
    }
    if (srcElType == int32Type && dstElType == halfType) {
        return ascendc::QuantMode::DEQF16;
    }
    if (srcElType == floatType && (dstElType == int8Type || dstElType == uint8Type)) {
        return ascendc::QuantMode::QF322B8_PRE;
    }
    if (srcElType == int32Type && (dstElType == int8Type || dstElType == uint8Type)) {
        return ascendc::QuantMode::REQ8;
    }
    // TODO: Add support for VDEQF16, VQF322B8_PRE, VREQ8
    return std::nullopt;
}

Value calculateCopyCount(
    ConvertRewriter& rewriter, Location& loc, const SmallVector<int64_t>& ubShape, const SmallVector<Value>& gmShape,
    const OperandRange& offsets, const OperandRange& realShape, size_t dim)
{
    ascir::ConstantOpBuilder consts(rewriter);
    Value tailElementsCount = rewriter.create<arith::SubIOp>(loc, gmShape[dim], offsets[dim]);
    tailElementsCount = rewriter.create<arith::MaxSIOp>(loc, tailElementsCount, consts.i32(0));
    return rewriter.create<arith::MinSIOp>(
        loc, realShape.empty() ? consts.i32(ubShape[dim]) : realShape[dim], tailElementsCount);
}

struct ConvertLoadToUB : ConvertOp<asctile::LoadOp> {
    using ConvertOp::ConvertOp;
    using ConvertOp::createTensorOp;

    LogicalResult matchAndRewrite(asctile::LoadOp op, ConvertRewriter& rewriter) const override
    {
        auto opType = op.getType();
        auto dstLoc = opType.getLoc();
        if (dstLoc != asctile::TileLocation::UB || op->hasAttr(asctile::attr::transposeDims))
            return failure();
        auto loc = op.getLoc();
        auto offsets = op.getOffsets();
        TensorInfo srcInfo = prepareTensorInfo(rewriter, loc, op.getBase(), offsets);
        auto dst = createTensorOp(rewriter, loc, opType, locationToPosition(dstLoc)).getResult();
        auto dstType = dst.getType();
        auto dstShape = dstType.getShape();
        ascir::ConstantOpBuilder consts(rewriter);
        auto const0 = consts.i32(0);
        auto const1 = consts.i32(1);
        auto padValue = rewriter.getRemappedValue(op.getPadValue());
        auto typeSizeValue = consts.i32(ascendc::getElementTypeSize(dstType));
        Value dstLastDim = consts.i32(dstShape.back());
        Value srcLastDim = srcInfo.shape.back();
        Value tailElementsLastDim = rewriter.create<arith::SubIOp>(loc, srcLastDim, offsets.back());
        auto tailNegCond = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, tailElementsLastDim, const0);
        Value tailElements = rewriter.create<arith::SelectOp>(loc, tailNegCond, const0, tailElementsLastDim);
        Value minTailElements = rewriter.create<arith::MinSIOp>(loc, dstLastDim, tailElements);
        Value blockLen, srcStrideElements;
        auto realShape = op.getRealShape();
        auto hasRealShape = !realShape.empty();
        if (hasRealShape) {
            Value realLastDim = rewriter.getRemappedValue(realShape.back());
            Value realTailElements = rewriter.create<arith::MinSIOp>(loc, realLastDim, tailElements);
            blockLen = rewriter.create<arith::MulIOp>(loc, realTailElements, typeSizeValue);
            srcStrideElements = rewriter.create<arith::SubIOp>(loc, srcLastDim, realTailElements);
        } else {
            blockLen = rewriter.create<arith::MulIOp>(loc, minTailElements, typeSizeValue);
            srcStrideElements = rewriter.create<arith::SubIOp>(loc, srcLastDim, minTailElements);
        }
        auto ubBlockSizeValue = consts.i32(ascendc::ubBlockSize);
        auto blockLenRemainder = rewriter.create<arith::RemSIOp>(loc, blockLen, ubBlockSizeValue);
        auto remainderIsZero = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, blockLenRemainder, const0);
        auto rightPadBytes = rewriter.create<arith::SelectOp>(
            loc, remainderIsZero, const0, rewriter.create<arith::SubIOp>(loc, ubBlockSizeValue, blockLenRemainder));
        auto alignedBlockSize = rewriter.create<arith::AddIOp>(loc, blockLen, rightPadBytes);
        auto rowSizeBytes = rewriter.create<arith::MulIOp>(loc, dstLastDim, typeSizeValue);
        if (hasRealShape) {
            auto hasGap =
                rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sgt, rowSizeBytes, alignedBlockSize);
            auto ifHasGap = rewriter.create<scf::IfOp>(loc, hasGap, false);
            ConvertRewriter::InsertionGuard guard(rewriter);
            rewriter.setInsertionPointToStart(ifHasGap.thenBlock());
            rewriter.create<ascendc::DuplicateL2Op>(loc, dst, padValue, const0);
        }
        Value blockCount = const1;
        if (dstShape.size() > 1) {
            // innerRows = max(0, min(srcRows, dstRows + offsetRows) - offsetRows)
            // If realShape is provided, further limit innerRows to realShape[0]
            auto srcRows = srcInfo.shape[0];
            auto dstRows = consts.i32(dstShape[0]);
            auto offsetRows = offsets[0];
            auto endPos = rewriter.create<arith::AddIOp>(loc, dstRows, offsetRows);
            auto clampedEnd = rewriter.create<arith::MinSIOp>(loc, srcRows, endPos);
            auto rowCount = rewriter.create<arith::SubIOp>(loc, clampedEnd, offsetRows);
            Value innerRows = rewriter.create<arith::MaxSIOp>(loc, const0, rowCount);
            if (!realShape.empty()) {
                Value realRows = rewriter.getRemappedValue(realShape[0]);
                innerRows = rewriter.create<arith::MinSIOp>(loc, innerRows, realRows);
            }
            blockCount = innerRows;
            if (hasRealShape) {
                auto padRows = rewriter.create<arith::SubIOp>(loc, dstRows, innerRows);
                auto padRowsIsPositive =
                    rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sgt, padRows, const0);
                auto ifPadRows = rewriter.create<scf::IfOp>(loc, padRowsIsPositive, false);
                ConvertRewriter::InsertionGuard guard(rewriter);
                rewriter.setInsertionPointToStart(ifPadRows.thenBlock());
                auto dstCols = consts.i32(dstShape.back());
                auto padOffset = rewriter.create<arith::MulIOp>(loc, innerRows, dstCols);
                auto padTensor = rewriter.create<ascendc::LocalTensorSubIndexOp>(loc, dstType, dst, padOffset);
                auto padCalCount = rewriter.create<arith::MulIOp>(loc, padRows, dstCols);
                auto dupOp = rewriter.create<ascendc::DuplicateL2Op>(loc, padTensor, padValue, padCalCount);
                dupOp->setAttr(ascendc::attr::calCountSet, rewriter.getUnitAttr());
            }
        }
        auto srcStride = rewriter.create<arith::MulIOp>(loc, srcStrideElements, typeSizeValue);
        auto rightPadElements = rewriter.create<arith::DivSIOp>(loc, rightPadBytes, typeSizeValue);
        auto dstStrideBytes = rewriter.create<arith::SubIOp>(loc, rowSizeBytes, alignedBlockSize);
        auto dstStrideDataBlocks = rewriter.create<arith::DivSIOp>(loc, dstStrideBytes, ubBlockSizeValue);
        auto ui32Type = rewriter.getIntegerType(32, false);
        auto dataCopyExtParams = rewriter.create<ascendc::ConstructOp>(
            loc, rewriter.getType<ascendc::DataCopyExtParamsType>(),
            ValueRange{blockCount, blockLen, srcStride, dstStrideDataBlocks, const0},
            rewriter.getTypeArrayAttr({rewriter.getIntegerType(16, false), ui32Type, ui32Type, ui32Type, ui32Type}));
        auto dataCopyPadExtParams = rewriter.create<ascendc::ConstructOp>(
            loc, rewriter.getType<ascendc::DataCopyPadExtParamsType>(getElementTypeOrSelf(dstType)),
            ValueRange{const1, const0, rightPadElements, padValue},
            rewriter.getTypeArrayAttr(
                {rewriter.getI32Type(), rewriter.getI32Type(), rewriter.getIntegerType(8, false), padValue.getType()}));
        rewriter.create<ascendc::DataCopyPadExtL0Op>(loc, dst, srcInfo.tensor, dataCopyExtParams, dataCopyPadExtParams);
        rewriter.replaceOp(op, dst);
        return success();
    }
};

struct ConvertLoadToUBWithTranspose : ConvertOp<asctile::LoadOp> {
    using ConvertOp::ConvertOp;
    using ConvertOp::createTensorOp;

    static void createTilePadding(
        ConvertRewriter& rewriter, Location loc, Value value, ValueRange copyCount, ArrayRef<int64_t> tensorShape,
        Value padValue)
    {
        ascir::ConstantOpBuilder consts(rewriter);
        auto const0 = consts.i32(0);
        Value needDuplicateFull = consts.i1(false);
        for (size_t i = copyCount.size() - 1; i + 1 > 1; --i) {
            Value padCount = rewriter.create<arith::SubIOp>(loc, consts.i32(tensorShape[i]), copyCount[i]);
            Value needPadding = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sgt, padCount, const0);
            needDuplicateFull = rewriter.create<arith::OrIOp>(loc, needDuplicateFull, needPadding);
        }
        Value firstDimPadding = rewriter.create<arith::SubIOp>(loc, consts.i32(tensorShape.front()), copyCount.front());
        Value needDuplicateLast =
            rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sgt, firstDimPadding, const0);
        auto ifDuplicateFullOp = rewriter.create<scf::IfOp>(loc, needDuplicateFull, true);
        {
            ConvertRewriter::InsertionGuard guard(rewriter);
            rewriter.setInsertionPointToStart(ifDuplicateFullOp.thenBlock());
            rewriter.create<ascendc::DuplicateL2Op>(loc, value, padValue, const0);
        }
        {
            ConvertRewriter::InsertionGuard guard(rewriter);
            rewriter.setInsertionPointToStart(ifDuplicateFullOp.elseBlock());
            auto ifDuplicateLast = rewriter.create<scf::IfOp>(loc, needDuplicateLast, false);
            {
                ConvertRewriter::InsertionGuard guard(rewriter);
                rewriter.setInsertionPointToStart(ifDuplicateLast.thenBlock());
                auto strides = getStrides(tensorShape);
                Value firstDimStride = consts.i32(strides.front());
                Value padOffset = rewriter.create<arith::MulIOp>(loc, firstDimStride, copyCount.front());
                auto padTensor =
                    rewriter.create<ascendc::LocalTensorSubIndexOp>(loc, value.getType(), value, padOffset);
                auto padCalCount = rewriter.create<arith::MulIOp>(loc, firstDimStride, firstDimPadding);
                auto dupOp = rewriter.create<ascendc::DuplicateL2Op>(loc, padTensor, padValue, padCalCount);
                dupOp->setAttr(ascendc::attr::calCountSet, rewriter.getUnitAttr());
            }
        }
    }

    template <typename T>
    static SmallVector<T> applyPermute(const SmallVector<T>& input, const SmallVector<int64_t>& permute)
    {
        SmallVector<T> result;
        for (auto i : permute) {
            result.push_back(input[i]);
        }
        return result;
    }
    template <typename T, typename A>
    static SmallVector<T> reversePermute(const SmallVector<A>& input, const SmallVector<int64_t>& permute)
    {
        SmallVector<T> result;
        result.assign(permute.size(), T{});
        for (size_t i = 0; i < permute.size(); ++i) {
            result[permute[i]] = static_cast<A>(input[i]);
        }
        return result;
    }

    static SmallVector<Value> getStrides(
        ConvertRewriter& rewriter, Location loc, ascir::ConstantOpBuilder& consts, ValueRange shape)
    {
        SmallVector<Value> result;
        result.assign(shape.size(), Value{});
        Value stride = consts.i32(1);
        for (size_t i = shape.size() - 1; i + 1 >= 1; --i) {
            result[i] = stride;
            if (i > 0) {
                stride = rewriter.create<arith::MulIOp>(loc, stride, shape[i]);
            }
        }
        return result;
    }
    static SmallVector<int64_t> getStrides(const ArrayRef<int64_t>& shape)
    {
        SmallVector<int64_t> result;
        result.assign(shape.size(), 0);
        int64_t stride = 1;
        for (size_t i = shape.size() - 1; i + 1 >= 1; --i) {
            result[i] = stride;
            stride *= shape[i];
        }
        return result;
    }

    LogicalResult matchAndRewrite(asctile::LoadOp op, ConvertRewriter& rewriter) const override
    {
        auto opType = op.getType();
        auto dstLoc = opType.getLoc();
        if (dstLoc != asctile::TileLocation::UB)
            return failure();
        auto transposeAttrs = op->getAttrOfType<DenseI32ArrayAttr>(asctile::attr::transposeDims);
        if (!transposeAttrs)
            return failure();
        auto loc = op.getLoc();
        TensorInfo srcInfo = prepareTensorInfo(rewriter, loc, op.getBase(), op.getOffsets());
        auto dst = createTensorOp(rewriter, loc, opType, locationToPosition(dstLoc)).getResult();
        auto dstShape = dst.getType().getShape();
        ascir::ConstantOpBuilder consts(rewriter);
        auto const1 = consts.i32(1);
        auto padValue = rewriter.getRemappedValue(op.getPadValue());
        SmallVector<int64_t> dimsOrder{static_cast<ArrayRef<int32_t>>(transposeAttrs)};
        auto dimCount = dimsOrder.size();

        SmallVector<int64_t> readShape = reversePermute<int64_t>(SmallVector<int64_t>{dstShape}, dimsOrder);
        SmallVector<Value> copyCount;
        for (size_t i = 0; i < dimCount; ++i) {
            copyCount.push_back(
                calculateCopyCount(rewriter, loc, readShape, srcInfo.shape, op.getOffsets(), op.getRealShape(), i));
        }
        SmallVector<Value> srcStrides = getStrides(rewriter, loc, consts, srcInfo.shape);
        SmallVector<int32_t> dstStrides = reversePermute<int32_t>(getStrides(dstShape), dimsOrder);
        SmallVector<int32_t> padLeft(dimCount, 0);
        SmallVector<int32_t> padRight(dimCount, 0);

        if (!op.getRealShape().empty()) {
            createTilePadding(rewriter, loc, dst, applyPermute(copyCount, dimsOrder), dstShape, padValue);
        }
        auto paramsType = rewriter.getType<ascendc::NdDmaParamsType>(srcInfo.type.getElementType(), dimCount);
        auto params = rewriter.create<ascendc::NdDmaParamsOp>(
            loc, paramsType, dimCount, padValue, copyCount, srcStrides, rewriter.getI32ArrayAttr(dstStrides),
            rewriter.getI32ArrayAttr(padLeft), rewriter.getI32ArrayAttr(padRight));
        rewriter.create<ascendc::DataCopyNdDmaOp>(loc, dst, srcInfo.tensor, params.getResult(), dimCount);
        rewriter.replaceOp(op, dst);
        return success();
    }
};

std::pair<Value, Value> alignToL1Block(ConvertRewriter& rewriter, Location loc, Value value, Value elementSize)
{
    ascir::ConstantOpBuilder consts(rewriter);
    auto blockSizeVal = consts.i32(cubeKBlockBytes);
    auto valueBytes = rewriter.create<arith::MulIOp>(loc, value, elementSize);
    auto ceilDiv = rewriter.create<arith::CeilDivSIOp>(loc, valueBytes, blockSizeVal);
    auto alignedBytes = rewriter.create<arith::MulIOp>(loc, ceilDiv, blockSizeVal);
    auto alignedValue = rewriter.create<arith::DivSIOp>(loc, alignedBytes, elementSize);
    return {alignedValue, alignedBytes};
}

struct ConvertLoadToL1 : ConvertOp<asctile::LoadOp> {
    using ConvertOp::ConvertOp;
    using ConvertOp::createTensorOp;

    LogicalResult matchAndRewrite(asctile::LoadOp op, ConvertRewriter& rewriter) const override
    {
        auto opType = op.getType();
        auto dstLoc = opType.getLoc();
        if (dstLoc != asctile::TileLocation::L1)
            return failure();
        auto loc = op.getLoc();
        TensorInfo srcInfo = prepareTensorInfo(rewriter, loc, op.getBase(), op.getOffsets());
        auto dst = createTensorOp(rewriter, loc, opType, locationToPosition(dstLoc)).getResult();
        auto dstShape = dst.getType().getShape();
        ascir::ConstantOpBuilder consts(rewriter);
        auto const0 = consts.i32(0);
        auto const1 = consts.i32(1);
        if (op->hasAttrOfType<UnitAttr>(asctile::attr::isBias)) {
            if (dstShape.size() != 1)
                return op.emitError() << "L1 load must be 1D for bias";
            auto calCount = consts.i32(dstShape[0]);
            rewriter.create<ascendc::DataCopyL2Op>(loc, dst, srcInfo.tensor, calCount);
            rewriter.replaceOp(op, dst);
            return success();
        }
        auto ui16Type = rewriter.getIntegerType(16, false);
        auto ui32Type = rewriter.getIntegerType(32, false);
        auto ui64Type = rewriter.getIntegerType(64, false);
        auto argTypes = rewriter.getTypeArrayAttr(
            TypeRange{ui16Type, ui16Type, ui32Type, ui64Type, ui32Type, ui16Type, ui16Type, ui64Type});
        bool isMatrixA = op->hasAttr(asctile::attr::isMatrixA);
        bool isTransposeA = op->hasAttrOfType<UnitAttr>(asctile::attr::transposeA);
        bool isTransposeB = op->hasAttrOfType<UnitAttr>(asctile::attr::transposeB);
        auto dstShapeCols = rewriter.create<arith::MinSIOp>(loc, srcInfo.shape[1], consts.i32(dstShape[1]));
        auto dstShapeRows = rewriter.create<arith::MinSIOp>(loc, srcInfo.shape[0], consts.i32(dstShape[0]));
        auto offsets = op.getOffsets();
        assert(offsets.size() == 2);
        Value availableRows = rewriter.create<arith::MaxSIOp>(
            loc, const0, rewriter.create<arith::SubIOp>(loc, srcInfo.shape[0], offsets[0]));
        Value availableCols = rewriter.create<arith::MaxSIOp>(
            loc, const0, rewriter.create<arith::SubIOp>(loc, srcInfo.shape[1], offsets[1]));
        Value nValue = rewriter.create<arith::MinSIOp>(loc, dstShapeRows, availableRows);
        Value dValue = rewriter.create<arith::MinSIOp>(loc, dstShapeCols, availableCols);
        constexpr int64_t maxSrcDValue = 65535;
        auto dstRowStride = consts.i32(dstShape[0]);
        auto needsLoop =
            rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sgt, srcInfo.shape[1], consts.i32(maxSrcDValue));
        auto ifOp = rewriter.create<scf::IfOp>(loc, needsLoop, /*withElseRegion=*/true);
        {
            ConvertRewriter::InsertionGuard guard(rewriter);
            rewriter.setInsertionPointToStart(ifOp.thenBlock());
            auto dstType = dst.getType();
            auto srcRows = srcInfo.shape[0];
            auto offsetRows = op.getOffsets().empty() ? const0 : op.getOffsets()[0];
            auto availableRows = rewriter.create<arith::SubIOp>(loc, srcRows, offsetRows);
            auto availableRowsPos = rewriter.create<arith::MaxSIOp>(loc, const0, availableRows);
            auto actualNValue = rewriter.create<arith::MinSIOp>(loc, nValue, availableRowsPos);
            auto forOp = rewriter.create<scf::ForOp>(loc, const0, actualNValue, const1);
            rewriter.setInsertionPointToStart(forOp.getBody());
            auto rowIdx = forOp.getInductionVar();
            auto srcRowOffset = rewriter.create<arith::MulIOp>(loc, rowIdx, srcInfo.shape[1]);
            auto srcTensorWithOffset =
                rewriter.create<ascendc::GlobalTensorSubIndexOp>(loc, srcInfo.type, srcInfo.tensor, srcRowOffset);
            auto dstRowOffset = rewriter.create<arith::MulIOp>(loc, rowIdx, consts.i32(ascendc::cubeBlockSize));
            auto dstTensorWithOffset = rewriter.create<ascendc::LocalTensorSubIndexOp>(loc, dstType, dst, dstRowOffset);
            auto nd2NzParams = rewriter.create<ascendc::ConstructOp>(
                loc, rewriter.getType<ascendc::Nd2NzParamsType>(),
                ValueRange{const1, const1, dValue, const0, dValue, dstRowStride, const1, const0}, argTypes);
            rewriter.create<ascendc::DataCopyL2Op>(loc, dstTensorWithOffset, srcTensorWithOffset, nd2NzParams);
            rewriter.setInsertionPointAfter(forOp);
            forOp->setAttr(asctile::attr::parallel, rewriter.getUnitAttr());
        }
        {
            ConvertRewriter::InsertionGuard guard(rewriter);
            rewriter.setInsertionPointToStart(ifOp.elseBlock());
            auto dstNzC0Stride =
                dstShape[0] == 1 && isMatrixA ?
                    dstRowStride :
                    consts.i32(static_cast<int64_t>(llvm::alignTo(dstShape[0], ascendc::cubeBlockSize)));
            auto nd2NzParams = rewriter.create<ascendc::ConstructOp>(
                loc, rewriter.getType<ascendc::Nd2NzParamsType>(),
                ValueRange{const1, nValue, dValue, const0, srcInfo.shape[1], dstNzC0Stride, const1, const0}, argTypes);
            rewriter.create<ascendc::DataCopyL2Op>(loc, dst, srcInfo.tensor, nd2NzParams);
        }
        auto elemType = dst.getType().getElementType();
        auto constArgTypes = rewriter.getTypeArrayAttr(TypeRange{ui16Type, ui16Type, ui16Type, elemType});
        auto elementSize = ascendc::getElementTypeSize(opType);
        int64_t cubeKBlockSize = cubeKBlockBytes / elementSize;
        auto c0Size = consts.i32(cubeKBlockSize);
        auto elemSizeVal = consts.i32(elementSize);
        auto blockSizeVal = consts.i32(cubeKBlockBytes);
        if ((isMatrixA && isTransposeA) || (!isMatrixA && !isTransposeB)) {
            auto [alignedNValue, alignedNValueBytes] = alignToL1Block(rewriter, loc, nValue, elemSizeVal);
            auto totalBytes = consts.i32(dstShape[0] * dstShape[1] * elementSize);
            auto validBytes = rewriter.create<arith::MulIOp>(loc, alignedNValue, consts.i32(dstShape[1] * elementSize));
            auto padBytes = rewriter.create<arith::SubIOp>(loc, totalBytes, validBytes);
            auto padBlocks = rewriter.create<arith::DivSIOp>(loc, padBytes, blockSizeVal);
            auto needsRowPad = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sgt, padBlocks, const0);
            auto rowPadIf = rewriter.create<scf::IfOp>(loc, needsRowPad);
            {
                ConvertRewriter::InsertionGuard guard(rewriter);
                rewriter.setInsertionPointToStart(rowPadIf.thenBlock());
                auto colBlocks = consts.i32(dstShape[1] / cubeKBlockSize);
                auto rowOffset = rewriter.create<arith::MulIOp>(loc, alignedNValue, c0Size);
                auto blockNum = rewriter.create<arith::SubIOp>(loc, consts.i32(dstShape[0]), alignedNValue);
                auto padTensor = rewriter.create<ascendc::LocalTensorSubIndexOp>(loc, dst.getType(), dst, rowOffset);
                auto params = rewriter.create<ascendc::ConstructOp>(
                    loc, rewriter.getType<ascendc::InitConstValueParamsType>(),
                    ValueRange{colBlocks, blockNum, alignedNValue, const0}, constArgTypes);
                rewriter.create<ascendc::FillOp>(loc, padTensor, params.getResult());
            }
        } else {
            auto [alignedDValue, alignedDValueBytes] = alignToL1Block(rewriter, loc, dValue, elemSizeVal);
            auto totalBytes = consts.i32(dstShape[1] * elementSize);
            auto padBytes = rewriter.create<arith::SubIOp>(loc, totalBytes, alignedDValueBytes);
            auto padBlocks = rewriter.create<arith::DivSIOp>(loc, padBytes, blockSizeVal);
            auto needsColPad = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sgt, padBlocks, const0);
            auto colPadIf = rewriter.create<scf::IfOp>(loc, needsColPad);
            {
                ConvertRewriter::InsertionGuard guard(rewriter);
                rewriter.setInsertionPointToStart(colPadIf.thenBlock());
                auto colOffset = rewriter.create<arith::MulIOp>(loc, alignedDValue, consts.i32(dstShape[0]));
                auto blockNum = rewriter.create<arith::MulIOp>(loc, padBlocks, consts.i32(dstShape[0]));
                auto padTensor = rewriter.create<ascendc::LocalTensorSubIndexOp>(loc, dst.getType(), dst, colOffset);
                auto params = rewriter.create<ascendc::ConstructOp>(
                    loc, rewriter.getType<ascendc::InitConstValueParamsType>(),
                    ValueRange{const1, blockNum, const0, const0}, constArgTypes);
                rewriter.create<ascendc::FillOp>(loc, padTensor, params.getResult());
            }
        }
        rewriter.replaceOp(op, dst);
        return success();
    }
};

struct ConvertStore : ConvertOp<asctile::StoreOp> {
    using ConvertOp::ConvertOp;
    using ConvertOp::createTensorOp;

    LogicalResult matchAndRewrite(asctile::StoreOp op, ConvertRewriter& rewriter) const override
    {
        auto value = op.getValue();
        assert(value.getType().getLoc() == asctile::TileLocation::UB && "Tile should be located in UB.");
        Value src = rewriter.getRemappedValue(value);
        auto srcType = cast<ascendc::BaseTensorType>(src.getType());
        SmallVector<Value> srcShape = getStaticShape(rewriter, srcType);
        if (srcShape.size() > 2)
            return failure();
        auto loc = op.getLoc();
        auto offsets = op.getOffsets();
        ascir::ConstantOpBuilder consts(rewriter);
        TensorInfo dstInfo = prepareTensorInfo(rewriter, loc, op.getBase(), offsets);
        auto const0 = consts.i32(0);
        Value typeSize = consts.i32(ascendc::getElementTypeSize(srcType));
        Value srcLastDim = srcShape.back();
        Value dstLastDim = dstInfo.shape.back();
        Value tailElementsLastDim = rewriter.create<arith::SubIOp>(loc, dstLastDim, offsets.back());
        auto tailNegCond = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, tailElementsLastDim, const0);
        Value tailElements = rewriter.create<arith::SelectOp>(loc, tailNegCond, const0, tailElementsLastDim);
        Value minTailElements;
        auto realShape = op.getRealShape();
        if (realShape.empty()) {
            minTailElements = rewriter.create<arith::MinSIOp>(loc, srcLastDim, tailElements);
        } else {
            Value realLastDim = realShape.back();
            minTailElements = rewriter.create<arith::MinSIOp>(loc, realLastDim, tailElements);
        }
        Value blockLen = rewriter.create<arith::MulIOp>(loc, minTailElements, typeSize);
        Value srcStrideElements = rewriter.create<arith::SubIOp>(loc, srcLastDim, minTailElements);
        Value dstStrideElements = rewriter.create<arith::SubIOp>(loc, dstLastDim, minTailElements);
        Value blockCount = consts.i32(1);
        if (realShape.size() > 1) {
            for (size_t i = 0; i + 1 < realShape.size(); ++i)
                blockCount = rewriter.create<arith::MulIOp>(loc, blockCount, realShape[i]);
        } else {
            for (size_t i = 0; i + 1 < srcShape.size(); ++i)
                blockCount = rewriter.create<arith::MulIOp>(loc, blockCount, srcShape[i]);
        }
        Value dataBlockSize = consts.i32(ascendc::ubBlockSize);
        Value srcStrideBytes = rewriter.create<arith::MulIOp>(loc, srcStrideElements, typeSize);
        Value srcStride = rewriter.create<arith::DivSIOp>(loc, srcStrideBytes, dataBlockSize);
        Value dstStride = rewriter.create<arith::MulIOp>(loc, dstStrideElements, typeSize);
        auto ui32Type = rewriter.getIntegerType(32, false);
        auto dataCopyExtParams = rewriter.create<ascendc::ConstructOp>(
            loc, rewriter.getType<ascendc::DataCopyExtParamsType>(),
            ValueRange{blockCount, blockLen, srcStride, dstStride, const0},
            rewriter.getTypeArrayAttr({rewriter.getIntegerType(16, false), ui32Type, ui32Type, ui32Type, ui32Type}));
        rewriter.replaceOpWithNewOp<ascendc::DataCopyPadExtL2Op>(op, dstInfo.tensor, src, dataCopyExtParams);
        return success();
    }
};

struct ConvertStoreHighDims : ConvertOp<asctile::StoreOp> {
    using ConvertOp::ConvertOp;
    using ConvertOp::createTensorOp;

    LogicalResult matchAndRewrite(asctile::StoreOp op, ConvertRewriter& rewriter) const override
    {
        auto loc = op.getLoc();
        auto offsets = op.getOffsets();
        auto value = op.getValue();
        TensorInfo dstInfo = prepareTensorInfo(rewriter, loc, op.getBase(), offsets);
        Value src = rewriter.getRemappedValue(value);
        auto srcType = cast<ascendc::BaseTensorType>(src.getType());
        assert(value.getType().getLoc() == asctile::TileLocation::UB && "Tile should be located in UB.");
        ascir::ConstantOpBuilder consts(rewriter);
        SmallVector<int64_t> srcShape{static_cast<ArrayRef<int64_t>>(srcType.getShape())};
        if (srcShape.size() <= 2 || srcShape.size() > 4)
            return failure();
        auto const0 = consts.i32(0);
        Value typeSize = consts.i32(ascendc::getElementTypeSize(srcType));
        Value srcLastDim = consts.i32(srcShape.back());
        Value dstLastDim = dstInfo.shape.back();
        Value minTailElements =
            calculateCopyCount(rewriter, loc, srcShape, dstInfo.shape, offsets, op.getRealShape(), srcShape.size() - 1);

        Value blockLen = rewriter.create<arith::MulIOp>(loc, minTailElements, typeSize);
        Value srcStrideElements = rewriter.create<arith::SubIOp>(loc, srcLastDim, minTailElements);
        Value dstStrideElements = rewriter.create<arith::SubIOp>(loc, dstLastDim, minTailElements);
        Value blockCount = consts.i32(1);
        blockCount =
            calculateCopyCount(rewriter, loc, srcShape, dstInfo.shape, offsets, op.getRealShape(), srcShape.size() - 2);
        Value dataBlockSize = consts.i32(ascendc::ubBlockSize);
        Value srcStrideBytes = rewriter.create<arith::MulIOp>(loc, srcStrideElements, typeSize);
        Value srcStride = rewriter.create<arith::DivSIOp>(loc, srcStrideBytes, dataBlockSize);
        Value dstStride = rewriter.create<arith::MulIOp>(loc, dstStrideElements, typeSize);
        auto ui32Type = rewriter.getIntegerType(32, false);
        auto dataCopyExtParams = rewriter.create<ascendc::ConstructOp>(
            loc, rewriter.getType<ascendc::DataCopyExtParamsType>(),
            ValueRange{blockCount, blockLen, srcStride, dstStride, const0},
            rewriter.getTypeArrayAttr({rewriter.getIntegerType(16, false), ui32Type, ui32Type, ui32Type, ui32Type}));
        Value loop1Size =
            calculateCopyCount(rewriter, loc, srcShape, dstInfo.shape, offsets, op.getRealShape(), srcShape.size() - 3);
        Value loop2Size = srcShape.size() == 3 ? consts.i32(1) :
                                                 calculateCopyCount(
                                                     rewriter, loc, srcShape, dstInfo.shape, offsets, op.getRealShape(),
                                                     srcShape.size() - 4);

        Value dim0SrcStride = rewriter.create<arith::MulIOp>(loc, consts.i32(srcShape[srcShape.size() - 1]), typeSize);
        Value dim1SrcStride =
            rewriter.create<arith::MulIOp>(loc, dim0SrcStride, consts.i32(srcShape[srcShape.size() - 2]));
        Value dim2SrcStride =
            rewriter.create<arith::MulIOp>(loc, dim1SrcStride, consts.i32(srcShape[srcShape.size() - 3]));

        Value dim0DstStride = rewriter.create<arith::MulIOp>(loc, dstInfo.shape[dstInfo.shape.size() - 1], typeSize);
        Value dim1DstStride =
            rewriter.create<arith::MulIOp>(loc, dim0DstStride, dstInfo.shape[dstInfo.shape.size() - 2]);
        Value dim2DstStride =
            rewriter.create<arith::MulIOp>(loc, dim1DstStride, dstInfo.shape[dstInfo.shape.size() - 3]);

        auto dataCopyOp =
            rewriter.replaceOpWithNewOp<ascendc::DataCopyPadExtL2Op>(op, dstInfo.tensor, src, dataCopyExtParams);
        auto ui64Type = rewriter.getIntegerType(64, false);
        auto params = rewriter.create<ascendc::ConstructOp>(
            loc, rewriter.getType<ascendc::LoopModeParamsType>(),
            ValueRange{loop1Size, loop2Size, dim1SrcStride, dim1DstStride, dim2SrcStride, dim2DstStride},
            rewriter.getTypeArrayAttr({ui32Type, ui32Type, ui64Type, ui64Type, ui64Type, ui64Type}));
        auto setParamsOp = rewriter.create<ascendc::SetLoopModeParaOp>(loc, params, ascendc::DataCopyMVType::UB_TO_OUT);
        rewriter.create<ascendc::ResetLoopModeParaOp>(loc, ascendc::DataCopyMVType::UB_TO_OUT);
        rewriter.moveOpBefore(setParamsOp, dataCopyOp);
        rewriter.moveOpBefore(params, setParamsOp);
        return success();
    }
};

struct ConvertStoreFixpipe : ConvertOp<asctile::StoreFixpipeOp> {
    using ConvertOp::ConvertOp;
    using ConvertOp::createTensorOp;

    LogicalResult matchAndRewrite(asctile::StoreFixpipeOp op, ConvertRewriter& rewriter) const override
    {
        auto loc = op.getLoc();
        auto offsets = op.getOffsets();
        auto value = op.getValue();
        TensorInfo dstInfo = prepareTensorInfo(rewriter, loc, op.getBase(), offsets);
        Value src = rewriter.getRemappedValue(value);
        auto srcType = cast<ascendc::BaseTensorType>(src.getType());
        assert(value.getType().getLoc() == asctile::TileLocation::L0C && "Tile should be located in L0C.");
        ascir::ConstantOpBuilder consts(rewriter);
        SmallVector<Value> srcShape = getStaticShape(rewriter, srcType);
        auto const0 = consts.i32(0);
        auto const1 = consts.i32(1);
        Value srcLastDim = srcShape.back();
        Value dstLastDim = dstInfo.shape.back();
        Value tailElementsLastDim = rewriter.create<arith::SubIOp>(loc, dstLastDim, offsets.back());
        auto tailNegCond = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, tailElementsLastDim, const0);
        Value tailElements = rewriter.create<arith::SelectOp>(loc, tailNegCond, const0, tailElementsLastDim);
        Value nSize = rewriter.create<arith::MinSIOp>(loc, srcLastDim, tailElements);
        Value mSize = srcShape[0];
        if (auto realShape = op.getRealShape(); !realShape.empty()) {
            Value realCols = rewriter.getRemappedValue(realShape.back());
            nSize = rewriter.create<arith::MinSIOp>(loc, nSize, realCols);
            Value realRows = rewriter.getRemappedValue(realShape[0]);
            mSize = rewriter.create<arith::MinSIOp>(loc, mSize, realRows);
        } else {
            mSize = rewriter.create<arith::MinSIOp>(loc, mSize, dstInfo.shape[0]);
        }
        auto srcStride = static_cast<int32_t>(llvm::alignTo<ascendc::cubeBlockSize>(srcType.getShape()[0]));
        auto paramsBuilder = emitasc::InitStructBuilder(rewriter.getType<ascendc::FixpipeParamsV220Type>())
                                 .addField("nSize", nSize)
                                 .addField("mSize", mSize)
                                 .addField("srcStride", consts.i32(srcStride))
                                 .addField("dstStride", dstInfo.shape[1]);
        if (op.getRelu())
            paramsBuilder.addField("reluEn", const1);
        if (op.getQuantize()) {
            auto mode = getQuantizeMode(srcType, dstInfo.type, rewriter);
            if (!mode) {
                return op.emitError() << "Unsupported quant mode from " << srcType.getElementType() << " to "
                                      << dstInfo.type.getElementType();
            }
            auto quantMode = rewriter.create<ascendc::ConstructOp>(
                loc, rewriter.getType<ascendc::QuantModesType>(), ValueRange{consts.i32(static_cast<int32_t>(*mode))},
                rewriter.getTypeArrayAttr(rewriter.getType<ascendc::QuantModesType>()), true, true);
            paramsBuilder.addField("quantPre", quantMode);
        }
        Value params = paramsBuilder.create(rewriter, loc);
        Value layout = rewriter.create<ascendc::ConstructOp>(
            loc, rewriter.getType<ascendc::CO2LayoutType>(),
            ValueRange{consts.i32(static_cast<int32_t>(ascendc::CO2Layout::ROW_MAJOR))}, ArrayAttr{}, true, true);
        auto fixPipeConfig = rewriter.create<ascendc::ConstructOp>(
            loc, rewriter.getType<ascendc::FixpipeConfigType>(), ValueRange{layout}, ArrayAttr{}, true, true);
        rewriter.replaceOpWithNewOp<ascendc::FixpipeOp>(op, dstInfo.tensor, src, params, fixPipeConfig);
        return success();
    }
};

struct ConvertCopyFixpipe : ConvertOp<asctile::CopyFixpipeOp> {
    using ConvertOp::ConvertOp;
    using ConvertOp::createTensorOp;

    LogicalResult matchAndRewrite(asctile::CopyFixpipeOp op, ConvertRewriter& rewriter) const override
    {
        auto base = op.getResult();
        auto loc = op.getLoc();
        auto value = op.getBase();
        Value src = rewriter.getRemappedValue(value);
        Value dst = createTensorOp(rewriter, loc, op.getType());
        auto srcType = cast<ascendc::BaseTensorType>(src.getType());
        assert(value.getType().getLoc() == asctile::TileLocation::L0C && "Tile should be located in L0C.");
        auto dstType = cast<ascendc::BaseTensorType>(dst.getType());
        assert(dstType.getElementType() != rewriter.getF32Type() && "dst type in L1 shouldn't be float32");
        ascir::ConstantOpBuilder consts(rewriter);
        SmallVector<Value> srcShape = getStaticShape(rewriter, srcType);
        SmallVector<Value> dstShape = getStaticShape(rewriter, dstType);
        auto const1 = consts.i32(1);
        Value linearOffset = linearizeOffset(rewriter, loc, dstShape, op.getOffsets());
        src = rewriter.create<ascendc::LocalTensorSubIndexOp>(loc, srcType, src, linearOffset);
        auto dstStride = rewriter.create<arith::MulIOp>(
            loc, srcShape[0], consts.i32(cubeKBlockBytes / ascendc::getElementTypeSize(op.getType())));
        auto paramsBuilder =
            emitasc::InitStructBuilder(
                ascendc::FixpipeParamsC310Type::get(
                    op.getContext(), ascendc::CO2LayoutAttr::get(op.getContext(), ascendc::CO2Layout::NZ)))
                .addField("nSize", srcShape[1])
                .addField("mSize", srcShape[0])
                .addField("srcStride", srcShape[0])
                .addField("dstStride", dstStride);
        if (op.getRelu())
            paramsBuilder.addField("reluEn", const1);
        if (op.getQuantize()) {
            auto mode = getQuantizeMode(srcType, dstType, rewriter);
            if (!mode) {
                return op.emitError() << "Unsupported quant mode from " << srcType.getElementType() << " to "
                                      << dstType.getElementType();
            }
            auto quantMode = rewriter.create<ascendc::ConstructOp>(
                loc, rewriter.getType<ascendc::QuantModesType>(), ValueRange{consts.i32(static_cast<int32_t>(*mode))},
                rewriter.getTypeArrayAttr(rewriter.getType<ascendc::QuantModesType>()), true, true);
            paramsBuilder.addField("quantPre", quantMode);
        }
        Value params = paramsBuilder.create(rewriter, loc);
        Value layout = rewriter.create<ascendc::ConstructOp>(
            loc, rewriter.getType<ascendc::CO2LayoutType>(),
            ValueRange{consts.i32(static_cast<int32_t>(ascendc::CO2Layout::NZ))}, ArrayAttr{}, true, true);
        auto fixPipeConfig = rewriter.create<ascendc::ConstructOp>(
            loc, rewriter.getType<ascendc::FixpipeConfigType>(), ValueRange{layout}, ArrayAttr{}, true, true);
        rewriter.create<ascendc::FixpipeOp>(loc, dst, src, params, fixPipeConfig);
        rewriter.replaceOp(op, dst);
        return success();
    }
};

Value buildLoadData2DV2Params(
    OpBuilder& builder, Location loc, ascir::ConstantOpBuilder& consts, bool isTensorA, bool isTransposeA,
    bool isTransposeB, int64_t cubeKBlockSize, ArrayRef<int64_t> srcShape, ArrayRef<int64_t> dstShape,
    Value mStartPosition, Value kStartPosition)
{
    int64_t mStep, kStep, srcStride, dstStride;
    bool ifTranspose;
    if ((isTensorA && !isTransposeA) || (!isTensorA && isTransposeB)) {
        auto mAlignL0 = llvm::alignTo(isTransposeB ? dstShape[1] : dstShape[0], ascendc::cubeBlockSize);
        auto mAlignL1 = llvm::alignTo(srcShape[0], ascendc::cubeBlockSize);
        auto kAlignL1 = llvm::alignTo(isTransposeB ? dstShape[0] : dstShape[1], cubeKBlockSize);
        mStep = llvm::divideCeilSigned(mAlignL0, ascendc::cubeBlockSize);
        kStep = llvm::divideCeilSigned(kAlignL1, cubeKBlockSize);
        srcStride = llvm::divideCeilSigned(mAlignL1, ascendc::cubeBlockSize);
        dstStride = llvm::divideCeilSigned(mAlignL0, ascendc::cubeBlockSize);
        ifTranspose = false;
    } else {
        auto mAlignL1 = llvm::alignTo(isTransposeA ? dstShape[0] : dstShape[1], ascendc::cubeBlockSize);
        auto kaAlignL0 = llvm::alignTo(isTransposeA ? dstShape[1] : dstShape[0], ascendc::cubeBlockSize);
        auto kaAlignL1 = llvm::alignTo(srcShape[0], ascendc::cubeBlockSize);
        mStep = llvm::divideCeilSigned(kaAlignL0, ascendc::cubeBlockSize);
        kStep = llvm::divideCeilSigned(mAlignL1, cubeKBlockSize);
        srcStride = llvm::divideCeilSigned(kaAlignL1, ascendc::cubeBlockSize);
        dstStride = llvm::divideCeilSigned(mAlignL1, ascendc::cubeBlockSize);
        ifTranspose = true;
    }
    auto paramsType = builder.getType<ascendc::LoadData2DParamsV2Type>();
    return emitasc::InitStructBuilder(paramsType)
        .addField("mStartPosition", mStartPosition)
        .addField("kStartPosition", kStartPosition)
        .addField("mStep", consts.i32(mStep))
        .addField("kStep", consts.i32(kStep))
        .addField("srcStride", consts.i32(srcStride))
        .addField("dstStride", consts.i32(dstStride))
        .addField("ifTranspose", consts.i1(ifTranspose))
        .create(builder, loc);
}

struct ConvertCopy : ConvertOp<asctile::CopyOp> {
    using ConvertOp::ConvertOp;
    using ConvertOp::createTensorOp;

    LogicalResult matchAndRewrite(asctile::CopyOp op, ConvertRewriter& rewriter) const override
    {
        auto opType = op.getType();
        auto dstPos = opType.getLoc();
        if (dstPos != asctile::TileLocation::L0A && dstPos != asctile::TileLocation::L0B &&
            dstPos != asctile::TileLocation::BT) {
            op.emitError() << "invalid destination location of the tile";
            return failure();
        }
        auto loc = op.getLoc();
        auto base = op.getBase();
        Value src = rewriter.getRemappedValue(base);
        auto srcType = src.getType();
        auto srcShape = base.getType().getShape();
        auto offsets = op.getOffsets();
        ascir::ConstantOpBuilder consts(rewriter);
        if (dstPos == asctile::TileLocation::BT) {
            auto srcLoc = base.getType().getLoc();
            if (srcLoc != asctile::TileLocation::L1) {
                op.emitError() << "BT destination requires L1 source for bias copy";
                return failure();
            }
            auto dst = createTensorOp(rewriter, loc, opType).getResult();
            auto dstShape = opType.getShape();
            if (srcShape.size() != 1)
                return op.emitError() << "bias must have 1D shape";
            if (!offsets.empty()) {
                src = rewriter.create<ascendc::LocalTensorSubIndexOp>(loc, srcType, src, offsets[0]);
            }
            int64_t typeSize = ascendc::getElementTypeSize(base.getType());
            int64_t blockLen = (dstShape[0] * typeSize) / cubeKBlockBytes;
            auto dataCopyParams = rewriter.create<ascendc::ConstructOp>(
                loc, rewriter.getType<ascendc::DataCopyParamsType>(),
                ValueRange{consts.i32(1), consts.i32(blockLen), consts.i32(0), consts.i32(0)});
            rewriter.create<ascendc::DataCopyL0Op>(loc, dst, src, dataCopyParams);
            rewriter.replaceOp(op, dst);
            return success();
        }
        assert(srcShape.size() == 2 && "supported only tensorShape with 2 dims");
        assert(offsets.size() == srcShape.size() && "must be one offset for each dimension");
        auto dst = createTensorOp(rewriter, loc, opType).getResult();
        auto dstType = dst.getType();
        auto dstShape = dstType.getShape();
        bool isTensorA = dstPos == asctile::TileLocation::L0A;
        bool isTransposeA = op->hasAttrOfType<UnitAttr>(asctile::attr::transposeA);
        bool isTransposeB = op->hasAttrOfType<UnitAttr>(asctile::attr::transposeB);
        bool isFloat32 = isa<Float32Type>(opType.getElementType());
        bool isBNoTransF32 = !isTensorA && isFloat32 && !isTransposeB;
        const int64_t cubeKBlockSize = cubeKBlockBytes / ascendc::getElementTypeSize(opType);
        const int64_t cubeBlockCols = !isBNoTransF32 ? cubeKBlockSize : ascendc::cubeBlockSize;
        Value mStartPosition = rewriter.create<arith::DivSIOp>(loc, offsets[0], consts.i32(ascendc::cubeBlockSize));
        Value kStartPosition = rewriter.create<arith::DivSIOp>(loc, offsets[1], consts.i32(cubeKBlockSize));
        Value params = buildLoadData2DV2Params(
            rewriter, loc, consts, isTensorA, isTransposeA, isTransposeB, cubeKBlockSize, srcShape, dstShape,
            mStartPosition, kStartPosition);
        rewriter.create<ascendc::LoadDataL0V2Op>(loc, dst, src, params);
        rewriter.replaceOp(op, dst);
        return success();
    }
};

struct ConvertGetValue : ConvertOp<asctile::GetValueOp> {
    using ConvertOp::ConvertOp;
    using ConvertOp::createTensorOp;

    LogicalResult matchAndRewrite(asctile::GetValueOp op, ConvertRewriter& rewriter) const override
    {
        auto base = op.getBase();
        auto loc = op.getLoc();
        auto tensorOp = base.getDefiningOp<asctile::TensorOp>();
        assert(tensorOp && "tensor must be created by asctile.tensor op");
        SmallVector<Value> srcShape = getTensorShape(rewriter, tensorOp);
        Value linearOffset = linearizeOffset(rewriter, loc, srcShape, op.getOffsets());
        Value src = rewriter.getRemappedValue(base);
        rewriter.replaceOpWithNewOp<ascendc::GlobalTensorGetValueOp>(op, op.getType(), src, linearOffset);
        return success();
    }
};

struct ConvertSetValue : ConvertOp<asctile::SetValueOp> {
    using ConvertOp::ConvertOp;
    using ConvertOp::createTensorOp;

    LogicalResult matchAndRewrite(asctile::SetValueOp op, ConvertRewriter& rewriter) const override
    {
        auto base = op.getBase();
        auto loc = op.getLoc();
        Value src = rewriter.getRemappedValue(op.getValue());
        if (auto srcType = dyn_cast<ascendc::LocalTensorType>(src.getType())) {
            ascir::ConstantOpBuilder consts(rewriter);
            src = rewriter.create<ascendc::LocalTensorGetValueOp>(loc, srcType.getElementType(), src, consts.i64(0));
        }
        auto tensorOp = base.getDefiningOp<asctile::TensorOp>();
        assert(tensorOp && "tensor must be created by asctile.tensor op");
        SmallVector<Value> dstShape = getTensorShape(rewriter, tensorOp);
        Value linearOffset = linearizeOffset(rewriter, loc, dstShape, op.getOffsets());
        Value dst = rewriter.getRemappedValue(base);
        Value offset = rewriter.create<emitc::CastOp>(loc, rewriter.getIntegerType(64, false), linearOffset);
        rewriter.replaceOpWithNewOp<ascendc::GlobalTensorSetValueOp>(op, dst, offset, src);
        return success();
    }
};

struct LowerAscTileDataTransferPass
    : public asclower::impl::LowerAscTileDataTransferBase<LowerAscTileDataTransferPass> {
    void runOnOperation() override
    {
        TensorTypeConverter converter;
        MLIRContext* context = &getContext();
        ConversionTarget target(*context);
        target.addIllegalOp<
            //
            asctile::LoadOp, asctile::StoreOp, asctile::CopyOp, asctile::StoreFixpipeOp, asctile::GetValueOp,
            asctile::SetValueOp, asctile::CopyFixpipeOp
            //
            >();
        target.addLegalDialect<
            ascendc::AscendCDialect, arith::ArithDialect, emitasc::EmitAscDialect, emitc::EmitCDialect,
            scf::SCFDialect>();
        target.addLegalOp<UnrealizedConversionCastOp>();
        RewritePatternSet patterns(context);
        patterns.insert<
            //
            ConvertLoadToUB, ConvertLoadToUBWithTranspose, ConvertLoadToL1, ConvertStore, ConvertStoreFixpipe,
            ConvertCopy, ConvertGetValue, ConvertSetValue, ConvertCopyFixpipe, ConvertStoreHighDims
            //
            >(converter, context);
        if (applyPartialConversion(getOperation(), target, std::move(patterns)).failed())
            signalPassFailure();
    }
};

} // namespace

std::unique_ptr<Pass> mlir::asclower::createLowerAscTileDataTransferPass()
{
    return std::make_unique<LowerAscTileDataTransferPass>();
}
