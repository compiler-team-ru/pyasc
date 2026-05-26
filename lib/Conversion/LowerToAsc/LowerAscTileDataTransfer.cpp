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
        if (!realShape.empty()) {
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
        auto hasGap = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sgt, rowSizeBytes, alignedBlockSize);
        auto ifHasGap = rewriter.create<scf::IfOp>(loc, hasGap, false);
        {
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
            auto padRows = rewriter.create<arith::SubIOp>(loc, dstRows, innerRows);
            auto padRowsIsPositive = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sgt, padRows, const0);
            auto ifPadRows = rewriter.create<scf::IfOp>(loc, padRowsIsPositive, false);
            {
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
        ArrayRef<int32_t> transposeDims = transposeAttrs;
        auto dims = transposeDims.size();
        SmallVector<int64_t> dimsOrder(dims, 0);
        SmallVector<int64_t> readShape(dims, 0);
        for (size_t i = 0; i < dims; ++i) {
            dimsOrder[i] = transposeDims[i];
            readShape[dimsOrder[i]] = dstShape[i];
        }
        SmallVector<Value> size(dims, const1);
        SmallVector<Value> strides(dims, Value());
        strides.back() = const1;
        for (int64_t i = static_cast<int64_t>(dims) - 2; i >= 0; i--) {
            auto mulOp = rewriter.create<arith::MulIOp>(loc, srcInfo.shape[i + 1], strides[i + 1]);
            strides[i] = mulOp;
        }
        SmallVector<Value> srcStride(dims, Value());
        SmallVector<int32_t> dstStride(dims, 0);
        SmallVector<int32_t> padLeft(dims, 0);
        SmallVector<int32_t> padRight(dims, 0);
        for (size_t i = 0; i < dims; ++i) {
            auto realShape = op.getRealShape();
            if (i < realShape.size())
                size[i] = rewriter.getRemappedValue(realShape[i]);
            else
                size[i] = consts.i32(readShape[i]);
            int32_t writeStride = 1;
            auto mappedDim = dimsOrder[i];
            for (size_t j = mappedDim + 1; j < dims; ++j)
                writeStride *= static_cast<int32_t>(dstShape[j]);
            srcStride[i] = strides[i];
            dstStride[i] = writeStride;
            padLeft[i] = 0;
            padRight[i] = 0;
        }
        auto paramsType = rewriter.getType<ascendc::NdDmaParamsType>(srcInfo.type.getElementType(), dims);
        auto params = rewriter.create<ascendc::NdDmaParamsOp>(
            loc, paramsType, dims, padValue, size, srcStride, rewriter.getI32ArrayAttr(dstStride),
            rewriter.getI32ArrayAttr(padLeft), rewriter.getI32ArrayAttr(padRight));
        rewriter.create<ascendc::DataCopyNdDmaOp>(loc, dst, srcInfo.tensor, params.getResult(), dims);
        rewriter.replaceOp(op, dst);
        return success();
    }
};

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
        bool isMatrixA = op->hasAttr(asctile::attr::isMatrixA);
        bool isTransposeB = op->hasAttrOfType<UnitAttr>(asctile::attr::transposeB);
        auto dstShapeCols = consts.i32(dstShape[1]);
        if (isMatrixA || isa<Float16Type, BFloat16Type>(opType.getElementType()) || isTransposeB) {
            auto dstShapeRows = consts.i32(dstShape[0]);
            Value nValue = dstShapeRows;
            Value dValue = dstShapeCols;
            if (auto realShape = op.getRealShape(); !realShape.empty()) {
                Value realRows = rewriter.getRemappedValue(realShape[0]);
                Value realCols = rewriter.getRemappedValue(realShape[1]);
                nValue = rewriter.create<arith::MinSIOp>(loc, dstShapeRows, realRows);
                dValue = rewriter.create<arith::MinSIOp>(loc, dstShapeCols, realCols);
            }
            auto nd2NzParams = rewriter.create<ascendc::ConstructOp>(
                loc, rewriter.getType<ascendc::Nd2NzParamsType>(),
                ValueRange{const1, nValue, dValue, const0, srcInfo.shape[1], dstShapeRows, const1, const0});
            rewriter.create<ascendc::DataCopyL2Op>(loc, dst, srcInfo.tensor, nd2NzParams);
        } else {
            auto ndNum = consts.i32(llvm::divideCeilSigned(dstShape[0], ascendc::cubeBlockSize));
            auto nValue = consts.i32(ascendc::cubeBlockSize);
            auto srcNdMatrixStride = rewriter.create<arith::MulIOp>(loc, nValue, srcInfo.shape[1]);
            int64_t fractal = (cubeKBlockBytes / ascendc::getElementTypeSize(opType)) * fractalNum;
            auto ceilAlignFractal = static_cast<int64_t>(llvm::alignTo(dstShape[1], fractal));
            auto dstNzMatrixStride = consts.i32(ascendc::cubeBlockSize * ceilAlignFractal);
            auto nd2NzParams = rewriter.create<ascendc::ConstructOp>(
                loc, rewriter.getType<ascendc::Nd2NzParamsType>(),
                ValueRange{
                    ndNum, nValue, dstShapeCols, srcNdMatrixStride, srcInfo.shape[1], nValue, const1,
                    dstNzMatrixStride});
            rewriter.create<ascendc::DataCopyL2Op>(loc, dst, srcInfo.tensor, nd2NzParams);
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
        auto loc = op.getLoc();
        auto offsets = op.getOffsets();
        auto value = op.getValue();
        TensorInfo dstInfo = prepareTensorInfo(rewriter, loc, op.getBase(), offsets);
        Value src = rewriter.getRemappedValue(value);
        auto srcType = cast<ascendc::BaseTensorType>(src.getType());
        assert(value.getType().getLoc() == asctile::TileLocation::UB && "Tile should be located in UB.");
        ascir::ConstantOpBuilder consts(rewriter);
        SmallVector<Value> srcShape = getStaticShape(rewriter, srcType);
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
        auto srcStride = static_cast<int32_t>(llvm::alignTo<ascendc::cubeBlockSize>(srcType.getShape()[0]));
        auto paramsBuilder = emitasc::InitStructBuilder(rewriter.getType<ascendc::FixpipeParamsV220Type>())
                                 .addField("nSize", nSize)
                                 .addField("mSize", srcShape[0])
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

struct ConvertCopy : ConvertOp<asctile::CopyOp> {
    using ConvertOp::ConvertOp;
    using ConvertOp::createTensorOp;

    LogicalResult matchAndRewrite(asctile::CopyOp op, ConvertRewriter& rewriter) const override
    {
        auto opType = op.getType();
        auto dstPos = opType.getLoc();
        if (dstPos != asctile::TileLocation::L0A && dstPos != asctile::TileLocation::L0B) {
            op.emitError() << "invalid destination location of the tile";
            return failure();
        }
        auto loc = op.getLoc();
        auto base = op.getBase();
        Value src = rewriter.getRemappedValue(base);
        auto srcType = src.getType();
        auto srcShape = base.getType().getShape();
        auto offsets = op.getOffsets();
        assert(srcShape.size() == 2 && "supported only tensorShape with 2 dims");
        assert(offsets.size() == srcShape.size() && "must be one offset for each dimension");
        ascir::ConstantOpBuilder consts(rewriter);
        bool isTensorA = dstPos == asctile::TileLocation::L0A;
        bool isTransposeA = op->hasAttrOfType<UnitAttr>(asctile::attr::transposeA);
        bool isTransposeB = op->hasAttrOfType<UnitAttr>(asctile::attr::transposeB);
        bool isBNoTransF32 = !isTensorA && isa<Float32Type>(opType.getElementType()) && !isTransposeB;
        const int64_t cubeKBlockSize = cubeKBlockBytes / ascendc::getElementTypeSize(opType);
        const int64_t cubeBlockRows = !isBNoTransF32 ? ascendc::cubeBlockSize : cubeKBlockSize;
        const int64_t cubeBlockCols = !isBNoTransF32 ? cubeKBlockSize : ascendc::cubeBlockSize;
        const int64_t fractalSize = cubeBlockRows * cubeBlockCols;
        const int64_t dstNzC0Stride = !isBNoTransF32 ? srcShape[0] : cubeBlockCols;
        const int64_t dValue = !isBNoTransF32 ? cubeBlockCols : srcShape[1];
        // Note! For this formula it is assumed that offsets are divisible by corresponding Cube block dimensions.
        Value colOffset = rewriter.create<arith::MulIOp>(loc, consts.i32(dstNzC0Stride), offsets[1]);
        Value rowOffset = rewriter.create<arith::MulIOp>(loc, offsets[0], consts.i32(dValue));
        Value linearOffset = rewriter.create<arith::AddIOp>(loc, colOffset, rowOffset);
        src = rewriter.create<ascendc::LocalTensorSubIndexOp>(loc, srcType, src, linearOffset);
        auto dst = createTensorOp(rewriter, loc, opType).getResult();
        auto dstType = dst.getType();
        auto dstShape = dstType.getShape();
        auto const0 = consts.i32(0);
        auto const1 = consts.i32(1);
        auto dstFracStride = llvm::divideCeilSigned(isTransposeB ? dstShape[0] : dstShape[1], cubeBlockCols);
        if (!isBNoTransF32) {
            // LoadData2DParams doesn't support ifTranspose for float32, LoadDataWithTranpose doesn't support TPosition
            // A2 on Ascend910_95. Used LoadData2DParamsV2
            if (isTransposeA && isa<Float32Type>(opType.getElementType())) {
                auto paramsType = rewriter.getType<ascendc::LoadData2DParamsV2Type>();
                auto mStep = consts.i32(llvm::divideCeilSigned(dstShape[1], ascendc::cubeBlockSize));
                auto kStep = consts.i32(
                    llvm::divideCeilSigned(llvm::divideCeilSigned(dstShape[0], cubeKBlockSize), fractalNum) *
                    fractalNum);
                auto dstStride = consts.i32(llvm::divideCeilSigned(dstShape[0], cubeKBlockSize * fractalNum));
                Value params = emitasc::InitStructBuilder(paramsType)
                                   .addField("mStep", mStep)
                                   .addField("kStep", kStep)
                                   .addField("srcStride", mStep)
                                   .addField("dstStride", dstStride)
                                   .addField("ifTranspose", consts.i1(true))
                                   .create(rewriter, loc);
                rewriter.create<ascendc::LoadDataL0V2Op>(loc, dst, src, params);
            } else {
                auto paramsType = rewriter.getType<ascendc::LoadData2DParamsType>();
                int64_t repeatTimes = llvm::divideCeilSigned(isTransposeB ? dstShape[1] : dstShape[0], cubeBlockRows);
                Value srcStrideParam = isTransposeA ? consts.i32(dstFracStride) : const1;
                Value dstGap = ((isTensorA && !isTransposeA) || isTransposeB) ? const0 : consts.i32(dstFracStride - 1);
                Value params = emitasc::InitStructBuilder(paramsType)
                                   .addField("repeatTimes", consts.i32(repeatTimes))
                                   .addField("srcStride", srcStrideParam)
                                   .addField("dstGap", dstGap)
                                   .addField("ifTranspose", consts.i1(isTransposeA || (!isTensorA && !isTransposeB)))
                                   .create(rewriter, loc);
                auto forOp = rewriter.create<scf::ForOp>(loc, const0, consts.i32(dstFracStride), const1);
                rewriter.setInsertionPointToStart(forOp.getBody());
                auto indVar = forOp.getInductionVar();
                int64_t srcStride =
                    (isTensorA && isTransposeA) ? (cubeBlockRows * cubeBlockCols) : (srcShape[0] * cubeBlockCols);
                auto iterSrcOffset = rewriter.create<arith::MulIOp>(loc, indVar, consts.i32(srcStride));
                auto subLocalL1 = rewriter.create<ascendc::LocalTensorSubIndexOp>(loc, srcType, src, iterSrcOffset);
                int64_t dstStride = (isTensorA || isTransposeB) ? repeatTimes * fractalSize : fractalSize;
                auto iterDstOffset = rewriter.create<arith::MulIOp>(loc, indVar, consts.i32(dstStride));
                auto subLocalL0 = rewriter.create<ascendc::LocalTensorSubIndexOp>(loc, dstType, dst, iterDstOffset);
                rewriter.create<ascendc::LoadDataG2LOp>(loc, subLocalL0, subLocalL1, params);
                rewriter.setInsertionPointAfter(forOp);
                forOp->setAttr(asctile::attr::parallel, rewriter.getUnitAttr());
            }
        } else {
            auto paramsTransposeType = rewriter.getType<ascendc::LoadData2dTransposeParamsType>();
            int64_t repeatTimes = llvm::divideCeilSigned(dstShape[1], cubeBlockRows * fractalNum);
            Value params = emitasc::InitStructBuilder(paramsTransposeType)
                               .addField("repeatTimes", consts.i32(repeatTimes))
                               .addField("srcStride", const1)
                               .addField("dstGap", const0)
                               .addField("dstFracGap", consts.i32(dstFracStride - 1))
                               .create(rewriter, loc);
            Value uBound = consts.i32(llvm::divideCeilSigned(dstShape[0], ascendc::cubeBlockSize));
            auto forOp = rewriter.create<scf::ForOp>(loc, const0, uBound, const1);
            rewriter.setInsertionPointToStart(forOp.getBody());
            auto indVar = forOp.getInductionVar();
            int64_t srcOffset = repeatTimes * fractalSize * fractalNum;
            auto iterSrcOffset = rewriter.create<arith::MulIOp>(loc, indVar, consts.i32(srcOffset));
            auto subLocalL1 = rewriter.create<ascendc::LocalTensorSubIndexOp>(loc, srcType, src, iterSrcOffset);
            int64_t dstOffset = dstFracStride * fractalSize * fractalNum;
            auto iterDstOffset = rewriter.create<arith::MulIOp>(loc, indVar, consts.i32(dstOffset));
            auto subLocalL0 = rewriter.create<ascendc::LocalTensorSubIndexOp>(loc, dstType, dst, iterDstOffset);
            rewriter.create<ascendc::LoadDataWithTransposeOp>(loc, subLocalL0, subLocalL1, params);
            rewriter.setInsertionPointAfter(forOp);
            forOp->setAttr(asctile::attr::parallel, rewriter.getUnitAttr());
        }
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
            ConvertCopy, ConvertGetValue, ConvertSetValue, ConvertCopyFixpipe
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
