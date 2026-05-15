/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ascir/Dialect/Asc/IR/Asc.h"
#include "ascir/Dialect/Asc/Utils/Utils.h"
#include "ascir/Conversion/LowerToAsc/Passes.h"
#include "ascir/Dialect/AscTile/IR/AscTile.h"
#include "ascir/Dialect/AscTile/Utils/Attributes.h"
#include "ascir/Dialect/EmitAsc/Utils/InitStructBuilder.h"
#include "ascir/Dialect/Utils/ConstantOpBuilder.h"
#include "ascir/Dialect/EmitAsc/IR/EmitAsc.h"

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

constexpr int64_t CUBE_K_BLOCK_BYTES = ascendc::ubBlockSize;
constexpr int64_t FRACTAL_NUM = 2;

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

Value calculateNumElements(OpBuilder& builder, Location loc, ArrayRef<Value> shape)
{
    assert(!shape.empty() && "shape must contain values");
    Value acc = shape.front();
    for (auto next : shape.drop_front())
        acc = builder.create<arith::MulIOp>(loc, acc, next);
    return acc;
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

struct ConvertLoad : ConvertOp<asctile::LoadOp> {
    using ConvertOp::ConvertOp;
    using ConvertOp::createTensorOp;

    LogicalResult convert(asctile::LoadOp op, ConvertRewriter& rewriter) const override
    {
        auto opType = op.getType();
        auto dstLoc = opType.getLoc();
        if (dstLoc != asctile::TileLocation::L1 && dstLoc != asctile::TileLocation::UB) {
            op.emitError() << "invalid destination location of the tile";
            return failure();
        }
        auto loc = op.getLoc();
        auto base = op.getBase();
        auto tensorOp = base.getDefiningOp<asctile::TensorOp>();
        assert(tensorOp && "tensor must be created by asctile.tensor op");
        SmallVector<Value> srcShape = getTensorShape(rewriter, tensorOp);
        Value src = rewriter.getRemappedValue(base);
        auto srcType = cast<ascendc::BaseTensorType>(src.getType());
        Value linearOffset = linearizeOffset(rewriter, loc, srcShape, op.getOffsets());
        src = rewriter.create<ascendc::GlobalTensorSubIndexOp>(loc, srcType, src, linearOffset);
        bool isMatrixA = op->hasAttr(asctile::attr::isMatrixA);
        auto dstTensorOp = createTensorOp(rewriter, loc, opType, locationToPosition(dstLoc));
        auto dst = dstTensorOp.getResult();
        auto dstType = dst.getType();
        auto dstShape = dstType.getShape();
        ascir::ConstantOpBuilder consts(rewriter);
        auto const0 = consts.i32(0);
        auto const1 = consts.i32(1);
        bool isTransposeB = op->hasAttrOfType<UnitAttr>(asctile::attr::transposeB);
        if (dstLoc == asctile::TileLocation::L1) {
            auto dstShapeCols = consts.i32(dstShape[1]);
            if (isMatrixA || isa<Float16Type, BFloat16Type>(opType.getElementType()) || isTransposeB) {
                auto dstShapeRows = consts.i32(dstShape[0]);
                auto nd2NzParams = rewriter.create<ascendc::ConstructOp>(
                    loc, rewriter.getType<ascendc::Nd2NzParamsType>(),
                    ValueRange{const1, dstShapeRows, dstShapeCols, const0, srcShape[1], dstShapeRows, const1, const0});
                rewriter.create<ascendc::DataCopyL2Op>(loc, dst, src, nd2NzParams);
            } else {
                auto ndNum = consts.i32(llvm::divideCeilSigned(dstShape[0], ascendc::cubeBlockSize));
                auto nValue = consts.i32(ascendc::cubeBlockSize);
                auto srcNdMatrixStride = rewriter.create<arith::MulIOp>(loc, nValue, srcShape[1]);
                int64_t fractal = (CUBE_K_BLOCK_BYTES / ascendc::getElementTypeSize(opType)) * FRACTAL_NUM;
                int64_t ceilAlignFractal = llvm::alignTo(dstShape[1], fractal);
                auto dstNzMatrixStride = consts.i32(ascendc::cubeBlockSize * ceilAlignFractal);
                auto nd2NzParams = rewriter.create<ascendc::ConstructOp>(
                    loc, rewriter.getType<ascendc::Nd2NzParamsType>(),
                    ValueRange{
                        ndNum, nValue, dstShapeCols, srcNdMatrixStride, srcShape[1], nValue, const1,
                        dstNzMatrixStride});
                rewriter.create<ascendc::DataCopyL2Op>(loc, dst, src, nd2NzParams);
            }
        } else {
            auto padValue = rewriter.getRemappedValue(op.getPadValue());
            auto typeSize = ascendc::getElementTypeSize(dstType);
            auto numElements = calculateNumElements(rewriter, loc, srcShape);
            Value dstLastDim = consts.i32(dstShape[dstShape.size() - 1]);
            Value srcLastDim = srcShape[srcShape.size() - 1];
            Value numElementsInBlock = consts.i32(ascendc::ubBlockSize / typeSize);
            Value typeSizeValue = consts.i32(typeSize);
            auto offsets = op.getOffsets();
            Value lastDimOffset = offsets.back();
            Value tailElementsLastDim = rewriter.create<arith::SubIOp>(loc, srcLastDim, lastDimOffset);
            auto tailNegCond =
                rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, tailElementsLastDim, const0);
            Value tailElements = rewriter.create<arith::SelectOp>(loc, tailNegCond, const0, tailElementsLastDim);
            Value minTailElements = rewriter.create<arith::MinSIOp>(loc, dstLastDim, tailElements);
            Value blockLen;
            Value srcStrideElements;
            Value rightPad;
            if (auto realShape = op.getRealShape(); !realShape.empty()) {
                auto realLastDim = realShape.back();
                Value realTailElements = rewriter.create<arith::MinSIOp>(loc, realLastDim, tailElements);
                blockLen = rewriter.create<arith::MulIOp>(loc, realTailElements, typeSizeValue);
                srcStrideElements = rewriter.create<arith::SubIOp>(loc, srcLastDim, realTailElements);
                rightPad = rewriter.create<arith::SubIOp>(loc, dstLastDim, realLastDim);
            } else {
                blockLen = rewriter.create<arith::MulIOp>(loc, minTailElements, typeSizeValue);
                srcStrideElements = rewriter.create<arith::SubIOp>(loc, srcLastDim, minTailElements);
                rightPad = rewriter.create<arith::SubIOp>(loc, dstLastDim, minTailElements);
            }
            Value srcStride = rewriter.create<arith::MulIOp>(loc, srcStrideElements, typeSizeValue);
            Value blockCount = const1;
            for (size_t i = 0; i + 1 < dstShape.size(); i++)
                blockCount = rewriter.create<arith::MulIOp>(loc, blockCount, consts.i32(dstShape[i]));
            auto ui32Type = rewriter.getIntegerType(32, false);
            auto dataCopyExtParams = rewriter.create<ascendc::ConstructOp>(
                loc, rewriter.getType<ascendc::DataCopyExtParamsType>(),
                ValueRange{blockCount, blockLen, srcStride, const0, const0},
                rewriter.getTypeArrayAttr(
                    {rewriter.getIntegerType(16, false), ui32Type, ui32Type, ui32Type, ui32Type}));
            auto dataCopyPadExtParams = rewriter.create<ascendc::ConstructOp>(
                loc, rewriter.getType<ascendc::DataCopyPadExtParamsType>(getElementTypeOrSelf(dstType)),
                ValueRange{const1, const0, rightPad, padValue},
                rewriter.getTypeArrayAttr(
                    {rewriter.getI32Type(), rewriter.getI32Type(), rewriter.getIntegerType(8, false),
                     padValue.getType()}));
            rewriter.create<ascendc::DataCopyPadExtL0Op>(loc, dst, src, dataCopyExtParams, dataCopyPadExtParams);
        }
        rewriter.replaceOp(op, dst);
        return success();
    }
};

struct ConvertStore : ConvertOp<asctile::StoreOp> {
    using ConvertOp::ConvertOp;
    using ConvertOp::createTensorOp;

    LogicalResult convert(asctile::StoreOp op, ConvertRewriter& rewriter) const override
    {
        auto base = op.getBase();
        auto tensorOp = base.getDefiningOp<asctile::TensorOp>();
        assert(tensorOp && "tensor must be created by asctile.tensor op");
        auto loc = op.getLoc();
        auto value = op.getValue();
        Value src = rewriter.getRemappedValue(value);
        Value dst = rewriter.getRemappedValue(base);
        auto srcType = cast<ascendc::BaseTensorType>(src.getType());
        assert(value.getType().getLoc() == asctile::TileLocation::UB && "Tile should be located in UB.");
        auto dstType = cast<ascendc::BaseTensorType>(dst.getType());
        ascir::ConstantOpBuilder consts(rewriter);
        SmallVector<Value> srcShape;
        for (auto dim : srcType.getShape()) {
            srcShape.push_back(consts.i32(dim));
        }
        SmallVector<Value> dstShape = getTensorShape(rewriter, tensorOp);
        auto const0 = consts.i32(0);
        auto offsets = op.getOffsets();
        Value linearOffset = linearizeOffset(rewriter, loc, dstShape, offsets);
        dst = rewriter.create<ascendc::GlobalTensorSubIndexOp>(loc, dstType, dst, linearOffset);
        Value typeSize = consts.i32(ascendc::getElementTypeSize(srcType));
        Value srcLastDim = srcShape[srcShape.size() - 1];
        Value dstLastDim = dstShape[dstShape.size() - 1];
        Value lastDimOffset = offsets.back();
        Value tailElementsLastDim = rewriter.create<arith::SubIOp>(loc, dstLastDim, lastDimOffset);
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
        rewriter.replaceOpWithNewOp<ascendc::DataCopyPadExtL2Op>(op, dst, src, dataCopyExtParams);
        return success();
    }
};

struct ConvertStoreFixpipe : ConvertOp<asctile::StoreFixpipeOp> {
    using ConvertOp::ConvertOp;
    using ConvertOp::createTensorOp;

    LogicalResult convert(asctile::StoreFixpipeOp op, ConvertRewriter& rewriter) const override
    {
        auto base = op.getBase();
        auto tensorOp = base.getDefiningOp<asctile::TensorOp>();
        assert(tensorOp && "tensor must be created by asctile.tensor op");
        auto loc = op.getLoc();
        auto value = op.getValue();
        Value src = rewriter.getRemappedValue(value);
        Value dst = rewriter.getRemappedValue(base);
        auto srcType = cast<ascendc::BaseTensorType>(src.getType());
        assert(value.getType().getLoc() == asctile::TileLocation::L0C && "Tile should be located in L0C.");
        auto dstType = cast<ascendc::BaseTensorType>(dst.getType());
        ascir::ConstantOpBuilder consts(rewriter);
        SmallVector<Value> srcShape;
        for (auto dim : srcType.getShape()) {
            srcShape.push_back(consts.i32(dim));
        }
        SmallVector<Value> dstShape = getTensorShape(rewriter, tensorOp);
        auto const0 = consts.i32(0);
        auto const1 = consts.i32(1);
        auto offsets = op.getOffsets();
        Value linearOffset = linearizeOffset(rewriter, loc, dstShape, offsets);
        dst = rewriter.create<ascendc::GlobalTensorSubIndexOp>(loc, dstType, dst, linearOffset);
        Value srcLastDim = srcShape[srcShape.size() - 1];
        Value dstLastDim = dstShape[dstShape.size() - 1];
        Value lastDimOffset = offsets.back();
        Value tailElementsLastDim = rewriter.create<arith::SubIOp>(loc, dstLastDim, lastDimOffset);
        auto tailNegCond = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, tailElementsLastDim, const0);
        Value tailElements = rewriter.create<arith::SelectOp>(loc, tailNegCond, const0, tailElementsLastDim);
        Value nSize = rewriter.create<arith::MinSIOp>(loc, srcLastDim, tailElements);
        auto srcStride = llvm::alignTo<ascendc::cubeBlockSize>(srcType.getShape()[0]);
        auto paramsBuilder = emitasc::InitStructBuilder(rewriter.getType<ascendc::FixpipeParamsV220Type>())
                                 .addField("nSize", nSize)
                                 .addField("mSize", srcShape[0])
                                 .addField("srcStride", consts.i32(srcStride))
                                 .addField("dstStride", dstShape[1]);
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
            ValueRange{consts.i32(static_cast<int32_t>(ascendc::CO2Layout::ROW_MAJOR))}, ArrayAttr{}, true, true);
        auto fixPipeConfig = rewriter.create<ascendc::ConstructOp>(
            loc, rewriter.getType<ascendc::FixpipeConfigType>(), ValueRange{layout}, ArrayAttr{}, true, true);
        rewriter.replaceOpWithNewOp<ascendc::FixpipeOp>(op, dst, src, params, fixPipeConfig);
        return success();
    }
};

struct ConvertCopyFixpipe : ConvertOp<asctile::CopyFixpipeOp> {
    using ConvertOp::ConvertOp;
    using ConvertOp::createTensorOp;

    LogicalResult convert(asctile::CopyFixpipeOp op, ConvertRewriter& rewriter) const override
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
        SmallVector<Value> srcShape;
        for (auto dim : srcType.getShape()) {
            srcShape.push_back(consts.i32(dim));
        }
        SmallVector<Value> dstShape;
        for (auto dim : dstType.getShape()) {
            dstShape.push_back(consts.i32(dim));
        }
        auto const1 = consts.i32(1);
        Value linearOffset = linearizeOffset(rewriter, loc, dstShape, op.getOffsets());
        src = rewriter.create<ascendc::LocalTensorSubIndexOp>(loc, srcType, src, linearOffset);
        auto dstStride = rewriter.create<arith::MulIOp>(
            loc, srcShape[0], consts.i32(CUBE_K_BLOCK_BYTES / ascendc::getElementTypeSize(op.getType())));
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
        auto fixPipeOp = rewriter.create<ascendc::FixpipeOp>(loc, dst, src, params, fixPipeConfig);
        rewriter.replaceOp(op, dst);
        return success();
    }
};

struct ConvertCopy : ConvertOp<asctile::CopyOp> {
    using ConvertOp::ConvertOp;
    using ConvertOp::createTensorOp;

    LogicalResult convert(asctile::CopyOp op, ConvertRewriter& rewriter) const override
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
        const int64_t cubeKBlockSize = CUBE_K_BLOCK_BYTES / ascendc::getElementTypeSize(opType);
        const int64_t cubeBlockRows = isTensorA ? ascendc::cubeBlockSize : cubeKBlockSize;
        const int64_t cubeBlockCols = isTensorA ? cubeKBlockSize : ascendc::cubeBlockSize;
        const int64_t fractalSize = cubeBlockRows * cubeBlockCols;
        // Note! For this formula it is assumed that offsets are divisible by corresponding Cube block dimensions.
        Value colOffset = rewriter.create<arith::MulIOp>(loc, consts.i32(srcShape[0]), offsets[1]);
        Value rowOffset = rewriter.create<arith::MulIOp>(loc, offsets[0], consts.i32(cubeBlockCols));
        Value linearOffset = rewriter.create<arith::AddIOp>(loc, colOffset, rowOffset);
        src = rewriter.create<ascendc::LocalTensorSubIndexOp>(loc, srcType, src, linearOffset);
        auto dstTensorOp = createTensorOp(rewriter, loc, opType);
        auto dst = dstTensorOp.getResult();
        auto dstType = dst.getType();
        auto dstShape = dstType.getShape();
        auto const0 = consts.i32(0);
        auto const1 = consts.i32(1);
        bool isTransposeA = op->hasAttrOfType<UnitAttr>(asctile::attr::transposeA);
        bool isTransposeB = op->hasAttrOfType<UnitAttr>(asctile::attr::transposeB);
        auto dstFracStride = llvm::divideCeilSigned(isTransposeB ? dstShape[0] : dstShape[1], cubeBlockCols);
        if (isTensorA || !isa<Float32Type>(opType.getElementType()) || isTransposeB) {
            // LoadData2DParams doesn't support ifTranspose for float32, LoadDataWithTranpose doesn't support TPosition
            // A2 on Ascend910_95. Used LoadData2DParamsV2
            if (isTransposeA && isa<Float32Type>(opType.getElementType())) {
                auto paramsType = rewriter.getType<ascendc::LoadData2DParamsV2Type>();
                auto mStep = consts.i32(llvm::divideCeilSigned(dstShape[1], ascendc::cubeBlockSize));
                auto kStep = consts.i32(
                    llvm::divideCeilSigned(llvm::divideCeilSigned(dstShape[0], cubeKBlockSize), FRACTAL_NUM) *
                    FRACTAL_NUM);
                auto dstStride = consts.i32(llvm::divideCeilSigned(dstShape[0], cubeKBlockSize * FRACTAL_NUM));
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
            int64_t repeatTimes = llvm::divideCeilSigned(dstShape[1], cubeBlockRows * FRACTAL_NUM);
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
            int64_t srcOffset = repeatTimes * fractalSize * FRACTAL_NUM;
            auto iterSrcOffset = rewriter.create<arith::MulIOp>(loc, indVar, consts.i32(srcOffset));
            auto subLocalL1 = rewriter.create<ascendc::LocalTensorSubIndexOp>(loc, srcType, src, iterSrcOffset);
            int64_t dstOffset = dstFracStride * fractalSize * FRACTAL_NUM;
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

    LogicalResult convert(asctile::GetValueOp op, ConvertRewriter& rewriter) const override
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

    LogicalResult convert(asctile::SetValueOp op, ConvertRewriter& rewriter) const override
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
            ConvertLoad, ConvertStore, ConvertStoreFixpipe, ConvertCopy, ConvertGetValue, ConvertSetValue,
            ConvertCopyFixpipe
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
