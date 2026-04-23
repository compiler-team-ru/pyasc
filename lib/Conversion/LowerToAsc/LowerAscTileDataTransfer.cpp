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

constexpr int CUBE_MN_BLOCK_SIZE = 16;
constexpr int CUBE_K_BLOCK_BYTES = 32;
constexpr int FRACTAL_NUM = 2;

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
    assert(tensorShape.size() <= 2 && "supported only tensorShape with dims <= 2");
    Value linearOffset = consts.i32(0);
    Value stride = consts.i32(1);
    for (size_t i = tensorShape.size(); i-- > 0;) {
        Value next = builder.create<arith::MulIOp>(loc, offsets[i], stride);
        linearOffset = builder.create<arith::AddIOp>(loc, linearOffset, next);
        stride = builder.create<arith::MulIOp>(loc, tensorShape[i], stride);
    }
    return linearOffset;
}

Value linearizeNzOffset(
    OpBuilder& builder, Location loc, ArrayRef<Value> tensorShape, ValueRange offsets, Value blockSize)
{
    ascir::ConstantOpBuilder consts(builder);
    assert(offsets.size() == tensorShape.size() && "must be one offset for each dimension");
    assert(tensorShape.size() <= 2 && "supported only tensorShape with dims <= 2");
    Value ZBlockSize = builder.create<arith::MulIOp>(loc, tensorShape[0], blockSize);
    Value fullZBlocks = builder.create<arith::DivSIOp>(loc, offsets[1], blockSize);
    Value linearOffset = builder.create<arith::MulIOp>(loc, ZBlockSize, fullZBlocks);
    Value lastZSubBlockSize = builder.create<arith::MulIOp>(loc, offsets[0], blockSize);
    linearOffset = builder.create<arith::AddIOp>(loc, linearOffset, lastZSubBlockSize);
    Value lastZBlockColOffset = builder.create<arith::RemSIOp>(loc, offsets[1], blockSize);
    linearOffset = builder.create<arith::AddIOp>(loc, linearOffset, lastZBlockColOffset);
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

std::optional<ascendc::QuantMode>
getQuantizeMode(ascendc::BaseTensorType srcType, ascendc::BaseTensorType dstType, ConvertRewriter& rewriter)
{
    auto srcElType = srcType.getElementType();
    auto dstElType = dstType.getElementType();
    auto floatType = rewriter.getF32Type();
    auto halfType = rewriter.getF16Type();
    auto int32Type = rewriter.getIntegerType(32);
    auto int8Type = rewriter.getIntegerType(8);
    auto uint8Type = rewriter.getIntegerType(8, false);
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
        auto dstTensorOp = createTensorOp(rewriter, loc, opType, locationToPosition(dstLoc));
        auto dst = dstTensorOp.getResult();
        auto dstType = dst.getType();
        auto dstShape = dstType.getShape();
        ascir::ConstantOpBuilder consts(rewriter);
        auto const0 = consts.i32(0);
        auto const1 = consts.i32(1);
        if (dstLoc == asctile::TileLocation::L1) {
            const int64_t cubeKBlockSize = CUBE_K_BLOCK_BYTES / ascendc::getElementTypeSize(opType);
            const int64_t cubeRowBlock = (op->hasAttr(asctile::attr::isMatrixA)) ? CUBE_MN_BLOCK_SIZE : cubeKBlockSize;
            auto dValue = consts.i32(dstShape[1]);
            if (op->hasAttr(asctile::attr::isMatrixA) || isa<Float16Type, BFloat16Type>(opType.getElementType())) {
                int64_t dstNzC0Stride = llvm::divideCeilSigned(dstShape[0], cubeRowBlock) * cubeRowBlock;
                auto nd2NzParams = rewriter.create<ascendc::ConstructOp>(
                    loc, rewriter.getType<ascendc::Nd2NzParamsType>(),
                    ValueRange{
                        const1, consts.i32(dstShape[0]), dValue, const0, srcShape[1], consts.i32(dstNzC0Stride), const1,
                        const0});
                rewriter.create<ascendc::DataCopyL2Op>(loc, dst, src, nd2NzParams);
            } else {
                int64_t ndNum = llvm::divideCeilSigned(dstShape[0], CUBE_MN_BLOCK_SIZE);
                auto nValue = consts.i32(CUBE_MN_BLOCK_SIZE);
                auto srcNdMatrixStride = rewriter.create<arith::MulIOp>(loc, nValue, dValue);
                int64_t fractal = cubeKBlockSize * FRACTAL_NUM;
                int64_t ceilAlignFractal = llvm::divideCeilSigned(dstShape[1], fractal) * fractal;
                auto dstNzMatrixStride = rewriter.create<arith::MulIOp>(loc, nValue, consts.i32(ceilAlignFractal));
                auto nd2NzParams = rewriter.create<ascendc::ConstructOp>(
                    loc, rewriter.getType<ascendc::Nd2NzParamsType>(),
                    ValueRange{
                        consts.i32(ndNum), nValue, dValue, srcNdMatrixStride, dValue, nValue, const1,
                        dstNzMatrixStride});
                rewriter.create<ascendc::DataCopyL2Op>(loc, dst, src, nd2NzParams);
            }
        } else {
            auto padValue = rewriter.getRemappedValue(op.getPadValue());
            auto typeSize = ascendc::getElementTypeSize(dstType);
            auto numElements = calculateNumElements(rewriter, loc, srcShape);
            Value tailElements = rewriter.create<arith::SubIOp>(loc, numElements, linearOffset);
            Value blockCount = dstShape.size() == 1 ? const1 : consts.i32(dstShape[0]);
            Value dstLastDim = consts.i32(dstShape[dstShape.size() - 1]);
            Value srcLastDim = srcShape[srcShape.size() - 1];
            Value strideElements = rewriter.create<arith::SubIOp>(loc, srcLastDim, dstLastDim);
            auto typeSizeValue = consts.i32(typeSize);
            Value srcStride = rewriter.create<arith::MulIOp>(loc, strideElements, typeSizeValue);
            Value numElementsInBlock = consts.i32(ascendc::ubBlockSize / typeSize);
            Value totalElementsInBlock = rewriter.create<arith::MulIOp>(loc, blockCount, numElementsInBlock);
            Value minTailElements = rewriter.create<arith::MinSIOp>(loc, dstLastDim, tailElements);
            Value blockLen = rewriter.create<arith::MulIOp>(loc, minTailElements, typeSizeValue);
            auto context = op.getContext();
            auto ui32Type = rewriter.getIntegerType(32, false);
            auto dataCopyExtParams = rewriter.create<ascendc::ConstructOp>(
                loc, rewriter.getType<ascendc::DataCopyExtParamsType>(),
                ValueRange{blockCount, blockLen, srcStride, const0, const0},
                rewriter.getTypeArrayAttr(
                    {rewriter.getIntegerType(16, false), ui32Type, ui32Type, ui32Type, ui32Type}));
            Value numPaddingElements = rewriter.create<arith::SubIOp>(loc, totalElementsInBlock, tailElements);
            Value rightPad = rewriter.create<arith::MaxSIOp>(loc, numPaddingElements, const0);
            auto dataCopyPadExtParams = rewriter.create<ascendc::ConstructOp>(
                loc, ascendc::DataCopyPadExtParamsType::get(context, cast<ShapedType>(dstType).getElementType()),
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
        Value linearOffset = linearizeOffset(rewriter, loc, dstShape, op.getOffsets());
        dst = rewriter.create<ascendc::GlobalTensorSubIndexOp>(loc, dstType, dst, linearOffset);
        auto numElements = calculateNumElements(rewriter, loc, dstShape);
        auto typeSize = ascendc::getElementTypeSize(srcType);
        auto typeSizeValue = consts.i32(typeSize);
        Value srcNumElements = consts.i32(calCount(src));
        Value tailElements = rewriter.create<arith::SubIOp>(loc, numElements, linearOffset);
        Value blockCount = srcShape.size() == 1 ? consts.i32(1) : srcShape[0];
        Value srcLastDim = srcShape[srcShape.size() - 1];
        Value dstLastDim = dstShape[dstShape.size() - 1];
        Value strideElements = rewriter.create<arith::SubIOp>(loc, dstLastDim, srcLastDim);
        Value dstStride = rewriter.create<arith::MulIOp>(loc, strideElements, typeSizeValue);
        Value minTailElements = rewriter.create<arith::MinSIOp>(loc, srcLastDim, tailElements);
        Value blockLen = rewriter.create<arith::MulIOp>(loc, minTailElements, typeSizeValue);
        auto context = op.getContext();
        auto ui32Type = rewriter.getIntegerType(32, false);
        auto dataCopyExtParams = rewriter.create<ascendc::ConstructOp>(
            loc, rewriter.getType<ascendc::DataCopyExtParamsType>(),
            ValueRange{blockCount, blockLen, const0, dstStride, const0},
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
        auto const1 = consts.i32(1);
        Value linearOffset = linearizeOffset(rewriter, loc, dstShape, op.getOffsets());
        dst = rewriter.create<ascendc::GlobalTensorSubIndexOp>(loc, dstType, dst, linearOffset);
        auto paramsBuilder = emitasc::InitStructBuilder(rewriter.getType<ascendc::FixpipeParamsV220Type>())
                                 .addField("nSize", srcShape[1])
                                 .addField("mSize", srcShape[0])
                                 .addField("srcStride", srcShape[0])
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
        ascir::ConstantOpBuilder consts(rewriter);
        SmallVector<Value> srcShape;
        for (auto dim : base.getType().getShape()) {
            srcShape.push_back(consts.i32(dim));
        }
        const auto cubeKBlockSize = CUBE_K_BLOCK_BYTES / ascendc::getElementTypeSize(opType);
        const auto cubeBlock = CUBE_MN_BLOCK_SIZE * cubeKBlockSize;
        const auto cubeRowBlock = (dstPos == asctile::TileLocation::L0A) ? CUBE_MN_BLOCK_SIZE : cubeKBlockSize;
        Value linearOffset = linearizeNzOffset(rewriter, loc, srcShape, op.getOffsets(), consts.i32(cubeRowBlock));
        src = rewriter.create<ascendc::LocalTensorSubIndexOp>(loc, srcType, src, linearOffset);
        auto dstTensorOp = createTensorOp(rewriter, loc, opType);
        auto dst = dstTensorOp.getResult();
        auto dstType = dst.getType();
        auto dstShape = dstType.getShape();
        auto const0 = consts.i32(0);
        auto const1 = consts.i32(1);
        auto paramsType = rewriter.getType<ascendc::LoadData2DParamsType>();
        if (dstPos == asctile::TileLocation::L0A) {
            auto dstNumElements = calCount(dst);
            Value repeatTimes = consts.i32(llvm::divideCeilSigned(dstNumElements, cubeBlock));
            Value params = emitasc::InitStructBuilder(paramsType)
                               .addField("repeatTimes", repeatTimes)
                               .addField("srcStride", const1)
                               .addField("dstGap", const0)
                               .addField("ifTranspose", consts.i1(0))
                               .create(rewriter, loc);
            rewriter.create<ascendc::LoadDataG2LOp>(loc, dst, src, params);
        } else if (dstPos == asctile::TileLocation::L0B) {
            auto dstFracGap = llvm::divideCeilSigned(dstShape[1], CUBE_MN_BLOCK_SIZE);
            if (isa<Float32Type>(opType.getElementType())) {
                auto paramsTransposeType = rewriter.getType<ascendc::LoadData2dTransposeParamsType>();
                int64_t fractalSize = CUBE_MN_BLOCK_SIZE * cubeKBlockSize;
                int64_t dstOffset = dstFracGap * fractalSize * FRACTAL_NUM;
                int64_t repeatTimes = llvm::divideCeilSigned(dstShape[1], cubeKBlockSize * FRACTAL_NUM);
                int64_t srcOffset = repeatTimes * fractalSize * FRACTAL_NUM;
                Value params = emitasc::InitStructBuilder(paramsTransposeType)
                                   .addField("repeatTimes", consts.i32(repeatTimes))
                                   .addField("srcStride", const1)
                                   .addField("dstGap", const0)
                                   .addField("dstFracGap", consts.i32(dstFracGap - 1))
                                   .create(rewriter, loc);
                Value uBound = consts.i32(llvm::divideCeilSigned(dstShape[0], CUBE_MN_BLOCK_SIZE));
                auto forOp = rewriter.create<scf::ForOp>(loc, const0, uBound, const1);
                rewriter.setInsertionPointToStart(forOp.getBody());
                auto indVar = forOp.getInductionVar();
                auto iterDstOffset = rewriter.create<arith::MulIOp>(loc, indVar, consts.i32(dstOffset));
                auto iterSrcOffset = rewriter.create<arith::MulIOp>(loc, indVar, consts.i32(srcOffset));
                auto subLocalL1 = rewriter.create<ascendc::LocalTensorSubIndexOp>(loc, srcType, src, iterSrcOffset);
                auto subLocalL0 = rewriter.create<ascendc::LocalTensorSubIndexOp>(loc, dstType, dst, iterDstOffset);
                rewriter.create<ascendc::LoadDataWithTransposeOp>(loc, subLocalL0, subLocalL1, params);
                rewriter.setInsertionPointAfter(forOp);
                forOp->setAttr(asctile::attr::parallel, UnitAttr::get(forOp->getContext()));
            } else {
                Value repeatTimes = consts.i32(dstFracGap);
                Value srcStride = rewriter.create<arith::CeilDivSIOp>(loc, srcShape[0], consts.i32(cubeKBlockSize));
                Value params = emitasc::InitStructBuilder(paramsType)
                                   .addField("repeatTimes", repeatTimes)
                                   .addField("srcStride", srcStride)
                                   .addField("dstGap", const0)
                                   .addField("ifTranspose", consts.i1(1))
                                   .create(rewriter, loc);
                Value uBound = consts.i32(llvm::divideCeilSigned(dstShape[0], cubeKBlockSize));
                auto forOp = rewriter.create<scf::ForOp>(loc, const0, uBound, const1);
                rewriter.setInsertionPointToStart(forOp.getBody());
                auto indVar = forOp.getInductionVar();
                auto cubeBlockSize = consts.i32(cubeBlock);
                auto dstOffset = rewriter.create<arith::MulIOp>(loc, cubeBlockSize, repeatTimes);
                auto srcOffset = cubeBlockSize;
                auto iterDstOffset = rewriter.create<arith::MulIOp>(loc, indVar, dstOffset);
                auto iterSrcOffset = rewriter.create<arith::MulIOp>(loc, indVar, srcOffset);
                auto subLocalL1 = rewriter.create<ascendc::LocalTensorSubIndexOp>(loc, srcType, src, iterSrcOffset);
                auto subLocalL0 = rewriter.create<ascendc::LocalTensorSubIndexOp>(loc, dstType, dst, iterDstOffset);
                rewriter.create<ascendc::LoadDataG2LOp>(loc, subLocalL0, subLocalL1, params);
                rewriter.setInsertionPointAfter(forOp);
                forOp->setAttr(asctile::attr::parallel, UnitAttr::get(forOp->getContext()));
            }
        } else {
            return op.emitError() << "dst tile location is not supported";
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
