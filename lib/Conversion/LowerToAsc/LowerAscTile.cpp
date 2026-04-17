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
#define GEN_PASS_DEF_LOWERASCTILE
#include "ascir/Conversion/LowerToAsc/Passes.h.inc"
} // namespace asclower
} // namespace mlir

using namespace mlir;
using namespace mlir::asclower;

namespace {

constexpr int ONE_BLK_FLOAT_NUM = 8;
constexpr int ONE_BLK_SIZE = 32;
constexpr int TOTAL_UB_SIZE = 256 * 1024;
constexpr int MAX_REPEAT = 255;
constexpr int BASIC_BLK_BSLENGTH = 8;
constexpr int HALF_SIZE_IN_BYTE = 2;
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

unsigned int calculateFinalTensorSize(const unsigned int& typeSize, int64_t calCount)
{
    unsigned int elementsPerBlock = ascendc::ubBlockSize / typeSize;
    unsigned int elementsPerRepeat = ascendc::repeatBlockSize / typeSize;
    unsigned int firstMaxRepeat = calCount / elementsPerRepeat;
    return llvm::divideCeilSigned(firstMaxRepeat, elementsPerBlock) * elementsPerBlock;
}

std::pair<int, int> unpack2DShape(ArrayRef<int64_t> shape)
{
    assert(shape.size() == 1 || shape.size() == 2);
    return {shape.size() == 2 ? shape[0] : 1, shape.back()};
}

bool check1D2DShape(Operation* op, ArrayRef<int64_t> shape)
{
    if (shape.size() != 1 && shape.size() != 2) {
        op->emitError() << "invalid dimension of input tensor";
        return false;
    }
    return true;
}

struct LoweringConversionTarget : public ConversionTarget {
    LoweringConversionTarget(TensorTypeConverter& converter, MLIRContext* context) : ConversionTarget(*context)
    {
        addIllegalOp<
            //
            asctile::TensorOp, asctile::LoadOp, asctile::GetValueOp, asctile::StoreOp, asctile::SetValueOp,
            asctile::SplatOp, asctile::ReluOp, asctile::CastOp, asctile::SoftmaxOp, asctile::MatmulOp,
            asctile::ReshapeOp, asctile::BroadcastOp, asctile::AddSOp, asctile::SubSOp, asctile::MulSOp,
            asctile::DivSOp, asctile::MinSOp, asctile::MaxSOp, asctile::ShLSOp, asctile::ShRSOp, asctile::ReduceAs1dOp,
            asctile::ReduceOp, asctile::AccumulatorOp, asctile::MatmulAccOp, asctile::CopyOp, asctile::StoreFixpipeOp
            //
            >();
        addLegalDialect<
            ascendc::AscendCDialect, arith::ArithDialect, emitasc::EmitAscDialect, emitc::EmitCDialect,
            scf::SCFDialect>();
        addLegalOp<UnrealizedConversionCastOp>();
    }
};

struct ConvertTensor : public ConvertOp<asctile::TensorOp> {
    using ConvertOp::converter;
    using ConvertOp::ConvertOp;

    LogicalResult convert(asctile::TensorOp op, ConvertRewriter& rewriter) const override
    {
        auto loc = op.getLoc();
        Value tensor = rewriter.create<ascendc::GlobalTensorOp>(loc, converter().convertType(op.getType()));
        rewriter.create<ascendc::GlobalTensorSetGlobalBufferOp>(loc, tensor, op.getBase(), /*size*/ Value{});
        rewriter.replaceOp(op, tensor);
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

struct ConvertSplat : public ConvertOp<asctile::SplatOp> {
    using ConvertOp::converter;
    using ConvertOp::ConvertOp;

    LogicalResult convert(asctile::SplatOp op, ConvertRewriter& rewriter) const override
    {
        ascir::ConstantOpBuilder consts(rewriter);
        auto loc = op.getLoc();
        Value dst = createTensorOp(rewriter, loc, op.getType());
        rewriter.create<ascendc::DuplicateL2Op>(loc, dst, op.getValue(), consts.i64(calCount(dst)));
        rewriter.replaceOp(op, dst);
        return success();
    }
};

struct ConvertRelu : ConvertOp<asctile::ReluOp> {
    using ConvertOp::ConvertOp;
    using ConvertOp::createTensorOp;

    LogicalResult convert(asctile::ReluOp op, ConvertRewriter& rewriter) const override
    {
        ascir::ConstantOpBuilder consts(rewriter);
        Location loc = op.getLoc();
        Value dst = createTensorOp(rewriter, loc, op.getType());
        Value src = rewriter.getRemappedValue(op.getOperand());
        rewriter.create<ascendc::ReluL2Op>(loc, dst, src, consts.i64(calCount(dst)));
        rewriter.replaceOp(op, dst);
        return success();
    }
};

struct ConvertCast : ConvertOp<asctile::CastOp> {
    using ConvertOp::ConvertOp;
    using ConvertOp::createTensorOp;

    static Type getElementType(Value shaped) { return cast<ShapedType>(shaped.getType()).getElementType(); }

    static ascendc::RoundMode getRoundMode(Type in, Type out)
    {
        if (isa<FloatType>(in) && isa<IntegerType>(out))
            return ascendc::RoundMode::CAST_TRUNC;
        return ascendc::RoundMode::CAST_NONE;
    }

    LogicalResult convert(asctile::CastOp op, ConvertRewriter& rewriter) const override
    {
        Location loc = op.getLoc();
        Value dst = createTensorOp(rewriter, loc, op.getType());
        ascir::ConstantOpBuilder consts(rewriter);
        Value src = rewriter.getRemappedValue(op.getIn());
        auto roundMode = getRoundMode(getElementType(src), getElementType(dst));
        rewriter.create<ascendc::CastL2Op>(loc, dst, src, roundMode, consts.i64(calCount(dst)));
        rewriter.replaceOp(op, dst);
        return success();
    }
};

struct ConvertSoftmax : ConvertOp<asctile::SoftmaxOp> {
    using ConvertOp::ConvertOp;
    using ConvertOp::createTensorOp;

    LogicalResult convert(asctile::SoftmaxOp op, ConvertRewriter& rewriter) const override
    {
        ascir::ConstantOpBuilder consts(rewriter);
        auto loc = op.getLoc();
        auto tensorType = op.getType();
        auto shape = tensorType.getShape();
        if (!check1D2DShape(op, shape))
            return failure();
        auto [height, width] = unpack2DShape(shape);
        auto src = rewriter.getRemappedValue(op.getOperand());
        auto dst = createTensorOp(rewriter, loc, tensorType);
        auto elemType = tensorType.getElementType();
        int64_t bufferSize = ascendc::ubBlockSize * height / ascendc::getElementTypeSize(tensorType);
        auto maxTensor = createTensorOp(rewriter, loc, bufferSize, elemType);
        auto sumTensor = createTensorOp(rewriter, loc, bufferSize, elemType);
        auto sharedBufTensor =
            createTensorOp(rewriter, loc, ascendc::getTypeSize(tensorType) * 2, rewriter.getIntegerType(8, false));
        auto tiling = rewriter.create<ascendc::ConstructOp>(loc, rewriter.getType<ascendc::SoftMaxTilingType>());
        Value shapeInfo = emitasc::InitStructBuilder(rewriter.getType<ascendc::SoftMaxShapeInfoType>())
                              .addField("srcM", consts.i32(height))
                              .addField("srcK", consts.i32(width))
                              .addField("oriSrcM", consts.i32(height))
                              .addField("oriSrcK", consts.i32(width))
                              .create(rewriter, loc);
        rewriter.create<ascendc::SoftMaxOp>(
            loc, false, false, false, dst, maxTensor, sumTensor, src, sharedBufTensor, tiling, shapeInfo);
        rewriter.replaceOp(op, dst);
        return success();
    }
};

struct ConvertRmsNorm : ConvertOp<asctile::RmsNormOp> {
    using ConvertOp::ConvertOp;
    using ConvertOp::createTensorOp;

    struct RmsNormTiling {
        int bLength;
        int sLength;
        int hLength;
        int originalHLength;
        float reciprocalOfHLength;
        int mainBshLength;
        int mainBsLength;
        int mainBsLengthAlign;
        int loopRound;
        int tailBshLength;
        int inputTailPos;
        int tailBsLength;
    };

    static int alignToBlock(const int inputValue, const int typeSize)
    {
        int alignUnit = ONE_BLK_SIZE / typeSize;
        return (inputValue + alignUnit - 1) / alignUnit * alignUnit;
    }

    // TODO: Refactor
    static RmsNormTiling getRmsNormTiling(ArrayRef<int64_t> shape, bool isBasicBlock, int typeSize)
    {
        auto [bLength, inHLength] = unpack2DShape(shape);
        auto sLength = 1;
        auto hLength = alignToBlock(inHLength, typeSize);
        auto bshLength = bLength * sLength * hLength;
        auto originalHLength = inHLength;
        auto reciprocalOfHLength = 1.0f / originalHLength;
        auto oneTmpSize = TOTAL_UB_SIZE / typeSize;
        auto alignBsLength = ONE_BLK_FLOAT_NUM;
        auto halfCoeff = (typeSize == sizeof(float) ? 1u : 2u);
        while (oneTmpSize > alignBsLength * hLength * halfCoeff + alignBsLength) {
            alignBsLength += ONE_BLK_FLOAT_NUM;
        }
        alignBsLength = alignBsLength == ONE_BLK_FLOAT_NUM ? ONE_BLK_FLOAT_NUM : alignBsLength - ONE_BLK_FLOAT_NUM;
        oneTmpSize =
            (typeSize == HALF_SIZE_IN_BYTE) ? (oneTmpSize - alignBsLength) / halfCoeff : (oneTmpSize - alignBsLength);
        auto inputXSize = bLength * sLength * hLength;
        if (oneTmpSize > inputXSize) {
            oneTmpSize = inputXSize;
        }
        auto bsLength = oneTmpSize / hLength;
        if (isBasicBlock) {
            bsLength = bsLength < BASIC_BLK_BSLENGTH ? 1 : bsLength / BASIC_BLK_BSLENGTH * BASIC_BLK_BSLENGTH;
        } else if (bsLength > MAX_REPEAT) {
            bsLength = MAX_REPEAT;
        }
        oneTmpSize = bsLength * hLength;
        auto mainBshLength = oneTmpSize;
        auto mainBsLength = oneTmpSize / hLength;
        auto mainBsLengthAlign = alignToBlock(oneTmpSize / hLength, typeSize);
        auto loopRound = inputXSize / oneTmpSize;
        auto inputTailSize = inputXSize % oneTmpSize;
        auto tailBshLength = inputTailSize;
        auto inputTailPos = inputXSize - inputTailSize;
        auto tailBsLength = inputTailSize / hLength;
        return {bLength,      sLength,           hLength,   originalHLength, reciprocalOfHLength, mainBshLength,
                mainBsLength, mainBsLengthAlign, loopRound, tailBshLength,   inputTailPos,        tailBsLength};
    }

    LogicalResult convert(asctile::RmsNormOp op, ConvertRewriter& rewriter) const override
    {
        ascir::ConstantOpBuilder consts(rewriter);
        auto loc = op.getLoc();
        auto tensorType = op.getType();
        auto shape = tensorType.getShape();
        if (!check1D2DShape(op, shape))
            return failure();
        auto src = rewriter.getRemappedValue(op.getInput());
        auto gammaTensor = rewriter.getRemappedValue(op.getGamma());
        auto epsilon = rewriter.getRemappedValue(op.getEpsilon());
        auto dst = createTensorOp(rewriter, loc, tensorType);
        constexpr bool isBasicBlock = false;
        RmsNormTiling tilingStruct = getRmsNormTiling(shape, isBasicBlock, ascendc::getElementTypeSize(tensorType));
        auto sharedBufTensor = createTensorOp(rewriter, loc, TOTAL_UB_SIZE, rewriter.getIntegerType(8, false));
        Value tiling = emitasc::InitStructBuilder(rewriter.getType<ascendc::RmsNormTilingType>())
                           .addField("bLength", consts.i32(tilingStruct.bLength))
                           .addField("sLength", consts.i32(tilingStruct.sLength))
                           .addField("hLength", consts.i32(tilingStruct.hLength))
                           .addField("originalHLength", consts.i32(tilingStruct.originalHLength))
                           .addField("reciprocalOfHLength", consts.f32(tilingStruct.reciprocalOfHLength))
                           .addField("mainBshLength", consts.i32(tilingStruct.mainBshLength))
                           .addField("mainBsLength", consts.i32(tilingStruct.mainBsLength))
                           .addField("mainBsLengthAlign", consts.i32(tilingStruct.mainBsLengthAlign))
                           .addField("loopRound", consts.i32(tilingStruct.loopRound))
                           .addField("tailBshLength", consts.i32(tilingStruct.tailBshLength))
                           .addField("inputTailPos", consts.i32(tilingStruct.inputTailPos))
                           .addField("tailBsLength", consts.i32(tilingStruct.tailBsLength))
                           .create(rewriter, loc);
        rewriter.create<ascendc::RmsNormOp>(loc, isBasicBlock, dst, src, gammaTensor, epsilon, tiling, sharedBufTensor);
        rewriter.replaceOp(op, dst);
        return success();
    }
};

struct ConvertMatmul : ConvertOp<asctile::MatmulOp> {
    using ConvertOp::ConvertOp;
    using ConvertOp::createTensorOp;

    LogicalResult convert(asctile::MatmulOp op, ConvertRewriter& rewriter) const override
    {
        ascir::ConstantOpBuilder consts(rewriter);
        auto loc = op.getLoc();
        auto dst = createTensorOp(rewriter, loc, op.getType());
        auto matrixA = rewriter.getRemappedValue(op.getMatrixA());
        auto matrixB = rewriter.getRemappedValue(op.getMatrixB());
        auto matrixATensorShape = cast<ascendc::LocalTensorType>(matrixA.getType()).getShape();
        auto matrixBTensorShape = cast<ascendc::LocalTensorType>(matrixB.getType()).getShape();
        assert(matrixATensorShape.size() == 2 && "matrix must be have dim = 2");
        assert(matrixBTensorShape.size() == 2 && "matrix must be have dim = 2");
        auto mmadParams = emitasc::InitStructBuilder(rewriter.getType<ascendc::MmadParamsType>())
                              .addField("m", consts.i32(matrixATensorShape[0]))
                              .addField("n", consts.i32(matrixBTensorShape[1]))
                              .addField("k", consts.i32(matrixBTensorShape[0]))
                              .create(rewriter, loc);
        rewriter.create<ascendc::MmadOp>(loc, dst, matrixA, matrixB, mmadParams);
        rewriter.replaceOp(op, dst);
        return success();
    }
};

struct ConvertReshape : ConvertOp<asctile::ReshapeOp> {
    using ConvertOp::ConvertOp;
    using ConvertOp::createTensorOp;

    LogicalResult convert(asctile::ReshapeOp op, ConvertRewriter& rewriter) const override
    {
        Location loc = op.getLoc();
        Value dst = createTensorOp(rewriter, loc, op.getType());
        Value src = rewriter.getRemappedValue(op.getIn());
        ascir::ConstantOpBuilder consts(rewriter);
        rewriter.create<ascendc::DataCopyL2Op>(loc, dst, src, consts.i64(calCount(dst)));
        rewriter.replaceOp(op, dst);
        return success();
    }
};

struct ConvertBroadcast : ConvertOp<asctile::BroadcastOp> {
    using ConvertOp::ConvertOp;
    using ConvertOp::createTensorOp;

    LogicalResult convert(asctile::BroadcastOp op, ConvertRewriter& rewriter) const override
    {
        ascir::ConstantOpBuilder consts(rewriter);
        auto loc = op.getLoc();
        auto dstType = op.getResult().getType();
        auto src = rewriter.getRemappedValue(op.getOperand());
        auto dst = createTensorOp(rewriter, loc, dstType.getShape(), dstType.getElementType());
        auto srcType = op.getOperand().getType();
        if (srcType.getNumElements() == 1) {
            Value dupSrc = src;
            if (!ascendc::isTargetPlatform95(op)) {
                dupSrc =
                    rewriter.create<ascendc::LocalTensorGetValueOp>(loc, srcType.getElementType(), src, consts.i64(0));
            }
            rewriter.create<ascendc::DuplicateL2Op>(loc, dst, dupSrc, consts.i64(0));
            rewriter.replaceOp(op, dst);
            return success();
        }
        auto srcShapeVec = srcType.getShape();
        auto dstShapeVec = dstType.getShape();
        if (srcShapeVec.size() > dstShapeVec.size() || srcShapeVec.size() == 0 || dstShapeVec.size() == 0)
            return op.emitError("Incompatible tensor shapes for Broadcast: [")
                .append(srcShapeVec)
                .append("] and [")
                .append(dstShapeVec)
                .append("]");
        SmallVector<Value> dstShape, srcShape;
        // Workaround: when dim<3 old Broadcast algorithm used (it need aligned data)
        for (size_t i = 0; i < 3U - dstShapeVec.size(); ++i) {
            dstShape.push_back(consts.i32(1));
        }
        // Pad srcShape with `1` to match dstShape
        for (size_t i = srcShapeVec.size(); i < dstShapeVec.size() + dstShape.size(); ++i) {
            srcShape.push_back(consts.i32(1));
        }
        for (size_t i = 0; i < srcShapeVec.size(); ++i) {
            srcShape.push_back(consts.i32(srcShapeVec[i]));
        }
        for (size_t i = 0; i < dstShapeVec.size(); ++i) {
            dstShape.push_back(consts.i32(dstShapeVec[i]));
        }
        assert(srcShape.size() == dstShape.size());
        rewriter.create<ascendc::BroadcastOp>(loc, dst, src, dstShape, srcShape);
        rewriter.replaceOp(op, dst);
        return success();
    }
};

template <typename TileOp, typename L2Op>
struct ConvertToL2 : ConvertOp<TileOp> {
    using ConvertOp<TileOp>::ConvertOp;
    using ConvertOp<TileOp>::calCount;
    using ConvertOp<TileOp>::createTensorOp;

    LogicalResult convert(TileOp op, ConvertRewriter& rewriter) const override
    {
        ascir::ConstantOpBuilder consts(rewriter);
        Location loc = op.getLoc();
        Value dst = createTensorOp(rewriter, loc, op.getType());
        Value lhs = rewriter.getRemappedValue(op->getOperand(0));
        Value rhs = rewriter.getRemappedValue(op->getOperand(1));
        rewriter.create<L2Op>(loc, dst, lhs, rhs, consts.i64(calCount(dst)));
        rewriter.replaceOp(op, dst);
        return success();
    }
};

template <typename TileOp, typename VecScalarOp, typename VectorOp = void>
struct ConvertVecScalarToL2 : ConvertOp<TileOp> {
    using ConvertOp<TileOp>::ConvertOp;
    using ConvertOp<TileOp>::calCount;
    using ConvertOp<TileOp>::createTensorOp;

    LogicalResult convert(TileOp op, ConvertRewriter& rewriter) const override
    {
        ascir::ConstantOpBuilder consts(rewriter);
        Location loc = op.getLoc();

        Value dst = createTensorOp(rewriter, loc, op.getType());
        Value lhs = rewriter.getRemappedValue(op->getOperand(0));
        Value rhs = rewriter.getRemappedValue(op->getOperand(1));
        constexpr bool requirePlatform95 = !std::is_same_v<VectorOp, void>;
        if (requirePlatform95 && !ascendc::isTargetPlatform95(op)) {
            Value dup = createTensorOp(rewriter, loc, op.getType());
            rewriter.create<ascendc::DuplicateL2Op>(loc, dup, rhs, consts.i64(calCount(dst)));
            rewriter.create<VectorOp>(loc, dst, lhs, dup, consts.i64(calCount(dst)));
        } else {
            rewriter.create<VecScalarOp>(loc, dst, lhs, rhs, consts.i64(calCount(dst)));
        }
        rewriter.replaceOp(op, dst);

        return success();
    }
};

struct ConvertReduceAs1d : ConvertOp<asctile::ReduceAs1dOp> {
    using ConvertOp::calCount;
    using ConvertOp::ConvertOp;
    using ConvertOp::createTensorOp;

    LogicalResult convert(asctile::ReduceAs1dOp op, ConvertRewriter& rewriter) const override
    {
        Type elemType = getElementTypeOrSelf(op.getType());
        unsigned int typeSize = ascendc::getTypeSize(elemType);
        unsigned int finalSize = calculateFinalTensorSize(typeSize, calCount(op.getOperand()));
        ascir::ConstantOpBuilder consts(rewriter);
        Location loc = op.getLoc();
        Value dst = createTensorOp(rewriter, loc, 1, elemType);
        Value src = rewriter.getRemappedValue(op.getOperand());
        Value tmpBuff = createTensorOp(rewriter, loc, static_cast<int64_t>(finalSize), elemType);
        Value count = consts.i64(calCount(op.getOperand()));
        auto kind = op.getKind();
        if (kind == asctile::ReduceKind::Sum)
            rewriter.create<ascendc::ReduceSumL2Op>(loc, dst, src, tmpBuff, count);
        else if (kind == asctile::ReduceKind::Max)
            rewriter.create<ascendc::ReduceMaxL2Op>(loc, dst, src, tmpBuff, count, consts.i64(0));
        else if (kind == asctile::ReduceKind::Min)
            rewriter.create<ascendc::ReduceMinL2Op>(loc, dst, src, tmpBuff, count, consts.i64(0));
        else
            return op.emitOpError() << "with " << asctile::stringifyReduceKind(kind) << " is not supported";
        if (isa<asctile::TileType>(op.getType()))
            rewriter.replaceOp(op, dst);
        else
            rewriter.replaceOpWithNewOp<ascendc::LocalTensorGetValueOp>(op, elemType, dst, consts.i64(0));
        return success();
    }
};

struct ConvertReduce : ConvertOp<asctile::ReduceOp> {
    using ConvertOp::ConvertOp;
    using ConvertOp::createTensorOp;

    static std::optional<ascendc::ReducePattern> findPattern(size_t length, uint64_t mask)
    {
        struct Pattern {
            size_t length;
            uint64_t mask;
            ascendc::ReducePattern value;
        };
        static constexpr Pattern patterns[] = {
            {1, 0b1, ascendc::ReducePattern::R},
            {2, 0b01, ascendc::ReducePattern::RA},
            {2, 0b10, ascendc::ReducePattern::AR},
            {3, 0b010, ascendc::ReducePattern::ARA},
            {3, 0b101, ascendc::ReducePattern::RAR},
            {4, 0b1010, ascendc::ReducePattern::ARAR},
            {4, 0b0101, ascendc::ReducePattern::RARA},
            {5, 0b10101, ascendc::ReducePattern::RARAR},
            {5, 0b01010, ascendc::ReducePattern::ARARA},
            {6, 0b101010, ascendc::ReducePattern::RARARA},
            {6, 0b010101, ascendc::ReducePattern::ARARAR},
            {7, 0b1010101, ascendc::ReducePattern::RARARAR},
            {7, 0b0101010, ascendc::ReducePattern::ARARARA},
            {8, 0b10101010, ascendc::ReducePattern::RARARARA},
            {8, 0b01010101, ascendc::ReducePattern::ARARARAR},
            {9, 0b010101010, ascendc::ReducePattern::ARARARARA},
        };
        for (const auto& p : patterns) {
            if (p.length == length && p.mask == mask)
                return p.value;
        }
        return std::nullopt;
    }

    static std::pair<SmallVector<int64_t>, std::optional<ascendc::ReducePattern>>
    getReductionParams(ArrayRef<int64_t> tensorShape, ArrayRef<int64_t> dims)
    {
        if (tensorShape.empty() || dims.empty())
            return {};
        SmallVector<bool> reduceDims(tensorShape.size(), false);
        for (auto dim : dims)
            reduceDims[dim] = true;
        SmallVector<int64_t> shape;
        bool reduceCurrent = reduceDims[0];
        int64_t accum = 1;
        uint64_t mask = 0;
        for (size_t i = 0; i <= tensorShape.size(); ++i) {
            if (i < tensorShape.size() && reduceDims[i] == reduceCurrent) {
                accum *= tensorShape[i];
                continue;
            }
            if (reduceCurrent)
                mask |= (1 << shape.size());
            shape.push_back(accum);
            if (i < tensorShape.size()) {
                accum = tensorShape[i];
                reduceCurrent = reduceDims[i];
            }
        }
        return std::pair(shape, findPattern(shape.size(), mask));
    }

    LogicalResult convert(asctile::ReduceOp op, ConvertRewriter& rewriter) const override
    {
        ascir::ConstantOpBuilder consts(rewriter);
        Location loc = op.getLoc();
        SmallVector<int64_t> reduceDims;
        for (auto attr : op.getDims()) {
            reduceDims.push_back(cast<IntegerAttr>(attr).getValue().getSExtValue());
        }
        auto srcType = op.getOperand().getType();
        auto [shape, pattern] = getReductionParams(srcType.getShape(), reduceDims);
        if (shape.empty() || !pattern)
            return emitError(loc, "Tensor of shape [")
                .append(srcType.getShape())
                .append("] have wrong reduction dimensions: ")
                .append(reduceDims);
        SmallVector<Value> srcShape;
        for (auto size : shape)
            srcShape.push_back(consts.i32(size));
        Value dst = createTensorOp(rewriter, loc, op.getType());
        Value src = rewriter.getRemappedValue(op.getOperand());
        Value tmpBuff = createTensorOp(rewriter, loc, srcType.getNumElements() * 4, rewriter.getIntegerType(8, false));
        auto kind = op.getKind();
        if (kind == asctile::ReduceKind::Sum)
            rewriter.create<ascendc::ReduceSumOp>(loc, dst, src, tmpBuff, srcShape, *pattern);
        else if (kind == asctile::ReduceKind::Max)
            rewriter.create<ascendc::ReduceMaxOp>(loc, dst, src, tmpBuff, srcShape, *pattern);
        else if (kind == asctile::ReduceKind::Min)
            rewriter.create<ascendc::ReduceMinOp>(loc, dst, src, tmpBuff, srcShape, *pattern);
        else if (kind == asctile::ReduceKind::Prod)
            rewriter.create<ascendc::ReduceProdOp>(loc, dst, src, tmpBuff, srcShape, *pattern);
        else
            return op.emitOpError() << "with " << asctile::stringifyReduceKind(kind) << " is not supported";
        rewriter.replaceOp(op, dst);
        return success();
    }
};

struct ConvertAccumulator : ConvertOp<asctile::AccumulatorOp> {
    using ConvertOp::ConvertOp;
    using ConvertOp::createTensorOp;

    LogicalResult convert(asctile::AccumulatorOp op, ConvertRewriter& rewriter) const override
    {
        auto type = op.getType();
        assert(type.getLoc() == asctile::TileLocation::L0C && "accumulator should be have tile location L0C");
        auto loc = op.getLoc();
        auto dst = createTensorOp(rewriter, loc, type);
        rewriter.replaceOp(op, dst);
        return success();
    }
};

struct ConvertMatmulAcc : ConvertOp<asctile::MatmulAccOp> {
    using ConvertOp::ConvertOp;

    LogicalResult convert(asctile::MatmulAccOp op, ConvertRewriter& rewriter) const override
    {
        ascir::ConstantOpBuilder consts(rewriter);
        auto loc = op.getLoc();
        auto dst = rewriter.getRemappedValue(op.getAcc());
        auto matrixA = rewriter.getRemappedValue(op.getMatrixA());
        auto matrixB = rewriter.getRemappedValue(op.getMatrixB());
        auto matrixATensorShape = cast<ascendc::LocalTensorType>(matrixA.getType()).getShape();
        auto matrixBTensorShape = cast<ascendc::LocalTensorType>(matrixB.getType()).getShape();
        auto accTensorShape = cast<ascendc::LocalTensorType>(dst.getType()).getShape();
        assert(matrixATensorShape.size() == 2 && "matrix must be have dim = 2");
        assert(matrixBTensorShape.size() == 2 && "matrix must be have dim = 2");
        assert(accTensorShape.size() == 2 && "accumulator must be have dim = 2");
        auto mmadParams = emitasc::InitStructBuilder(rewriter.getType<ascendc::MmadParamsType>())
                              .addField("m", consts.i32(matrixATensorShape[0]))
                              .addField("n", consts.i32(matrixBTensorShape[1]))
                              .addField("k", consts.i32(matrixBTensorShape[0]))
                              .addField("isBias", consts.i32(1))
                              .create(rewriter, loc);
        rewriter.create<ascendc::MmadOp>(loc, dst, matrixA, matrixB, mmadParams);
        rewriter.eraseOp(op);
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
            auto srcElType = srcType.getElementType();
            auto dstElType = dstType.getElementType();
            auto floatType = rewriter.getF32Type();
            auto halfType = rewriter.getF16Type();
            auto int32Type = rewriter.getIntegerType(32);
            auto int8Type = rewriter.getIntegerType(8);
            auto uint8Type = rewriter.getIntegerType(8, false);
            int32_t mode;
            if (srcElType == floatType && dstElType == halfType) {
                mode = static_cast<int32_t>(ascendc::QuantMode::F322F16);
            } else if (srcElType == floatType && dstElType == rewriter.getBF16Type()) {
                mode = static_cast<int32_t>(ascendc::QuantMode::F322BF16);
            } else if (srcElType == int32Type && dstElType == halfType) {
                mode = static_cast<int32_t>(ascendc::QuantMode::DEQF16);
            } else if (srcElType == floatType && (dstElType == int8Type || dstElType == uint8Type)) {
                mode = static_cast<int32_t>(ascendc::QuantMode::QF322B8_PRE);
            } else if (srcElType == int32Type && (dstElType == int8Type || dstElType == uint8Type)) {
                mode = static_cast<int32_t>(ascendc::QuantMode::REQ8);
            } else {
                // TODO: Add support for VDEQF16, VQF322B8_PRE, VREQ8
                op.emitError() << "Unsupported quant mode from " << srcElType << " to " << dstElType;
                return failure();
            }
            auto quantMode = rewriter.create<ascendc::ConstructOp>(
                loc, rewriter.getType<ascendc::QuantModesType>(), ValueRange{consts.i32(mode)},
                rewriter.getTypeArrayAttr(rewriter.getType<ascendc::QuantModesType>()), true, true);
            paramsBuilder.addField("quantPre", quantMode);
        }
        Value params = paramsBuilder.create(rewriter, loc);
        Value layout = rewriter.create<ascendc::ConstructOp>(
            loc, rewriter.getType<ascendc::CO2LayoutType>(), ValueRange{const1}, ArrayAttr{}, true, true);
        auto fixPipeConfig = rewriter.create<ascendc::ConstructOp>(
            loc, rewriter.getType<ascendc::FixpipeConfigType>(), ValueRange{layout}, ArrayAttr{}, true, true);
        rewriter.replaceOpWithNewOp<ascendc::FixpipeOp>(op, dst, src, params, fixPipeConfig);
        return success();
    }
};

struct LowerAscTilePass : public asclower::impl::LowerAscTileBase<LowerAscTilePass> {
    void runOnOperation() override
    {
        func::FuncOp funcOp = getOperation();
        TensorTypeConverter converter;
        MLIRContext* context = &getContext();
        LoweringConversionTarget target(converter, context);
        RewritePatternSet patterns(context);
        patterns.insert<
            //
            ConvertTensor, ConvertLoad, ConvertGetValue, ConvertStore, ConvertSetValue, ConvertSplat, ConvertRelu,
            ConvertCast, ConvertMatmul, ConvertReshape, ConvertBroadcast, ConvertSoftmax, ConvertRmsNorm,
            ConvertAccumulator, ConvertMatmulAcc, ConvertCopy, ConvertStoreFixpipe, ConvertReduceAs1d, ConvertReduce,
            ConvertToL2<asctile::AddSOp, ascendc::AddsL2Op>,
            ConvertVecScalarToL2<asctile::SubSOp, ascendc::SubsL2Op, ascendc::SubL2Op>,
            ConvertToL2<asctile::MulSOp, ascendc::MulsL2Op>,
            ConvertVecScalarToL2<asctile::DivSOp, ascendc::DivsL2Op, ascendc::DivL2Op>,
            ConvertToL2<asctile::MinSOp, ascendc::MinsL2Op>, ConvertToL2<asctile::MaxSOp, ascendc::MaxsL2Op>,
            ConvertToL2<asctile::ShLSOp, ascendc::ShiftLeftL2Op>, ConvertToL2<asctile::ShRSOp, ascendc::ShiftRightL2Op>
            //
            >(converter, context);
        if (applyPartialConversion(funcOp, target, std::move(patterns)).failed())
            signalPassFailure();
    }
};

} // namespace

std::unique_ptr<Pass> mlir::asclower::createLowerAscTilePass() { return std::make_unique<LowerAscTilePass>(); }
