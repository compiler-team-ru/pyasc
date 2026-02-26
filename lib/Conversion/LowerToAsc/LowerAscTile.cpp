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
#include "ascir/Dialect/Utils/ConstantOpBuilder.h"
#include "ascir/Dialect/EmitAsc/IR/EmitAsc.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
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

struct LoweringConversionTarget : public ConversionTarget {
    LoweringConversionTarget(TensorTypeConverter& converter, MLIRContext* context) : ConversionTarget(*context)
    {
        addIllegalDialect<asctile::AscTileDialect>();
        addLegalDialect<ascendc::AscendCDialect, arith::ArithDialect, emitasc::EmitAscDialect>();
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

struct ConvertLoad : ConvertOp<asctile::LoadOp> {
    using ConvertOp::ConvertOp;
    using ConvertOp::createTensorOp;

    LogicalResult convert(asctile::LoadOp op, ConvertRewriter& rewriter) const override
    {
        auto base = op.getBase();
        auto loc = op.getLoc();
        auto tensorOp = base.getDefiningOp<asctile::TensorOp>();
        assert(tensorOp && "tensor must be created by asctile.tensor op");
        auto dstTensorOp = createTensorOp(rewriter, loc, op.getType());
        auto dst = dstTensorOp.getResult();
        SmallVector<Value> srcTensorShape = getTensorShape(rewriter, tensorOp);
        auto dstTensorShape = dstTensorOp.getType().getShape();
        Value linearOffset = linearizeOffset(rewriter, loc, srcTensorShape, op.getOffsets());
        Value src = rewriter.getRemappedValue(base);
        auto srcType = cast<ascendc::BaseTensorType>(src.getType());
        auto dstType = dst.getType();
        auto numElements = calculateNumElements(rewriter, loc, srcTensorShape);
        ascir::ConstantOpBuilder consts(rewriter);
        src = rewriter.create<ascendc::GlobalTensorSubIndexOp>(loc, srcType, src, linearOffset);
        auto padValue = rewriter.getRemappedValue(op.getPadValue());
        auto typeSize = ascendc::getElementTypeSize(dstType);
        auto const0 = consts.i32(0);
        auto const1 = consts.i32(1);
        Value dstNumElements = consts.i32(calCount(dst));
        Value tailElements = rewriter.create<arith::SubIOp>(loc, numElements, linearOffset);
        Value blockCount = dstTensorShape.size() == 1 ? const1 : consts.i32(dstTensorShape[0]);
        Value dstLastDim = consts.i32(dstTensorShape[dstTensorShape.size() - 1]);
        Value srcLastDim = srcTensorShape[srcTensorShape.size() - 1];
        Value strideElements = rewriter.create<arith::SubIOp>(loc, srcLastDim, dstLastDim);
        auto typeSizeValue = consts.i32(typeSize);
        Value srcStride = rewriter.create<arith::MulIOp>(loc, strideElements, typeSizeValue);
        Value numElementsInBlock = consts.i32(ascendc::ubBlockSize / typeSize);
        Value totalElementsInBlock = rewriter.create<arith::MulIOp>(loc, blockCount, numElementsInBlock);
        Value minTailElements = rewriter.create<arith::MinSIOp>(loc, dstLastDim, tailElements);
        Value blockLen = rewriter.create<arith::MulIOp>(loc, minTailElements, typeSizeValue);
        auto context = op.getContext();
        auto dataCopyExtParams = rewriter.create<ascendc::ConstructOp>(
            loc, rewriter.getType<ascendc::DataCopyExtParamsType>(),
            ValueRange{blockCount, blockLen, srcStride, const0, const0},
            rewriter.getTypeArrayAttr(
                {rewriter.getI32Type(), rewriter.getIntegerType(32, false), rewriter.getI32Type(),
                 rewriter.getI32Type(), rewriter.getI32Type()}));
        Value numPaddingElements = rewriter.create<arith::SubIOp>(loc, totalElementsInBlock, tailElements);
        Value rightPad = rewriter.create<arith::MaxSIOp>(loc, numPaddingElements, const0);
        auto dataCopyPadExtParams = rewriter.create<ascendc::ConstructOp>(
            loc, ascendc::DataCopyPadExtParamsType::get(context, cast<ShapedType>(dstType).getElementType()),
            ValueRange{const1, const0, rightPad, padValue},
            rewriter.getTypeArrayAttr(
                {rewriter.getI32Type(), rewriter.getI32Type(), rewriter.getIntegerType(8, false), padValue.getType()}));
        rewriter.create<ascendc::DataCopyPadExtL0Op>(loc, dst, src, dataCopyExtParams, dataCopyPadExtParams);
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
        Value src = rewriter.getRemappedValue(op.getValue());
        Value dst = rewriter.getRemappedValue(op.getBase());
        auto srcType = cast<ascendc::BaseTensorType>(src.getType());
        auto dstType = cast<ascendc::BaseTensorType>(dst.getType());
        auto srcTensorShape = srcType.getShape();
        SmallVector<Value> dstTensorShape = getTensorShape(rewriter, tensorOp);
        Value linearOffset = linearizeOffset(rewriter, loc, dstTensorShape, op.getOffsets());
        dst = rewriter.create<ascendc::GlobalTensorSubIndexOp>(loc, dstType, dst, linearOffset);
        ascir::ConstantOpBuilder consts(rewriter);
        auto const0 = consts.i32(0);
        auto numElements = calculateNumElements(rewriter, loc, dstTensorShape);
        auto typeSize = ascendc::getElementTypeSize(srcType);
        auto typeSizeValue = consts.i32(typeSize);
        Value srcNumElements = consts.i32(calCount(src));
        Value tailElements = rewriter.create<arith::SubIOp>(loc, numElements, linearOffset);
        Value blockCount = srcTensorShape.size() == 1 ? consts.i32(1) : consts.i32(srcTensorShape[0]);
        Value srcLastDim = consts.i32(srcTensorShape[srcTensorShape.size() - 1]);
        Value dstLastDim = dstTensorShape[dstTensorShape.size() - 1];
        Value strideElements = rewriter.create<arith::SubIOp>(loc, dstLastDim, srcLastDim);
        Value dstStride = rewriter.create<arith::MulIOp>(loc, strideElements, typeSizeValue);
        Value minTailElements = rewriter.create<arith::MinSIOp>(loc, srcLastDim, tailElements);
        Value blockLen = rewriter.create<arith::MulIOp>(loc, minTailElements, typeSizeValue);
        auto context = op.getContext();
        auto dataCopyExtParams = rewriter.create<ascendc::ConstructOp>(
            loc, rewriter.getType<ascendc::DataCopyExtParamsType>(),
            ValueRange{blockCount, blockLen, const0, dstStride, const0},
            rewriter.getTypeArrayAttr(
                {rewriter.getI32Type(), rewriter.getIntegerType(32, false), rewriter.getI32Type(),
                 rewriter.getI32Type(), rewriter.getI32Type()}));
        rewriter.replaceOpWithNewOp<ascendc::DataCopyPadExtL2Op>(op, dst, src, dataCopyExtParams);
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
    using ConvertOp<asctile::ReluOp>::ConvertOp;
    using ConvertOp<asctile::ReluOp>::createTensorOp;

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

struct ConvertSelect : ConvertOp<asctile::SelectOp> {
    using ConvertOp<asctile::SelectOp>::ConvertOp;
    using ConvertOp<asctile::SelectOp>::createTensorOp;

    LogicalResult convert(asctile::SelectOp op, ConvertRewriter& rewriter) const override
    {
        ascir::ConstantOpBuilder consts(rewriter);
        auto loc = op.getLoc();
        auto dst = createTensorOp(rewriter, loc, op.getType());
        auto selMask = rewriter.getRemappedValue(op.getSelMask());
        auto src0 = rewriter.getRemappedValue(op.getSrc0());
        auto src1 = rewriter.getRemappedValue(op.getSrc1());
        auto zero = consts.i64(0);
        rewriter.create<ascendc::SelectL0Op>(
            loc, dst, selMask, src0, src1, ascendc::SELMODE::VSEL_TENSOR_TENSOR_MODE,
            rewriter.create<emitasc::MaskOp>(loc, zero, zero), zero,
            rewriter.create<ascendc::ConstructOp>(loc, rewriter.getType<ascendc::BinaryRepeatParamsType>()));
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

template <typename TileOp, typename L2Op>
struct ConvertReduce : ConvertOp<TileOp> {
    using ConvertOp<TileOp>::ConvertOp;
    using ConvertOp<TileOp>::createTensorOp;
    using ConvertOp<TileOp>::calCount;

    LogicalResult convert(TileOp op, ConvertRewriter& rewriter) const override
    {
        unsigned int typeSize = ascendc::getTypeSize(op.getType());
        unsigned int finalSize = calculateFinalTensorSize(typeSize, calCount(op.getOperand()));
        ascir::ConstantOpBuilder consts(rewriter);
        Location loc = op.getLoc();
        Value dst = createTensorOp(rewriter, loc, static_cast<int64_t>(typeSize), op.getType());
        Value src = rewriter.getRemappedValue(op.getOperand());

        Value tmpBuff = createTensorOp(rewriter, loc, static_cast<int64_t>(finalSize), op.getType());
        if constexpr (std::is_same_v<L2Op, ascendc::ReduceSumL2Op>)
            rewriter.create<L2Op>(loc, dst, src, tmpBuff, consts.i64(calCount(op.getOperand())));
        else
            rewriter.create<L2Op>(loc, dst, src, tmpBuff, consts.i64(calCount(op.getOperand())), consts.i64(0));
        rewriter.replaceOpWithNewOp<ascendc::LocalTensorGetValueOp>(op, op.getType(), dst, consts.i64(0));
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
            ConvertTensor, ConvertLoad, ConvertStore, ConvertSplat, ConvertRelu, ConvertCast, ConvertSelect,
            ConvertToL2<asctile::AddsOp, ascendc::AddsL2Op>, ConvertToL2<asctile::MulsOp, ascendc::MulsL2Op>,
            ConvertToL2<asctile::ShLSOp, ascendc::ShiftLeftL2Op>, ConvertToL2<asctile::ShRSOp, ascendc::ShiftRightL2Op>,
            ConvertReduce<asctile::ReduceSumAs1dOp, ascendc::ReduceSumL2Op>,
            ConvertReduce<asctile::ReduceMaxAs1dOp, ascendc::ReduceMaxL2Op>,
            ConvertReduce<asctile::ReduceMinAs1dOp, ascendc::ReduceMinL2Op>
            //
            >(converter, context);
        if (applyPartialConversion(funcOp, target, std::move(patterns)).failed())
            signalPassFailure();
    }
};

} // namespace

std::unique_ptr<Pass> mlir::asclower::createLowerAscTilePass() { return std::make_unique<LowerAscTilePass>(); }
