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
#include "ascir/Dialect/Asc/Utils/Utils.h"
#include "ascir/Dialect/AscTile/IR/AscTile.h"
#include "ascir/Dialect/Utils/ConstantOpBuilder.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

#include "Common.h"

namespace mlir {
namespace asclower {
#define GEN_PASS_DEF_LOWERASCTILEI1
#include "ascir/Conversion/LowerToAsc/Passes.h.inc"
} // namespace asclower
} // namespace mlir

using namespace mlir;
using namespace mlir::asclower;

namespace {

struct LoweringConversionTarget : public ConversionTarget {
    LoweringConversionTarget(TensorTypeConverter &converter, MLIRContext *context) : ConversionTarget(*context)
    {
        addIllegalOp<asctile::CmpOp, asctile::CmpSOp, asctile::SelectOp>();
        addLegalDialect<arith::ArithDialect, ascendc::AscendCDialect>();
        addLegalOp<UnrealizedConversionCastOp>();
    }
};

ascendc::CMPMODE getCmpMode(asctile::CompareMode mode)
{
    switch (mode) {
        case asctile::CompareMode::EQ:
            return ascendc::CMPMODE::EQ;
        case asctile::CompareMode::NE:
            return ascendc::CMPMODE::NE;
        case asctile::CompareMode::LT:
            return ascendc::CMPMODE::LT;
        case asctile::CompareMode::LE:
            return ascendc::CMPMODE::LE;
        case asctile::CompareMode::GT:
            return ascendc::CMPMODE::GT;
        case asctile::CompareMode::GE:
            return ascendc::CMPMODE::GE;
    }
    llvm_unreachable("unexpected cmpmode");
}

struct ConvertCmp : public ConvertOp<asctile::CmpOp> {
    using ConvertOp::converter;
    using ConvertOp::ConvertOp;

    LogicalResult convert(asctile::CmpOp op, ConvertRewriter &rewriter) const override
    {
        auto loc = op.getLoc();
        Value src0 = rewriter.getRemappedValue(op.getLhs());
        Value src1 = rewriter.getRemappedValue(op.getRhs());
        auto srcType = cast<ShapedType>(src0.getType());
        Value zero = ascir::ConstantOpBuilder(rewriter).i64(0);
        if (isa<IntegerType>(srcType.getElementType()) && !ascendc::isTargetPlatform95(op)) {
            unsigned bitWidth = srcType.getElementTypeBitWidth();
            if (bitWidth != 16 && bitWidth != 32)
                return op.emitOpError("can only be lowered with i16 or i32 tile operands");
            auto castToType = bitWidth == 16 ? rewriter.getF16Type() : rewriter.getF32Type();
            auto src0Casted = createTensorOp(rewriter, loc, srcType.getShape(), castToType);
            auto src1Casted = createTensorOp(rewriter, loc, srcType.getShape(), castToType);
            rewriter.create<ascendc::CastL2Op>(loc, src0Casted, src0, ascendc::RoundMode::CAST_NONE, zero);
            rewriter.create<ascendc::CastL2Op>(loc, src1Casted, src1, ascendc::RoundMode::CAST_NONE, zero);
            src0 = src0Casted;
            src1 = src1Casted;
        }
        auto srcNumElems = srcType.getNumElements();
        I1ReplacementType replType(op.getContext());
        auto dstShape = llvm::divideCeil(srcNumElems, replType.width);
        Value dst = createTensorOp(rewriter, loc, dstShape, replType.iType);
        dst = createReCastOp(rewriter, loc, dst, dstShape, replType.uiType);
        ascendc::CMPMODE cmpMode = getCmpMode(op.getCmpMode());
        rewriter.create<ascendc::CompareL0Op>(
            loc, dst, src0, src1, cmpMode, zero, zero,
            rewriter.create<ascendc::ConstructOp>(loc, rewriter.getType<ascendc::BinaryRepeatParamsType>()));
        rewriter.replaceOp(op, dst);
        return success();
    }
};

struct ConvertCmpS : ConvertOp<asctile::CmpSOp> {
    using ConvertOp::ConvertOp;
    using ConvertOp::createTensorOp;

    LogicalResult convert(asctile::CmpSOp op, ConvertRewriter &rewriter) const override
    {
        ascir::ConstantOpBuilder consts(rewriter);
        auto loc = op.getLoc();
        auto value = rewriter.getRemappedValue(op.getValue());
        auto base = rewriter.getRemappedValue(op.getBase());
        auto srcType = op.getBase().getType();
        I1ReplacementType replType(op.getContext());
        auto dstShape = llvm::divideCeil(srcType.getNumElements(), replType.width);
        Value dst = createTensorOp(rewriter, loc, dstShape, replType.iType);
        dst = createReCastOp(rewriter, loc, dst, dstShape, replType.uiType);
        auto mode = getCmpMode(op.getCmpMode());
        Value zero = ascir::ConstantOpBuilder(rewriter).i64(0);
        rewriter.create<ascendc::CompareScalarL0Op>(
            loc, dst, base, value, mode, zero, zero,
            rewriter.create<ascendc::ConstructOp>(loc, rewriter.getType<ascendc::UnaryRepeatParamsType>()));
        rewriter.replaceOp(op, dst);
        return success();
    }
};

struct ConvertSelect : ConvertOp<asctile::SelectOp> {
    using ConvertOp::ConvertOp;
    using ConvertOp::createTensorOp;

    LogicalResult convert(asctile::SelectOp op, ConvertRewriter &rewriter) const override
    {
        ascir::ConstantOpBuilder consts(rewriter);
        auto loc = op.getLoc();
        auto dst = createTensorOp(rewriter, loc, op.getType());
        auto sel = rewriter.getRemappedValue(op.getSelMask());
        I1ReplacementType replType(op.getContext());
        sel = createReCastOp(rewriter, loc, sel, cast<ShapedType>(sel.getType()).getShape(), replType.uiType);
        auto src0 = rewriter.getRemappedValue(op.getSrc0());
        auto src1 = rewriter.getRemappedValue(op.getSrc1());
        auto zero = consts.i64(0);
        rewriter.create<ascendc::SelectL0Op>(
            loc, dst, sel, src0, src1, ascendc::SELMODE::VSEL_TENSOR_TENSOR_MODE, zero, zero,
            rewriter.create<ascendc::ConstructOp>(loc, rewriter.getType<ascendc::BinaryRepeatParamsType>()));
        rewriter.replaceOp(op, dst);
        return success();
    }
};

struct LowerAscTileI1Pass : public asclower::impl::LowerAscTileI1Base<LowerAscTileI1Pass> {
    void runOnOperation() override
    {
        func::FuncOp funcOp = getOperation();
        TensorTypeConverter converter;
        MLIRContext *context = &getContext();
        LoweringConversionTarget target(converter, context);
        RewritePatternSet patterns(context);
        patterns.insert<ConvertCmp, ConvertCmpS, ConvertSelect>(converter, context);
        if (applyPartialConversion(funcOp, target, std::move(patterns)).failed())
            signalPassFailure();
    }
};

} // namespace

std::unique_ptr<Pass> mlir::asclower::createLowerAscTileI1Pass()
{
    return std::make_unique<LowerAscTileI1Pass>();
}
