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
#include "ascir/Conversion/LowerToAsc/Passes.h"
#include "ascir/Dialect/Utils/ConstantOpBuilder.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

#include "Common.h"

namespace mlir {
namespace asclower {
#define GEN_PASS_DEF_LOWERARITH
#include "ascir/Conversion/LowerToAsc/Passes.h.inc"
} // namespace asclower
} // namespace mlir

using namespace mlir;
using namespace mlir::asclower;

namespace {

struct LoweringConversionTarget : public ConversionTarget {
    LoweringConversionTarget(TensorTypeConverter &converter, MLIRContext *context) : ConversionTarget(*context)
    {
        addDynamicallyLegalOp<
            //
            arith::ConstantOp, arith::BitcastOp, arith::TruncFOp, arith::TruncIOp, arith::UIToFPOp, arith::ExtSIOp,
            arith::ExtFOp, arith::SIToFPOp, arith::FPToSIOp, arith::NegFOp
            //
            >([&](Operation *op) { return converter.isLegal(op); });
        addLegalDialect<ascendc::AscendCDialect>();
    }
};

struct ConvertSplatConstant : ConvertOp<arith::ConstantOp> {
    using ConvertOp::ConvertOp;
    using ConvertOp::createTensorOp;

    LogicalResult convert(arith::ConstantOp op, ConvertRewriter &rewriter) const override
    {
        if (!isa_and_present<SplatElementsAttr>(op.getValue()))
            return failure();
        ascir::ConstantOpBuilder consts(rewriter);
        auto dense = dyn_cast<DenseElementsAttr>(op.getValue());
        Value scalar = rewriter.create<arith::ConstantOp>(op.getLoc(), dense.getSplatValue<TypedAttr>());
        Location loc = op.getLoc();
        Value dst = createTensorOp(rewriter, loc, op.getType());
        rewriter.create<ascendc::DuplicateL2Op>(loc, dst, scalar, consts.i64(0));
        rewriter.replaceOp(op, dst);
        return success();
    }
};

struct ConvertDenseConstant : ConvertOp<arith::ConstantOp> {
    using ConvertOp::ConvertOp;
    using ConvertOp::createTensorOp;

    LogicalResult convert(arith::ConstantOp op, ConvertRewriter &rewriter) const override
    {
        auto dense = dyn_cast<DenseElementsAttr>(op.getValue());
        if (!dense || dense.isSplat())
            return failure();
        ascir::ConstantOpBuilder consts(rewriter);
        Value dst = createTensorOp(rewriter, op.getLoc(), op.getType());
        for (auto [i, value] : llvm::enumerate(dense.getValues<TypedAttr>())) {
            Location uloc = rewriter.getUnknownLoc();
            Value cst = rewriter.create<arith::ConstantOp>(uloc, value);
            rewriter.create<ascendc::LocalTensorSetValueOp>(uloc, cst, dst, consts.index(i));
        }
        rewriter.replaceOp(op, dst);
        return success();
    }
};

struct ConvertBitcast : public ConvertOp<arith::BitcastOp> {
    using ConvertOp::converter;
    using ConvertOp::ConvertOp;

    LogicalResult convert(arith::BitcastOp op, ConvertRewriter &rewriter) const override
    {
        rewriter.replaceOpWithNewOp<ascendc::LocalTensorReinterpretCastOp>(op, converter().convertType(op.getType()),
                                                                           rewriter.getRemappedValue(op.getIn()));
        return success();
    }
};

template <typename ArithOp>
struct ConvertCast : public ConvertOp<ArithOp> {
    using ConvertOp<ArithOp>::ConvertOp;
    using ConvertOp<ArithOp>::createTensorOp;

    static Type getElementType(Value tensor) { return cast<ShapedType>(tensor.getType()).getElementType(); }

    static ascendc::RoundMode getRoundMode(Type in, Type out)
    {
        if (isa<FloatType>(in) && isa<IntegerType>(out))
            return ascendc::RoundMode::CAST_TRUNC;
        return ascendc::RoundMode::CAST_NONE;
    }

    LogicalResult convert(ArithOp op, ConvertRewriter &rewriter) const override
    {
        Location loc = op.getLoc();
        Value dst = createTensorOp(rewriter, loc, op.getType());
        ascir::ConstantOpBuilder consts(rewriter);
        Value src = rewriter.getRemappedValue(op.getIn());
        auto roundMode = getRoundMode(getElementType(src), getElementType(dst));
        rewriter.create<ascendc::CastL2Op>(loc, dst, src, roundMode, consts.i64(1));
        rewriter.replaceOp(op, dst);
        return success();
    }
};

struct ConvertNegF : public ConvertOp<arith::NegFOp> {
    using ConvertOp::converter;
    using ConvertOp::ConvertOp;
    using ConvertOp::createTensorOp;

    LogicalResult convert(arith::NegFOp op, ConvertRewriter &rewriter) const override
    {
        ascir::ConstantOpBuilder consts(rewriter);
        auto loc = op.getLoc();
        Value dst = createTensorOp(rewriter, loc, op.getType());
        auto src = rewriter.getRemappedValue(op.getOperand());
        auto floatTy = cast<FloatType>(cast<ShapedType>(op.getType()).getElementType());
        auto cm1 =
            rewriter.create<arith::ConstantFloatOp>(loc, llvm::APFloat(floatTy.getFloatSemantics(), "-1"), floatTy);
        rewriter.create<ascendc::MulsL2Op>(loc, dst, src, cm1, consts.i64(1));
        rewriter.replaceOp(op, dst);
        return success();
    }
};

struct LowerArithPass : public asclower::impl::LowerArithBase<LowerArithPass> {
    void runOnOperation() override
    {
        func::FuncOp funcOp = getOperation();
        TensorTypeConverter converter;
        MLIRContext *context = &getContext();
        LoweringConversionTarget target(converter, context);
        RewritePatternSet patterns(context);
        patterns.insert<
            //
            ConvertSplatConstant, ConvertDenseConstant, ConvertBitcast, ConvertNegF, ConvertCast<arith::TruncFOp>,
            ConvertCast<arith::TruncIOp>, ConvertCast<arith::ExtFOp>, ConvertCast<arith::ExtSIOp>,
            ConvertCast<arith::SIToFPOp>, ConvertCast<arith::UIToFPOp>, ConvertCast<arith::FPToSIOp>
            //
            >(converter, context);
        if (applyPartialConversion(funcOp, target, std::move(patterns)).failed())
            signalPassFailure();
    }
};

} // namespace

std::unique_ptr<Pass> mlir::asclower::createLowerArithPass()
{
    return std::make_unique<LowerArithPass>();
}
