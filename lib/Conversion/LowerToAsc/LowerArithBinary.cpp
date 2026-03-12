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
#include "ascir/Dialect/AscTile/IR/AscTile.h"
#include "ascir/Dialect/Utils/ConstantOpBuilder.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

#include "Common.h"

namespace mlir {
namespace asclower {
#define GEN_PASS_DEF_LOWERARITHBINARY
#include "ascir/Conversion/LowerToAsc/Passes.h.inc"
} // namespace asclower
} // namespace mlir

using namespace mlir;
using namespace mlir::asclower;

namespace {

struct LoweringConversionTarget : public ConversionTarget {
    LoweringConversionTarget(TensorTypeConverter &converter, MLIRContext *context) : ConversionTarget(*context)
    {
        addLegalDialect<ascendc::AscendCDialect, arith::ArithDialect>();
        addDynamicallyLegalOp<
            //
            arith::AddFOp, arith::AddIOp, arith::SubFOp, arith::SubIOp, arith::MulFOp, arith::MulIOp, arith::DivFOp,
            arith::MaxSIOp, arith::MinSIOp, arith::MaximumFOp, arith::MinimumFOp, arith::MaxNumFOp, arith::MinNumFOp,
            arith::AndIOp, arith::OrIOp
            //
            >([&](Operation *op) { return converter.isLegal(op); });
        addLegalOp<UnrealizedConversionCastOp>();
    }
};

template <typename ArithOp, typename L3Op>
struct ConvertToL3 : ConvertOp<ArithOp> {
    using ConvertOp<ArithOp>::ConvertOp;
    using ConvertOp<ArithOp>::createTensorOp;

    LogicalResult convert(ArithOp op, ConvertRewriter &rewriter) const override
    {
        Location loc = op.getLoc();
        auto dst = createTensorOp(rewriter, loc, op.getType());
        rewriter.create<L3Op>(loc, dst, rewriter.getRemappedValue(op.getLhs()), rewriter.getRemappedValue(op.getRhs()));
        rewriter.replaceOp(op, dst);
        return success();
    }
};

template <typename ArithOp, typename L2Op>
struct ConvertToL2 : ConvertOp<ArithOp> {
    using ConvertOp<ArithOp>::ConvertOp;
    using ConvertOp<ArithOp>::createTensorOp;
    using ConvertOp<ArithOp>::calCount;

    LogicalResult convert(ArithOp op, ConvertRewriter &rewriter) const override
    {
        ascir::ConstantOpBuilder consts(rewriter);
        Location loc = op.getLoc();
        auto dst = createTensorOp(rewriter, loc, op.getType());
        rewriter.create<L2Op>(loc, dst, rewriter.getRemappedValue(op.getLhs()), rewriter.getRemappedValue(op.getRhs()),
                              consts.i64(calCount(dst)));
        rewriter.replaceOp(op, dst);
        return success();
    }
};

template <typename ArithOp, typename L2Op>
struct ConvertBitwiseToL2 : public ConvertOp<ArithOp> {
    using ConvertOp<ArithOp>::ConvertOp;
    using ConvertOp<ArithOp>::createTensorOp;
    using ConvertOp<ArithOp>::createReCastOp;

    LogicalResult convert(ArithOp op, ConvertRewriter &rewriter) const override
    {
        ascir::ConstantOpBuilder consts(rewriter);
        auto shapedTy = cast<asctile::TileType>(op.getType());
        auto resType = shapedTy.getElementType();
        I1ReplacementType replType(op.getContext());
        auto supportedElemTy = replType.iType;
        auto needCast = resType != supportedElemTy;
        auto src0 = rewriter.getRemappedValue(op->getOperand(0));
        auto src1 = rewriter.getRemappedValue(op->getOperand(1));
        Location loc = op.getLoc();
        Value dst = createTensorOp(rewriter, loc, shapedTy.getShape(), resType);
        needCast &= cast<ShapedType>(dst.getType()).getElementType() != supportedElemTy;
        if (needCast) {
            SmallVector<int64_t> newShape(shapedTy.getShape());
            newShape[0] *= shapedTy.getElementTypeBitWidth();
            newShape[0] /= supportedElemTy.getIntOrFloatBitWidth();
            src0 = createReCastOp(rewriter, loc, src0, newShape, supportedElemTy);
            src1 = createReCastOp(rewriter, loc, src1, newShape, supportedElemTy);
            dst = createReCastOp(rewriter, loc, dst, newShape, supportedElemTy);
        }
        rewriter.create<L2Op>(loc, dst, src0, src1, consts.i64(1));
        if (needCast) {
            dst = createReCastOp(rewriter, loc, dst, shapedTy.getShape(), resType);
        }
        rewriter.replaceOp(op, dst);
        return success();
    }
};

struct LowerArithBinaryPass : public asclower::impl::LowerArithBinaryBase<LowerArithBinaryPass> {
    void runOnOperation() override
    {
        func::FuncOp funcOp = getOperation();
        TensorTypeConverter converter;
        MLIRContext *context = &getContext();
        LoweringConversionTarget target(converter, context);
        RewritePatternSet patterns(context);
        patterns.insert<
            //
            ConvertToL3<arith::AddFOp, ascendc::AddL3Op>, ConvertToL3<arith::AddIOp, ascendc::AddL3Op>,
            ConvertToL3<arith::SubFOp, ascendc::SubL3Op>, ConvertToL3<arith::SubIOp, ascendc::SubL3Op>,
            ConvertToL3<arith::MulFOp, ascendc::MulL3Op>, ConvertToL3<arith::MulIOp, ascendc::MulL3Op>,
            ConvertToL3<arith::DivFOp, ascendc::DivL3Op>, ConvertToL2<arith::MaximumFOp, ascendc::MaxL2Op>,
            ConvertToL2<arith::MaxSIOp, ascendc::MaxL2Op>, ConvertToL2<arith::MinSIOp, ascendc::MinL2Op>,
            ConvertToL2<arith::MinimumFOp, ascendc::MinL2Op>, ConvertToL2<arith::MaxNumFOp, ascendc::MaxL2Op>,
            ConvertToL2<arith::MinNumFOp, ascendc::MinL2Op>, ConvertBitwiseToL2<arith::AndIOp, ascendc::AndL2Op>,
            ConvertBitwiseToL2<arith::OrIOp, ascendc::OrL2Op>
            //
            >(converter, context);
        if (applyPartialConversion(funcOp, target, std::move(patterns)).failed())
            signalPassFailure();
    }
};

} // namespace

std::unique_ptr<Pass> mlir::asclower::createLowerArithBinaryPass()
{
    return std::make_unique<LowerArithBinaryPass>();
}
