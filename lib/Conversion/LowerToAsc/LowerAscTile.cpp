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

#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
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

struct LoweringConversionTarget : public ConversionTarget {
    LoweringConversionTarget(TensorTypeConverter &converter, MLIRContext *context) : ConversionTarget(*context)
    {
        addIllegalDialect<asctile::AscTileDialect>();
        addLegalDialect<ascendc::AscendCDialect, arith::ArithDialect>();
        addLegalOp<UnrealizedConversionCastOp>();
    }
};

struct ConvertTensor : public ConvertOp<asctile::TensorOp> {
    using ConvertOp::converter;
    using ConvertOp::ConvertOp;

    LogicalResult convert(asctile::TensorOp op, ConvertRewriter &rewriter) const override
    {
        auto loc = op.getLoc();
        Value tensor = rewriter.create<ascendc::GlobalTensorOp>(loc, converter().convertType(op.getType()));
        rewriter.create<ascendc::GlobalTensorSetGlobalBufferOp>(loc, tensor, rewriter.getRemappedValue(op.getBase()),
                                                                /*size*/ Value {});
        rewriter.replaceOp(op, tensor);
        return success();
    }
};

struct ConvertSplat : public ConvertOp<asctile::SplatOp> {
    using ConvertOp::converter;
    using ConvertOp::ConvertOp;

    LogicalResult convert(asctile::SplatOp op, ConvertRewriter &rewriter) const override
    {
        ascir::ConstantOpBuilder consts(rewriter);
        auto loc = op.getLoc();
        Value dst = createTensorOp(rewriter, op.getLoc(), op.getType());
        rewriter.create<ascendc::DuplicateL2Op>(op.getLoc(), dst, op.getValue(), consts.i64(calCount(dst)));
        rewriter.replaceOp(op, dst);
        return success();
    }
};

struct ConvertRelu : ConvertOp<asctile::ReluOp> {
    using ConvertOp<asctile::ReluOp>::ConvertOp;
    using ConvertOp<asctile::ReluOp>::createTensorOp;

    LogicalResult convert(asctile::ReluOp op, ConvertRewriter &rewriter) const override
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

template <typename TileOp, typename L2Op>
struct ConvertToL2 : ConvertOp<TileOp> {
    using ConvertOp<TileOp>::ConvertOp;
    using ConvertOp<TileOp>::calCount;
    using ConvertOp<TileOp>::createTensorOp;

    LogicalResult convert(TileOp op, ConvertRewriter &rewriter) const override
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

struct LowerAscTilePass : public asclower::impl::LowerAscTileBase<LowerAscTilePass> {
    void runOnOperation() override
    {
        func::FuncOp funcOp = getOperation();
        TensorTypeConverter converter;
        MLIRContext *context = &getContext();
        LoweringConversionTarget target(converter, context);
        RewritePatternSet patterns(context);
        patterns.insert<
            //
            ConvertTensor, ConvertSplat, ConvertRelu, ConvertToL2<asctile::AddsOp, ascendc::AddsL2Op>,
            ConvertToL2<asctile::MulsOp, ascendc::MulsL2Op>
            //
            >(converter, context);
        if (applyPartialConversion(funcOp, target, std::move(patterns)).failed())
            signalPassFailure();
    }
};

} // namespace

std::unique_ptr<Pass> mlir::asclower::createLowerAscTilePass()
{
    return std::make_unique<LowerAscTilePass>();
}
