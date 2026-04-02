/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ascir/Dialect/AscTile/IR/AscTile.h"
#include "ascir/Dialect/AscTile/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir {
namespace asctile {
#define GEN_PASS_DEF_UNSCALARIZEREDUCTION
#include "ascir/Dialect/AscTile/Transforms/Passes.h.inc"
} // namespace asctile
} // namespace mlir

using namespace mlir;

namespace {

constexpr const char* const unscalarizeAttr = "asctile.op_to_unscalarize";

template <typename OpT, typename FloatOp, typename IntOp>
struct ConvertTileScalar : OpConversionPattern<OpT> {
    using OpConversionPattern<OpT>::OpConversionPattern;

    LogicalResult
    matchAndRewrite(OpT op, typename OpT::Adaptor adaptor, ConversionPatternRewriter& rewriter) const override
    {
        auto splatOp = rewriter.create<asctile::SplatOp>(op.getValue().getLoc(), op.getType(), op.getValue());
        splatOp->setAttr(unscalarizeAttr, rewriter.getUnitAttr());
        if (isa<IntegerType>(op.getValue().getType())) {
            rewriter.replaceOpWithNewOp<IntOp>(op, adaptor.getBase(), splatOp);
            return success();
        }
        if (isa<FloatType>(op.getValue().getType())) {
            rewriter.replaceOpWithNewOp<FloatOp>(op, adaptor.getBase(), splatOp);
            return success();
        }
        return failure();
    }
};

template <typename OpT>
struct ConvertReduce : OpConversionPattern<OpT> {
    using OpConversionPattern<OpT>::OpConversionPattern;
    using OpConversionPattern<OpT>::getTypeConverter;

    LogicalResult
    matchAndRewrite(OpT op, typename OpT::Adaptor adaptor, ConversionPatternRewriter& rewriter) const override
    {
        auto resultType = getTypeConverter()->convertType(op.getType());
        auto newOp = rewriter.replaceOpWithNewOp<OpT>(op, resultType, adaptor.getOperands(), op->getAttrs());
        newOp->removeAttr(unscalarizeAttr);
        return success();
    }
};

struct ConvertSplat : OpConversionPattern<asctile::SplatOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(
        asctile::SplatOp op, asctile::SplatOp::Adaptor adaptor, ConversionPatternRewriter& rewriter) const override
    {
        rewriter.replaceOpWithNewOp<asctile::BroadcastOp>(op, op.getType(), adaptor.getValue());
        return success();
    }
};

bool allUnscalarizableUsers(Operation* op)
{
    return llvm::all_of(op->getUsers(), [](Operation* user) {
        return isa<
            asctile::AddSOp, asctile::SubSOp, asctile::MulSOp, asctile::DivSOp, asctile::MinSOp, asctile::MaxSOp,
            asctile::SplatOp>(user);
    });
}

template <typename ReduceOp>
void markOps(func::FuncOp root)
{
    root.walk([](ReduceOp op) {
        if (isa<asctile::TileType>(op.getType()) || !allUnscalarizableUsers(op))
            return;
        op->setAttr(unscalarizeAttr, UnitAttr::get(op.getContext()));
        for (auto* user : op->getUsers())
            user->setAttr(unscalarizeAttr, UnitAttr::get(op.getContext()));
    });
}

Value addUnrealizedCast(OpBuilder& builder, Type type, ValueRange inputs, Location loc)
{
    return builder.create<UnrealizedConversionCastOp>(loc, type, inputs).getResult(0);
}

struct UnscalarizeReductionPass : public asctile::impl::UnscalarizeReductionBase<UnscalarizeReductionPass> {
    void runOnOperation() override
    {
        func::FuncOp op = getOperation();
        markOps<asctile::ReduceMinAs1dOp>(op);
        markOps<asctile::ReduceMaxAs1dOp>(op);
        markOps<asctile::ReduceSumAs1dOp>(op);
        TypeConverter converter;
        converter.addConversion([](Type type) { return std::optional<Type>{type}; });
        converter.addConversion([](IntegerType type) { return asctile::TileType::get(1, type); });
        converter.addConversion([](FloatType type) { return asctile::TileType::get(1, type); });
        converter.addArgumentMaterialization(addUnrealizedCast);
        converter.addSourceMaterialization(addUnrealizedCast);
        converter.addTargetMaterialization(addUnrealizedCast);
        MLIRContext* context = &getContext();
        ConversionTarget target(*context);
        target.markUnknownOpDynamicallyLegal([](Operation* op) { return !op->hasAttr(unscalarizeAttr); });
        RewritePatternSet patterns(context);
        patterns.insert<
            //
            ConvertTileScalar<asctile::AddSOp, arith::AddFOp, arith::AddIOp>,
            ConvertTileScalar<asctile::SubSOp, arith::SubFOp, arith::SubIOp>,
            ConvertTileScalar<asctile::MulSOp, arith::MulFOp, arith::MulIOp>,
            ConvertTileScalar<asctile::DivSOp, arith::DivFOp, arith::DivSIOp>,
            ConvertTileScalar<asctile::MinSOp, arith::MinimumFOp, arith::MinSIOp>,
            ConvertTileScalar<asctile::MaxSOp, arith::MaximumFOp, arith::MaxSIOp>,
            //
            ConvertReduce<asctile::ReduceMaxAs1dOp>, ConvertReduce<asctile::ReduceMinAs1dOp>,
            ConvertReduce<asctile::ReduceSumAs1dOp>, ConvertSplat
            //
            >(converter, context);
        if (applyFullConversion(op, target, std::move(patterns)).failed())
            signalPassFailure();
    }
};

} // namespace

std::unique_ptr<Pass> mlir::asctile::createUnscalarizeReductionPass()
{
    return std::make_unique<UnscalarizeReductionPass>();
}
