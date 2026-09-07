/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ascir/Dialect/AscTile/IR/AscTile.h"
#include "ascir/Dialect/AscTile/Transforms/Passes.h"
#include "ascir/Dialect/AscTile/Utils/Utils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace asctile {
#define GEN_PASS_DEF_APPLYHOMOMORPHISM
#include "ascir/Dialect/AscTile/Transforms/Passes.h.inc"
} // namespace asctile
} // namespace mlir

using namespace mlir;

namespace {

// cast(select(mask, splat(x), splat(y))) -> select(mask, splat(cast(x)), splat(cast(y)))
struct PushCastThroughSelect : OpRewritePattern<asctile::CastOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult matchAndRewrite(asctile::CastOp op, PatternRewriter& rewriter) const override
    {
        if (op.getRoundMode() != asctile::RoundMode::Default)
            return failure();
        auto select = op.getOperand().getDefiningOp<arith::SelectOp>();
        if (!select)
            return failure();
        auto lhs = asctile::materializeSplatValue(rewriter, select.getTrueValue());
        auto rhs = asctile::materializeSplatValue(rewriter, select.getFalseValue());
        if (!lhs || !rhs)
            return failure();
        auto tensor = op.getType();
        auto lhsCasted = convertScalarToDtype(rewriter, lhs.getLoc(), lhs, tensor.getElementType(), false);
        auto rhsCasted = convertScalarToDtype(rewriter, rhs.getLoc(), rhs, tensor.getElementType(), false);
        if (lhsCasted == lhs || rhsCasted == rhs) // convertScalarToDtype failed
            return failure();
        Value newLhs = rewriter.create<tensor::SplatOp>(lhsCasted.getLoc(), tensor, lhsCasted);
        Value newRhs = rewriter.create<tensor::SplatOp>(lhsCasted.getLoc(), tensor, rhsCasted);
        rewriter.replaceOpWithNewOp<arith::SelectOp>(op, tensor, select.getCondition(), newLhs, newRhs);
        return success();
    }
};

// op(splat(x), ...) -> splat(op(x, ...))
struct PushSplatThroughScalarizable : OpTraitRewritePattern<OpTrait::Scalarizable> {
    using OpTraitRewritePattern::OpTraitRewritePattern;

    LogicalResult matchAndRewrite(Operation* op, PatternRewriter& rewriter) const override
    {
        if (!isPure(op) || op->getNumOperands() == 0 || op->getNumRegions() != 0)
            return failure();
        SmallVector<OpFoldResult, 4> splatOperands;
        for (auto operand : op->getOperands()) {
            auto splat = asctile::getSplatValue(operand);
            if (!splat)
                return failure();
            splatOperands.push_back(splat);
        }
        SmallVector<Type, 2> splatResults;
        for (auto type : op->getResultTypes()) {
            auto tensor = dyn_cast<asctile::LocalTensorType>(type);
            if (!tensor)
                return failure();
            splatResults.push_back(tensor.getElementType());
        }
        SmallVector<Value, 4> newOperands;
        for (auto ofr : splatOperands) {
            if (auto value = dyn_cast<Value>(ofr))
                newOperands.push_back(value);
            else
                newOperands.push_back(
                    rewriter.create<arith::ConstantOp>(op->getLoc(), cast<TypedAttr>(cast<Attribute>(ofr))));
        }
        Operation* newOp = rewriter.create(
            op->getLoc(), op->getName().getIdentifier(), newOperands, splatResults, op->getAttrs(),
            op->getSuccessors());
        SmallVector<Value, 2> newResults;
        for (auto [scalarResult, tensorType] : llvm::zip_equal(newOp->getResults(), op->getResultTypes())) {
            Value splat = rewriter.create<tensor::SplatOp>(op->getLoc(), tensorType, scalarResult);
            newResults.push_back(splat);
        }
        rewriter.replaceOp(op, newResults);
        return success();
    }
};

struct ApplyHomomorphismPass : public asctile::impl::ApplyHomomorphismBase<ApplyHomomorphismPass> {
    void runOnOperation() override
    {
        func::FuncOp funcOp = getOperation();
        MLIRContext* context = &getContext();
        RewritePatternSet patterns(context);
        patterns.add<PushCastThroughSelect, PushSplatThroughScalarizable>(context);
        if (applyPatternsAndFoldGreedily(funcOp, std::move(patterns)).failed())
            signalPassFailure();
    }
};

} // namespace

std::unique_ptr<Pass> mlir::asctile::createApplyHomomorphismPass() { return std::make_unique<ApplyHomomorphismPass>(); }
