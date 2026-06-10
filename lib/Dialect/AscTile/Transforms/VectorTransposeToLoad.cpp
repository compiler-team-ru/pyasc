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
#include "ascir/Dialect/AscTile/Utils/Attributes.h"
#include "ascir/Dialect/Utils/ConstantOpBuilder.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace asctile {
#define GEN_PASS_DEF_VECTORTRANSPOSETOLOAD
#include "ascir/Dialect/AscTile/Transforms/Passes.h.inc"
} // namespace asctile
} // namespace mlir

using namespace mlir;
using namespace mlir::asctile;

namespace {

struct VectorTransposeToLoad : OpRewritePattern<asctile::TransposeOp> {
    using OpRewritePattern<asctile::TransposeOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(TransposeOp op, PatternRewriter& rewriter) const override
    {
        ascir::ConstantOpBuilder consts(rewriter);
        auto loadOp = op.getOperand().getDefiningOp<asctile::LoadOp>();
        if (!loadOp || op.getType().getLoc() != TileLocation::UB || !op.getOperand().hasOneUse()) {
            return failure();
        }
        auto shape = op.getOperand().getType().getShape();
        auto newLoadOp = rewriter.replaceOpWithNewOp<asctile::LoadOp>(
            op, op.getType(), loadOp.getBase(), loadOp.getOffsets(), loadOp.getPadValue(), loadOp.getRealShape());
        rewriter.startOpModification(newLoadOp);
        SmallVector<int32_t> dimOrder;
        for (auto value : op.getDims().getAsValueRange<IntegerAttr>()) {
            dimOrder.push_back(static_cast<int32_t>(value.getSExtValue()));
        }
        newLoadOp->setAttr(asctile::attr::transposeDims, rewriter.getDenseI32ArrayAttr(dimOrder));
        rewriter.finalizeOpModification(newLoadOp);
        return success();
    }
};

struct VectorTransposeToLoadPass : public asctile::impl::VectorTransposeToLoadBase<VectorTransposeToLoadPass> {
    void runOnOperation() override
    {
        auto op = getOperation();
        MLIRContext* context = &getContext();
        RewritePatternSet patterns(context);
        patterns.add<VectorTransposeToLoad>(context);
        if (applyPatternsAndFoldGreedily(op, std::move(patterns)).failed()) {
            signalPassFailure();
            return;
        }
        op->walk([this](asctile::TransposeOp op) {
            if (op.getType().getLoc() == TileLocation::UB) {
                op.emitOpError() << "not supported for UB";
                signalPassFailure();
            }
        });
    }
};

} // namespace

std::unique_ptr<Pass> mlir::asctile::createVectorTransposeToLoadPass()
{
    return std::make_unique<VectorTransposeToLoadPass>();
}
