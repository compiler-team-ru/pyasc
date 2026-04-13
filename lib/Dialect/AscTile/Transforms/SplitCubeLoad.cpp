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

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace asctile {
#define GEN_PASS_DEF_SPLITCUBELOAD
#include "ascir/Dialect/AscTile/Transforms/Passes.h.inc"
} // namespace asctile
} // namespace mlir

using namespace mlir;
using namespace mlir::asctile;

namespace {

struct ConvertLoadGMToL0 : OpRewritePattern<asctile::LoadOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult matchAndRewrite(asctile::LoadOp op, PatternRewriter &rewriter) const override
    {
        auto opType = op.getType();
        auto tileLoc = opType.getLoc();
        if (tileLoc != TileLocation::L0A && tileLoc != TileLocation::L0B) {
            return failure();
        }
        auto l1Type = TileType::get(opType.getShape(), opType.getElementType(), TileLocation::L1);
        Value l1Tile =
            rewriter.create<asctile::LoadOp>(op.getLoc(), l1Type, op.getBase(), op.getOffsets(), op.getPadValue());
        Value zero = rewriter.create<arith::ConstantOp>(op.getLoc(), rewriter.getI32IntegerAttr(0));
        SmallVector<Value> offsets(opType.getShape().size(), zero);
        rewriter.replaceOpWithNewOp<asctile::CopyOp>(op, op.getType(), l1Tile, offsets);
        return success();
    }
};

struct MarkTileOperandInMmad : OpRewritePattern<asctile::LoadOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult matchAndRewrite(asctile::LoadOp op, PatternRewriter &rewriter) const override
    {
        auto opType = op.getType();
        auto tileLoc = opType.getLoc();
        if (opType.getLoc() != TileLocation::L1 || op->use_empty() || op->hasAttr(attr::isMatrixA)) {
            return failure();
        }

        std::optional<bool> isTensorA;
        for (auto &use : op->getUses()) {
            auto copyOp = dyn_cast<asctile::CopyOp>(use.getOwner());
            if (!copyOp) {
                op.emitError() << "L1 tile is expected to be used for copy operations only.";
                return failure();
            }
            auto l0TileLoc = copyOp.getType().getLoc();
            if (l0TileLoc != TileLocation::L0A && l0TileLoc != TileLocation::L0B) {
                auto diag = op.emitError() << "L1 tile copy to L0A/L0B location is expected only.";
                diag.attachNote(copyOp->getLoc()) << "used here unexpectedly";
                return failure();
            }
            if (!isTensorA.has_value()) {
                isTensorA = l0TileLoc == TileLocation::L0A;
            } else if (isTensorA.value() != (l0TileLoc == TileLocation::L0A)) {
                auto diag = op.emitError()
                            << "The same L1 tile should be copied only to tiles in same L0A/L0B location.";
                diag.attachNote(copyOp->getLoc()) << "copied here unexpectedly";
                return failure();
            }
        }
        if (isTensorA.value()) {
            rewriter.modifyOpInPlace(op, [&]() { op->setAttr(attr::isMatrixA, rewriter.getUnitAttr()); });
            return success();
        }

        // If we are here it means that processed L1 tile is for B.
        return failure();
    }
};

struct SplitCubeLoadPass : public asctile::impl::SplitCubeLoadBase<SplitCubeLoadPass> {
    void runOnOperation() override
    {
        func::FuncOp funcOp = getOperation();
        MLIRContext *context = &getContext();
        RewritePatternSet patterns(context);
        patterns.insert<ConvertLoadGMToL0, MarkTileOperandInMmad>(context);
        if (applyPatternsAndFoldGreedily(funcOp, std::move(patterns)).failed())
            signalPassFailure();
    }
};

} // namespace

std::unique_ptr<Pass> mlir::asctile::createSplitCubeLoadPass()
{
    return std::make_unique<SplitCubeLoadPass>();
}
