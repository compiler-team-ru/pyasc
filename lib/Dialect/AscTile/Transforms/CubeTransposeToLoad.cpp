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
#define GEN_PASS_DEF_CUBETRANSPOSETOLOAD
#include "ascir/Dialect/AscTile/Transforms/Passes.h.inc"
} // namespace asctile
} // namespace mlir

using namespace mlir;
using namespace mlir::asctile;

namespace {

struct CubeTransposeToLoad : OpRewritePattern<asctile::TransposeOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult matchAndRewrite(asctile::TransposeOp op, PatternRewriter& rewriter) const override
    {
        auto opType = op.getType();
        auto tileLoc = opType.getLoc();
        if (tileLoc != TileLocation::L0A && tileLoc != TileLocation::L0B)
            return failure();
        auto copyOp = op.getIn().getDefiningOp<asctile::CopyOp>();
        if (!copyOp || !copyOp->hasOneUse())
            return failure();
        auto attr = tileLoc == TileLocation::L0A ? asctile::attr::transposeA : asctile::attr::transposeB;
        auto copyOpType = copyOp.getType();
        auto shape = copyOpType.getShape();
        if (shape.size() != 2)
            return failure();
        Type newType = TileType::get({shape[1], shape[0]}, opType.getElementType(), copyOpType.getLoc());
        auto newCopyOp = rewriter.create<asctile::CopyOp>(op.getLoc(), newType, copyOp.getBase(), copyOp.getOffsets());
        rewriter.replaceOp(copyOp, newCopyOp);
        rewriter.startOpModification(newCopyOp);
        newCopyOp->setAttr(attr, rewriter.getUnitAttr());
        rewriter.finalizeOpModification(newCopyOp);
        auto loadOp = newCopyOp.getBase().getDefiningOp();
        if (!loadOp || !loadOp->hasOneUse())
            return failure();
        rewriter.startOpModification(loadOp);
        loadOp->setAttr(attr, rewriter.getUnitAttr());
        rewriter.finalizeOpModification(loadOp);
        rewriter.replaceOp(op, newCopyOp);
        return success();
    }
};

struct CubeTransposeToLoadPass : public asctile::impl::CubeTransposeToLoadBase<CubeTransposeToLoadPass> {
    void runOnOperation() override
    {
        func::FuncOp funcOp = getOperation();
        MLIRContext* context = &getContext();
        RewritePatternSet patterns(context);
        patterns.insert<CubeTransposeToLoad>(context);
        if (applyPatternsAndFoldGreedily(funcOp, std::move(patterns)).failed()) {
            signalPassFailure();
            return;
        }
        funcOp->walk([this](asctile::TransposeOp op) {
            auto location = op.getType().getLoc();
            if (location == TileLocation::L0A || location == TileLocation::L0B) {
                op.emitOpError() << "not supported on cube";
                signalPassFailure();
            }
        });
    }
};

} // namespace

std::unique_ptr<Pass> mlir::asctile::createCubeTransposeToLoadPass()
{
    return std::make_unique<CubeTransposeToLoadPass>();
}
