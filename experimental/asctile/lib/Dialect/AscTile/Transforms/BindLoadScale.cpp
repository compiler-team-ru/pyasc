/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "asctile/Dialect/AscTile/IR/AscTile.h"
#include "asctile/Dialect/AscTile/Transforms/Passes.h"
#include "asctile/Dialect/AscTile/Utils/Attributes.h"

#include "ascir/Dialect/Asc/Utils/Utils.h"
#include "ascir/Dialect/Utils/ConstantOpBuilder.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace asctile {
#define GEN_PASS_DEF_BINDLOADSCALE
#include "asctile/Dialect/AscTile/Transforms/Passes.h.inc"
} // namespace asctile
} // namespace mlir

using namespace mlir;
using namespace mlir::asctile;

namespace {

struct ReplaceBindScale : public OpRewritePattern<asctile::BindScaleOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult matchAndRewrite(asctile::BindScaleOp op, PatternRewriter& rewriter) const override
    {
        auto loadBaseOp = op.getBase().getDefiningOp<asctile::CopyOp>();
        if (!loadBaseOp)
            return failure();
        auto loadScaleOp = op.getScale().getDefiningOp<asctile::CopyOp>();
        if (!loadScaleOp)
            return failure();
        if (loadBaseOp.getDistrib() != loadScaleOp.getDistrib())
            return op.emitError("base and scale distrib modes not match");
        if (loadScaleOp->hasAttrOfType<UnitAttr>(asctile::attr::transposeAL0) ||
            loadScaleOp->hasAttrOfType<UnitAttr>(asctile::attr::transposeAL1)) {
            return op.emitError("Transposing scale in L0 not supported");
        }
        auto loadOp = loadScaleOp.getBase().getDefiningOp<asctile::LoadOp>();
        bool isMatrixA = loadScaleOp.getType().getLoc() == asctile::TensorLocation::L0A;
        if (!loadOp) {
            return failure();
        }
        loadOp->setAttr(asctile::attr::loadScale, UnitAttr::get(loadOp.getContext()));
        if (isMatrixA) {
            loadOp->setAttr(asctile::attr::isMatrixA, UnitAttr::get(loadOp.getContext()));
        }

        DominanceInfo di;
        bool baseLast = ascendc::opPrecedes(loadScaleOp, loadBaseOp, di);
        rewriter.setInsertionPoint(baseLast ? loadBaseOp : loadScaleOp);
        auto newOp = rewriter.create<asctile::CopyScaleOp>(
            op->getLoc(), loadBaseOp.getResult().getType(), loadBaseOp.getBase(), ValueRange{loadBaseOp.getOffsets()},
            loadScaleOp.getBase(), ValueRange{loadScaleOp.getOffsets()}, loadBaseOp.getDistribAttr());
        rewriter.replaceOp(op, newOp);
        return success();
    }
};

struct BindLoadScalePass : public asctile::impl::BindLoadScaleBase<BindLoadScalePass> {
    void runOnOperation() override
    {
        func::FuncOp funcOp = getOperation();
        MLIRContext* context = &getContext();
        RewritePatternSet patterns(context);
        patterns.add<ReplaceBindScale>(context);
        if (applyPatternsAndFoldGreedily(funcOp, std::move(patterns)).failed())
            signalPassFailure();
    }
};

} // namespace

std::unique_ptr<Pass> mlir::asctile::createBindLoadScalePass() { return std::make_unique<BindLoadScalePass>(); }
