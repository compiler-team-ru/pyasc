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

#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace mlir {
namespace asctile {
#define GEN_PASS_DEF_PREPARECVSTRATEGY
#include "asctile/Dialect/AscTile/Transforms/Passes.h.inc"
} // namespace asctile
} // namespace mlir

using namespace mlir;
using namespace mlir::asctile;

namespace {

enum struct CopyVerdict { Accept, Fail, Skip };

CopyVerdict getResultShape(CopyOp op, SplitMode split, SmallVectorImpl<int64_t>& shape)
{
    if (auto copySplit = op.getSplit(); copySplit.value_or(split) != split) {
        op.emitOpError() << "has incompatible 'split' argument: " << stringifySplitMode(*copySplit);
        return CopyVerdict::Fail;
    }
    auto srcType = op.getBase().getType();
    auto dstType = op.getType();
    if (srcType.getLoc() != TensorLocation::L0C || dstType.getLoc() != TensorLocation::UB)
        return CopyVerdict::Skip;
    assert(srcType.getRank() == dstType.getRank());
    if (srcType.getRank() != 2) {
        op->emitError() << "Splitting by axis is only supported for 2D tensors, got " << srcType.getRank() << "D";
        return CopyVerdict::Fail;
    }
    auto srcShape = srcType.getShape();
    auto dstShape = dstType.getShape();
    if (srcShape != dstShape) {
        op->emitError() << "Splitting by axis is only supported for full shape " << srcShape << ", got " << dstShape;
        return CopyVerdict::Fail;
    }
    if (split == SplitMode::SplitByM) {
        if (dstShape.front() % 2 != 0) {
            op->emitError() << "Splitting by M axis requires that it be a multiple of 2, got " << dstShape.front();
            return CopyVerdict::Fail;
        }
        shape.push_back(dstShape.front() / 2);
    } else {
        shape.push_back(dstShape.front());
    }
    if (split == SplitMode::SplitByN) {
        if (dstShape.back() % 32 != 0) {
            op->emitError() << "Splitting by N axis requires that it be a multiple of 32, got " << dstShape.back();
            return CopyVerdict::Fail;
        }
        shape.push_back(dstShape.back() / 2);
    } else {
        shape.push_back(dstShape.back());
    }
    return CopyVerdict::Accept;
}

struct PrepareCVStrategyPass : public asctile::impl::PrepareCVStrategyBase<PrepareCVStrategyPass> {
    void ensureStrategy(CVStrategyOp root);

    void runOnOperation() override
    {
        getOperation().walk([this](CVStrategyOp op) { ensureStrategy(op); });
    }
};

void PrepareCVStrategyPass::ensureStrategy(CVStrategyOp root)
{
    auto filter = [root](Operation* op) { return op->getParentOfType<CVStrategyOp>() == root; };
    root.walk([&](CopyOp op) {
        SmallVector<int64_t, 2> shape;
        auto verdict = getResultShape(op, root.getSplit(), shape);
        if (verdict == CopyVerdict::Fail) {
            signalPassFailure();
            return;
        }
        if (verdict == CopyVerdict::Skip)
            return;
        assert(verdict == CopyVerdict::Accept);
        SetVector<Operation*> users;
        users.insert(op);
        getForwardSlice(op.getResult(), &users, ForwardSliceOptions(filter));
        Builder builder(op);
        auto needSplitAttr = builder.getAttr<SplitModeAttr>(root.getSplit());
        auto splitShapeAttr = builder.getDenseI64ArrayAttr(shape);
        for (auto* user : users) {
            if (user->hasAttr(attr::needSplit)) {
                user->emitError("Splitting is already requested by other asctile.copy op");
                signalPassFailure();
                return;
            }
            user->setAttr(attr::needSplit, needSplitAttr);
            user->setAttr(attr::splitShape, splitShapeAttr);
        }
    });
}

} // namespace

std::unique_ptr<Pass> mlir::asctile::createPrepareCVStrategyPass() { return std::make_unique<PrepareCVStrategyPass>(); }
