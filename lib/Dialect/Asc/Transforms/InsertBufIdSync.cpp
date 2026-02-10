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
#include "ascir/Dialect/Asc/Transforms/Passes.h"
#include "ascir/Dialect/Utils/ConstantOpBuilder.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

namespace mlir {
namespace ascendc {
#define GEN_PASS_DEF_INSERTBUFIDSYNC
#include "ascir/Dialect/Asc/Transforms/Passes.h.inc"
} // namespace ascendc
} // namespace mlir

using namespace mlir;

namespace {

using VisitedOpsSet = std::set<std::pair<Operation*, int32_t>>;

void insertGetRlsBuf(Operation* op, ascendc::Pipe pipe, int32_t bufId)
{
    OpBuilder builder(op);
    ascir::ConstantOpBuilder consts(builder);
    builder.create<ascendc::GetBufOp>(op->getLoc(), pipe, consts.i32(bufId), false);
    builder.setInsertionPointAfter(op);
    builder.create<ascendc::RlsBufOp>(op->getLoc(), pipe, consts.i32(bufId), false);
}

void recursiveVisit(Value arg, int32_t bufId, VisitedOpsSet& visitedOps)
{
    for (auto& use : arg.getUses()) {
        auto userOp = use.getOwner();
        if (visitedOps.find({userOp, bufId}) != visitedOps.end())
            continue;
        if (auto vecOp = dyn_cast<ascendc::VectorOp>(userOp)) {
            visitedOps.insert({vecOp, bufId});
            insertGetRlsBuf(vecOp, ascendc::Pipe::PIPE_V, bufId);
            for (auto operand : vecOp->getOperands()) {
                recursiveVisit(operand, bufId, visitedOps);
            }
        } else if (auto copyOp = dyn_cast<ascendc::DataCopyOp>(userOp)) {
            if (copyOp.getDirection() == ascendc::CopyDirection::gm_ubuf) {
                visitedOps.insert({copyOp, bufId});
                insertGetRlsBuf(copyOp, ascendc::Pipe::PIPE_MTE2, bufId);
            }
        }
    }
}

void syncOperations(Region& region, VisitedOpsSet& visitedOps, int32_t& bufId)
{
    for (Block& block : region) {
        for (Operation& op : llvm::make_early_inc_range(block)) {
            for (Region& inner : op.getRegions()) {
                syncOperations(inner, visitedOps, bufId);
            }
            auto copyOp = dyn_cast<ascendc::DataCopyOp>(op);
            if (!copyOp || copyOp.getDirection() == ascendc::CopyDirection::gm_ubuf)
                continue;
            visitedOps.insert({copyOp, bufId});
            insertGetRlsBuf(copyOp, ascendc::Pipe::PIPE_MTE3, bufId);
            recursiveVisit(copyOp.getSrc(), bufId, visitedOps);
            bufId++;
        }
    }
}

class InsertBufIdSyncPass : public ascendc::impl::InsertBufIdSyncBase<InsertBufIdSyncPass> {
    void runOnOperation() override
    {
        func::FuncOp funcOp = getOperation();
        if (funcOp.isDeclaration()) {
            return;
        }
        MLIRContext* context = &getContext();
        RewritePatternSet patterns(context);
        VisitedOpsSet visitedOps;
        int32_t bufId = 0;
        syncOperations(funcOp.getRegion(), visitedOps, bufId);
    }
};
} // namespace

namespace mlir {
namespace ascendc {
std::unique_ptr<Pass> createInsertBufIdSyncPass() { return std::make_unique<InsertBufIdSyncPass>(); }
} // namespace ascendc
} // namespace mlir
