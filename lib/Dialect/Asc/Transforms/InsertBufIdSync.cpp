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
#include "ascir/Dialect/Asc/Utils/Attributes.h"
#include "ascir/Dialect/Utils/ConstantOpBuilder.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Interfaces/LoopLikeInterface.h"

namespace mlir {
namespace ascendc {
#define GEN_PASS_DEF_INSERTBUFIDSYNC
#include "ascir/Dialect/Asc/Transforms/Passes.h.inc"
} // namespace ascendc
} // namespace mlir

using namespace mlir;

namespace {

using VisitedOpsSet = std::set<std::pair<Operation*, int32_t>>;
using VisitedBufIdMap = DenseMap<Operation*, SmallVector<int32_t>>;

void insertGetRlsBuf(Operation* op, ascendc::Pipe pipe, int32_t bufId)
{
    OpBuilder builder(op);
    ascir::ConstantOpBuilder consts(builder);
    builder.create<ascendc::GetBufOp>(op->getLoc(), pipe, consts.i32(bufId), false);
    builder.setInsertionPointAfter(op);
    builder.create<ascendc::RlsBufOp>(op->getLoc(), pipe, consts.i32(bufId), false);
}

void recursiveVisit(Operation* op, int32_t bufId, VisitedOpsSet& visitedOps, VisitedBufIdMap& bufIdMap)
{
    if (!op)
        return;
    if (visitedOps.find({op, bufId}) != visitedOps.end())
        return;
    visitedOps.insert({op, bufId});
    if (auto vecOp = dyn_cast<ascendc::VectorOp>(op)) {
        bufIdMap[vecOp].push_back(bufId);
        insertGetRlsBuf(vecOp, ascendc::Pipe::PIPE_V, bufId);
    } else if (auto copyOp = dyn_cast<ascendc::DataCopyOp>(op)) {
        bufIdMap[copyOp].push_back(bufId);
        auto direction = copyOp.getDirection();
        if (direction == ascendc::CopyDirection::gm_ubuf) {
            insertGetRlsBuf(copyOp, ascendc::Pipe::PIPE_MTE2, bufId);
        } else if (direction == ascendc::CopyDirection::ubuf_gm) {
            insertGetRlsBuf(copyOp, ascendc::Pipe::PIPE_MTE3, bufId);
        } else if (direction == ascendc::CopyDirection::ubuf_ubuf) {
            insertGetRlsBuf(copyOp, ascendc::Pipe::PIPE_V, bufId);
        }
    }
    if (auto loopOp = dyn_cast<LoopLikeOpInterface>(op)) {
        for (auto* region : loopOp.getLoopRegions()) {
            for (Operation& blockOp : region->getOps()) {
                recursiveVisit(&blockOp, bufId, visitedOps, bufIdMap);
            }
        }
    }
    for (auto operand : op->getOperands()) {
        if (!isa<ascendc::LocalTensorType>(operand.getType()))
            continue;
        for (auto* user : operand.getUsers()) {
            recursiveVisit(user, bufId, visitedOps, bufIdMap);
        }
        recursiveVisit(operand.getDefiningOp(), bufId, visitedOps, bufIdMap);
    }
}

void syncOperations(Region& region, VisitedOpsSet& visitedOps, int32_t& bufId, VisitedBufIdMap& bufIdMap)
{
    for (Block& block : region) {
        for (Operation& op : llvm::make_early_inc_range(block)) {
            for (Region& inner : op.getRegions()) {
                syncOperations(inner, visitedOps, bufId, bufIdMap);
            }
            auto copyOp = dyn_cast<ascendc::DataCopyOp>(op);
            if (!copyOp || copyOp.getDirection() == ascendc::CopyDirection::gm_ubuf ||
                copyOp.getDirection() == ascendc::CopyDirection::ubuf_ubuf)
                continue;
            recursiveVisit(copyOp, bufId, visitedOps, bufIdMap);
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
        VisitedOpsSet visitedOps;
        VisitedBufIdMap bufIdMap;
        int32_t bufId = 0;
        syncOperations(funcOp.getRegion(), visitedOps, bufId, bufIdMap);
        for (auto& [op, bufIdVec] : bufIdMap) {
            OpBuilder builder(op);
            op->setAttr(ascendc::attr::bufId, builder.getI32ArrayAttr(bufIdVec));
        }
    }
};
} // namespace

namespace mlir {
namespace ascendc {
std::unique_ptr<Pass> createInsertBufIdSyncPass() { return std::make_unique<InsertBufIdSyncPass>(); }
} // namespace ascendc
} // namespace mlir
