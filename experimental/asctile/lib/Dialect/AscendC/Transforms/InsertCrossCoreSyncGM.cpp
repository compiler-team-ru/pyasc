/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "asctile/Dialect/AscendC/Transforms/Passes.h"
#include "asctile/Dialect/AscendC/Utils/Attributes.h"
#include "asctile/Dialect/AscendC/Utils/Utils.h"

#include "ascir/Dialect/Asc/IR/Asc.h"

#include "CrossCoreSyncUtils.h"

namespace mlir {
namespace ascendc {
#define GEN_PASS_DEF_INSERTCROSSCORESYNCGM
#include "asctile/Dialect/AscendC/Transforms/Passes.h.inc"
} // namespace ascendc
} // namespace mlir

using namespace mlir;
using namespace mlir::ascendc;

namespace {

struct GMFlagInfo {
    int32_t flagId = -1;
    bool isUBToGM = false;
};

bool isL0CToGMCopy(Operation* op)
{
    return isDataCopyOp(op, [](ascendc::TPosition src, ascendc::TPosition dst) {
        return src == ascendc::TPosition::CO1 && dst == ascendc::TPosition::GM;
    });
}

bool isUBToGMCopy(Operation* op)
{
    return isDataCopyOp(op, [](ascendc::TPosition src, ascendc::TPosition dst) {
        return src == ascendc::TPosition::VECCALC && dst == ascendc::TPosition::GM;
    });
}

Value getRootGlobalTensor(Value v)
{
    while (auto subOp = v.getDefiningOp<ascendc::GlobalTensorSubIndexOp>())
        v = subOp.getTensor();
    return v;
}

GMFlagInfo findGMLoadFlag(Operation* groupOp, const llvm::DenseMap<Value, GMFlagInfo>& gmRootsWithFlags)
{
    GMFlagInfo info;
    groupOp->walk([&](Operation* op) {
        if (op == groupOp)
            return WalkResult::advance();
        if (isDataCopyOp(op, [](ascendc::TPosition, ascendc::TPosition dst) { return dst == ascendc::TPosition::GM; }))
            return WalkResult::advance();
        for (Value operand : op->getOperands()) {
            if (!isa<ascendc::GlobalTensorType>(operand.getType()))
                continue;
            Value root = getRootGlobalTensor(operand);
            if (auto it = gmRootsWithFlags.find(root); it != gmRootsWithFlags.end()) {
                info = it->second;
                return WalkResult::interrupt();
            }
        }
        return WalkResult::advance();
    });
    return info;
}

struct InsertCrossCoreSyncGMPass : public ascendc::impl::InsertCrossCoreSyncGMBase<InsertCrossCoreSyncGMPass> {
    void runOnOperation() override
    {
        func::FuncOp funcOp = getOperation();
        SmallVector<Operation*> groupOps = collectGroupOps(funcOp);
        if (groupOps.size() < 2) {
            funcOp->removeAttr(attr::crossCoreFlagId);
            return;
        }
        OpBuilder builder(funcOp.getContext());
        int32_t crossCoreFlagId = 0;
        if (auto attr = funcOp->getAttrOfType<IntegerAttr>(ascendc::attr::crossCoreFlagId))
            crossCoreFlagId = static_cast<int32_t>(attr.getValue().getSExtValue());
        llvm::DenseMap<Value, GMFlagInfo> gmRootsWithFlags;
        for (auto* groupOp : groupOps) {
            if (!gmRootsWithFlags.empty()) {
                GMFlagInfo info = findGMLoadFlag(groupOp, gmRootsWithFlags);
                if (info.flagId >= 0) {
                    Block& body = groupOp->getRegion(0).front();
                    builder.setInsertionPointToStart(&body);
                    createWaitFlag(builder, groupOp->getLoc(), info.flagId, ascendc::Pipe::PIPE_S);
                    if (info.isUBToGM)
                        createWaitFlag(builder, groupOp->getLoc(), info.flagId + maxTensorId, ascendc::Pipe::PIPE_S);
                }
            }
            Block& body = groupOp->getRegion(0).front();
            auto emitProducerFlags = [&](llvm::function_ref<bool(Operation*)> pred, bool emitPair, bool isUBToGM) {
                SmallVector<Operation*> ops = collectOps(groupOp, pred);
                if (ops.empty())
                    return;
                builder.setInsertionPoint(body.getTerminator());
                for (auto* op : ops) {
                    ascendc::Pipe producerPipe = ascendc::getOpPipeExt(op);
                    int32_t flagId = crossCoreFlagId;
                    crossCoreFlagId = (crossCoreFlagId + 1) % maxTensorId;
                    createSetFlag(builder, op->getLoc(), flagId, producerPipe);
                    if (emitPair)
                        createSetFlag(builder, op->getLoc(), flagId + maxTensorId, producerPipe);
                    Value root = getRootGlobalTensor(cast<ascendc::DataCopyOp>(op).getDst());
                    gmRootsWithFlags[root] = {flagId, isUBToGM};
                }
            };
            emitProducerFlags(isL0CToGMCopy, true, false);
            emitProducerFlags(isUBToGMCopy, false, true);
        }
        funcOp->removeAttr(attr::crossCoreFlagId);
    }
};

} // namespace

std::unique_ptr<Pass> mlir::ascendc::createInsertCrossCoreSyncGMPass()
{
    return std::make_unique<InsertCrossCoreSyncGMPass>();
}
