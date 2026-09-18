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

#include "mlir/Dialect/SCF/IR/SCF.h"

#include "CrossCoreSyncUtils.h"

namespace mlir {
namespace ascendc {
#define GEN_PASS_DEF_INSERTCROSSCORESYNC
#include "asctile/Dialect/AscendC/Transforms/Passes.h.inc"
} // namespace ascendc
} // namespace mlir

using namespace mlir;
using namespace mlir::ascendc;

namespace {

bool isSyncTriggerOp(Operation* op)
{
    return isDataCopyOp(op, [](ascendc::TPosition src, ascendc::TPosition dst) {
        return (src == ascendc::TPosition::VECCALC &&
                (dst == ascendc::TPosition::A1 || dst == ascendc::TPosition::B1)) ||
               (src == ascendc::TPosition::CO1 && dst == ascendc::TPosition::VECCALC);
    });
}

Operation* findGroupAncestor(Operation* op)
{
    for (Operation* p = op->getParentOp(); p; p = p->getParentOp())
        if (isa<ascendc::IfAICOp, ascendc::IfAIVOp>(p))
            return p;
    return nullptr;
}

Operation* findSyncInsertionPoint(Operation* user)
{
    for (Operation* p = user->getParentOp(); p; p = p->getParentOp()) {
        if (isa<ascendc::IfAICOp, ascendc::IfAIVOp>(p))
            return user;
    }
    return user;
}

SmallVector<Operation*> collectUsers(Operation* dstDefiningOp, bool triggerIsAIV, Operation* afterOp)
{
    SmallVector<Operation*> users;
    bool seen = !afterOp;
    dstDefiningOp->getParentOfType<func::FuncOp>()->walk([&](Operation* op) {
        if (op == afterOp) {
            seen = true;
            return;
        }
        if (!seen || isa<ascendc::IfAICOp, ascendc::IfAIVOp>(op))
            return;
        if (!llvm::any_of(op->getOperands(), [&](Value v) { return v.getDefiningOp() == dstDefiningOp; }))
            return;
        if (Operation* group = findGroupAncestor(op); isa_and_present<ascendc::IfAICOp>(group) == triggerIsAIV)
            users.push_back(op);
    });
    return users;
}

void processUsers(OpBuilder& builder, ArrayRef<Operation*> users, int32_t flagId)
{
    Operation* first = users.front();
    Operation* firstGroup = findGroupAncestor(first);
    Operation* last = first;
    for (auto* user : users) {
        if (findGroupAncestor(user) != firstGroup)
            break;
        last = user;
    }
    builder.setInsertionPoint(findSyncInsertionPoint(first));
    createWaitFlag(builder, first->getLoc(), flagId, ascendc::getOpPipeExt(first));
    Operation* setFlagPos = findSyncInsertionPoint(last);
    builder.setInsertionPointAfter(isa<scf::ForOp>(setFlagPos) ? setFlagPos : last);
    createSetFlag(builder, last->getLoc(), flagId, ascendc::getOpPipeExt(last));
}

template <typename FlagOp>
void insertFlagGroup(OpBuilder& builder, Operation* op, bool isAIV, int32_t flagId, Pipe pipe)
{
    Location loc = op->getLoc();
    ascir::ConstantOpBuilder consts(builder);
    Operation* group = isAIV ? builder.create<ascendc::IfAIVOp>(loc, TypeRange{}, ValueRange{}).getOperation() :
                               builder.create<ascendc::IfAICOp>(loc, TypeRange{}, ValueRange{}).getOperation();
    builder.createBlock(&group->getRegion(0));
    builder.create<FlagOp>(loc, consts.i32(flagId), crossCoreMode, pipe);
    builder.create<ascendc::YieldOp>(loc);
}

struct InsertCrossCoreSyncPass : public ascendc::impl::InsertCrossCoreSyncBase<InsertCrossCoreSyncPass> {
    void runOnOperation() override
    {
        func::FuncOp funcOp = getOperation();
        SmallVector<Operation*> groupOps = collectGroupOps(funcOp);
        if (groupOps.size() < 2)
            return;
        OpBuilder builder(funcOp.getContext());
        int32_t crossCoreFlagId = 0;
        llvm::DenseMap<Operation*, int32_t> dstFlagMap;
        for (auto* groupOp : groupOps) {
            bool isAIV = isa<ascendc::IfAIVOp>(groupOp);
            SmallVector<Operation*> syncOps = collectOps(groupOp, isSyncTriggerOp);
            if (syncOps.empty())
                continue;
            scf::ForOp forOp = groupOp->getParentOfType<scf::ForOp>();
            for (auto* op : syncOps) {
                Operation* dstDefiningOp = cast<ascendc::DataCopyOp>(op).getDst().getDefiningOp();
                if (!dstDefiningOp)
                    continue;
                ascendc::Pipe producerPipe = ascendc::getOpPipeExt(op);
                auto it = dstFlagMap.find(dstDefiningOp);
                bool isSecondTrigger = it != dstFlagMap.end();
                int32_t flagId = isSecondTrigger ? it->second : crossCoreFlagId;
                SmallVector<Operation*> users = collectUsers(dstDefiningOp, isAIV, op);
                if (users.empty() && !isSecondTrigger)
                    continue;
                if (!isSecondTrigger) {
                    crossCoreFlagId = (crossCoreFlagId + 1) % maxTensorId;
                    dstFlagMap[dstDefiningOp] = flagId;
                }
                if (isSecondTrigger || forOp) {
                    builder.setInsertionPoint(op);
                    createWaitFlag(builder, op->getLoc(), flagId, producerPipe);
                }
                if (!users.empty()) {
                    builder.setInsertionPointAfter(op);
                    createSetFlag(builder, op->getLoc(), flagId, producerPipe);
                    processUsers(builder, users, flagId);
                }
                if (!isSecondTrigger && forOp) {
                    builder.setInsertionPoint(forOp);
                    insertFlagGroup<ascendc::CrossCoreSetFlagOp>(
                        builder, groupOp, !isAIV, flagId, ascendc::Pipe::PIPE_S);
                    builder.setInsertionPointAfter(forOp);
                    insertFlagGroup<ascendc::CrossCoreWaitFlagOp>(builder, groupOp, isAIV, flagId, producerPipe);
                }
            }
        }
        funcOp->setAttr(attr::crossCoreFlagId, builder.getI32IntegerAttr(crossCoreFlagId));
    }
};

} // namespace

std::unique_ptr<Pass> mlir::ascendc::createInsertCrossCoreSyncPass()
{
    return std::make_unique<InsertCrossCoreSyncPass>();
}
