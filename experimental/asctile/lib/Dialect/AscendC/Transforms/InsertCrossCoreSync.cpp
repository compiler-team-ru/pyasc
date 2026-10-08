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
#include "asctile/Dialect/AscendC/Utils/Utils.h"

#include "ascir/Dialect/Asc/IR/Asc.h"
#include "ascir/Dialect/Asc/Transforms/Passes.h"
#include "ascir/Dialect/Utils/ConstantOpBuilder.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

namespace mlir {
namespace ascendc {
#define GEN_PASS_DEF_INSERTCROSSCORESYNC
#include "asctile/Dialect/AscendC/Transforms/Passes.h.inc"
} // namespace ascendc
} // namespace mlir

using namespace mlir;

namespace {

constexpr uint8_t crossCoreMode = 4;
constexpr int32_t maxTensorId = 16;

bool isDataCopyOp(Operation* op, llvm::function_ref<bool(ascendc::TPosition, ascendc::TPosition)> pred)
{
    auto copyOp = dyn_cast<ascendc::DataCopyOp>(op);
    if (!copyOp)
        return false;
    auto direction = copyOp.getDirection();
    if (!direction)
        return false;
    auto [src, dst] = *direction;
    return pred(src, dst);
}

bool isSyncTriggerOp(Operation* op)
{
    return isDataCopyOp(op, [](ascendc::TPosition src, ascendc::TPosition dst) {
        return (src == ascendc::TPosition::VECCALC &&
                (dst == ascendc::TPosition::A1 || dst == ascendc::TPosition::B1)) ||
               (src == ascendc::TPosition::CO1 && dst == ascendc::TPosition::VECCALC);
    });
}

bool isL0CToGMCopy(Operation* op)
{
    return isDataCopyOp(op, [](ascendc::TPosition src, ascendc::TPosition dst) {
        return src == ascendc::TPosition::CO1 && dst == ascendc::TPosition::GM;
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

SmallVector<Operation*> collectUsers(Operation* dstDefiningOp, bool triggerIsAIV, Operation* afterOp = nullptr)
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

Value getRootGlobalTensor(Value v)
{
    while (auto subOp = v.getDefiningOp<ascendc::GlobalTensorSubIndexOp>())
        v = subOp.getTensor();
    return v;
}

int32_t findGMLoadFlag(Operation* groupOp, const llvm::DenseMap<Value, int32_t>& gmRootsWithFlags)
{
    int32_t flagId = -1;
    groupOp->walk([&](Operation* op) {
        if (op == groupOp || isa<ascendc::FixpipeOp>(op))
            return WalkResult::advance();
        for (Value operand : op->getOperands()) {
            if (!isa<ascendc::GlobalTensorType>(operand.getType()))
                continue;
            Value root = getRootGlobalTensor(operand);
            if (auto it = gmRootsWithFlags.find(root); it != gmRootsWithFlags.end()) {
                flagId = it->second;
                return WalkResult::interrupt();
            }
        }
        return WalkResult::advance();
    });
    return flagId;
}

void createSetFlag(OpBuilder& builder, Location loc, int32_t flagId, ascendc::Pipe pipe)
{
    ascir::ConstantOpBuilder consts(builder);
    builder.create<ascendc::CrossCoreSetFlagOp>(loc, consts.i32(flagId), crossCoreMode, pipe);
}

void createWaitFlag(OpBuilder& builder, Location loc, int32_t flagId, ascendc::Pipe pipe)
{
    ascir::ConstantOpBuilder consts(builder);
    builder.create<ascendc::CrossCoreWaitFlagOp>(loc, consts.i32(flagId), crossCoreMode, pipe);
}

template <typename FlagOp>
void insertFlagGroup(OpBuilder& builder, Operation* op, bool isAIV, int32_t flagId, ascendc::Pipe pipe)
{
    Location loc = op->getLoc();
    ascir::ConstantOpBuilder consts(builder);
    Operation* group = isAIV ? builder.create<ascendc::IfAIVOp>(loc, TypeRange{}, ValueRange{}).getOperation() :
                               builder.create<ascendc::IfAICOp>(loc, TypeRange{}, ValueRange{}).getOperation();
    builder.createBlock(&group->getRegion(0));
    builder.create<FlagOp>(loc, consts.i32(flagId), crossCoreMode, pipe);
    builder.create<ascendc::YieldOp>(loc);
}

SmallVector<Operation*> collectGroupOps(func::FuncOp funcOp)
{
    SmallVector<Operation*> groups;
    funcOp.walk<WalkOrder::PreOrder>([&](Operation* op) {
        if (isa<ascendc::IfAICOp, ascendc::IfAIVOp>(op)) {
            groups.push_back(op);
            return WalkResult::skip();
        }
        return WalkResult::advance();
    });
    return groups;
}

SmallVector<Operation*> collectOps(Operation* groupOp, llvm::function_ref<bool(Operation*)> pred)
{
    SmallVector<Operation*> ops;
    groupOp->walk([&](Operation* op) {
        if (pred(op))
            ops.push_back(op);
    });
    return ops;
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

struct InsertCrossCoreSyncPass : public ascendc::impl::InsertCrossCoreSyncBase<InsertCrossCoreSyncPass> {
    void runOnOperation() override
    {
        func::FuncOp funcOp = getOperation();
        SmallVector<Operation*> groupOps = collectGroupOps(funcOp);
        if (groupOps.size() < 2)
            return;
        OpBuilder builder(funcOp.getContext());
        int32_t nextFlagId = 0;
        llvm::DenseMap<Operation*, int32_t> dstFlagMap;
        llvm::DenseMap<Value, int32_t> gmRootsWithFlags;
        for (auto* groupOp : groupOps) {
            bool isAIV = isa<ascendc::IfAIVOp>(groupOp);
            if (!gmRootsWithFlags.empty()) {
                int32_t gmFlagId = findGMLoadFlag(groupOp, gmRootsWithFlags);
                if (gmFlagId >= 0) {
                    Block& body = groupOp->getRegion(0).front();
                    builder.setInsertionPointToStart(&body);
                    createWaitFlag(builder, groupOp->getLoc(), gmFlagId, ascendc::Pipe::PIPE_S);
                }
            }
            SmallVector<Operation*> syncOps = collectOps(groupOp, isSyncTriggerOp);
            if (!syncOps.empty()) {
                scf::ForOp forOp = groupOp->getParentOfType<scf::ForOp>();
                for (auto* op : syncOps) {
                    Operation* dstDefiningOp = cast<ascendc::DataCopyOp>(op).getDst().getDefiningOp();
                    if (!dstDefiningOp)
                        continue;
                    ascendc::Pipe producerPipe = ascendc::getOpPipeExt(op);
                    auto it = dstFlagMap.find(dstDefiningOp);
                    bool isSecondTrigger = it != dstFlagMap.end();
                    int32_t flagId = isSecondTrigger ? it->second : nextFlagId;
                    SmallVector<Operation*> users = collectUsers(dstDefiningOp, isAIV, op);
                    if (users.empty() && !isSecondTrigger)
                        continue;
                    if (!isSecondTrigger) {
                        nextFlagId = (nextFlagId + 1) % maxTensorId;
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
            SmallVector<Operation*> gmOps = collectOps(groupOp, isL0CToGMCopy);
            if (gmOps.empty())
                continue;
            Block& body = groupOp->getRegion(0).front();
            builder.setInsertionPoint(body.getTerminator());
            for (auto* op : gmOps) {
                ascendc::Pipe producerPipe = ascendc::getOpPipeExt(op);
                int32_t flagId1 = nextFlagId;
                nextFlagId = (nextFlagId + 1) % maxTensorId;
                int32_t flagId2 = flagId1 + maxTensorId;
                createSetFlag(builder, op->getLoc(), flagId1, producerPipe);
                createSetFlag(builder, op->getLoc(), flagId2, producerPipe);
                gmRootsWithFlags[getRootGlobalTensor(cast<ascendc::FixpipeOp>(op).getDst())] = flagId1;
            }
        }
    }
};

} // namespace

std::unique_ptr<Pass> mlir::ascendc::createInsertCrossCoreSyncPass()
{
    return std::make_unique<InsertCrossCoreSyncPass>();
}
