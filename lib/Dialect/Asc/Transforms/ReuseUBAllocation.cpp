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
#include "ascir/Dialect/Asc/Utils/Utils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Dominance.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir {
namespace ascendc {
#define GEN_PASS_DEF_REUSEUBALLOCATION
#include "ascir/Dialect/Asc/Transforms/Passes.h.inc"
} // namespace ascendc
} // namespace mlir

using namespace mlir;

namespace {

constexpr const char* const eraseMeAttr = "asc.erase_me";

using TensorOp = ascendc::LocalTensorAutoOp;

enum class TargetUser { FirstUser, LastUser };

void appendImplicitUsers(Operation* op, SmallVectorImpl<Operation*>& allUsers)
{
    for (auto* user : op->getUsers()) {
        auto users = user->getUsers();
        if (isa<CastOpInterface>(user) && !users.empty()) {
            allUsers.append(users.begin(), users.end());
            appendImplicitUsers(user, allUsers);
        }
    }
}

DenseSet<scf::ForOp> collectNearestForOps(TensorOp tensorOp)
{
    DenseSet<scf::ForOp> forOps;
    SmallVector<Operation*> users(tensorOp->getUsers());
    appendImplicitUsers(tensorOp, users);
    for (Operation* user : users) {
        auto forOp = user->getParentOfType<scf::ForOp>();
        if (forOp)
            forOps.insert(forOp);
    }
    return forOps;
}

template <TargetUser target>
Operation* findUser(Operation* tensorOp, DominanceInfo& di)
{
    if (tensorOp->getUsers().empty())
        return nullptr;
    if (tensorOp->hasOneUse())
        return *tensorOp->user_begin();

    SmallVector<Operation*> users(tensorOp->getUsers());
    appendImplicitUsers(tensorOp, users);

    llvm::stable_sort(users, [&](Operation* lhs, Operation* rhs) {
        Block* lhsBlk = lhs->getBlock();
        Block* rhsBlk = rhs->getBlock();
        if (lhsBlk == rhsBlk)
            return lhs->isBeforeInBlock(rhs);
        Block* dtr = di.findNearestCommonDominator(lhsBlk, rhsBlk);
        Operation* lhsAnc = dtr->findAncestorOpInBlock(*lhs);
        Operation* rhsAnc = dtr->findAncestorOpInBlock(*rhs);
        if (lhsAnc == rhsAnc) {
            Region* lhsRegion = lhsBlk->getParent();
            Region* rhsRegion = rhsBlk->getParent();
            if (lhsRegion != rhsRegion)
                return lhsRegion->getRegionNumber() < rhsRegion->getRegionNumber();
            for (Block& block : *lhsRegion) {
                if (&block == lhsBlk)
                    return true;
                if (&block == rhsBlk)
                    return false;
            }
            return false;
        }
        return lhsAnc->isBeforeInBlock(rhsAnc);
    });

    if constexpr (target == TargetUser::FirstUser)
        return users.front();
    if constexpr (target == TargetUser::LastUser)
        return users.back();

    llvm_unreachable("Unknown target user");
}

struct ReuseUBAllocationPass : public ascendc::impl::ReuseUBAllocationBase<ReuseUBAllocationPass> {
    ReuseUBAllocationPass(const ascendc::ReuseUBAllocationOptions& options) : ReuseUBAllocationBase(options) {}

    static bool isTensorGreaterOrEqual(TensorOp firstOp, TensorOp secondOp)
    {
        return ascendc::getTypeSize(firstOp.getType()) >= ascendc::getTypeSize(secondOp.getType());
    }

    bool reusable(TensorOp op) const
    {
        return op.getPosition() == ascendc::TPosition::VECCALC && op.getType().hasStaticShape();
    }

    bool testOnReuse(TensorOp bottomTensor, TensorOp topTensor, Block* block, DominanceInfo& di) const
    {
        if (!reuseInOut) {
            bool bottomIsInOut = bottomTensor.getInput() || bottomTensor.getOutput();
            bool topIsInOut = topTensor.getInput() || topTensor.getOutput();
            if (bottomIsInOut && topIsInOut) {
                auto bottomForOps = collectNearestForOps(bottomTensor);
                auto topForOps = collectNearestForOps(topTensor);
                for (scf::ForOp forOp : bottomForOps) {
                    if (topForOps.contains(forOp))
                        return false;
                }
            }
        }

        auto* last = findUser<TargetUser::LastUser>(topTensor, di);
        auto* first = findUser<TargetUser::FirstUser>(bottomTensor, di);
        auto memEffectOp = dyn_cast<MemoryEffectOpInterface>(last);
        if (!last || !first || isa<scf::YieldOp>(last) ||
            (isa<CastOpInterface>(last) &&
             ((memEffectOp && !memEffectOp.hasEffect<MemoryEffects::Allocate>()) || !memEffectOp)))
            return false;
        auto* firstUser = block->findAncestorOpInBlock(*first);
        auto* lastUser = block->findAncestorOpInBlock(*last);
        if (lastUser == firstUser && bottomTensor.getType().getElementType() == topTensor.getType().getElementType()) {
            auto isTensorInDiffRegions = [&](Region& r1, Region& r2) {
                return (r1.isAncestor(bottomTensor->getParentRegion()) &&
                        r2.isAncestor(topTensor->getParentRegion())) ||
                       (r1.isAncestor(topTensor->getParentRegion()) && r2.isAncestor(bottomTensor->getParentRegion()));
            };
            if (auto op = dyn_cast<scf::WhileOp>(lastUser))
                return isTensorInDiffRegions(op.getBefore(), op.getAfter());
            if (auto op = dyn_cast<scf::IfOp>(lastUser))
                return isTensorInDiffRegions(op.getThenRegion(), op.getElseRegion());
            if (isa<ascendc::UnaryOp, ascendc::BinaryOp, ascendc::VecScalarOp>(lastUser))
                return true;
            return false;
        }
        return lastUser->isBeforeInBlock(firstUser);
    }

    template <typename CastOp>
    void reuse(SmallVectorImpl<TensorOp>& tensorOpList, Block* block, DominanceInfo& di)
    {
        while (tensorOpList.size() > 1) {
            TensorOp topTensorOp = tensorOpList.pop_back_val();
            for (auto& tensorOp : llvm::reverse(tensorOpList)) {
                if (!reusable(tensorOp) || !reusable(topTensorOp) || !testOnReuse(tensorOp, topTensorOp, block, di))
                    continue;
                if (tensorOp->getBlock() != topTensorOp->getBlock()) {
                    constexpr bool reuseGreedily = true;
                    if (!reuseGreedily)
                        continue;
                    if (topTensorOp->getBlock() != block)
                        topTensorOp->moveBefore(block, block->begin());
                }
                assert(topTensorOp->getNumResults() == 1);
                assert(tensorOp->getNumResults() == 1);
                auto reuseTensorOp = [](TensorOp op, Operation* newTensor) {
                    op->replaceAllUsesWith(newTensor);
                    op->setAttr(eraseMeAttr, UnitAttr::get(op.getContext()));
                };
                OpBuilder builder(tensorOp);
                if (isTensorGreaterOrEqual(topTensorOp, tensorOp)) {
                    builder.setInsertionPointAfter(topTensorOp);
                    auto reinterpretCastOpToSecond =
                        builder.create<CastOp>(tensorOp->getLoc(), tensorOp.getType(), topTensorOp->getResult(0));
                    reuseTensorOp(tensorOp, reinterpretCastOpToSecond);
                    tensorOp = topTensorOp;
                } else {
                    tensorOp->moveBefore(topTensorOp);
                    builder.setInsertionPointAfter(tensorOp);
                    auto reinterpretCastOpToFirst =
                        builder.create<CastOp>(topTensorOp->getLoc(), topTensorOp.getType(), tensorOp->getResult(0));
                    reuseTensorOp(topTensorOp, reinterpretCastOpToFirst);
                }
                break;
            }
        }
    }

    void runOnOperation() override
    {
        func::FuncOp funcOp = getOperation();
        DominanceInfo& di = getAnalysis<DominanceInfo>();
        funcOp.walk([&](Block* block) {
            SmallVector<TensorOp> tensorOpList;
            DenseMap<TensorOp, Operation*> lastUserMap;
            block->walk<WalkOrder::PreOrder>([&](TensorOp tensor) {
                auto* lastUser = findUser<TargetUser::LastUser>(tensor, di);
                if (!lastUser)
                    return;
                lastUserMap[tensor] = lastUser;
                if (llvm::all_of(tensor->getUsers(), [](Operation* user) {
                        return !isa<scf::YieldOp>(user) || !isa<scf::ForOp>(user->getParentOp());
                    }))
                    tensorOpList.push_back(tensor);
            });

            llvm::sort(tensorOpList, [&](const TensorOp& lhs, const TensorOp& rhs) {
                auto* leftOp = block->findAncestorOpInBlock(*lastUserMap[lhs]);
                auto* rightOp = block->findAncestorOpInBlock(*lastUserMap[rhs]);
                return rightOp->isBeforeInBlock(leftOp);
            });

            SmallVector<TensorOp> lastUserInWhileOpList;
            SmallVector<TensorOp> lastUserInIfOpList;
            for (auto& tensorOp : tensorOpList) {
                auto* lastUser = block->findAncestorOpInBlock(*lastUserMap[tensorOp]);
                if (isa<scf::WhileOp>(lastUser))
                    lastUserInWhileOpList.push_back(tensorOp);
                else if (isa<scf::IfOp>(lastUser))
                    lastUserInIfOpList.push_back(tensorOp);
            }
            reuse<ascendc::LocalTensorReinterpretCastOp>(lastUserInWhileOpList, block, di);
            reuse<ascendc::LocalTensorReinterpretCastOp>(lastUserInIfOpList, block, di);

            SmallVector<TensorOp> usedTensorOpList;
            for (auto& tensorOp : tensorOpList) {
                if (!tensorOp->hasAttr(eraseMeAttr))
                    usedTensorOpList.push_back(tensorOp);
            }

            reuse<ascendc::LocalTensorReinterpretCastOp>(usedTensorOpList, block, di);
            block->walk([](TensorOp op) {
                if (op->hasAttr(eraseMeAttr))
                    op.erase();
            });
        });
    }
};

} // namespace

namespace mlir {
namespace ascendc {
std::unique_ptr<Pass> createReuseUBAllocationPass(bool reuseInOut)
{
    ReuseUBAllocationOptions options;
    options.reuseInOut = reuseInOut;
    return std::make_unique<ReuseUBAllocationPass>(options);
}
} // namespace ascendc
} // namespace mlir
