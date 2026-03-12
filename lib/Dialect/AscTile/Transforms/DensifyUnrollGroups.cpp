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

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Support/LLVM.h"

namespace mlir {
namespace asctile {
#define GEN_PASS_DEF_DENSIFYUNROLLGROUPS
#include "ascir/Dialect/AscTile/Transforms/Passes.h.inc"
} // namespace asctile
} // namespace mlir

using namespace mlir;
using namespace mlir::asctile;

namespace {

// See llvm-project@86b69c31/mlir/lib/Interfaces/SideEffectInterfaces.cpp
template <typename... EffectTys>
bool effect(Operation *op, Value value)
{
    auto memOp = dyn_cast<MemoryEffectOpInterface>(op);
    if (!memOp)
        return false;
    SmallVector<SideEffects::EffectInstance<MemoryEffects::Effect>, 4> effects;
    memOp.getEffects(effects);
    return llvm::any_of(effects, [&](MemoryEffects::EffectInstance &effect) {
        if (effect.getValue() != value)
            return false;
        return isa<EffectTys...>(effect.getEffect());
    });
}

class Densifier {
    using UnrollGroup = SmallVector<Operation *>;

    enum CanMoveKind {
        Before,
        After,
    };

    DominanceInfo dom;
    std::unordered_map<int64_t, std::unordered_map<Block *, UnrollGroup>> groups;
    bool separateBlocks;

    static bool allOpsSameBlock(const UnrollGroup &group)
    {
        if (group.empty())
            return false;
        Block *block = group.front()->getBlock();
        return !llvm::any_of(group, [block](Operation *op) { return op->getBlock() != block; });
    }

    bool haveMemoryEffects(Operation *op1, Operation *op2, Value value) const
    {
        if (effect<MemoryEffects::Read, MemoryEffects::Write>(op1, value) && effect<MemoryEffects::Write>(op2, value))
            return true;
        if (effect<MemoryEffects::Write>(op1, value) && effect<MemoryEffects::Read>(op2, value))
            return true;
        return false;
    }

    template <CanMoveKind kind>
    bool canMove(Operation *toMove, Operation *baseOp) const
    {
        if (toMove == baseOp)
            return false;
        Operation *op1, *op2;
        if constexpr (kind == CanMoveKind::Before) {
            op1 = baseOp;
            op2 = toMove;
        } else if constexpr (kind == CanMoveKind::After) {
            op1 = toMove;
            op2 = baseOp;
        } else {
            llvm_unreachable("unexpected CanMoveKind value");
        }
        for (Value opnd : toMove->getOperands()) {
            if (!dom.dominates(opnd, baseOp))
                return false;
            if (haveMemoryEffects(op1, op2, opnd))
                return false;
        }
        return true;
    }

    void densifyAtTop(UnrollGroup &group)
    {
        if (!allOpsSameBlock(group))
            return;
        llvm::sort(group, [](Operation *lhs, Operation *rhs) { return lhs->isBeforeInBlock(rhs); });
        auto *firstOp = group.front();
        auto remainingOps = MutableArrayRef(group).drop_front();
        while (!remainingOps.empty()) {
            auto *toMove = remainingOps.front();
            remainingOps = remainingOps.drop_front();
            auto prevOps = llvm::make_range(std::next(Block::iterator(firstOp)), Block::iterator(toMove));
            auto it =
                llvm::find_if(prevOps, [this, toMove](Operation &base) { return canMove<Before>(toMove, &base); });
            if (it != prevOps.end())
                toMove->moveBefore(&*it);
            firstOp = toMove;
        }
    }

    void densifyAtBottom(UnrollGroup &group)
    {
        if (!allOpsSameBlock(group))
            return;
        llvm::sort(group, [](Operation *lhs, Operation *rhs) { return !lhs->isBeforeInBlock(rhs); });
        auto *lastOp = group.front();
        auto remainingOps = MutableArrayRef(group).drop_front();
        while (!remainingOps.empty()) {
            auto *toMove = remainingOps.front();
            remainingOps = remainingOps.drop_front();
            SmallVector<Operation *> nextOps;
            for (auto &op : llvm::make_range(std::next(Block::iterator(toMove)), std::next(Block::iterator(lastOp))))
                nextOps.push_back(&op);
            if (nextOps.empty())
                continue;
            auto it = std::find_if(nextOps.rbegin(), nextOps.rend(),
                                   [this, toMove](Operation *base) { return canMove<After>(toMove, base); });
            if (it != nextOps.rend())
                toMove->moveAfter(*it);
            lastOp = toMove;
        }
    }

  public:
    enum Target {
        Skip,
        AtTop,
        AtBottom,
    };

    explicit Densifier(bool separateBlocks = true) : separateBlocks(separateBlocks) {}
    ~Densifier() = default;

    void add(int64_t group, Operation *op)
    {
        Block *block = separateBlocks ? op->getBlock() : nullptr;
        groups[group][block].push_back(op);
    }

    void run(function_ref<Target(ArrayRef<Operation *>)> selectTargetFn)
    {
        for (auto groupIt = groups.begin(); groupIt != groups.end(); ++groupIt) {
            auto &groupBlocks = groupIt->second;
            for (auto blockIt = groupBlocks.begin(); blockIt != groupBlocks.end(); ++blockIt) {
                UnrollGroup &group = blockIt->second;
                auto target = selectTargetFn(group);
                if (target == Skip)
                    continue;
                if (target == AtTop)
                    densifyAtTop(group);
                else if (target == AtBottom)
                    densifyAtBottom(group);
                else
                    llvm_unreachable("unexpected Densifier::Target value");
            }
        }
    }
};

Densifier::Target selectTarget(ArrayRef<Operation *> group)
{
    if (group.empty())
        return Densifier::Skip;
    if (isa<asctile::LoadOp>(group.front()))
        return Densifier::AtTop;
    if (isa<asctile::StoreOp>(group.front()))
        return Densifier::AtBottom;
    return Densifier::Skip;
}

struct DensifyUnrollGroupsPass : public asctile::impl::DensifyUnrollGroupsBase<DensifyUnrollGroupsPass> {
    void runOnOperation() override
    {
        auto op = getOperation();
        Densifier densifier;
        op.walk([&](Operation *op) {
            if (auto a = op->getAttrOfType<IntegerAttr>(attr::unrollGroup)) {
                int64_t group = a.getValue().getSExtValue();
                densifier.add(group, op);
                op->removeAttr(attr::unrollGroup);
            }
        });
        densifier.run(selectTarget);
    }
};

} // namespace

std::unique_ptr<Pass> mlir::asctile::createDensifyUnrollGroupsPass()
{
    return std::make_unique<DensifyUnrollGroupsPass>();
}
