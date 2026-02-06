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

class Densifier {
    using UnrollGroup = SmallVector<Operation*>;

    DominanceInfo dom;
    std::unordered_map<int64_t, UnrollGroup> groups;

    static bool allOpsSameBlock(const UnrollGroup& group)
    {
        if (group.empty())
            return false;
        Block* block = group.front()->getBlock();
        return !llvm::any_of(group, [block](Operation* op) { return op->getBlock() != block; });
    }

    // Densify an unroll group by moving all operations up to the earliest one
    void densifyAtTop(UnrollGroup& group)
    {
        if (!allOpsSameBlock(group))
            return;
        llvm::sort(group, [](Operation* lhs, Operation* rhs) { return lhs->isBeforeInBlock(rhs); });
        auto* firstOp = group.front();
        auto toMove = ArrayRef(group).drop_front();
        for (auto* op : toMove) {
            if (llvm::all_of(
                    op->getOperands(), [this, firstOp](Value operand) { return dom.dominates(operand, firstOp); }))
                op->moveAfter(firstOp);
        }
    }

    // Densify an unroll group by moving all operations down to the latest one
    void densifyAtBottom(UnrollGroup& group)
    {
        // TODO: implement
    }

public:
    enum Target {
        Skip,
        AtTop,
        AtBottom,
    };

    Densifier() = default;
    ~Densifier() = default;

    void add(int64_t group, Operation* op) { groups[group].push_back(op); }

    void run(function_ref<Target(ArrayRef<Operation*>)> selectTargetFn)
    {
        for (auto it : groups) {
            auto target = selectTargetFn(it.second);
            if (target == Skip)
                continue;
            if (target == AtTop)
                densifyAtTop(it.second);
            else if (target == AtBottom)
                densifyAtBottom(it.second);
            else
                llvm_unreachable("unexpected Densifier::Target value");
        }
    }
};

Densifier::Target selectTarget(ArrayRef<Operation*> group)
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
        op.walk([&](Operation* op) {
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
