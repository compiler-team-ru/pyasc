/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ascir/Dialect/AscTile/IR/AscTile.h"
#include "ascir/Dialect/AscTile/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/IRMapping.h"

namespace mlir {
namespace asctile {
#define GEN_PASS_DEF_MERGECVGROUPS
#include "ascir/Dialect/AscTile/Transforms/Passes.h.inc"
} // namespace asctile
} // namespace mlir

using namespace mlir;
using namespace mlir::asctile;

namespace {

enum class GroupType {
    Cube,
    Vector,
    Neither,
};

GroupType classifyGroup(Operation* op)
{
    if (isa<asctile::CubeGroupOp>(op))
        return GroupType::Cube;
    if (isa<asctile::VectorGroupOp>(op))
        return GroupType::Vector;
    return GroupType::Neither;
}

SmallVector<SmallVector<Operation*>> collectRuns(Block& block)
{
    SmallVector<SmallVector<Operation*>> runs;
    for (Operation& op : block) {
        GroupType type = classifyGroup(&op);
        if (runs.empty() ||
            !runs.back().empty() && (type == GroupType::Neither || classifyGroup(runs.back().front()) != type)) {
            runs.emplace_back();
        }
        if (type != GroupType::Neither)
            runs.back().push_back(&op);
    }
    return runs;
}

void mergeRun(SmallVector<Operation*>& groups)
{
    if (groups.size() < 2)
        return;
    Operation* firstGroup = groups.front();
    Location loc = firstGroup->getLoc();
    GroupType type = classifyGroup(firstGroup);
    DenseSet<Value> intermediateValues;
    for (Operation* group : groups) {
        for (Value result : group->getResults())
            intermediateValues.insert(result);
    }
    SmallVector<Value> externalOperands;
    DenseSet<Value> seenOperands;
    for (Operation* group : groups) {
        for (Value operand : group->getOperands()) {
            if (intermediateValues.contains(operand))
                continue;
            if (seenOperands.insert(operand).second)
                externalOperands.push_back(operand);
        }
    }
    DenseSet<Operation*> groupSet(groups.begin(), groups.end());
    SmallVector<Value> externalResults;
    SmallVector<Type> resultTypes;
    for (Operation* group : groups)
        for (Value result : group->getResults())
            if (llvm::any_of(result.getUses(), [&](OpOperand& use) { return !groupSet.contains(use.getOwner()); })) {
                externalResults.push_back(result);
                resultTypes.push_back(result.getType());
            }
    OpBuilder builder(firstGroup);
    Operation* mergedGroup;
    if (type == GroupType::Cube)
        mergedGroup = builder.create<CubeGroupOp>(loc, resultTypes, externalOperands);
    else
        mergedGroup = builder.create<VectorGroupOp>(loc, resultTypes, externalOperands);
    Block* mergedBlock = &mergedGroup->getRegion(0).emplaceBlock();
    IRMapping mapping;
    builder.setInsertionPointToEnd(mergedBlock);
    for (Operation* group : groups) {
        Block& groupBlock = group->getRegion(0).front();
        auto yieldOp = cast<asctile::YieldOp>(groupBlock.getTerminator());
        for (Operation& innerOp : groupBlock.without_terminator()) {
            Operation* cloned = builder.clone(innerOp, mapping);
            for (auto [i, result] : llvm::enumerate(innerOp.getResults()))
                mapping.map(result, cloned->getResult(i));
        }
        for (auto [i, result] : llvm::enumerate(group->getResults())) {
            Value yieldOperand = yieldOp.getOperand(i);
            Value mappedYieldOperand = mapping.lookupOrDefault(yieldOperand);
            mapping.map(result, mappedYieldOperand);
        }
    }
    SmallVector<Value> yieldOperands;
    for (Value externalResult : externalResults)
        yieldOperands.push_back(mapping.lookup(externalResult));
    builder.create<asctile::YieldOp>(loc, yieldOperands);
    for (Value intermediateValue : intermediateValues) {
        if (!llvm::is_contained(externalResults, intermediateValue) && mapping.contains(intermediateValue))
            intermediateValue.replaceAllUsesWith(mapping.lookup(intermediateValue));
    }
    for (auto [i, externalResult] : llvm::enumerate(externalResults))
        externalResult.replaceAllUsesWith(mergedGroup->getResult(i));
    for (Operation* group : groups)
        group->erase();
}

struct MergeCVGroupsPass : public asctile::impl::MergeCVGroupsBase<MergeCVGroupsPass> {
    void runOnOperation() override
    {
        getOperation().walk([&](Block* block) {
            auto runs = collectRuns(*block);
            for (auto& groups : runs) {
                mergeRun(groups);
            }
        });
    }
};

} // namespace

std::unique_ptr<Pass> mlir::asctile::createMergeCVGroupsPass() { return std::make_unique<MergeCVGroupsPass>(); }
