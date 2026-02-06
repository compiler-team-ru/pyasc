/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ascir/Dialect/AscTile/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

namespace mlir {
namespace asctile {
#define GEN_PASS_DEF_PROMOTEPUREOPS
#include "ascir/Dialect/AscTile/Transforms/Passes.h.inc"
} // namespace asctile
} // namespace mlir

using namespace mlir;
using namespace mlir::asctile;

namespace {

class Promoter {
    static bool canPromote(Operation *op) { return isPure(op) && !op->mightHaveTrait<OpTrait::IsTerminator>(); }

    // If any of the operands is defined within the same block as op then op can
    // be moved immediately after the last one of them. Otherwise, op can be moved
    // to the beginning of its parent block.
    void promote(Operation *op)
    {
        Block *block = op->getBlock();
        SmallVector<Value> opndInBlock;
        llvm::copy_if(op->getOperands(), std::back_inserter(opndInBlock),
                      [block](Value operand) { return operand.getDefiningOp() && block == operand.getParentBlock(); });
        if (opndInBlock.empty()) {
            auto &frontOp = *block->begin();
            op->moveBefore(&frontOp);
            return;
        }
        auto *lastOp = llvm::max_element(opndInBlock, [](const Value &lhs, const Value &rhs) {
                           return lhs.getDefiningOp()->isBeforeInBlock(rhs.getDefiningOp());
                       })->getDefiningOp();
        op->moveAfter(lastOp);
    }

  public:
    Promoter() = default;
    ~Promoter() = default;

    void runWithinBlock(Block *block)
    {
        SmallVector<Operation *> ops;
        for (auto &op : *block) {
            if (canPromote(&op))
                ops.push_back(&op);
        }
        for (auto *op : ops)
            promote(op);
    }
};

struct PromotePureOpsPass : public asctile::impl::PromotePureOpsBase<PromotePureOpsPass> {
    void runOnOperation() override
    {
        auto op = getOperation();
        Promoter promoter;
        op.walk([&](scf::ForOp op) { promoter.runWithinBlock(op.getBody()); });
    }
};

} // namespace

std::unique_ptr<Pass> mlir::asctile::createPromotePureOpsPass()
{
    return std::make_unique<PromotePureOpsPass>();
}
