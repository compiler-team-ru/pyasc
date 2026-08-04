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

namespace mlir {
namespace ascendc {
#define GEN_PASS_DEF_INSERTCROSSCORESYNC
#include "ascir/Dialect/Asc/Transforms/Passes.h.inc"
} // namespace ascendc
} // namespace mlir

using namespace mlir;

namespace {

constexpr uint8_t crossCoreMode = 4;

ascendc::Pipe getLastOpPipe(Operation* groupOp)
{
    Block& block = groupOp->getRegion(0).front();
    if (llvm::hasSingleElement(block))
        return ascendc::Pipe::PIPE_V;
    auto& lastOp = *std::prev(block.end(), 2);
    if (isa<ascendc::MmadOp, ascendc::MmadWithBiasOp>(lastOp))
        return ascendc::Pipe::PIPE_M;
    if (isa<ascendc::FixpipeOp>(lastOp))
        return ascendc::Pipe::PIPE_FIX;
    if (isa<ascendc::CopyToL0Op>(lastOp))
        return ascendc::Pipe::PIPE_MTE1;
    if (isa<ascendc::FillOp>(lastOp))
        return ascendc::Pipe::PIPE_MTE2;
    if (auto copyOp = dyn_cast<ascendc::DataCopyOp>(lastOp)) {
        auto direction = copyOp.getDirection();
        if (direction == ascendc::CopyDirection::GlobalToLocal)
            return ascendc::Pipe::PIPE_MTE2;
        if (direction == ascendc::CopyDirection::LocalToGlobal)
            return ascendc::Pipe::PIPE_MTE3;
    }
    if (isa<ascendc::LocalTensorGetValueOp, ascendc::LocalTensorSetValueOp>(lastOp))
        return ascendc::Pipe::PIPE_S;
    return ascendc::Pipe::PIPE_V;
}

struct InsertCrossCoreSyncPass : public ascendc::impl::InsertCrossCoreSyncBase<InsertCrossCoreSyncPass> {
    void runOnOperation() override
    {
        func::FuncOp funcOp = getOperation();
        SmallVector<Operation*> groupOps;
        funcOp.walk<WalkOrder::PreOrder>([&](Operation* op) {
            if (isa<ascendc::IfAICOp, ascendc::IfAIVOp>(op)) {
                groupOps.push_back(op);
                return WalkResult::skip();
            }
            return WalkResult::advance();
        });
        if (groupOps.size() < 2)
            return;
        OpBuilder builder(funcOp.getContext());
        ascir::ConstantOpBuilder consts(builder);
        for (size_t i = 1; i < groupOps.size(); ++i) {
            Operation* prev = groupOps[i - 1];
            Operation* next = groupOps[i];
            if (prev->getName() == next->getName())
                continue;
            ascendc::Pipe setPipe = getLastOpPipe(prev);
            Block& prevBlock = prev->getRegion(0).front();
            Operation* prevYield = prevBlock.getTerminator();
            builder.setInsertionPoint(prevYield);
            builder.create<ascendc::CrossCoreSetFlagOp>(prev->getLoc(), consts.i32(0), crossCoreMode, setPipe);
            if (isa<ascendc::IfAICOp>(prev)) {
                builder.create<ascendc::CrossCoreSetFlagOp>(prev->getLoc(), consts.i32(16), crossCoreMode, setPipe);
            }
            Block& nextBlock = next->getRegion(0).front();
            builder.setInsertionPointToStart(&nextBlock);
            builder.create<ascendc::CrossCoreWaitFlagOp>(
                next->getLoc(), consts.i32(0), crossCoreMode, ascendc::Pipe::PIPE_S);
        }
    }
};

} // namespace

std::unique_ptr<Pass> mlir::ascendc::createInsertCrossCoreSyncPass()
{
    return std::make_unique<InsertCrossCoreSyncPass>();
}
