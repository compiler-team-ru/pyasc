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

#include "ascir/Dialect/Asc/IR/Asc.h"
#include "ascir/Dialect/Asc/Utils/Attributes.h"
#include "ascir/Dialect/Asc/Utils/Utils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Matchers.h"

namespace mlir {
namespace ascendc {
#define GEN_PASS_DEF_COMPUTEMEMORYCONSUMPTION
#include "asctile/Dialect/AscendC/Transforms/Passes.h.inc"
} // namespace ascendc
} // namespace mlir

using namespace mlir;
using namespace mlir::ascendc;

namespace {

StringLiteral positionToStr(TPosition position)
{
    switch (position) {
        case ascendc::TPosition::A1:
        case ascendc::TPosition::B1:
            return "L1";
        case ascendc::TPosition::A2:
            return "L0A";
        case ascendc::TPosition::B2:
            return "L0B";
        case ascendc::TPosition::CO1:
            return "L0C";
        case ascendc::TPosition::VECIN:
        case ascendc::TPosition::VECOUT:
        case ascendc::TPosition::VECCALC:
            return "UB";
        case ascendc::TPosition::C2:
            return "BT";
        default:
            llvm_unreachable("unexpected TPosition value");
    }
}

auto calculateMemoryConsumption(ModuleOp moduleOp)
{
    std::map<StringLiteral, int64_t> mem;
    moduleOp.walk([&mem](ascendc::LocalTensorV3Op op) {
        mem[positionToStr(op.getPos())] += getElementTypeSize(op.getType()) * static_cast<int64_t>(op.getTileSize());
    });
    moduleOp.walk([&mem](ascendc::TPipeInitBufferOp op) {
        if (auto tbufType = dyn_cast<ascendc::TBufType>(op.getBuffer().getType())) {
            APInt lengthAttr;
            if (matchPattern(op.getLength(), m_ConstantInt(&lengthAttr))) {
                mem[positionToStr(tbufType.getTPosition())] += lengthAttr.getSExtValue();
            }
        }
    });
    moduleOp.walk([&mem](ascendc::TPipeInitQueueOp op) {
        auto queueType = op.getQueue().getType();
        TPosition position;
        if (auto queType = dyn_cast<ascendc::QueueType>(queueType)) {
            position = queType.getPosition();
        } else if (auto queBindType = dyn_cast<ascendc::QueBindType>(queueType)) {
            position = queBindType.getSrcPosition();
        } else {
            return;
        }
        APInt numAttr, lengthAttr;
        if (matchPattern(op.getNum(), m_ConstantInt(&numAttr)) &&
            matchPattern(op.getLength(), m_ConstantInt(&lengthAttr))) {
            mem[positionToStr(position)] += numAttr.getSExtValue() * lengthAttr.getSExtValue();
        }
    });
    return mem;
}

struct ComputeMemoryConsumptionPass : public ascendc::impl::ComputeMemoryConsumptionBase<ComputeMemoryConsumptionPass> {
    void runOnOperation() override
    {
        ModuleOp moduleOp = getOperation();
        auto mem = calculateMemoryConsumption(moduleOp);
        Builder builder(moduleOp);
        SmallVector<NamedAttribute, 6> attrs;
        for (auto [key, value] : mem)
            attrs.emplace_back(builder.getStringAttr(key), builder.getI64IntegerAttr(value));
        moduleOp->setAttr(attr::memoryConsumed, builder.getDictionaryAttr(attrs));
    }
};

} // namespace

std::unique_ptr<Pass> mlir::ascendc::createComputeMemoryConsumptionPass()
{
    return std::make_unique<ComputeMemoryConsumptionPass>();
}
