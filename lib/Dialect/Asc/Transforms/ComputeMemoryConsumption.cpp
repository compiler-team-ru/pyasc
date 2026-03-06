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
#include "ascir/Dialect/Asc/Utils/Utils.h"
#include "ascir/Dialect/Asc/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Dominance.h"

namespace mlir {
namespace ascendc {
#define GEN_PASS_DEF_COMPUTEMEMORYCONSUMPTION
#include "ascir/Dialect/Asc/Transforms/Passes.h.inc"
} // namespace ascendc
} // namespace mlir

using namespace mlir;
using namespace mlir::ascendc;

namespace {

constexpr const char* ubConsumedAttrName = "asc.ub_consumed";

int64_t getSizeInBytes(LocalTensorType type)
{
    assert(type.hasStaticShape() && "Expected static shape");
    return llvm::alignTo<ubBlockSize>(ascendc::getTypeSize(type));
}

int64_t calculateUBConsumed(ModuleOp moduleOp)
{
    int64_t ubConsumed = 0;
    moduleOp.walk([&ubConsumed](ascendc::LocalTensorV3Op op) {
        if (op.getPos() == TPosition::VECCALC)
            ubConsumed += getSizeInBytes(op.getType());
    });
    return ubConsumed;
}

struct ComputeMemoryConsumptionPass : public ascendc::impl::ComputeMemoryConsumptionBase<ComputeMemoryConsumptionPass> {
    void runOnOperation() override
    {
        ModuleOp moduleOp = getOperation();
        int64_t ubConsumed = calculateUBConsumed(moduleOp);
        Builder builder(moduleOp);
        moduleOp->setAttr(ubConsumedAttrName, builder.getI64IntegerAttr(ubConsumed));
    }
};

} // namespace

std::unique_ptr<Pass> mlir::ascendc::createComputeMemoryConsumptionPass()
{
    return std::make_unique<ComputeMemoryConsumptionPass>();
}
