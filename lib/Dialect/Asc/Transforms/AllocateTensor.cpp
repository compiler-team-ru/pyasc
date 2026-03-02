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

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace ascendc {
#define GEN_PASS_DEF_ALLOCATETENSOR
#include "ascir/Dialect/Asc/Transforms/Passes.h.inc"
} // namespace ascendc
} // namespace mlir

using namespace mlir;

namespace {

class AllocateTensorPass : public ascendc::impl::AllocateTensorBase<AllocateTensorPass> {
    void runOnOperation() override
    {
        func::FuncOp funcOp = getOperation();
        if (funcOp.isDeclaration()) {
            return;
        }
        uint32_t addr = 0;
        funcOp.walk<WalkOrder::PreOrder>([this, &addr](ascendc::LocalTensorAutoOp op) {
            auto type = op.getType();
            if (!type.hasStaticShape()) {
                op.emitError() << "must be static shape";
                signalPassFailure();
            }
            OpBuilder builder(op);
            auto localTensorV3Op = builder.create<ascendc::LocalTensorV3Op>(
                op.getLoc(), type, ascendc::TPosition::VECCALC, addr, type.getNumElements());
            op->replaceAllUsesWith(localTensorV3Op);
            op.erase();
            addr += ascendc::getTypeSize(type);
        });
    }
};

} // namespace

namespace mlir {
namespace ascendc {
std::unique_ptr<Pass> createAllocateTensorPass() { return std::make_unique<AllocateTensorPass>(); }
} // namespace ascendc
} // namespace mlir
