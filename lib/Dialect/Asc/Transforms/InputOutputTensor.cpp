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
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace ascendc {
#define GEN_PASS_DEF_INPUTOUTPUTTENSOR
#include "ascir/Dialect/Asc/Transforms/Passes.h.inc"
} // namespace ascendc
} // namespace mlir

using namespace mlir;
using namespace mlir::ascendc;

namespace {

void createDataCopyIfNeeded(Operation* op)
{
    for (auto& use : op->getUses()) {
        auto copyOp = dyn_cast<ascendc::DataCopyOp>(use.getOwner());
        if (!copyOp || copyOp.getDirection() != ascendc::CopyDirection::ubuf_gm)
            return;
        OpBuilder builder(op);
        ascir::ConstantOpBuilder consts(builder);
        auto type = cast<ascendc::BaseTensorType>(use.get().getType());
        auto dst = builder.create<ascendc::LocalTensorAutoOp>(
            op->getLoc(), type, /*input*/ false,
            /*output*/ true, ValueRange{});
        builder.setInsertionPointAfter(op);
        Value calCount = consts.i64(type.getNumElements());
        builder.create<ascendc::DataCopyL2Op>(op->getLoc(), dst, use.get(), calCount);
        copyOp.setSrc(dst);
    }
}

void setInOutTensors(func::FuncOp funcOp)
{
    funcOp.walk([](ascendc::LocalTensorAutoOp op) {
        bool input = false;
        bool output = false;
        for (Operation* user : op->getUsers()) {
            if (auto copyOp = dyn_cast<ascendc::DataCopyOp>(user)) {
                auto dir = copyOp.getDirection();
                if (dir == ascendc::CopyDirection::gm_ubuf) {
                    input = true;
                    continue;
                }
                if (dir == ascendc::CopyDirection::ubuf_gm) {
                    output = true;
                    continue;
                }
            }
        }
        op.setInput(input);
        op.setOutput(output);
    });
    funcOp.walk([](scf::ForOp op) { createDataCopyIfNeeded(op); });
    funcOp.walk([](scf::IfOp op) { createDataCopyIfNeeded(op); });
}

void fixInOutTensor(func::FuncOp& funcOp)
{
    funcOp.walk([](ascendc::LocalTensorAutoOp inTensor) {
        if (!inTensor.getInput() || inTensor.getOutput())
            return;
        auto loc = inTensor.getLoc();
        OpBuilder builder(inTensor);
        auto tensorType = inTensor.getResult().getType();
        inTensor.setOutput(false);
        for (auto& use : inTensor->getUses()) {
            auto* owner = use.getOwner();
            auto copyOp = dyn_cast<ascendc::DataCopyOp>(owner);
            if (!copyOp || copyOp.getDirection() != ascendc::CopyDirection::ubuf_gm)
                return builder.setInsertionPoint(owner);
            ascir::ConstantOpBuilder consts(builder);
            Value calCount = consts.i64(tensorType.getNumElements());
            auto outTensor = builder.create<ascendc::LocalTensorAutoOp>(
                loc, tensorType, /*input*/ false,
                /*output*/ true, ValueRange{});
            builder.create<ascendc::DataCopyL2Op>(loc, outTensor, inTensor, calCount);
            owner->setOperand(use.getOperandNumber(), outTensor);
        }
    });
}

struct InputOutputTensorPass : public ascendc::impl::InputOutputTensorBase<InputOutputTensorPass> {
    void runOnOperation() override
    {
        func::FuncOp funcOp = getOperation();
        if (funcOp.isDeclaration()) {
            return;
        }
        setInOutTensors(funcOp);
        fixInOutTensor(funcOp);
        MLIRContext* context = &getContext();
        RewritePatternSet patterns(context);
        ascendc::LocalTensorAutoOp::getCanonicalizationPatterns(patterns, context);
        if (applyPatternsAndFoldGreedily(funcOp, std::move(patterns)).failed()) {
            signalPassFailure();
            return;
        }
    }
};

} // namespace

namespace mlir {
namespace ascendc {
std::unique_ptr<Pass> createInputOutputTensorPass() { return std::make_unique<InputOutputTensorPass>(); }
} // namespace ascendc
} // namespace mlir
