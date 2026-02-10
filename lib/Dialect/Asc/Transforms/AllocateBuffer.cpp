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
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace ascendc {
#define GEN_PASS_DEF_ALLOCATEBUFFER
#include "ascir/Dialect/Asc/Transforms/Passes.h.inc"
} // namespace ascendc
} // namespace mlir

using namespace mlir;

namespace {

struct TBufAllocation : OpRewritePattern<ascendc::LocalTensorAutoOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult matchAndRewrite(ascendc::LocalTensorAutoOp op, PatternRewriter& rewriter) const override
    {
        ascir::ConstantOpBuilder consts(rewriter);
        auto type = op.getType();
        auto loc = op.getLoc();
        Value pipe = rewriter.create<ascendc::PipeOp>(loc);
        auto bufferTy = ascendc::TBufType::get(op.getContext(), ascendc::TPosition::VECCALC);
        Value buffer = rewriter.create<ascendc::TBufOp>(loc, bufferTy);
        Value length;
        if (type.hasStaticShape()) {
            length = consts.i64(type.getNumElements() * type.getElementTypeBitWidth() / CHAR_BIT);
        } else {
            assert(op->getNumOperands() != 0 && "must have operands for dynamic shape");
            length = consts.i64(type.getElementTypeBitWidth() / CHAR_BIT);
            for (auto dim : op.getDynamicShape()) {
                length = rewriter.create<arith::MulIOp>(loc, length, dim);
            }
        }
        rewriter.create<ascendc::TPipeInitBufferOp>(loc, pipe, buffer, length);
        rewriter.replaceOpWithNewOp<ascendc::TBufGetTensorOp>(op, type, buffer);
        return success();
    }
};

class AllocateBufferPass : public ascendc::impl::AllocateBufferBase<AllocateBufferPass> {
    void runOnOperation() override
    {
        func::FuncOp funcOp = getOperation();
        if (funcOp.isDeclaration()) {
            return;
        }
        MLIRContext* context = &getContext();
        RewritePatternSet patterns(context);
        patterns.add<TBufAllocation>(context);
        if (applyPatternsAndFoldGreedily(funcOp, std::move(patterns)).failed()) {
            signalPassFailure();
        }
    }
};
} // namespace

namespace mlir {
namespace ascendc {
std::unique_ptr<Pass> createAllocateBufferPass() { return std::make_unique<AllocateBufferPass>(); }
} // namespace ascendc
} // namespace mlir
