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
#include "ascir/Conversion/LowerToAsc/Passes.h"
#include "ascir/Dialect/AscTile/IR/AscTile.h"
#include "ascir/Dialect/Utils/ConstantOpBuilder.h"
#include "ascir/Dialect/EmitAsc/IR/EmitAsc.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"

#include "Common.h"

namespace mlir {
namespace asclower {
#define GEN_PASS_DEF_LOWERATOMIC
#include "ascir/Conversion/LowerToAsc/Passes.h.inc"
} // namespace asclower
} // namespace mlir

using namespace mlir;
using namespace mlir::asclower;

namespace {

struct LoweringConversionTarget : public ConversionTarget {
    LoweringConversionTarget(TensorTypeConverter &converter, MLIRContext *context) : ConversionTarget(*context)
    {
        addIllegalOp<asctile::AtomicRMWOp>();
        addLegalOp<asctile::StoreOp>();
        addLegalDialect<ascendc::AscendCDialect>();
    }
};

struct ConvertAtomicRMW : ConvertOp<asctile::AtomicRMWOp> {
    using ConvertOp<asctile::AtomicRMWOp>::ConvertOp;

    LogicalResult convert(asctile::AtomicRMWOp op, ConvertRewriter &rewriter) const override
    {
        auto loc = op.getLoc();
        auto base = op.getBase();
        auto value = op.getValue();
        auto offsets = op.getOffsets();
        auto elementType = value.getType().getElementType();
        auto kind = op.getKind();

        switch (kind) {
            case asctile::AtomicKind::Add:
                rewriter.create<ascendc::SetAtomicAddOp>(loc, elementType);
                break;
            case asctile::AtomicKind::Max:
                rewriter.create<ascendc::SetAtomicMaxOp>(loc, elementType);
                break;
            case asctile::AtomicKind::Min:
                rewriter.create<ascendc::SetAtomicMinOp>(loc, elementType);
                break;
            default:
                llvm_unreachable("unexpected asctile::AtomicKind value");
        }

        rewriter.replaceOpWithNewOp<asctile::StoreOp>(op, value, base, offsets);
        rewriter.create<ascendc::SetAtomicNoneOp>(loc);
        return success();
    }
};

struct LowerAtomicPass : public asclower::impl::LowerAtomicBase<LowerAtomicPass> {
    void runOnOperation() override
    {
        func::FuncOp funcOp = getOperation();
        TensorTypeConverter converter;
        MLIRContext *context = &getContext();
        LoweringConversionTarget target(converter, context);
        RewritePatternSet patterns(context);
        patterns.insert<ConvertAtomicRMW>(converter, context);
        if (applyPartialConversion(funcOp, target, std::move(patterns)).failed())
            signalPassFailure();
    }
};

} // namespace

std::unique_ptr<Pass> mlir::asclower::createLowerAtomicPass()
{
    return std::make_unique<LowerAtomicPass>();
}
