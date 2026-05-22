/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024.
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * */

#include "ascir/Conversion/LowerToAsc/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"

#include "Common.h"

namespace mlir {
namespace asclower {
#define GEN_PASS_DEF_LOWERSCF
#include "ascir/Conversion/LowerToAsc/Passes.h.inc"
} // namespace asclower
} // namespace mlir

using namespace mlir;
using namespace mlir::asclower;

namespace {

struct LowerSCFPass : public asclower::impl::LowerSCFBase<LowerSCFPass> {
    void runOnOperation() override
    {
        func::FuncOp funcOp = getOperation();
        TensorTypeConverter converter;
        MLIRContext* context = &getContext();
        ConversionTarget target(*context);
        target.addDynamicallyLegalOp<scf::IfOp, scf::ForOp, scf::YieldOp>(
            [&converter](Operation* op) { return converter.isLegal(op); });
        target.addLegalOp<UnrealizedConversionCastOp>();
        RewritePatternSet patterns(context);
        scf::populateSCFStructuralTypeConversionsAndLegality(converter, patterns, target);
        if (applyPartialConversion(funcOp, target, std::move(patterns)).failed())
            signalPassFailure();
    }
};

} // namespace

std::unique_ptr<Pass> mlir::asclower::createLowerSCFPass() { return std::make_unique<LowerSCFPass>(); }
