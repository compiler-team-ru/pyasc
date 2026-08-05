/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ascir/Conversion/LowerToAsc/Passes.h"
#include "ascir/Dialect/Asc/IR/Asc.h"
#include "ascir/Dialect/Asc/Utils/Utils.h"
#include "ascir/Dialect/AscTile/IR/AscTile.h"
#include "ascir/Dialect/Utils/ConstantOpBuilder.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

#include "Common.h"

namespace mlir {
namespace asclower {
#define GEN_PASS_DEF_LOWERASCTILEI1
#include "ascir/Conversion/LowerToAsc/Passes.h.inc"
} // namespace asclower
} // namespace mlir

using namespace mlir;
using namespace mlir::asclower;

namespace {

ascendc::CMPMODE getCmpMode(asctile::CompareMode mode)
{
    switch (mode) {
        case asctile::CompareMode::EQ:
            return ascendc::CMPMODE::EQ;
        case asctile::CompareMode::NE:
            return ascendc::CMPMODE::NE;
        case asctile::CompareMode::LT:
            return ascendc::CMPMODE::LT;
        case asctile::CompareMode::LE:
            return ascendc::CMPMODE::LE;
        case asctile::CompareMode::GT:
            return ascendc::CMPMODE::GT;
        case asctile::CompareMode::GE:
            return ascendc::CMPMODE::GE;
    }
    llvm_unreachable("unexpected cmpmode");
}

struct ConvertCmpS : ConvertOp<asctile::CmpSOp> {
    using ConvertOp::ConvertOp;
    using ConvertOp::createTensorOp;

    LogicalResult matchAndRewrite(asctile::CmpSOp op, ConvertRewriter& rewriter) const override
    {
        ascir::ConstantOpBuilder consts(rewriter);
        auto loc = op.getLoc();
        auto value = rewriter.getRemappedValue(op.getValue());
        auto base = rewriter.getRemappedValue(op.getBase());
        auto srcType = cast<ShapedType>(op.getBase().getType());
        if (isa<IntegerType>(srcType.getElementType())) {
            unsigned bitWidth = srcType.getElementTypeBitWidth();
            if (bitWidth != 8 && bitWidth != 16 && bitWidth != 32)
                return op.emitOpError("can only be lowered with i8, i16 or i32 tensor operands");
            auto castToType = bitWidth == 32 ? rewriter.getF32Type() : rewriter.getF16Type();
            auto baseCasted = createTensorOp(rewriter, loc, srcType.getShape(), castToType);
            rewriter.create<ascendc::CastL2Op>(
                loc, baseCasted, base, ascendc::RoundMode::CAST_NONE, consts.i64(srcType.getNumElements()));
            base = baseCasted;
            value = rewriter.create<arith::SIToFPOp>(loc, castToType, value);
        }
        I1ReplacementType replType(op.getContext());
        auto dstShape = llvm::divideCeilSigned(srcType.getNumElements(), replType.width);
        Value dst = createTensorOp(rewriter, loc, dstShape, replType.iType);
        dst = createReCastOp(rewriter, loc, dst, dstShape, replType.uiType);
        auto mode = getCmpMode(op.getCmpMode());
        Value zero = consts.i64(0);
        rewriter.create<ascendc::CompareScalarL0Op>(
            loc, dst, base, value, mode, zero, zero,
            rewriter.create<ascendc::ConstructOp>(loc, rewriter.getType<ascendc::UnaryRepeatParamsType>()));
        rewriter.replaceOp(op, dst);
        return success();
    }
};

struct LowerAscTileI1Pass : public asclower::impl::LowerAscTileI1Base<LowerAscTileI1Pass> {
    void runOnOperation() override
    {
        func::FuncOp funcOp = getOperation();
        TensorTypeConverter converter;
        MLIRContext* context = &getContext();
        ConversionTarget target(*context);
        target.addIllegalOp<asctile::CmpSOp>();
        target.addLegalDialect<arith::ArithDialect, ascendc::AscendCDialect>();
        target.addLegalOp<UnrealizedConversionCastOp>();
        RewritePatternSet patterns(context);
        patterns.insert<ConvertCmpS>(converter, context);
        if (applyPartialConversion(funcOp, target, std::move(patterns)).failed())
            signalPassFailure();
    }
};

} // namespace

std::unique_ptr<Pass> mlir::asclower::createLowerAscTileI1Pass() { return std::make_unique<LowerAscTileI1Pass>(); }
