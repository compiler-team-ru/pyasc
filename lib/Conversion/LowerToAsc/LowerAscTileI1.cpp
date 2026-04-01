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

struct LoweringConversionTarget : public ConversionTarget {
    LoweringConversionTarget(TensorTypeConverter &converter, MLIRContext *context) : ConversionTarget(*context)
    {
        addIllegalOp<asctile::CmpSOp, asctile::SelectOp>();
        addLegalDialect<ascendc::AscendCDialect, arith::ArithDialect, emitasc::EmitAscDialect>();
        addLegalOp<UnrealizedConversionCastOp>();
    }
};

struct ConvertCmpSOp : ConvertOp<asctile::CmpSOp> {
    using ConvertOp::ConvertOp;
    using ConvertOp::createTensorOp;

    static std::optional<ascendc::CMPMODE> getCmpMode(asctile::CompareMode pred)
    {
        switch (pred) {
            case asctile::CompareMode::LT:
                return ascendc::CMPMODE::LT;
            case asctile::CompareMode::GT:
                return ascendc::CMPMODE::GT;
            case asctile::CompareMode::EQ:
                return ascendc::CMPMODE::EQ;
            case asctile::CompareMode::LE:
                return ascendc::CMPMODE::LE;
            case asctile::CompareMode::GE:
                return ascendc::CMPMODE::GE;
            case asctile::CompareMode::NE:
                return ascendc::CMPMODE::NE;
            default:
                return std::nullopt;
        }
    }

    LogicalResult convert(asctile::CmpSOp op, ConvertRewriter &rewriter) const override
    {
        ascir::ConstantOpBuilder consts(rewriter);
        auto loc = op.getLoc();
        auto value = rewriter.getRemappedValue(op.getValue());
        auto base = rewriter.getRemappedValue(op.getBase());
        auto srcTy = op.getBase().getType();
        auto srcVecTy = cast<ShapedType>(srcTy);
        auto srcNumElems = srcVecTy.getNumElements();
        I1ReplacementType replType(op.getContext());
        auto dstShape = llvm::divideCeil(srcNumElems, replType.width);
        Value dst = createTensorOp(rewriter, loc, dstShape, replType.iType);
        dst = createReCastOp(rewriter, loc, dst, dstShape, replType.uiType);
        auto mode = getCmpMode(op.getCmpMode());
        if (!mode)
            llvm_unreachable("Unexpected predicate type!");
        auto count = calCount(op.getBase());
        rewriter.create<ascendc::CompareScalarL2Op>(loc, dst, base, value, *mode, consts.i64(count));
        rewriter.replaceOp(op, dst);
        return success();
    }
};

struct ConvertSelect : ConvertOp<asctile::SelectOp> {
    using ConvertOp::ConvertOp;
    using ConvertOp::createTensorOp;

    LogicalResult convert(asctile::SelectOp op, ConvertRewriter &rewriter) const override
    {
        ascir::ConstantOpBuilder consts(rewriter);
        auto loc = op.getLoc();
        auto dst = createTensorOp(rewriter, loc, op.getType());
        auto sel = rewriter.getRemappedValue(op.getSelMask());
        I1ReplacementType replType(op.getContext());
        sel = createReCastOp(rewriter, loc, sel, cast<ShapedType>(sel.getType()).getShape(), replType.uiType);
        auto src0 = rewriter.getRemappedValue(op.getSrc0());
        auto src1 = rewriter.getRemappedValue(op.getSrc1());
        auto zero = consts.i64(0);
        rewriter.create<ascendc::SelectL0Op>(
            loc, dst, sel, src0, src1, ascendc::SELMODE::VSEL_TENSOR_TENSOR_MODE,
            rewriter.create<emitasc::MaskOp>(loc, zero, zero), zero,
            rewriter.create<ascendc::ConstructOp>(loc, rewriter.getType<ascendc::BinaryRepeatParamsType>()));
        rewriter.replaceOp(op, dst);
        return success();
    }
};

struct LowerAscTileI1Pass : public asclower::impl::LowerAscTileI1Base<LowerAscTileI1Pass> {
    void runOnOperation() override
    {
        func::FuncOp funcOp = getOperation();
        TensorTypeConverter converter;
        MLIRContext *context = &getContext();
        LoweringConversionTarget target(converter, context);
        RewritePatternSet patterns(context);
        patterns.insert<ConvertSelect, ConvertCmpSOp>(converter, context);
        if (applyPartialConversion(funcOp, target, std::move(patterns)).failed())
            signalPassFailure();
    }
};

} // namespace

std::unique_ptr<Pass> mlir::asclower::createLowerAscTileI1Pass()
{
    return std::make_unique<LowerAscTileI1Pass>();
}
