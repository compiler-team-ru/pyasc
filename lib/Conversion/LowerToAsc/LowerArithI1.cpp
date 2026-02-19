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
#include "ascir/Dialect/Utils/ConstantOpBuilder.h"
#include "ascir/Dialect/EmitAsc/IR/EmitAsc.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"

#include "Common.h"

namespace mlir {
namespace asclower {
#define GEN_PASS_DEF_LOWERARITHI1
#include "ascir/Conversion/LowerToAsc/Passes.h.inc"
} // namespace asclower
} // namespace mlir

using namespace mlir;
using namespace mlir::asclower;

namespace {

struct LoweringConversionTarget : public ConversionTarget {
    LoweringConversionTarget(TensorTypeConverter& converter, MLIRContext* context) : ConversionTarget(*context)
    {
        addLegalDialect<ascendc::AscendCDialect, arith::ArithDialect, emitasc::EmitAscDialect>();
        addDynamicallyLegalOp<arith::SelectOp, arith::CmpFOp, arith::CmpIOp>(
            [&](Operation* op) { return converter.isLegal(op); });
        addLegalOp<arith::ConstantOp, arith::ExtUIOp, UnrealizedConversionCastOp>();
    }
};

struct ConvertSelect : public ConvertOp<arith::SelectOp> {
    using ConvertOp::converter;
    using ConvertOp::ConvertOp;
    using ConvertOp::createReCastOp;
    using ConvertOp::createTensorOp;

    LogicalResult convert(arith::SelectOp op, ConvertRewriter& rewriter) const override
    {
        ascir::ConstantOpBuilder consts(rewriter);
        auto sel = rewriter.getRemappedValue(op.getCondition());
        auto src0 = rewriter.getRemappedValue(op.getTrueValue());
        auto src1 = rewriter.getRemappedValue(op.getFalseValue());
        Location loc = op.getLoc();
        auto type = cast<ShapedType>(src0.getType());
        Value dst = createTensorOp(rewriter, loc, op.getType());
        Value realDst = dst;
        Type newType = type.getElementType();
        if (!isa<ShapedType>(sel.getType())) {
            auto srcShape = type.getShape();
            auto newCondType = rewriter.getI16Type();
            auto extui = rewriter.create<arith::ExtUIOp>(loc, newCondType, sel);
            auto newCondRes = createTensorOp(rewriter, loc, srcShape, newCondType);
            rewriter.create<ascendc::DuplicateL2Op>(loc, newCondRes, extui, consts.i64(0));
            sel = createReCastOp(rewriter, loc, newCondRes, srcShape, newCondType);
        }
        if (type.getElementType().isInteger(32) || type.getElementType().isIndex()) {
            newType = rewriter.getF32Type();
        } else if (type.getElementType().isInteger(16)) {
            newType = rewriter.getF16Type();
        }
        src0 = createReCastOp(rewriter, loc, src0, type.getShape(), newType);
        src1 = createReCastOp(rewriter, loc, src1, type.getShape(), newType);
        dst = createReCastOp(rewriter, loc, dst, type.getShape(), newType);
        rewriter.create<ascendc::SelectL2Op>(
            loc, dst, sel, src0, src1, ascendc::SELMODE::VSEL_CMPMASK_SPR, consts.i64(0));
        rewriter.replaceOp(op, realDst);
        return success();
    }
};

struct ConvertCmpI : public ConvertOp<arith::CmpIOp> {
    using ConvertOp::ConvertOp;
    using ConvertOp::createTensorOp;

    LogicalResult convert(arith::CmpIOp op, ConvertRewriter& rewriter) const override
    {
        ascir::ConstantOpBuilder consts(rewriter);
        auto src0 = rewriter.getRemappedValue(op.getLhs());
        auto src1 = rewriter.getRemappedValue(op.getRhs());
        auto srcTy = src0.getType();
        auto loc = op.getLoc();
        auto srcVecTy = cast<ShapedType>(srcTy);
        auto srcNumElems = srcVecTy.getNumElements();
        // Each dest tensor i16 element contains 16 comparison results.
        auto dstShape = llvm::divideCeil(srcNumElems, I1ReplacementType::width);
        unsigned bitWidth = srcVecTy.getElementTypeBitWidth();
        assert((bitWidth == 16 || bitWidth == 32) && "Only cast for i16 and i32 is supported");
        auto castToType = bitWidth == 16 ? rewriter.getF16Type() : rewriter.getF32Type();
        auto src0Casted = createTensorOp(rewriter, loc, srcVecTy.getShape(), castToType);
        auto src1Casted = createTensorOp(rewriter, loc, srcVecTy.getShape(), castToType);
        rewriter.create<ascendc::CastL2Op>(loc, src0Casted, src0, ascendc::RoundMode::CAST_NONE, consts.i64(1));
        rewriter.create<ascendc::CastL2Op>(loc, src1Casted, src1, ascendc::RoundMode::CAST_NONE, consts.i64(1));
        Value dst = createTensorOp(rewriter, loc, dstShape, rewriter.getI16Type());
        ascendc::CMPMODE cmpMode;
        switch (op.getPredicate()) {
        case arith::CmpIPredicate::eq:
            cmpMode = ascendc::CMPMODE::EQ;
            break;
        case arith::CmpIPredicate::ne:
            cmpMode = ascendc::CMPMODE::NE;
            break;
        case arith::CmpIPredicate::slt:
        case arith::CmpIPredicate::ult:
            cmpMode = ascendc::CMPMODE::LT;
            break;
        case arith::CmpIPredicate::sle:
        case arith::CmpIPredicate::ule:
            cmpMode = ascendc::CMPMODE::LE;
            break;
        case arith::CmpIPredicate::sgt:
        case arith::CmpIPredicate::ugt:
            cmpMode = ascendc::CMPMODE::GT;
            break;
        case arith::CmpIPredicate::sge:
        case arith::CmpIPredicate::uge:
            cmpMode = ascendc::CMPMODE::GE;
            break;
        default:
            llvm_unreachable("Unexpected predicate type!");
        }
        auto [maskH, maskL] = getMask(srcVecTy);
        auto mask = rewriter.create<emitasc::MaskOp>(loc, consts.i64(maskH), consts.i64(maskL));
        auto repeatParams = rewriter.create<ascendc::ConstructOp>(
            loc, rewriter.getType<ascendc::BinaryRepeatParamsType>(),
            ValueRange{
                consts.i64(dstBlkStride), consts.i64(src0BlkStride), consts.i64(src1BlkStride),
                consts.i64(dstRepStride), consts.i64(src0RepStride), consts.i64(src1RepStride)});
        rewriter.create<ascendc::CompareL0Op>(
            loc, dst, src0, src1, cmpMode, mask, consts.i64(getRepeatTimes(srcVecTy)), repeatParams);
        rewriter.replaceOp(op, dst);
        return success();
    }
};

struct ConvertCmpF : public ConvertOp<arith::CmpFOp> {
    using ConvertOp::converter;
    using ConvertOp::ConvertOp;

    LogicalResult convert(arith::CmpFOp op, ConvertRewriter& rewriter) const override
    {
        ascir::ConstantOpBuilder consts(rewriter);
        auto src0 = rewriter.getRemappedValue(op.getLhs());
        auto src1 = rewriter.getRemappedValue(op.getRhs());
        auto srcTy = src0.getType();
        auto loc = op.getLoc();
        auto srcVecTy = cast<ShapedType>(srcTy);
        auto srcNumElems = srcVecTy.getNumElements();
        // Each dest tensor i16 element contains 16 comparison results.
        auto dstShape = llvm::divideCeil(srcNumElems, 16);
        Value dst = createTensorOp(rewriter, loc, dstShape, rewriter.getI16Type());
        ascendc::CMPMODE cmpMode;
        switch (op.getPredicate()) {
        case arith::CmpFPredicate::OEQ:
        case arith::CmpFPredicate::UEQ:
            cmpMode = ascendc::CMPMODE::EQ;
            break;
        case arith::CmpFPredicate::ONE:
        case arith::CmpFPredicate::UNE:
            cmpMode = ascendc::CMPMODE::NE;
            break;
        case arith::CmpFPredicate::OLT:
        case arith::CmpFPredicate::ULT:
            cmpMode = ascendc::CMPMODE::LT;
            break;
        case arith::CmpFPredicate::OLE:
        case arith::CmpFPredicate::ULE:
            cmpMode = ascendc::CMPMODE::LE;
            break;
        case arith::CmpFPredicate::OGT:
        case arith::CmpFPredicate::UGT:
            cmpMode = ascendc::CMPMODE::GT;
            break;
        case arith::CmpFPredicate::OGE:
        case arith::CmpFPredicate::UGE:
            cmpMode = ascendc::CMPMODE::GE;
            break;
        default:
            llvm_unreachable("Unexpected predicate type!");
        }

        auto [maskH, maskL] = getMask(srcVecTy);
        auto mask = rewriter.create<emitasc::MaskOp>(loc, consts.i64(maskH), consts.i64(maskL));
        auto repeatParams = rewriter.create<ascendc::ConstructOp>(
            loc, rewriter.getType<ascendc::BinaryRepeatParamsType>(),
            ValueRange{
                consts.i64(dstBlkStride), consts.i64(src0BlkStride), consts.i64(src1BlkStride),
                consts.i64(dstRepStride), consts.i64(src0RepStride), consts.i64(src1RepStride)});
        rewriter.create<ascendc::CompareL0Op>(
            loc, dst, src0, src1, cmpMode, mask, consts.i64(getRepeatTimes(srcVecTy)), repeatParams);
        rewriter.replaceOp(op, dst);
        return success();
    }
};

struct LowerArithI1Pass : public asclower::impl::LowerArithI1Base<LowerArithI1Pass> {
    void runOnOperation() override
    {
        func::FuncOp funcOp = getOperation();
        TensorTypeConverter converter;
        MLIRContext* context = &getContext();
        LoweringConversionTarget target(converter, context);
        RewritePatternSet patterns(context);
        patterns.insert<
            //
            ConvertSelect, ConvertCmpF, ConvertCmpI
            //
            >(converter, context);
        if (applyPartialConversion(funcOp, target, std::move(patterns)).failed())
            signalPassFailure();
    }
};

} // namespace

std::unique_ptr<Pass> mlir::asclower::createLowerArithI1Pass() { return std::make_unique<LowerArithI1Pass>(); }
