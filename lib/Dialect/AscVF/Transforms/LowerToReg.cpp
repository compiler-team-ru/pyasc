/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ascir/Dialect/Asc/IR/Asc.h"
#include "ascir/Dialect/Asc/Utils/Utils.h"
#include "ascir/Dialect/AscVF/IR/AscVF.h"
#include "ascir/Dialect/AscVF/Transforms/Passes.h"
#include "ascir/Dialect/EmitAsc/IR/EmitAsc.h"
#include "ascir/Dialect/Utils/ConstantOpBuilder.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/TypeSwitch.h"

namespace mlir {
namespace ascvf {
#define GEN_PASS_DEF_LOWERTOREG
#include "ascir/Dialect/AscVF/Transforms/Passes.h.inc"
} // namespace ascvf
} // namespace mlir

using namespace mlir;

namespace {

ascendc::UpdateMaskOp createUpdateMask(OpBuilder& builder, Value calCountVar, Type type)
{
    auto updateMask = builder.create<ascendc::UpdateMaskOp>(
        builder.getUnknownLoc(), builder.getType<ascendc::MaskRegType>(), calCountVar, type);
    return updateMask;
}

ascendc::CreateMaskOp createMask(OpBuilder& builder, Type type, ascendc::MaskPattern pattern)
{
    auto op = builder.create<ascendc::CreateMaskOp>(
        builder.getUnknownLoc(), builder.getType<ascendc::MaskRegType>(), type, pattern);
    return op;
}

ascendc::RegTensorOp createRegTensor(OpBuilder builder, Type elemType)
{
    auto regTensorOp = builder.create<ascendc::RegTensorOp>(
        builder.getUnknownLoc(), ascendc::RegTensorType::get(builder.getContext(), elemType));
    return regTensorOp;
}

ascvf::LoadOp createLoad(OpBuilder builder, Value dstReg, Value srcTensor, Value offset)
{
    return builder.create<ascvf::LoadOp>(builder.getUnknownLoc(), dstReg, srcTensor, offset);
}

ascvf::StoreOp createStore(OpBuilder builder, Value dstTensor, Value srcReg, Value offset, Value mask)
{
    return builder.create<ascvf::StoreOp>(builder.getUnknownLoc(), dstTensor, offset, srcReg, mask);
}

template <typename ReduceOpType>
Value getNeutralElement(ascir::ConstantOpBuilder& consts, Type elemType)
{
    if constexpr (std::is_same_v<ReduceOpType, ascendc::ReduceMaxL2Op>) {
        if (elemType.isF32())
            return consts.f32(-std::numeric_limits<float>::infinity());
        if (elemType.isF16())
            return consts.f16(-std::numeric_limits<float>::infinity());
        if (elemType.isInteger(32))
            return consts.i32(-std::numeric_limits<int>::infinity());
    }
    if constexpr (std::is_same_v<ReduceOpType, ascendc::ReduceSumL2Op>) {
        if (elemType.isF32())
            return consts.f32(0);
        if (elemType.isF16())
            return consts.f16(0);
        if (elemType.isInteger(32))
            return consts.i32(0);
    }
    if constexpr (std::is_same_v<ReduceOpType, ascendc::ReduceMinL2Op>) {
        if (elemType.isF32())
            return consts.f32(std::numeric_limits<float>::infinity());
        if (elemType.isF16())
            return consts.f16(std::numeric_limits<float>::infinity());
        if (elemType.isInteger(32))
            return consts.i32(std::numeric_limits<int>::infinity());
    }
    llvm_unreachable("unknown neutral element");
}

class TranslatorFactory {
    Value calCount, oneRepeatSize;
    Type elemType;

public:
    TranslatorFactory(Value calCount, Value oneRepeatSize, Type elemType)
        : calCount(calCount), oneRepeatSize(oneRepeatSize), elemType(elemType)
    {}
    ~TranslatorFactory() = default;

    Value createRepeatTimes(OpBuilder builder)
    {
        auto repeatTimes = builder.create<arith::CeilDivSIOp>(
            builder.getUnknownLoc(), builder.getIndexType(), calCount, oneRepeatSize);
        return repeatTimes.getResult();
    }

    template <typename T>
    auto binary()
    {
        return [&](ascendc::BinaryL2Op binaryOp) {
            OpBuilder builder(binaryOp);
            ascir::ConstantOpBuilder consts(builder);
            auto src0Reg = createRegTensor(builder, elemType);
            auto src1Reg = createRegTensor(builder, elemType);
            auto dstReg = createRegTensor(builder, elemType);

            auto calCountVar = builder.create<emitasc::VariableOp>(
                builder.getUnknownLoc(), MemRefType::get(1, builder.getIntegerType(32U, false)), calCount);
            auto repeatTimes = createRepeatTimes(builder);
            auto maskAll = createMask(builder, elemType, ascendc::MaskPattern::ALL);
            auto loop = createLoop(builder, repeatTimes);
            builder.setInsertionPointToStart(loop.getBody());
            auto updateMask = createUpdateMask(builder, calCountVar.getResult(), elemType);
            auto mulOp = builder.create<arith::MulIOp>(builder.getUnknownLoc(), loop.getInductionVar(), oneRepeatSize);
            createLoad(builder, src0Reg, binaryOp.getSrc0(), mulOp);
            createLoad(builder, src1Reg, binaryOp.getSrc1(), mulOp);
            builder.create<T>(builder.getUnknownLoc(), dstReg, src0Reg, src1Reg, maskAll);
            createStore(builder, binaryOp.getDst(), dstReg, mulOp, updateMask.getResult());

            binaryOp.erase();
        };
    }

    template <typename T>
    auto unary()
    {
        return [&](ascendc::UnaryL2Op unaryOp) {
            OpBuilder builder(unaryOp);
            ascir::ConstantOpBuilder consts(builder);
            auto srcReg = createRegTensor(builder, elemType);
            auto dstReg = createRegTensor(builder, elemType);

            auto calCountVar = builder.create<emitasc::VariableOp>(
                builder.getUnknownLoc(), MemRefType::get(1, builder.getIntegerType(32U, false)), calCount);
            auto repeatTimes = createRepeatTimes(builder);
            auto maskAll = createMask(builder, elemType, ascendc::MaskPattern::ALL);
            auto loop = createLoop(builder, repeatTimes);
            builder.setInsertionPointToStart(loop.getBody());
            auto updateMask = createUpdateMask(builder, calCountVar.getResult(), elemType);
            auto mulOp = builder.create<arith::MulIOp>(builder.getUnknownLoc(), loop.getInductionVar(), oneRepeatSize);
            createLoad(builder, srcReg, unaryOp.getSrc(), mulOp);
            builder.create<T>(builder.getUnknownLoc(), dstReg, srcReg, maskAll);
            createStore(builder, unaryOp.getDst(), dstReg, mulOp, updateMask.getResult());

            unaryOp.erase();
        };
    }

    template <typename T>
    auto vecScalar()
    {
        return [&](ascendc::VecScalarL2Op vecScalarOp) {
            OpBuilder builder(vecScalarOp);
            ascir::ConstantOpBuilder consts(builder);
            auto srcReg = createRegTensor(builder, elemType);
            auto dstReg = createRegTensor(builder, elemType);

            auto calCountVar = builder.create<emitasc::VariableOp>(
                builder.getUnknownLoc(), MemRefType::get(1, builder.getIntegerType(32U, false)), calCount);
            auto repeatTimes = createRepeatTimes(builder);
            auto loop = createLoop(builder, repeatTimes);
            builder.setInsertionPointToStart(loop.getBody());
            auto updateMask = createUpdateMask(builder, calCountVar.getResult(), elemType);
            auto mulOp = builder.create<arith::MulIOp>(builder.getUnknownLoc(), loop.getInductionVar(), oneRepeatSize);
            createLoad(builder, srcReg, vecScalarOp.getSrc(), mulOp);
            builder.create<T>(builder.getUnknownLoc(), dstReg, srcReg, vecScalarOp.getScalar(), updateMask.getResult());
            createStore(builder, vecScalarOp.getDst(), dstReg, mulOp, updateMask.getResult());

            vecScalarOp.erase();
        };
    }

    template <typename T>
    auto vecScalarWithDuplicate()
    {
        return [&](ascendc::VecScalarL2Op vecScalarOp) {
            OpBuilder builder(vecScalarOp);
            ascir::ConstantOpBuilder consts(builder);
            auto srcReg = createRegTensor(builder, elemType);
            auto dupReg = createRegTensor(builder, elemType);
            auto dstReg = createRegTensor(builder, elemType);

            auto calCountVar = builder.create<emitasc::VariableOp>(
                builder.getUnknownLoc(), MemRefType::get(1, builder.getIntegerType(32U, false)), calCount);
            auto repeatTimes = createRepeatTimes(builder);
            auto maskAll = createMask(builder, elemType, ascendc::MaskPattern::ALL);
            auto loop = createLoop(builder, repeatTimes);
            builder.setInsertionPointToStart(loop.getBody());
            auto updateMask = createUpdateMask(builder, calCountVar.getResult(), elemType);
            auto mulOp = builder.create<arith::MulIOp>(builder.getUnknownLoc(), loop.getInductionVar(), oneRepeatSize);
            createLoad(builder, srcReg, vecScalarOp.getSrc(), mulOp);
            builder.create<ascendc::DuplicateRegOp>(builder.getUnknownLoc(), dupReg, vecScalarOp.getScalar(), maskAll);
            builder.create<T>(builder.getUnknownLoc(), dstReg, srcReg, dupReg, maskAll);
            createStore(builder, vecScalarOp.getDst(), dstReg, mulOp, updateMask.getResult());

            vecScalarOp.erase();
        };
    }

    template <typename ReduceL2Op, typename AccumulateRegOp, typename ReduceRegOp>
    auto reduce()
    {
        return [&](ReduceL2Op reduceOp) {
            OpBuilder builder(reduceOp);
            ascir::ConstantOpBuilder consts(builder);
            auto srcReg = createRegTensor(builder, elemType);
            auto dstReg = createRegTensor(builder, elemType);
            auto accReg = createRegTensor(builder, elemType);
            auto acc0Reg = createRegTensor(builder, elemType);
            Value neutral = getNeutralElement<ReduceL2Op>(consts, elemType);
            builder.create<ascendc::DuplicateScalarRegOp>(builder.getUnknownLoc(), accReg, neutral);
            auto maskAll = createMask(builder, elemType, ascendc::MaskPattern::ALL);
            auto repeatTimes = builder.create<arith::DivSIOp>(
                builder.getUnknownLoc(), builder.getIndexType(), calCount, oneRepeatSize);
            auto calCountVar = builder.create<emitasc::VariableOp>(
                builder.getUnknownLoc(), MemRefType::get(1, builder.getIntegerType(32U, false)), calCount);
            auto loop = createLoop(builder, repeatTimes);
            builder.setInsertionPoint(loop);
            builder.setInsertionPointToStart(loop.getBody());
            createUpdateMask(builder, calCountVar.getResult(), elemType);
            auto mulOp = builder.create<arith::MulIOp>(builder.getUnknownLoc(), loop.getInductionVar(), oneRepeatSize);
            createLoad(builder, srcReg, reduceOp.getSrc(), mulOp);
            builder.create<AccumulateRegOp>(builder.getUnknownLoc(), accReg, accReg, srcReg, maskAll);

            builder.setInsertionPointAfter(loop);
            auto remOp = builder.create<arith::RemSIOp>(builder.getUnknownLoc(), calCount, oneRepeatSize);
            auto cmpOp = builder.create<arith::CmpIOp>(
                builder.getUnknownLoc(), arith::CmpIPredicate::ne, remOp.getResult(), consts.index(0));
            auto ifOp = builder.create<scf::IfOp>(builder.getUnknownLoc(), cmpOp.getResult(), false);
            builder.setInsertionPointToStart(ifOp.getBody());
            builder.create<ascendc::DuplicateScalarRegOp>(builder.getUnknownLoc(), acc0Reg, neutral);
            auto lastIter = builder.create<arith::MulIOp>(builder.getUnknownLoc(), repeatTimes, oneRepeatSize);
            createLoad(builder, srcReg, reduceOp.getSrc(), lastIter);
            auto tailMask = createUpdateMask(builder, calCountVar.getResult(), elemType);
            builder.create<ascendc::SelectRegOp>(builder.getUnknownLoc(), acc0Reg, srcReg, acc0Reg, tailMask);
            builder.create<AccumulateRegOp>(builder.getUnknownLoc(), accReg, accReg, acc0Reg, maskAll);

            builder.setInsertionPointAfter(ifOp);
            builder.create<ReduceRegOp>(builder.getUnknownLoc(), dstReg, accReg, maskAll);
            auto maskOne = createMask(builder, elemType, ascendc::MaskPattern::VL1);
            createStore(builder, reduceOp.getDst(), dstReg, consts.index(0), maskOne);
            reduceOp.erase();
        };
    }

    auto duplicate()
    {
        return [&](ascendc::DuplicateL2Op duplicateOp) {
            OpBuilder builder(duplicateOp);
            ascir::ConstantOpBuilder consts(builder);
            auto srcReg = createRegTensor(builder, elemType);
            auto dstReg = createRegTensor(builder, elemType);
            auto tmpReg = createRegTensor(builder, elemType);
            createLoad(builder, srcReg, duplicateOp.getScalar(), consts.index(0));
            auto maskAll = createMask(builder, elemType, ascendc::MaskPattern::ALL);
            builder.create<ascendc::DuplicateRegOp>(builder.getUnknownLoc(), tmpReg, srcReg, maskAll);

            auto calCountVar = builder.create<emitasc::VariableOp>(
                builder.getUnknownLoc(), MemRefType::get(1, builder.getIntegerType(32U, false)), calCount);
            auto repeatTimes = createRepeatTimes(builder);

            auto loop = createLoop(builder, repeatTimes);
            builder.setInsertionPoint(loop.getBody()->getTerminator());
            auto updateMask = createUpdateMask(builder, calCountVar.getResult(), elemType);
            auto mulOp = builder.create<arith::MulIOp>(builder.getUnknownLoc(), loop.getInductionVar(), oneRepeatSize);
            auto store = createStore(builder, duplicateOp.getDst(), tmpReg, mulOp, updateMask);
            duplicateOp.erase();
        };
    }

private:
    ascvf::VFForOp createLoop(OpBuilder builder, Value ub)
    {
        return builder.create<ascvf::VFForOp>(builder.getUnknownLoc(), ub);
    }
};

Value createGetVecLen(OpBuilder builder, Operation* op)
{
    if (auto vecLen = ascendc::getVecLen(op)) {
        ascir::ConstantOpBuilder consts(builder);
        return consts.index(vecLen.value());
    }
    auto getVecLenIndex = builder.create<ascendc::GetVecLenOp>(builder.getUnknownLoc(), builder.getIndexType());
    return getVecLenIndex.getResult();
}

Value createOneRepeatSize(OpBuilder builder, Value getVecLenIndex, Type groupType)
{
    ascir::ConstantOpBuilder consts(builder);
    auto sizeIndex = consts.index(ascendc::getTypeSize(groupType));
    auto oneRepeatSize =
        builder.create<arith::DivSIOp>(builder.getUnknownLoc(), builder.getIndexType(), getVecLenIndex, sizeIndex);
    return oneRepeatSize;
}

void lowerToMicro(ascvf::VecScopeOp vecScopeOp, Value calCount, Type groupType)
{
    auto builder = OpBuilder::atBlockBegin(vecScopeOp.getBody());
    auto getVecLenIndex = createGetVecLen(builder, vecScopeOp);
    Value oneRepeatSizeIndex = createOneRepeatSize(builder, getVecLenIndex, groupType);
    if (!calCount.getType().isIndex()) {
        auto castOp = builder.create<arith::IndexCastOp>(builder.getUnknownLoc(), builder.getIndexType(), calCount);
        calCount = castOp.getResult();
    }
    TranslatorFactory factory(calCount, oneRepeatSizeIndex, groupType);

    vecScopeOp.walk([&](Operation* op) {
        llvm::TypeSwitch<Operation*>(op)
            .Case<ascendc::ReduceMaxL2Op>(
                factory.reduce<ascendc::ReduceMaxL2Op, ascendc::MaxRegOp, ascendc::ReduceMaxRegOp>())
            .Case<ascendc::ReduceSumL2Op>(
                factory.reduce<ascendc::ReduceSumL2Op, ascendc::AddRegOp, ascendc::ReduceSumRegOp>())
            .Case<ascendc::ReduceMinL2Op>(
                factory.reduce<ascendc::ReduceMinL2Op, ascendc::MinRegOp, ascendc::ReduceMinRegOp>())
            .Case<ascendc::DuplicateL2Op>(factory.duplicate())
            // BinaryOp
            .Case<ascendc::AddL2Op>(factory.binary<ascendc::AddRegOp>())
            .Case<ascendc::AndL2Op>(factory.binary<ascendc::AndRegOp>())
            .Case<ascendc::DivL2Op>(factory.binary<ascendc::DivRegOp>())
            .Case<ascendc::MaxL2Op>(factory.binary<ascendc::MaxRegOp>())
            .Case<ascendc::MinL2Op>(factory.binary<ascendc::MinRegOp>())
            .Case<ascendc::MulL2Op>(factory.binary<ascendc::MulRegOp>())
            .Case<ascendc::MulAddDstL2Op>(factory.binary<ascendc::MulAddDstRegOp>())
            .Case<ascendc::OrL2Op>(factory.binary<ascendc::OrRegOp>())
            .Case<ascendc::PreluL2Op>(factory.binary<ascendc::PreluRegOp>())
            .Case<ascendc::SubL2Op>(factory.binary<ascendc::SubRegOp>())
            // UnaryOp
            .Case<ascendc::AbsL2Op>(factory.unary<ascendc::AbsRegOp>())
            .Case<ascendc::ExpL2Op>(factory.unary<ascendc::ExpRegOp>())
            .Case<ascendc::LnL2Op>(factory.unary<ascendc::LnRegOp>())
            .Case<ascendc::NegL2Op>(factory.unary<ascendc::NegRegOp>())
            .Case<ascendc::NotL2Op>(factory.unary<ascendc::NotRegOp>())
            .Case<ascendc::ReluL2Op>(factory.unary<ascendc::ReluRegOp>())
            .Case<ascendc::SqrtL2Op>(factory.unary<ascendc::SqrtRegOp>())
            // VecScalarOp
            .Case<ascendc::LeakyReluL2Op>(factory.vecScalar<ascendc::LeakyReluRegOp>())
            .Case<ascendc::ShiftLeftL2Op>(factory.vecScalar<ascendc::ShiftLeftsRegOp>())
            .Case<ascendc::ShiftRightL2Op>(factory.vecScalar<ascendc::ShiftRightsRegOp>())
            // VecScalarWithDuplicate
            .Case<ascendc::AddsL2Op>(factory.vecScalarWithDuplicate<ascendc::AddRegOp>())
            .Case<ascendc::DivsL2Op>(factory.vecScalarWithDuplicate<ascendc::DivRegOp>())
            .Case<ascendc::MaxsL2Op>(factory.vecScalarWithDuplicate<ascendc::MaxRegOp>())
            .Case<ascendc::MinsL2Op>(factory.vecScalarWithDuplicate<ascendc::MinRegOp>())
            .Case<ascendc::MulsL2Op>(factory.vecScalarWithDuplicate<ascendc::MulRegOp>())
            .Case<ascendc::SubsL2Op>(factory.vecScalarWithDuplicate<ascendc::SubRegOp>());
    });
}

ascvf::VecScopeOp wrapInVecScope(OpBuilder& builder, const SmallVector<Operation*>& ops)
{
    auto vecScope = builder.create<ascvf::VecScopeOp>(builder.getUnknownLoc());
    auto* blockVecScope = &vecScope.getRegion().emplaceBlock();

    builder.setInsertionPointToStart(blockVecScope);

    for (auto* op : ops) {
        builder.clone(*op);
    }
    builder.create<ascvf::YieldOp>(builder.getUnknownLoc());
    for (auto* op : ops) {
        op->erase();
    }
    return vecScope;
}

struct LowerToRegPass : public ascvf::impl::LowerToRegBase<LowerToRegPass> {
    void runOnOperation() override
    {
        func::FuncOp funcOp = getOperation();
        funcOp.walk([](ascvf::VFGroupOp fusedOp) {
            Block* block = fusedOp.getBody();
            SmallVector<Operation*> ops;
            for (auto& op : block->without_terminator()) {
                ops.emplace_back(&op);
            }

            OpBuilder builder(fusedOp.getContext());
            builder.setInsertionPointToStart(block);

            auto vecScope = wrapInVecScope(builder, ops);
            lowerToMicro(vecScope, fusedOp.getCalCount(), fusedOp.getGroupType());
        });
    }
};

} // namespace

std::unique_ptr<Pass> mlir::ascvf::createLowerToRegPass() { return std::make_unique<LowerToRegPass>(); }
