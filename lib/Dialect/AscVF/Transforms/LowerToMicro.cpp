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
#include "ascir/Dialect/AscVF/IR/AscVF.h"
#include "ascir/Dialect/Asc/Transforms/Passes.h"
#include "ascir/Dialect/Asc/Utils/Utils.h"
#include "ascir/Dialect/EmitAsc/IR/EmitAsc.h"
#include "ascir/Dialect/Utils/ConstantOpBuilder.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/TypeSwitch.h"

namespace mlir {
namespace ascvf {
#define GEN_PASS_DEF_LOWERTOMICRO
#include "ascir/Dialect/AscVF/Transforms/Passes.h.inc"
} // namespace ascvf
} // namespace mlir

using namespace mlir;

namespace {

ascendc::UpdateMaskOp createUpdateMask(ascvf::VFForOp loop, Value calCount, Type type)
{
    OpBuilder builder(loop);
    auto calCountVar = builder.create<emitasc::VariableOp>(
        builder.getUnknownLoc(), MemRefType::get(1, builder.getIntegerType(32U, false)), calCount);
    builder.setInsertionPointToStart(loop.getBody());
    auto updateMask = builder.create<ascendc::UpdateMaskOp>(
        builder.getUnknownLoc(), builder.getType<ascendc::MaskRegType>(), calCountVar.getResult(), type);
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

ascvf::LoadMicroOp createLoad(OpBuilder builder, Value dstReg, Value srcTensor, Value offset)
{
    return builder.create<ascvf::LoadMicroOp>(builder.getUnknownLoc(), dstReg, srcTensor, offset);
}

ascvf::StoreMicroOp createStore(OpBuilder builder, Value dstTensor, Value srcReg, Value offset, Value mask)
{
    return builder.create<ascvf::StoreMicroOp>(builder.getUnknownLoc(), dstTensor, offset, srcReg, mask);
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
    Value calCount, repeatTimes, oneRepeatSizeIndex;
    Type elemType;

public:
    TranslatorFactory(Value calCount, Value repeatTimes, Value oneRepeatSizeIndex, Type elemType)
        : calCount(calCount), repeatTimes(repeatTimes), oneRepeatSizeIndex(oneRepeatSizeIndex), elemType(elemType)
    {}
    ~TranslatorFactory() = default;

    template <typename T>
    auto binary()
    {
        return [&](ascendc::BinaryL2Op binaryOp) {
            OpBuilder builder(binaryOp);
            ascir::ConstantOpBuilder consts(builder);
            auto src0Reg = createRegTensor(builder, elemType);
            auto src1Reg = createRegTensor(builder, elemType);
            auto dstReg = createRegTensor(builder, elemType);

            auto loop = createLoop(builder);
            builder.setInsertionPointToStart(loop.getBody());

            auto updateMask = createUpdateMask(loop, calCount, elemType);
            auto mulOp =
                builder.create<arith::MulIOp>(builder.getUnknownLoc(), loop.getInductionVar(), oneRepeatSizeIndex);
            createLoad(builder, src0Reg, binaryOp.getSrc0(), mulOp);
            createLoad(builder, src1Reg, binaryOp.getSrc1(), mulOp);
            builder.create<T>(builder.getUnknownLoc(), dstReg, src0Reg, src1Reg, updateMask.getResult());
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
            auto loop = createLoop(builder);
            builder.setInsertionPointToStart(loop.getBody());

            auto updateMask = createUpdateMask(loop, calCount, elemType);
            auto mulOp =
                builder.create<arith::MulIOp>(builder.getUnknownLoc(), loop.getInductionVar(), oneRepeatSizeIndex);
            createLoad(builder, srcReg, unaryOp.getSrc(), mulOp);
            builder.create<T>(builder.getUnknownLoc(), dstReg, srcReg, updateMask.getResult());
            createStore(builder, unaryOp.getDst(), dstReg, mulOp, updateMask.getResult());

            unaryOp.erase();
        };
    }

    template <typename ReduceL2Op, typename AccumulateMicroOp, typename ReduceMicroOp>
    auto reduce()
    {
        return [&](ReduceL2Op reduceOp) {
            OpBuilder builder(reduceOp);
            ascir::ConstantOpBuilder consts(builder);
            auto srcReg = createRegTensor(builder, elemType);
            auto dstReg = createRegTensor(builder, elemType);
            auto accReg = createRegTensor(builder, elemType);
            Value neutral = getNeutralElement<ReduceL2Op>(consts, elemType);
            auto duplicateOp =
                builder.create<ascendc::DuplicateScalarMicroOp>(builder.getUnknownLoc(), accReg, neutral);

            auto loop = createLoop(builder);
            builder.setInsertionPoint(loop);
            builder.setInsertionPointToStart(loop.getBody());
            auto mulOp =
                builder.create<arith::MulIOp>(builder.getUnknownLoc(), loop.getInductionVar(), oneRepeatSizeIndex);
            auto updateMask = createUpdateMask(loop, calCount, elemType);
            auto load = createLoad(builder, srcReg, reduceOp.getSrc(), mulOp);
            builder.create<AccumulateMicroOp>(builder.getUnknownLoc(), accReg, accReg, srcReg, updateMask);

            builder.setInsertionPointAfter(loop);
            auto maskAll = createMask(builder, elemType, ascendc::MaskPattern::ALL);
            builder.create<ReduceMicroOp>(builder.getUnknownLoc(), dstReg, accReg, maskAll);
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
            builder.create<ascendc::DuplicateMicroOp>(builder.getUnknownLoc(), tmpReg, srcReg, maskAll);

            auto loop = createLoop(builder);
            auto updateMask = createUpdateMask(loop, calCount, elemType);
            builder.setInsertionPoint(loop.getBody()->getTerminator());
            auto mulOp =
                builder.create<arith::MulIOp>(builder.getUnknownLoc(), loop.getInductionVar(), oneRepeatSizeIndex);
            auto store = createStore(builder, duplicateOp.getDst(), tmpReg, mulOp, updateMask);
            duplicateOp.erase();
        };
    }

private:
    ascvf::VFForOp createLoop(OpBuilder builder)
    {
        ascir::ConstantOpBuilder consts(builder);
        auto loop = builder.create<ascvf::VFForOp>(builder.getUnknownLoc(), repeatTimes);
        return loop;
    }
};

std::pair<Value, Value> createRepeatTimes(OpBuilder builder, Value calCount, Type groupType)
{
    ascir::ConstantOpBuilder consts(builder);
    auto getVecLenIndex = builder.create<ascendc::GetVecLenOp>(builder.getUnknownLoc(), builder.getIndexType());
    auto sizeIndex = consts.index(ascendc::getTypeSize(groupType));
    auto div = builder.create<arith::DivSIOp>(
        builder.getUnknownLoc(), builder.getIndexType(), getVecLenIndex.getResult(), sizeIndex);
    auto oneRepeatSizeIndex = div.getResult();
    if (!calCount.getType().isIndex()) {
        auto castOp = builder.create<arith::IndexCastOp>(builder.getUnknownLoc(), builder.getIndexType(), calCount);
        calCount = castOp.getResult();
    }
    auto repeatTimes = builder.create<arith::CeilDivSIOp>(
        builder.getUnknownLoc(), builder.getIndexType(), calCount, oneRepeatSizeIndex);
    return {oneRepeatSizeIndex, repeatTimes};
}

void lowerToMicro(ascvf::VecScopeOp vecScopeOp, Value calCount, Type groupType)
{
    auto builder = OpBuilder::atBlockBegin(vecScopeOp.getBody());
    Value oneRepeatSizeIndex, repeatTimes;
    std::tie(oneRepeatSizeIndex, repeatTimes) = createRepeatTimes(builder, calCount, groupType);
    TranslatorFactory factory(calCount, repeatTimes, oneRepeatSizeIndex, groupType);

    vecScopeOp.walk([&](Operation* op) {
        llvm::TypeSwitch<Operation*>(op)
            .Case<ascendc::ReduceMaxL2Op>(
                factory.reduce<ascendc::ReduceMaxL2Op, ascendc::MaxMicroOp, ascendc::ReduceMaxMicroOp>())
            .Case<ascendc::ReduceSumL2Op>(
                factory.reduce<ascendc::ReduceSumL2Op, ascendc::AddMicroOp, ascendc::ReduceSumMicroOp>())
            .Case<ascendc::ReduceMinL2Op>(
                factory.reduce<ascendc::ReduceMinL2Op, ascendc::MinMicroOp, ascendc::ReduceMinMicroOp>())
            .Case<ascendc::DuplicateL2Op>(factory.duplicate())
            // BinaryOp
            .Case<ascendc::AddL2Op>(factory.binary<ascendc::AddMicroOp>())
            .Case<ascendc::AndL2Op>(factory.binary<ascendc::AndMicroOp>())
            .Case<ascendc::DivL2Op>(factory.binary<ascendc::DivMicroOp>())
            .Case<ascendc::MaxL2Op>(factory.binary<ascendc::MaxMicroOp>())
            .Case<ascendc::MinL2Op>(factory.binary<ascendc::MinMicroOp>())
            .Case<ascendc::MulL2Op>(factory.binary<ascendc::MulMicroOp>())
            .Case<ascendc::MulAddDstL2Op>(factory.binary<ascendc::MulAddDstMicroOp>())
            .Case<ascendc::OrL2Op>(factory.binary<ascendc::OrMicroOp>())
            .Case<ascendc::PreluL2Op>(factory.binary<ascendc::PreluMicroOp>())
            .Case<ascendc::SubL2Op>(factory.binary<ascendc::SubMicroOp>())
            // UnaryOp
            .Case<ascendc::AbsL2Op>(factory.unary<ascendc::AbsMicroOp>())
            .Case<ascendc::ExpL2Op>(factory.unary<ascendc::ExpMicroOp>())
            .Case<ascendc::LnL2Op>(factory.unary<ascendc::LnMicroOp>())
            .Case<ascendc::NegL2Op>(factory.unary<ascendc::NegMicroOp>())
            .Case<ascendc::NotL2Op>(factory.unary<ascendc::NotMicroOp>())
            .Case<ascendc::ReluL2Op>(factory.unary<ascendc::ReluMicroOp>())
            .Case<ascendc::SqrtL2Op>(factory.unary<ascendc::SqrtMicroOp>());
    });
}

ascvf::VecScopeOp wrapInVecScope(OpBuilder& builder, SmallVector<Operation*> ops)
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

struct LowerToMicroPass : public ascvf::impl::LowerToMicroBase<LowerToMicroPass> {
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

namespace mlir {
namespace ascvf {
std::unique_ptr<Pass> createLowerToMicroPass() { return std::make_unique<LowerToMicroPass>(); }
} // namespace ascvf
} // namespace mlir
