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
#include "ascir/Dialect/Asc/Transforms/Passes.h"
#include "ascir/Dialect/Asc/Utils/Attributes.h"
#include "ascir/Dialect/Asc/Utils/Utils.h"
#include "ascir/Dialect/EmitAsc/IR/EmitAsc.h"
#include "ascir/Dialect/Utils/ConstantOpBuilder.h"
#include "ascir/Dialect/Utils/Utils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Dominance.h"
#include "llvm/ADT/TypeSwitch.h"

namespace mlir {
namespace ascendc {
#define GEN_PASS_DEF_FUSEVFBLOCK
#include "ascir/Dialect/Asc/Transforms/Passes.h.inc"
} // namespace ascendc
} // namespace mlir

using namespace mlir;

namespace {

using OpGroup = SmallVector<Operation*>;

ValueVector deduplicate(ArrayRef<Value> values)
{
    ValueVector result;
    ValueSet unique(values.begin(), values.end());
    for (auto value : values) {
        auto it = unique.find(value);
        if (it == unique.end())
            continue;
        result.push_back(value);
        unique.erase(it);
    }
    return result;
}

bool isFusible(Operation* op)
{
    if (auto barrier = dyn_cast<ascendc::PipeBarrierOp>(op)) {
        auto pipe = barrier.getPipe();
        return (pipe == ascendc::Pipe::PIPE_V) || (pipe == ascendc::Pipe::PIPE_ALL);
    }
    return isa<
        // Vector binary operations (L2)
        ascendc::AddL2Op, ascendc::AndL2Op, ascendc::DivL2Op, ascendc::FusedAbsSubL2Op, ascendc::FusedExpSubL2Op,
        ascendc::SubL2Op, ascendc::MaxL2Op, ascendc::MinL2Op, ascendc::MulL2Op, ascendc::MulAddDstL2Op, ascendc::OrL2Op,
        ascendc::PreluL2Op,
        // Vector unary operations (L2)
        ascendc::AbsL2Op, ascendc::ExpL2Op, ascendc::LnL2Op, ascendc::NegL2Op, ascendc::NotL2Op, ascendc::ReluL2Op,
        ascendc::SqrtL2Op>(op);
}

Value getCalCount(Operation* op)
{
    if (auto binaryOp = dyn_cast<ascendc::BinaryL2Op>(op)) {
        return binaryOp.getCalCount();
    }
    if (auto unaryOp = dyn_cast<ascendc::UnaryL2Op>(op)) {
        return unaryOp.getCalCount();
    }
    return Value{};
}

Type getType(Operation* op)
{
    if (auto binaryOp = dyn_cast<ascendc::BinaryL2Op>(op)) {
        if (auto tensor = dyn_cast<ascendc::LocalTensorType>(binaryOp.getDst().getType())) {
            return tensor.getElementType();
        }
    }
    if (auto unaryOp = dyn_cast<ascendc::UnaryL2Op>(op)) {
        if (auto tensor = dyn_cast<ascendc::LocalTensorType>(unaryOp.getDst().getType())) {
            return tensor.getElementType();
        }
    }
    llvm_unreachable("was not expected this type");
}

bool isSameGroup(Operation* firstOp, Operation* secondOp)
{
    Value val1 = getCalCount(firstOp);
    Value val2 = getCalCount(secondOp);
    if (val1 && val2) {
        std::optional<int64_t> number1 = getConstantIntValue(val1);
        std::optional<int64_t> number2 = getConstantIntValue(val2);
        if (number1.has_value() && number2.has_value()) {
            if (number1.value() != number2.value()) {
                return false;
            }
        } else if (val1 != val2) {
            return false;
        }
    }
    return getType(firstOp) == getType(secondOp);
}

void findGroupsImpl(Region& region, std::vector<OpGroup>& groups)
{
    OpGroup currentGroup;
    for (auto& op : region.getOps()) {
        for (auto& nestedRegion : op.getRegions()) {
            findGroupsImpl(nestedRegion, groups);
        }
        if (isFusible(&op)) {
            if (isa<ascendc::PipeBarrierOp>(op)) {
                // First operation in group must not be PipeBarrierOp
                if (!currentGroup.empty())
                    currentGroup.push_back(&op);
            } else {
                if (!currentGroup.empty() && !isSameGroup(currentGroup.front(), &op)) {
                    groups.push_back(currentGroup);
                    currentGroup.clear();
                }
                currentGroup.push_back(&op);
            }
        } else if (!currentGroup.empty()) {
            groups.push_back(currentGroup);
            currentGroup.clear();
        }
    }
    if (!currentGroup.empty()) {
        groups.push_back(currentGroup);
    }
}

// Find binary_l2, unary_l2 operations that may be executed together
std::vector<OpGroup> findOperationGroups(Region& region)
{
    // 1. The same calCount
    // 2. Between ops absent other operations
    // 3. Contains more than 1 operation
    std::vector<OpGroup> groups;
    findGroupsImpl(region, groups);

    // Erase PipeBarrierOp from the end of each group.
    // If group is empty then erase it.
    auto it = groups.begin();
    while (it != groups.end()) {
        auto& group = *it;
        while (!group.empty() && isa<ascendc::PipeBarrierOp>(group.back())) {
            group.pop_back();
        }
        if (group.size() < 2)
            it = groups.erase(it);
        else
            ++it;
    }
    return groups;
}

// Find local tensors that need be copy in RegTensor
ValueVector getInputLocalTensors(const OpGroup& group)
{
    // If tensor is input but before it is output then don't insert her
    ValueMap<bool> isInputLocalTensor;
    auto firstInsert = [&isInputLocalTensor](const Value& key, bool value) {
        if (!isInputLocalTensor.count(key)) {
            isInputLocalTensor.insert({key, value});
        }
    };
    for (auto* op : group) {
        if (auto binaryOp = dyn_cast<ascendc::BinaryL2Op>(op)) {
            firstInsert(binaryOp.getSrc0(), true);
            firstInsert(binaryOp.getSrc1(), true);
            firstInsert(binaryOp.getDst(), false);
        } else if (auto unaryOp = dyn_cast<ascendc::UnaryL2Op>(op)) {
            firstInsert(unaryOp.getSrc(), true);
            firstInsert(unaryOp.getDst(), false);
        }
    }
    ValueVector inputLocalTensors;
    for (const auto& [tensor, isInput] : isInputLocalTensor) {
        if (isInput)
            inputLocalTensors.push_back(tensor);
    }
    return deduplicate(inputLocalTensors);
}

// Find local tensors that need be copy out from RegTensor
ValueVector getOutputLocalTensors(const OpGroup& group)
{
    ValueVector outputLocalTensors;
    for (auto* op : group) {
        if (auto binaryOp = dyn_cast<ascendc::BinaryL2Op>(op)) {
            outputLocalTensors.push_back(binaryOp.getDst());
        } else if (auto unaryOp = dyn_cast<ascendc::UnaryL2Op>(op)) {
            outputLocalTensors.push_back(unaryOp.getDst());
        }
    }
    return deduplicate(outputLocalTensors);
}

// Find local tensors that are used as dst or src
ValueVector getUsedLocalTensors(const OpGroup& group)
{
    ValueVector usedLocalTensors;
    for (auto* op : group) {
        if (auto binaryOp = dyn_cast<ascendc::BinaryL2Op>(op)) {
            usedLocalTensors.push_back(binaryOp.getDst());
            usedLocalTensors.push_back(binaryOp.getSrc0());
            usedLocalTensors.push_back(binaryOp.getSrc1());
        } else if (auto unaryOp = dyn_cast<ascendc::UnaryL2Op>(op)) {
            usedLocalTensors.push_back(unaryOp.getDst());
            usedLocalTensors.push_back(unaryOp.getSrc());
        }
    }
    return deduplicate(usedLocalTensors);
}

Operation* getVecBlockLocation(DominanceInfo& di, const OpGroup& group)
{
    Operation* firstUser = *std::min_element(
        group.begin(), group.end(), [&](Operation* lhs, Operation* rhs) { return ascendc::opPrecedes(lhs, rhs, di); });
    return firstUser;
}

ascendc::UpdateMaskOp createUpdateMask(OpBuilder& builder, Value count, Type type)
{
    return builder.create<ascendc::UpdateMaskOp>(
        builder.getUnknownLoc(), builder.getType<ascendc::MaskRegType>(), count, type);
}

ArrayRef<int64_t> getShape(Value value)
{
    if (auto tensor = dyn_cast<ascendc::LocalTensorType>(value.getType())) {
        return tensor.getShape();
    }
    return {};
}

ValueMap<Value> createRegTensors(OpBuilder& builder, ArrayRef<Value> usedTensors, Type groupType)
{
    ValueMap<Value> regTensors;
    for (auto value : usedTensors) {
        auto regTensorOp = builder.create<ascendc::RegTensorOp>(
            builder.getUnknownLoc(), ascendc::RegTensorType::get(builder.getContext(), groupType));
        regTensors[value] = regTensorOp.getResult();
    }
    return regTensors;
}

class TranslatorFactory {
    ValueMap<Value>& regTensors;
    Value mask;

public:
    TranslatorFactory(ValueMap<Value>& regTensors, Value mask) : regTensors(regTensors), mask(mask) {}

    template <typename T>
    auto binary()
    {
        return [&](ascendc::BinaryL2Op binaryOp) {
            OpBuilder builder(binaryOp);
            builder.create<T>(
                builder.getUnknownLoc(), regTensors[binaryOp.getDst()], regTensors[binaryOp.getSrc0()],
                regTensors[binaryOp.getSrc1()], mask);
            binaryOp.erase();
        };
    }

    template <typename T>
    auto unary()
    {
        return [&](ascendc::UnaryL2Op unaryOp) {
            OpBuilder builder(unaryOp);
            builder.create<T>(
                builder.getUnknownLoc(), regTensors[unaryOp.getDst()], regTensors[unaryOp.getSrc()], mask);
            unaryOp.erase();
        };
    }
};

void processGroup(OpBuilder& builder, OpGroup& group, DominanceInfo& di)
{
    // Collect group info
    auto* loc = getVecBlockLocation(di, group);
    auto usedTensors = getUsedLocalTensors(group);
    Value calCount = getCalCount(group.front());
    Type groupType = getType(group.front());

    builder.setInsertionPoint(loc);
    auto vecScope = builder.create<emitasc::VecScopeOp>(builder.getUnknownLoc());
    auto* blockVecScope = &vecScope.getRegion().emplaceBlock();

    // Create loop
    ascir::ConstantOpBuilder consts(builder);
    builder.setInsertionPointToStart(blockVecScope);
    auto calCountVar = builder.create<emitasc::VariableOp>(
        builder.getUnknownLoc(), MemRefType::get(1, builder.getIntegerType(32U, false)), calCount);
    auto getVecLenIndex = builder.create<ascendc::GetVecLenOp>(builder.getUnknownLoc(), builder.getIndexType());
    auto sizeIndex = consts.index(groupType.getIntOrFloatBitWidth() / 8);
    auto div = builder.create<arith::DivSIOp>(
        builder.getUnknownLoc(), builder.getIndexType(), getVecLenIndex.getResult(), sizeIndex);
    auto oneRepeatSizeIndex = div.getResult();
    auto calCountIndex = builder.create<arith::IndexCastOp>(builder.getUnknownLoc(), builder.getIndexType(), calCount);
    auto repeatTimes = builder.create<arith::CeilDivSIOp>(
        builder.getUnknownLoc(), builder.getIndexType(), calCountIndex, oneRepeatSizeIndex);
    auto zeroIndex = consts.index(0);
    auto oneIndex = consts.index(1);
    auto loop = builder.create<scf::ForOp>(builder.getUnknownLoc(), zeroIndex, repeatTimes.getResult(), oneIndex);
    loop->setAttr(ascendc::attr::vecScopeLoop, builder.getUnitAttr());

    // Materialize group operations in loop body
    builder.setInsertionPointToStart(loop.getBody());
    for (auto*& op : group) {
        auto* tmp = builder.clone(*op);
        op->erase();
        op = tmp;
    }

    ValueMap<Value> addrTensors;
    builder.setInsertionPoint(vecScope);
    for (auto value : usedTensors) {
        auto type = MemRefType::get(getShape(value), groupType, {}, static_cast<int>(ascendc::AddressSpace::ubuf));
        auto getPhyAddrOp = builder.create<ascendc::LocalTensorGetPhyAddrV2Op>(builder.getUnknownLoc(), type, value);
        addrTensors[value] = getPhyAddrOp.getResult();
    }

    builder.setInsertionPointToStart(blockVecScope);
    ValueMap<Value> regTensors = createRegTensors(builder, usedTensors, groupType);

    builder.setInsertionPointToStart(loop.getBody());
    auto updateMask = createUpdateMask(builder, calCountVar.getResult(), groupType);
    auto mask = updateMask.getResult();
    auto mu = builder.create<arith::MulIOp>(builder.getUnknownLoc(), loop.getInductionVar(), oneRepeatSizeIndex);

    // DataCopyLoad
    auto inputTensors = getInputLocalTensors(group);
    for (auto& value : inputTensors) {
        auto resultType =
            MemRefType::get(getShape(value), groupType, {}, static_cast<int>(ascendc::AddressSpace::ubuf));
        auto srcAddr = builder.create<emitasc::PtrOffsetOp>(
            builder.getUnknownLoc(), resultType, addrTensors[value], nullptr, mu.getResult());
        builder.create<ascendc::DataCopyLoadOp>(builder.getUnknownLoc(), regTensors[value], srcAddr);
    }

    TranslatorFactory factory(regTensors, mask);
    for (auto* op : group) {
        llvm::TypeSwitch<Operation*>(op)
            // BinaryOp
            .Case<ascendc::AddL2Op>(factory.binary<ascendc::AddMicroOp>())
            .Case<ascendc::AndL2Op>(factory.binary<ascendc::AndMicroOp>())
            .Case<ascendc::DivL2Op>(factory.binary<ascendc::DivMicroOp>())
            .Case<ascendc::FusedAbsSubL2Op>(factory.binary<ascendc::FusedAbsSubMicroOp>())
            .Case<ascendc::FusedExpSubL2Op>(factory.binary<ascendc::FusedExpSubMicroOp>())
            .Case<ascendc::SubL2Op>(factory.binary<ascendc::SubMicroOp>())
            .Case<ascendc::MaxL2Op>(factory.binary<ascendc::MaxMicroOp>())
            .Case<ascendc::MinL2Op>(factory.binary<ascendc::MinMicroOp>())
            .Case<ascendc::MulL2Op>(factory.binary<ascendc::MulMicroOp>())
            .Case<ascendc::MulAddDstL2Op>(factory.binary<ascendc::MulAddDstMicroOp>())
            .Case<ascendc::OrL2Op>(factory.binary<ascendc::OrMicroOp>())
            .Case<ascendc::PreluL2Op>(factory.binary<ascendc::PreluMicroOp>())
            // UnaryOp
            .Case<ascendc::AbsL2Op>(factory.unary<ascendc::AbsMicroOp>())
            .Case<ascendc::ExpL2Op>(factory.unary<ascendc::ExpMicroOp>())
            .Case<ascendc::LnL2Op>(factory.unary<ascendc::LnMicroOp>())
            .Case<ascendc::NegL2Op>(factory.unary<ascendc::NegMicroOp>())
            .Case<ascendc::NotL2Op>(factory.unary<ascendc::NotMicroOp>())
            .Case<ascendc::ReluL2Op>(factory.unary<ascendc::ReluMicroOp>())
            .Case<ascendc::SqrtL2Op>(factory.unary<ascendc::SqrtMicroOp>())
            // PipeBarrierOp
            .Case<ascendc::PipeBarrierOp>([](auto barrier) { barrier->erase(); });
    }

    // DataCopyStore
    builder.setInsertionPoint(loop.getBody()->getTerminator());
    auto outputTensors = getOutputLocalTensors(group);
    for (auto& value : outputTensors) {
        auto resultType =
            MemRefType::get(getShape(value), groupType, {}, static_cast<int>(ascendc::AddressSpace::ubuf));
        auto dstAddr = builder.create<emitasc::PtrOffsetOp>(
            builder.getUnknownLoc(), resultType, addrTensors[value], nullptr, mu.getResult());
        builder.create<ascendc::DataCopyStoreOp>(builder.getUnknownLoc(), dstAddr, regTensors[value], mask);
    }
    builder.setInsertionPointToEnd(blockVecScope);
    builder.create<emitasc::YieldOp>(builder.getUnknownLoc());
}

void wrapInVFGroupOp(OpBuilder& builder, OpGroup& group)
{
    if (group.empty())
        return;
    builder.setInsertionPoint(group.back());
    ValueRange inputs = getInputLocalTensors(group);
    ValueRange outputs = getOutputLocalTensors(group);
    auto fusedOp = builder.create<emitasc::VFGroupOp>(builder.getUnknownLoc(), outputs, inputs);
    auto& block = fusedOp.getRegion().emplaceBlock();

    builder.setInsertionPointToEnd(&block);
    for (auto* op : group) {
        builder.clone(*op);
        op->erase();
    }
    builder.create<emitasc::YieldOp>(builder.getUnknownLoc());
}

struct FuseVFBlockPass : public ascendc::impl::FuseVFBlockBase<FuseVFBlockPass> {
    void runOnOperation() override
    {
        func::FuncOp funcOp = getOperation();
        if (funcOp.isDeclaration())
            return;

        OpBuilder builder(funcOp);
        for (auto& group : findOperationGroups(funcOp.getRegion())) {
            wrapInVFGroupOp(builder, group);
        }

        // TODO: split on 2 passes

        DominanceInfo di;
        funcOp.walk([&](emitasc::VFGroupOp fusedOps) {
            OpGroup group;
            for (auto& op : fusedOps.getBody()->without_terminator()) {
                group.emplace_back(&op);
            }

            fusedOps.getBody()->getTerminator()->erase();

            processGroup(builder, group, di);

            builder.setInsertionPointToEnd(fusedOps.getBody());
            builder.create<emitasc::YieldOp>(builder.getUnknownLoc());
        });
    }
};

} // namespace

namespace mlir {

namespace ascendc {
std::unique_ptr<Pass> createFuseVFBlockPass() { return std::make_unique<FuseVFBlockPass>(); }
} // namespace ascendc
} // namespace mlir
