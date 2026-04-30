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

constexpr const char* orderAttr = "order";

struct OpGroup {
    SmallVector<Operation*> ops;
    Value calCount;
    Type groupType;
    OpGroup(ArrayRef<Operation*> ops, Value calCount, Type groupType)
        : ops(ops), calCount(calCount), groupType(groupType)
    {}
};

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
    if (auto duplicate = dyn_cast<ascendc::DuplicateL2Op>(op)) {
        return isa<ascendc::LocalTensorType>(duplicate.getScalar().getType());
    }
    return isa<
        // Reduce operation (L2)
        ascendc::ReduceMaxL2Op, ascendc::ReduceMinL2Op, ascendc::ReduceSumL2Op,
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
    return llvm::TypeSwitch<Operation*, Value>(op)
        .Case<ascendc::BinaryL2Op, ascendc::UnaryL2Op, ascendc::DuplicateL2Op>([](auto op) { return op.getCalCount(); })
        .Case<ascendc::ReduceMaxL2Op, ascendc::ReduceMinL2Op, ascendc::ReduceSumL2Op>(
            [](auto op) { return op.getCount(); })
        .Default([](Operation* op) { return Value{}; });
}

Type getType(Operation* op)
{
    return llvm::TypeSwitch<Operation*, Type>(op)
        .Case<
            ascendc::BinaryL2Op, ascendc::UnaryL2Op, ascendc::ReduceMaxL2Op, ascendc::ReduceMinL2Op,
            ascendc::ReduceSumL2Op, ascendc::DuplicateL2Op>([](auto op) {
            assert(isa<ascendc::LocalTensorType>(op.getDst().getType()));
            return getElementTypeOrSelf(op.getDst());
        })
        .Default([](Operation*) {
            llvm_unreachable("was not expected this type");
            return Type{};
        });
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
    auto append = [&groups](SmallVectorImpl<Operation*>& ops) {
        if (!ops.empty()) {
            groups.emplace_back(ops, getCalCount(ops.front()), getType(ops.front()));
            ops.clear();
        }
    };

    SmallVector<Operation*> ops;
    for (auto& op : region.getOps()) {
        for (auto& nestedRegion : op.getRegions()) {
            findGroupsImpl(nestedRegion, groups);
        }
        if (isFusible(&op)) {
            if (!ops.empty() && !isSameGroup(ops.front(), &op)) {
                append(ops);
            }
            ops.emplace_back(&op);
        } else {
            append(ops);
        }
    }
    append(ops);
}

// Find binary_l2, unary_l2 operations that may be executed together
std::vector<OpGroup> findOperationGroups(Region& region)
{
    // 1. The same calCount
    // 2. Between ops absent other operations
    // 3. Contains more than 1 operation
    std::vector<OpGroup> groups;
    findGroupsImpl(region, groups);

    std::vector<OpGroup> filtered_groups;
    llvm::copy_if(
        groups, std::back_inserter(filtered_groups), [](const OpGroup& group) { return group.ops.size() >= 2; });
    return filtered_groups;
}

// Find local tensors that need be copy in RegTensor
ValueVector getInputLocalTensors(ArrayRef<Operation*> group)
{
    // If tensor is input but before it is output then don't insert her
    ValueMap<bool> isInputLocalTensor;
    for (auto* op : group) {
        if (auto opWithSrc = dyn_cast<ascendc::OpWithSrc>(op)) {
            for (auto& src : opWithSrc.getSrcTensors()) {
                isInputLocalTensor.try_emplace(src, true);
            }
        } else if (auto duplicateOp = dyn_cast<ascendc::DuplicateL2Op>(op)) {
            isInputLocalTensor.try_emplace(duplicateOp.getScalar(), true); // expected local tensor
        }
        if (auto opWithDst = dyn_cast<ascendc::OpWithDst>(op)) {
            isInputLocalTensor.try_emplace(opWithDst.getDst(), false);
        }
    }
    ValueVector inputLocalTensors;
    for (const auto& [tensor, isInput] : isInputLocalTensor) {
        if (isInput)
            inputLocalTensors.emplace_back(tensor);
    }
    return deduplicate(inputLocalTensors);
}

// Find local tensors that need be copy out from RegTensor
ValueVector getOutputLocalTensors(ArrayRef<Operation*> group)
{
    ValueVector outputLocalTensors;
    for (auto* op : group) {
        llvm::TypeSwitch<Operation*>(op)
            .Case<
                ascendc::BinaryL2Op, ascendc::UnaryL2Op, ascendc::ReduceMaxL2Op, ascendc::ReduceMinL2Op,
                ascendc::ReduceSumL2Op, ascendc::DuplicateL2Op>(
                [&](auto op) { outputLocalTensors.push_back(op.getDst()); });
    }
    return deduplicate(outputLocalTensors);
}

// Find local tensors that are used as dst or src
ValueVector getUsedLocalTensors(emitasc::VecScopeOp vecScope)
{
    ValueVector usedLocalTensors;
    vecScope.walk([&](Operation* op) {
        if (auto load = dyn_cast<emitasc::LoadMicroOp>(op)) {
            usedLocalTensors.emplace_back(load.getSrcTensor());
        } else if (auto store = dyn_cast<emitasc::StoreMicroOp>(op)) {
            usedLocalTensors.emplace_back(store.getDstTensor());
        }
    });
    return deduplicate(usedLocalTensors);
}

enum class Order { ConstantOp, VariableOp, UpdateMaskOp, RegTensorOp, CreateMaskOp, DuplicateOp };

void setOrder(Operation* op, Order order)
{
    Builder builder(op->getContext());
    op->setAttr(orderAttr, builder.getI32IntegerAttr(static_cast<int>(order)));
}

std::optional<int64_t> getOrder(Operation* op)
{
    if (auto attr = op->getAttr(orderAttr)) {
        if (auto order = dyn_cast<IntegerAttr>(attr)) {
            return order.getValue().getSExtValue();
        }
    }
    return std::nullopt;
}

ascendc::UpdateMaskOp createUpdateMask(scf::ForOp loop, Value calCount, Type type)
{
    OpBuilder builder(loop);
    auto calCountVar = builder.create<emitasc::VariableOp>(
        builder.getUnknownLoc(), MemRefType::get(1, builder.getIntegerType(32U, false)), calCount);
    setOrder(calCountVar, Order::VariableOp);
    builder.setInsertionPointToStart(loop.getBody());
    auto updateMask = builder.create<ascendc::UpdateMaskOp>(
        builder.getUnknownLoc(), builder.getType<ascendc::MaskRegType>(), calCountVar.getResult(), type);
    setOrder(updateMask, Order::UpdateMaskOp);
    return updateMask;
}

ascendc::CreateMaskOp createMask(OpBuilder& builder, Type type, ascendc::MaskPattern pattern)
{
    auto op = builder.create<ascendc::CreateMaskOp>(
        builder.getUnknownLoc(), builder.getType<ascendc::MaskRegType>(), type, pattern);
    setOrder(op, Order::CreateMaskOp);
    return op;
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

ascendc::RegTensorOp createRegTensor(OpBuilder builder, Type elemType)
{
    auto regTensorOp = builder.create<ascendc::RegTensorOp>(
        builder.getUnknownLoc(), ascendc::RegTensorType::get(builder.getContext(), elemType));
    setOrder(regTensorOp, Order::RegTensorOp);
    return regTensorOp;
}

emitasc::LoadMicroOp createLoad(OpBuilder builder, Value dstReg, Value srcTensor)
{
    return builder.create<emitasc::LoadMicroOp>(builder.getUnknownLoc(), dstReg, srcTensor);
}

emitasc::StoreMicroOp createStore(OpBuilder builder, Value dstTensor, Value srcReg, Value mask)
{
    return builder.create<emitasc::StoreMicroOp>(builder.getUnknownLoc(), dstTensor, srcReg, mask);
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
    Value calCount;
    Type elemType;

public:
    TranslatorFactory(Value calCount, Type elemType) : calCount(calCount), elemType(elemType) {}
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
            createLoad(builder, src0Reg, binaryOp.getSrc0());
            createLoad(builder, src1Reg, binaryOp.getSrc1());
            builder.create<T>(builder.getUnknownLoc(), dstReg, src0Reg, src1Reg, updateMask.getResult());
            createStore(builder, binaryOp.getDst(), dstReg, updateMask.getResult());

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
            createLoad(builder, srcReg, unaryOp.getSrc());
            builder.create<T>(builder.getUnknownLoc(), dstReg, srcReg, updateMask.getResult());
            createStore(builder, unaryOp.getDst(), dstReg, updateMask.getResult());

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
            setOrder(neutral.getDefiningOp(), Order::ConstantOp);
            auto duplicateOp =
                builder.create<ascendc::DuplicateScalarMicroOp>(builder.getUnknownLoc(), accReg, neutral);
            setOrder(duplicateOp, Order::DuplicateOp);

            auto loop = createLoop(builder);
            builder.setInsertionPoint(loop);
            builder.setInsertionPointToStart(loop.getBody());
            auto updateMask = createUpdateMask(loop, calCount, elemType);
            auto load = createLoad(builder, srcReg, reduceOp.getSrc());
            builder.create<AccumulateMicroOp>(builder.getUnknownLoc(), accReg, accReg, srcReg, updateMask);

            builder.setInsertionPointAfter(loop);
            auto maskAll = createMask(builder, elemType, ascendc::MaskPattern::ALL);
            builder.create<ReduceMicroOp>(builder.getUnknownLoc(), dstReg, accReg, maskAll);
            auto maskOne = createMask(builder, elemType, ascendc::MaskPattern::VL1);
            createStore(builder, reduceOp.getDst(), dstReg, maskOne);
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
            createLoad(builder, srcReg, duplicateOp.getScalar());
            auto maskAll = createMask(builder, elemType, ascendc::MaskPattern::ALL);
            builder.create<ascendc::DuplicateMicroOp>(builder.getUnknownLoc(), tmpReg, srcReg, maskAll);

            auto loop = createLoop(builder);
            auto updateMask = createUpdateMask(loop, calCount, elemType);
            builder.setInsertionPoint(loop.getBody()->getTerminator());
            auto store = createStore(builder, duplicateOp.getDst(), tmpReg, updateMask);
            duplicateOp.erase();
        };
    }

private:
    scf::ForOp createLoop(OpBuilder builder)
    {
        ascir::ConstantOpBuilder consts(builder);
        auto zero = consts.index(0);
        setOrder(zero.getDefiningOp(), Order::ConstantOp);
        auto one = consts.index(1);
        setOrder(one.getDefiningOp(), Order::ConstantOp);
        auto loop = builder.create<scf::ForOp>(builder.getUnknownLoc(), zero, zero, one);
        loop->setAttr(ascendc::attr::vecScopeLoop, builder.getUnitAttr());
        return loop;
    }
};

bool belong(Block* block, Block* parentBlock, DominanceInfo& di)
{
    assert(block && parentBlock);
    auto* commonBlock = di.findNearestCommonDominator(block, parentBlock);
    return commonBlock == parentBlock;
}

void eraseUnusedOutputs(emitasc::VFGroupOp groupOp, MutableOperandRange outputs)
{
    SmallVector<unsigned int> deleted;
    DominanceInfo di;
    for (auto& opnd : outputs) {
        auto users = opnd.get().getUsers();
        Block* body = groupOp.getBody();
        bool usesInsideBlock = std::all_of(users.begin(), users.end(), [&](Operation* use) {
            return belong(use->getBlock(), body, di) || use == groupOp || ascendc::opPrecedes(use, groupOp, di);
        });
        if (usesInsideBlock) {
            // insert indices in ascending order
            deleted.emplace_back(opnd.getOperandNumber());
        }
    }
    while (!deleted.empty()) {
        // delete indices in descending order
        outputs.erase(deleted.back());
        deleted.pop_back();
    }
}

void wrapInVFGroupOp(OpBuilder& builder, OpGroup& group)
{
    assert(!group.ops.empty());
    auto& ops = group.ops;
    builder.setInsertionPoint(ops.back());
    ValueVector inputs = getInputLocalTensors(ops);
    ValueVector outputs = getOutputLocalTensors(ops);

    auto fusedOp = builder.create<emitasc::VFGroupOp>(builder.getUnknownLoc(), outputs, inputs, group.calCount);
    auto& block = fusedOp.getRegion().emplaceBlock();

    builder.setInsertionPointToEnd(&block);
    for (auto* op : ops) {
        builder.clone(*op);
        op->erase();
    }
    builder.create<emitasc::YieldOp>(builder.getUnknownLoc());
}

void lowerToMicro(emitasc::VecScopeOp vecScopeOp, Value calCount, Type elemType)
{
    OpBuilder builder(vecScopeOp.getContext());
    builder.setInsertionPointToStart(vecScopeOp.getBody());
    TranslatorFactory factory(calCount, elemType);

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
            .Case<ascendc::SqrtL2Op>(factory.unary<ascendc::SqrtMicroOp>())
            .Default([](Operation* op) {
                if (isa<emitasc::YieldOp, emitasc::VecScopeOp>(op)) {
                    return;
                }
                op->dump();
                llvm_unreachable("unsupported case");
            });
    });
}

void hoistOperations(emitasc::VecScopeOp vecScopeOp)
{
    vecScopeOp.walk([](Block* block) {
        SmallVector<std::pair<int, Operation*>> ops;
        for (auto& op : block->getOperations()) {
            auto order = getOrder(&op);
            if (order.has_value()) {
                ops.push_back({order.value(), &op});
            }
        }

        llvm::sort(ops, std::greater<>());
        for (auto [order, op] : ops) {
            op->moveBefore(block, block->getOperations().begin());
        }
    });
}

void merge(scf::ForOp firstLoop, scf::ForOp secondLoop)
{
    OpBuilder builder(firstLoop.getContext());
    SmallVector<Operation*> opList;
    for (auto& op : secondLoop.getBody()->without_terminator()) {
        opList.emplace_back(&op);
    }
    for (auto& op : opList) {
        op->moveBefore(firstLoop.getBody()->getTerminator());
    }
    secondLoop.erase();
}

void fuseLoops(emitasc::VecScopeOp vecScopeOp)
{
    auto& ops = vecScopeOp.getBody()->getOperations();
    if (ops.size() < 2)
        return;
    auto it = ops.begin();
    while (std::next(it) != ops.end()) {
        bool flag = false;
        if (auto curLoop = dyn_cast<scf::ForOp>(*it)) {
            auto nextIt = std::next(it);
            if (auto nextLoop = dyn_cast<scf::ForOp>(*nextIt)) {
                merge(curLoop, nextLoop);
                flag = true;
            }
        }
        if (!flag)
            ++it;
    }
}

void loadStoreElimination(emitasc::VFGroupOp groupOp)
{
    ValueSet inputTensors;
    for (auto tensor : groupOp.getSrcList()) {
        inputTensors.insert(tensor);
    }

    ValueSet outputTensors;
    for (auto tensor : groupOp.getDstList()) {
        outputTensors.insert(tensor);
    }

    SmallVector<SmallVector<Operation*>> groups;
    auto append_group = [&groups](SmallVector<Operation*>& group) {
        if (!group.empty()) {
            groups.emplace_back(group);
            group.clear();
        }
    };

    groupOp.walk([&](Block* block) {
        SmallVector<Operation*> group;
        for (auto& op : block->getOperations()) {
            if (isa<emitasc::LoadMicroOp, emitasc::StoreMicroOp>(op)) {
                group.emplace_back(&op);
            }
        }
        append_group(group);
    });

    std::unordered_set<Operation*> needDelete;

    for (auto& group : groups) {
        ValueMap<SmallVector<Operation*>> tensorToMemOp;
        // Erase unnecessary emitasc.load:
        //   emitasc.store %addr, %reg0
        //   ...
        //   emitasc.load %reg1, %addr
        //   op(%reg1)
        // Replace to:
        //   emitasc.store %addr, %reg0
        //   ...
        //   op(%reg0)
        for (auto* op : group) {
            if (auto load = dyn_cast<emitasc::LoadMicroOp>(op)) {
                tensorToMemOp[load.getSrcTensor()].emplace_back(load);
            } else if (auto store = dyn_cast<emitasc::StoreMicroOp>(op)) {
                tensorToMemOp[store.getDstTensor()].emplace_back(store);
            }
        }

        for (auto [tensor, group] : tensorToMemOp) {
            Value srcReg{};
            for (auto It = group.begin(); It != group.end(); ++It) {
                if (auto storeOp = dyn_cast<emitasc::StoreMicroOp>(*It)) {
                    srcReg = storeOp.getSrcReg();
                } else if (auto loadOp = dyn_cast<emitasc::LoadMicroOp>(*It)) {
                    Value dstReg = loadOp.getDstReg();
                    if (srcReg) {
                        dstReg.replaceAllUsesWith(srcReg);
                        needDelete.insert(loadOp);
                    }
                }
            }
        }
    }

    for (auto op : needDelete) {
        op->erase();
    }

    needDelete.clear();

    ValueMap<SmallVector<Operation*>> tensorToGlobalMemOp;

    groupOp.walk([&](Operation* op) {
        if (auto load = dyn_cast<emitasc::LoadMicroOp>(op)) {
            tensorToGlobalMemOp[load.getSrcTensor()].emplace_back(load);
        }
        if (auto store = dyn_cast<emitasc::StoreMicroOp>(op)) {
            tensorToGlobalMemOp[store.getDstTensor()].emplace_back(store);
        }
    });

    for (auto [tensor, group] : tensorToGlobalMemOp) {
        bool rewrite = false;
        if (auto store = dyn_cast<emitasc::StoreMicroOp>(group.back())) {
            // we can delete last storeOp if tensor is not output
            if (!outputTensors.count(store.getDstTensor())) {
                needDelete.insert(store);
            }
        }
        std::reverse(group.begin(), group.end());

        // don't rewrite the same memory
        for (auto It = group.begin(); It != group.end(); ++It) {
            if (auto store = dyn_cast<emitasc::StoreMicroOp>(*It)) {
                if (rewrite) {
                    needDelete.insert(store);
                }
                rewrite = true;
            } else {
                rewrite = false;
            }
        }
    }

    for (auto op : needDelete) {
        op->erase();
    }
}

void mergeSameOperations(emitasc::VFGroupOp groupOp)
{
    std::unordered_set<Operation*> needDelete;

    // Merge load from one memory into one regTensor
    groupOp.walk([&](Block* block) {
        ValueMap<Value> loadedTensors;
        for (auto& op : block->getOperations()) {
            if (auto load = dyn_cast<emitasc::LoadMicroOp>(op)) {
                if (loadedTensors.count(load.getSrcTensor())) {
                    load.getDstReg().replaceAllUsesWith(
                        loadedTensors[load.getSrcTensor()]); // TODO: replace only in block
                    needDelete.insert(load);
                } else {
                    loadedTensors[load.getSrcTensor()] = load.getDstReg();
                }
            }
        }
    });

    for (auto* op : needDelete) {
        op->erase();
    }
}

void eliminateCommonMask(emitasc::VFGroupOp groupOp)
{
    std::unordered_set<Operation*> needDelete;

    groupOp.walk([&](Block* block) {
        Operation* firstUpdateMask = nullptr;
        for (auto& op : block->getOperations()) {
            if (auto updateMask = dyn_cast<ascendc::UpdateMaskOp>(op)) {
                if (firstUpdateMask) {
                    updateMask.replaceAllUsesWith(firstUpdateMask->getResult(0));
                    needDelete.insert(updateMask);
                } else {
                    firstUpdateMask = &op;
                }
            }
        }
    });

    groupOp.walk([&](Block* block) {
        std::map<ascendc::MaskPattern, ascendc::CreateMaskOp> createMaskMap;
        for (auto& op : block->getOperations()) {
            if (auto createMask = dyn_cast<ascendc::CreateMaskOp>(op)) {
                auto it = createMaskMap.find(createMask.getMask());
                if (it == createMaskMap.end()) {
                    createMaskMap[createMask.getMask()] = createMask;
                } else {
                    createMask.replaceAllUsesWith(it->second.getResult());
                    needDelete.insert(createMask);
                }
            }
        }
    });

    for (auto* op : needDelete) {
        op->erase();
    }
}

ValueMap<Value> setAddress(emitasc::VecScopeOp vecScopeOp, ArrayRef<Value> usedTensors, Type groupType)
{
    ValueMap<Value> addrTensors;
    OpBuilder builder(vecScopeOp);
    for (auto value : usedTensors) {
        auto type = MemRefType::get(getShape(value), groupType, {}, static_cast<int>(ascendc::AddressSpace::ubuf));
        auto getPhyAddrOp = builder.create<ascendc::LocalTensorGetPhyAddrV2Op>(builder.getUnknownLoc(), type, value);
        addrTensors[value] = getPhyAddrOp.getResult();
    }
    return addrTensors;
}

std::pair<Value, Value> createRepeatTimes(OpBuilder builder, Value calCount, Type groupType)
{
    ascir::ConstantOpBuilder consts(builder);
    auto getVecLenIndex = builder.create<ascendc::GetVecLenOp>(builder.getUnknownLoc(), builder.getIndexType());
    auto sizeIndex = consts.index(ascendc::getTypeSize(groupType));
    auto div = builder.create<arith::DivSIOp>(
        builder.getUnknownLoc(), builder.getIndexType(), getVecLenIndex.getResult(), sizeIndex);
    auto oneRepeatSizeIndex = div.getResult();
    if (!calCount.getType().isIndex()) {
        calCount =
            builder.create<arith::IndexCastOp>(builder.getUnknownLoc(), builder.getIndexType(), calCount).getResult();
    }
    auto repeatTimes = builder.create<arith::CeilDivSIOp>(
        builder.getUnknownLoc(), builder.getIndexType(), calCount, oneRepeatSizeIndex);
    return {oneRepeatSizeIndex, repeatTimes};
}

void materialize(emitasc::VecScopeOp vecScopeOp, Value calCount, Type groupType)
{
    ValueMap<Value> addrTensors;
    // materialize getPhyAddr
    SmallVector<Value> usedTensors = getUsedLocalTensors(vecScopeOp);
    addrTensors = setAddress(vecScopeOp, usedTensors, groupType);

    OpBuilder builder(vecScopeOp.getContext());
    builder.setInsertionPointToStart(vecScopeOp.getBody());

    Value oneRepeatSizeIndex, repeatTimes;
    std::tie(oneRepeatSizeIndex, repeatTimes) = createRepeatTimes(builder, calCount, groupType);

    // materialize DataCopyLoad, DataCopyStore
    vecScopeOp->walk([&](scf::ForOp loop) {
        OpBuilder builder(loop.getContext());

        loop.setUpperBound(repeatTimes);

        builder.setInsertionPoint(loop);
        builder.setInsertionPointToStart(loop.getBody());
        auto mulOp = builder.create<arith::MulIOp>(builder.getUnknownLoc(), loop.getInductionVar(), oneRepeatSizeIndex);
        SmallVector<Operation*> ops;
        for (auto& op : loop.getBody()->getOperations()) {
            ops.emplace_back(&op);
        }
        for (auto* op : ops) {
            builder.setInsertionPoint(op);
            if (auto load = dyn_cast<emitasc::LoadMicroOp>(op)) {
                Value tensor = load.getSrcTensor();
                auto resultType =
                    MemRefType::get(getShape(tensor), groupType, {}, static_cast<int>(ascendc::AddressSpace::ubuf));
                auto srcAddr = builder.create<emitasc::PtrOffsetOp>(
                    builder.getUnknownLoc(), resultType, addrTensors[tensor], IntegerAttr{}, mulOp.getResult());
                builder.create<ascendc::DataCopyLoadOp>(builder.getUnknownLoc(), load.getDstReg(), srcAddr);
                load.erase();
            } else if (auto store = dyn_cast<emitasc::StoreMicroOp>(op)) {
                Value tensor = store.getDstTensor();
                auto resultType =
                    MemRefType::get(getShape(tensor), groupType, {}, static_cast<int>(ascendc::AddressSpace::ubuf));
                auto dstAddr = builder.create<emitasc::PtrOffsetOp>(
                    builder.getUnknownLoc(), resultType, addrTensors[tensor], IntegerAttr{}, mulOp.getResult());
                builder.create<ascendc::DataCopyStoreOp>(
                    builder.getUnknownLoc(), dstAddr, store.getSrcReg(), store.getMask());
                store.erase();
            }
        }
    });
    vecScopeOp->walk([&](Operation* op) {
        OpBuilder builder(op);
        ascir::ConstantOpBuilder consts(builder);
        if (auto load = dyn_cast<emitasc::LoadMicroOp>(op)) {
            Value tensor = load.getSrcTensor();
            auto resultType =
                MemRefType::get(getShape(tensor), groupType, {}, static_cast<int>(ascendc::AddressSpace::ubuf));
            builder.create<ascendc::DataCopyLoadOp>(builder.getUnknownLoc(), load.getDstReg(), addrTensors[tensor]);
            load.erase();
        } else if (auto store = dyn_cast<emitasc::StoreMicroOp>(op)) {
            Value tensor = store.getDstTensor();
            auto resultType =
                MemRefType::get(getShape(tensor), groupType, {}, static_cast<int>(ascendc::AddressSpace::ubuf));
            builder.create<ascendc::DataCopyStoreOp>(
                builder.getUnknownLoc(), addrTensors[tensor], store.getSrcReg(), store.getMask());
            store.erase();
        }
    });
}

emitasc::VecScopeOp wrapInVecScope(OpBuilder& builder, SmallVector<Operation*> ops)
{
    auto vecScope = builder.create<emitasc::VecScopeOp>(builder.getUnknownLoc());
    auto* blockVecScope = &vecScope.getRegion().emplaceBlock();

    builder.setInsertionPointToStart(blockVecScope);

    for (auto* op : ops) {
        builder.clone(*op);
    }
    builder.create<emitasc::YieldOp>(builder.getUnknownLoc());
    for (auto* op : ops) {
        op->erase();
    }
    return vecScope;
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
        funcOp.walk([&](emitasc::VFGroupOp fusedOp) {
            eraseUnusedOutputs(fusedOp, fusedOp.getDstListMutable());

            Block* block = fusedOp.getBody();
            SmallVector<Operation*> ops;
            for (auto& op : block->without_terminator()) {
                ops.emplace_back(&op);
            }

            OpBuilder builder(fusedOp.getContext());
            builder.setInsertionPointToStart(block);

            auto vecScope = wrapInVecScope(builder, ops);
            lowerToMicro(vecScope, fusedOp.getCalCount(), fusedOp.getGroupType());
            hoistOperations(vecScope);
            fuseLoops(vecScope);
            loadStoreElimination(fusedOp);
            mergeSameOperations(fusedOp); // use for x + y + x sample with double load the same RegTensor
            eliminateCommonMask(fusedOp);
            // TODO: add reuse RegTensors
            materialize(vecScope, fusedOp.getCalCount(), fusedOp.getGroupType());

            fusedOp.walk([](Operation* op) { op->removeAttr(orderAttr); });
        });
    }
};

} // namespace

namespace mlir {

namespace ascendc {
std::unique_ptr<Pass> createFuseVFBlockPass() { return std::make_unique<FuseVFBlockPass>(); }
} // namespace ascendc
} // namespace mlir
