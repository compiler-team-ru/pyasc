/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "asctile/Dialect/AscTile/IR/AscTile.h"
#include "asctile/Dialect/AscTile/Transforms/Passes.h"
#include "asctile/Dialect/AscTile/Utils/Attributes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir {
namespace asctile {
#define GEN_PASS_DEF_PREPARECVSTRATEGY
#include "asctile/Dialect/AscTile/Transforms/Passes.h.inc"
} // namespace asctile
} // namespace mlir

using namespace mlir;
using namespace mlir::asctile;

namespace {

class CVStrategyModel {
    enum struct CopyVerdict { Accept, Fail, Skip };

    enum struct UserKind : bool { Split, Unsplit };

    using SplitShape = SmallVector<int64_t, 2>;

    struct SplitState {
        Value value;
        SplitShape shape;
        UserKind kind = UserKind::Split;
    };

    CVStrategyOp root;
    SplitMode split;
    unsigned axis;
    SmallVector<SplitState, 4> worklist;
    DenseMap<Operation*, SplitShape> annotations;
    DenseMap<Operation*, SmallVector<SplitShape, 4>> operandStates;

    static unsigned getSplitAxis(SplitMode split) { return split == SplitMode::SplitByM ? 0 : 1; }

    CopyVerdict getResultShape(CopyOp op, SmallVectorImpl<int64_t>& shape)
    {
        if (auto copySplit = op.getSplit(); copySplit.value_or(split) != split) {
            op.emitOpError() << "has incompatible 'split' argument: " << stringifySplitMode(*copySplit);
            return CopyVerdict::Fail;
        }
        auto srcType = op.getBase().getType();
        auto dstType = op.getType();
        if (srcType.getLoc() != TensorLocation::L0C || dstType.getLoc() != TensorLocation::UB)
            return CopyVerdict::Skip;
        assert(srcType.getRank() == dstType.getRank());
        if (srcType.getRank() != 2) {
            op.emitOpError() << "requires 2D tensors, got " << srcType.getRank() << "D";
            return CopyVerdict::Fail;
        }
        auto srcShape = srcType.getShape();
        auto dstShape = dstType.getShape();
        if (srcShape != dstShape) {
            op.emitOpError() << "requires the full source shape " << srcShape << ", got " << dstShape;
            return CopyVerdict::Fail;
        }
        if (split == SplitMode::SplitByM) {
            if (dstShape.front() % 2 != 0) {
                op.emitOpError() << "requires an M dimension divisible by 2, got " << dstShape.front();
                return CopyVerdict::Fail;
            }
            shape.push_back(dstShape.front() / 2);
        } else {
            shape.push_back(dstShape.front());
        }
        if (split == SplitMode::SplitByN) {
            if (dstShape.back() % 32 != 0) {
                op.emitOpError() << "requires an N dimension divisible by 32, got " << dstShape.back();
                return CopyVerdict::Fail;
            }
            shape.push_back(dstShape.back() / 2);
        } else {
            shape.push_back(dstShape.back());
        }
        return CopyVerdict::Accept;
    }

    LogicalResult getTensorShape(Operation* op, Value value, SmallVectorImpl<int64_t>& shape)
    {
        auto type = dyn_cast<LocalTensorType>(value.getType());
        if (!type || type.getRank() != 2)
            return op->emitOpError() << "requires 2D local tensors, got " << value.getType();
        shape.append(type.getShape().begin(), type.getShape().end());
        return success();
    }

    LogicalResult planAnnotation(ArrayRef<int64_t> shape, Operation* op)
    {
        if (auto existingSplit = op->getAttrOfType<SplitModeAttr>(attr::needSplit);
            existingSplit && existingSplit.getValue() != split)
            return op->emitOpError("has conflicting CV split mode requests");
        if (auto existingShape = op->getAttrOfType<DenseI64ArrayAttr>(attr::splitShape);
            existingShape && existingShape.asArrayRef() != shape)
            return op->emitOpError("has conflicting CV split shape requests");
        auto [it, inserted] = annotations.try_emplace(op, shape.begin(), shape.end());
        if (!inserted && it->second != shape)
            return op->emitOpError("has conflicting CV split requests");
        return success();
    }

    LogicalResult recordOperandState(const SplitState& state, Operation* user)
    {
        auto& shapes = operandStates[user];
        if (shapes.empty())
            shapes.resize(user->getNumOperands());
        for (auto [index, operand] : llvm::enumerate(user->getOperands())) {
            if (operand != state.value)
                continue;
            if (!shapes[index].empty() && shapes[index] != state.shape)
                return user->emitOpError() << "has conflicting split shapes for operand " << index;
            shapes[index] = state.shape;
        }
        return success();
    }

    LogicalResult getCopyOutputShape(const SplitState& state, CopyOp op, SmallVectorImpl<int64_t>& shape)
    {
        SplitShape inputShape;
        SplitShape resultShape;
        if (getTensorShape(op, state.value, inputShape).failed() ||
            getTensorShape(op, op.getResult(), resultShape).failed())
            return failure();
        if (inputShape != resultShape)
            return op.emitOpError("must preserve the tensor shape");
        shape.append(state.shape.begin(), state.shape.end());
        return success();
    }

    LogicalResult getElementwiseOutputShape(const SplitState& state, Operation* op, SmallVectorImpl<int64_t>& shape)
    {
        SplitShape inputShape;
        if (getTensorShape(op, state.value, inputShape).failed())
            return failure();
        for (Value operand : op->getOperands()) {
            if (operand == state.value || !isa<LocalTensorType>(operand.getType()))
                continue;
            SplitShape operandShape;
            if (getTensorShape(op, operand, operandShape).failed() || operandShape != inputShape)
                return op->emitOpError("must use tensors with the same shape");
        }
        for (Value result : op->getResults()) {
            SplitShape resultShape;
            if (getTensorShape(op, result, resultShape).failed() || resultShape != inputShape)
                return op->emitOpError("must preserve the input tensor shape");
        }
        shape.append(state.shape.begin(), state.shape.end());
        return success();
    }

    LogicalResult getReduceOutputShape(const SplitState& state, ReduceOp op, SmallVectorImpl<int64_t>& shape)
    {
        SplitShape inputShape;
        SplitShape resultShape;
        if (getTensorShape(op, op.getOperand(), inputShape).failed() ||
            getTensorShape(op, op.getResult(), resultShape).failed())
            return failure();
        for (Attribute attr : op.getDims()) {
            auto dimension = cast<IntegerAttr>(attr).getInt();
            if (dimension < 0)
                dimension += 2;
            if (dimension == axis)
                return op.emitOpError() << "cannot reduce the active split axis " << axis;
        }
        if (resultShape[axis] != inputShape[axis])
            return op.emitOpError("must preserve the active split axis");
        shape.append(resultShape.begin(), resultShape.end());
        shape[axis] = state.shape[axis];
        return success();
    }

    LogicalResult getReshapeOutputShape(const SplitState& state, ReshapeOp op, SmallVectorImpl<int64_t>& shape)
    {
        SplitShape inputShape;
        SplitShape resultShape;
        if (getTensorShape(op, op.getIn(), inputShape).failed() ||
            getTensorShape(op, op.getOut(), resultShape).failed())
            return failure();
        if (inputShape[axis] != resultShape[axis])
            return op.emitOpError("must preserve the active split axis");
        shape.append(resultShape.begin(), resultShape.end());
        shape[axis] = state.shape[axis];
        return success();
    }

    LogicalResult getBroadcastOutputShape(const SplitState& state, BroadcastOp op, SmallVectorImpl<int64_t>& shape)
    {
        SplitShape inputShape;
        SplitShape resultShape;
        if (getTensorShape(op, op.getOperand(), inputShape).failed() ||
            getTensorShape(op, op.getResult(), resultShape).failed())
            return failure();
        shape.append(resultShape.begin(), resultShape.end());
        if (inputShape[axis] == resultShape[axis])
            shape[axis] = state.shape[axis];
        else
            return op.emitOpError("cannot broadcast the active split axis");
        return success();
    }

    LogicalResult getOutputShape(const SplitState& state, Operation* op, SmallVectorImpl<int64_t>& shape)
    {
        if (isa<StoreOp>(op)) {
            shape.append(state.shape.begin(), state.shape.end());
            return success();
        }
        if (auto copyOp = dyn_cast<CopyOp>(op)) {
            if (state.kind == UserKind::Unsplit && (copyOp.getBase().getType().getLoc() != TensorLocation::UB ||
                                                    copyOp.getType().getLoc() != TensorLocation::L1))
                return op->emitOpError("is not UB->L1 transfer, hence cannot use split tensor");
            return getCopyOutputShape(state, copyOp, shape);
        }
        if (state.kind == UserKind::Unsplit)
            return op->emitOpError("only asctile.store and asctile.copy (UB->L1) can use asctile.cv_strategy results");
        if (isa<DumpTensorOp>(op))
            return success();
        if (op->hasTrait<OpTrait::Elementwise>())
            return getElementwiseOutputShape(state, op, shape);
        if (auto reduceOp = dyn_cast<ReduceOp>(op))
            return getReduceOutputShape(state, reduceOp, shape);
        if (auto reshapeOp = dyn_cast<ReshapeOp>(op))
            return getReshapeOutputShape(state, reshapeOp, shape);
        if (auto broadcastOp = dyn_cast<BroadcastOp>(op))
            return getBroadcastOutputShape(state, broadcastOp, shape);
        return op->emitOpError("cannot be used inside the CV strategy body");
    }

    LogicalResult propagateUser(const SplitState& state, Operation* user)
    {
        SplitShape outputShape;
        if (recordOperandState(state, user).failed())
            return failure();
        if (user->getNumRegions() != 0 || user->getNumSuccessors() != 0)
            return user->emitOpError("has regions or successors and is not supported by CV strategy propagation");
        if (getOutputShape(state, user, outputShape).failed())
            return failure();
        if (annotations.contains(user)) {
            if (annotations[user] != outputShape) {
                return user->emitOpError() << "has conflicting CV split requests, the surrounding CV strategy is "
                                           << stringifySplitMode(split) << " splitting with shape " << outputShape;
            }
        } else if (planAnnotation(outputShape, user).failed()) {
            return failure();
        }
        if (state.kind == UserKind::Split) {
            for (Value result : user->getResults())
                worklist.push_back({result, outputShape});
        }
        return success();
    }

public:
    explicit CVStrategyModel(CVStrategyOp root) : root(root), split(root.getSplit()), axis(getSplitAxis(split)) {};
    ~CVStrategyModel() = default;

    LogicalResult ensure()
    {
        for (auto op : root.getOps<CopyOp>()) {
            SplitShape shape;
            auto verdict = getResultShape(op, shape);
            if (verdict == CopyVerdict::Fail) {
                return failure();
            }
            if (verdict == CopyVerdict::Skip)
                continue;
            assert(verdict == CopyVerdict::Accept);
            if (planAnnotation(shape, op).failed())
                return failure();
            worklist.push_back({op.getResult(), shape});
        }
        while (!worklist.empty()) {
            auto state = worklist.pop_back_val();
            for (auto* user : state.value.getUsers()) {
                if (state.kind == UserKind::Split) {
                    if (user->getParentOfType<CVStrategyOp>() != root)
                        continue;
                    if (auto yieldOp = dyn_cast<YieldOp>(user)) {
                        if (planAnnotation({}, yieldOp).failed())
                            return failure();
                        for (auto [result, operand] : llvm::zip_equal(root.getResults(), yieldOp.getOperands())) {
                            if (operand == state.value)
                                worklist.push_back({result, state.shape, UserKind::Unsplit});
                        }
                        continue;
                    }
                }
                if (propagateUser(state, user).failed())
                    return failure();
            }
        }
        OpBuilder builder(root);
        auto needSplitAttr = builder.getAttr<SplitModeAttr>(split);
        for (auto& [op, shape] : annotations) {
            op->setAttr(attr::needSplit, needSplitAttr);
            op->setAttr(attr::splitShape, builder.getDenseI64ArrayAttr(shape));
        }
        return success();
    }
};

struct PrepareCVStrategyPass : public asctile::impl::PrepareCVStrategyBase<PrepareCVStrategyPass> {
    void runOnOperation() override
    {
        getOperation().walk([this](CVStrategyOp op) {
            CVStrategyModel model(op);
            if (model.ensure().failed())
                signalPassFailure();
        });
    }
};

} // namespace

std::unique_ptr<Pass> mlir::asctile::createPrepareCVStrategyPass() { return std::make_unique<PrepareCVStrategyPass>(); }
