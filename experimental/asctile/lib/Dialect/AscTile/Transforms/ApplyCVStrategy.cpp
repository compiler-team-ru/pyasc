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
#include "asctile/Dialect/AscTile/Utils/Utils.h"

#include "ascir/Dialect/Asc/IR/Asc.h"
#include "ascir/Dialect/Utils/ConstantOpBuilder.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir {
namespace asctile {
#define GEN_PASS_DEF_APPLYCVSTRATEGY
#include "asctile/Dialect/AscTile/Transforms/Passes.h.inc"
} // namespace asctile
} // namespace mlir

using namespace mlir;
using namespace mlir::asctile;

namespace {

std::pair<DistribModeAttr, ArrayRef<int64_t>> getSplitInfo(Operation* op)
{
    return {
        op->getAttrOfType<DistribModeAttr>(attr::needSplit),
        op->getAttrOfType<DenseI64ArrayAttr>(attr::splitShape).asArrayRef()};
}

SmallVector<Value, 2> splitOffsets(OpBuilder& builder, ValueRange offsets, DistribMode split, ArrayRef<int64_t> shape)
{
    auto rank = shape.size();
    assert(offsets.size() == rank && "offset count must match tensor rank");
    SmallVector<Value, 2> newOffsets;
    auto iType = offsets.front().getType();
    auto loc = builder.getUnknownLoc();
    Value subBlockIdx = builder.create<ascendc::GetSubBlockIdxOp>(loc, iType);
    unsigned axis = getSplitAxis(split, rank);
    for (auto [index, offset] : llvm::enumerate(offsets)) {
        if (index != axis) {
            newOffsets.push_back(offset);
            continue;
        }
        Value halfSize = builder.create<arith::ConstantOp>(loc, builder.getIntegerAttr(iType, shape[axis]));
        Value addOffset = builder.create<arith::MulIOp>(loc, halfSize, subBlockIdx);
        newOffsets.push_back(builder.create<arith::AddIOp>(loc, offset, addOffset));
    }
    return newOffsets;
}

Value splitTensor(Value opnd, DistribMode split, ArrayRef<int64_t> sizes, ConversionPatternRewriter& rewriter)
{
    auto tensor = dyn_cast<LocalTensorType>(opnd.getType());
    if (!tensor || tensor.getShape() == sizes)
        return opnd;
    assert(tensor.getLoc() == TensorLocation::UB && "tensor.extract_slice must use UB tensor");
    auto newType = tensor.clone(sizes);
    OpFoldResult splat = getSplatValue(opnd);
    if (auto attr = dyn_cast_if_present<Attribute>(splat))
        return rewriter.create<arith::ConstantOp>(opnd.getLoc(), SplatElementsAttr::get(newType, attr));
    if (auto value = dyn_cast_if_present<Value>(splat))
        return rewriter.create<tensor::SplatOp>(opnd.getLoc(), newType, value);
    ascir::ConstantOpBuilder consts(rewriter);
    SmallVector<Value, 2> zeros(sizes.size(), consts.index(0));
    auto offsets = splitOffsets(rewriter, zeros, split, sizes);
    auto staticOffsets = rewriter.getDenseI64ArrayAttr(SmallVector(sizes.size(), ShapedType::kDynamic));
    SmallVector<int64_t> strides = computeStrides(sizes);
    return rewriter.create<tensor::ExtractSliceOp>(
        opnd.getLoc(), newType, opnd, offsets, /*sizes*/ ValueRange{}, /*strides*/ ValueRange{}, staticOffsets,
        rewriter.getDenseI64ArrayAttr(sizes), rewriter.getDenseI64ArrayAttr(strides));
}

SmallVector<int64_t, 2> getOperandSplitShape(Value operand, DistribMode split)
{
    auto tensorShape = cast<LocalTensorType>(operand.getType()).getShape();
    SmallVector<int64_t, 2> shape(tensorShape.begin(), tensorShape.end());
    auto axis = getSplitAxis(split, shape.size());
    shape[axis] /= 2;
    return shape;
}

struct SplitCopy : OpConversionPattern<CopyOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(CopyOp op, CopyOp::Adaptor, ConversionPatternRewriter& rewriter) const override
    {
        auto [split, shape] = getSplitInfo(op);
        auto srcLoc = op.getBase().getType().getLoc();
        auto dstLoc = op.getType().getLoc();
        auto src = rewriter.getRemappedValue(op.getBase());
        if (srcLoc == TensorLocation::L0C && dstLoc == TensorLocation::UB) {
            rewriter.replaceOpWithNewOp<CopyOp>(op, op.getType().clone(shape), src, op.getOffsets(), split);
            return success();
        }
        if (srcLoc == TensorLocation::UB && dstLoc == TensorLocation::L1) {
            auto offsets = splitOffsets(rewriter, op.getOffsets(), split.getValue(), shape);
            rewriter.replaceOpWithNewOp<CopyOp>(op, op.getType(), src, offsets, DistribModeAttr{});
            return success();
        }
        return op->emitOpError("is not eligible for the CV strategy with splitting");
    }
};

struct SplitStore : OpConversionPattern<StoreOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(StoreOp op, StoreOp::Adaptor, ConversionPatternRewriter& rewriter) const override
    {
        auto [split, shape] = getSplitInfo(op);
        auto offsets = splitOffsets(rewriter, op.getOffsets(), split.getValue(), shape);
        if (auto realShape = op.getRealShape(); !realShape.empty()) {
            // TODO: Implement real shape clamping
            return op->emitOpError("with real shape is not eligible for the CV strategy with splitting");
        }
        rewriter.replaceOpWithNewOp<StoreOp>(
            op, rewriter.getRemappedValue(op.getValue()), op.getBase(), offsets, ValueRange{});
        return success();
    }
};

struct SplitElementwise : OpTraitConversionPattern<OpTrait::Elementwise> {
    using OpTraitConversionPattern::OpTraitConversionPattern;

    LogicalResult matchAndRewrite(Operation* op, ArrayRef<Value>, ConversionPatternRewriter& rewriter) const override
    {
        if (op->getNumRegions() != 0 || op->getNumSuccessors() != 0)
            return op->emitOpError("is not expected inside asctile.cv_strategy body");
        auto [split, shape] = getSplitInfo(op);
        SmallVector<Value, 4> operands;
        for (Value opnd : op->getOperands()) {
            operands.push_back(splitTensor(rewriter.getRemappedValue(opnd), split.getValue(), shape, rewriter));
        }
        SmallVector<Type, 2> results;
        for (Type result : op->getResultTypes()) {
            auto tensor = dyn_cast<LocalTensorType>(result);
            if (!tensor || tensor.getShape() == shape)
                results.push_back(result);
            else
                results.push_back(tensor.clone(shape));
        }
        auto* newOp = rewriter.create(op->getLoc(), op->getName().getIdentifier(), operands, results, op->getAttrs());
        newOp->removeAttr(attr::needSplit);
        newOp->removeAttr(attr::splitShape);
        rewriter.replaceOp(op, newOp);
        return success();
    }
};

struct SplitReduce : OpConversionPattern<ReduceOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(ReduceOp op, ReduceOp::Adaptor, ConversionPatternRewriter& rewriter) const override
    {
        auto [split, resultShape] = getSplitInfo(op);
        auto operandShape = getOperandSplitShape(op.getOperand(), split.getValue());
        auto operand =
            splitTensor(rewriter.getRemappedValue(op.getOperand()), split.getValue(), operandShape, rewriter);
        auto resultType = op.getType().clone(resultShape);
        rewriter.replaceOpWithNewOp<ReduceOp>(op, resultType, operand, op.getDims(), op.getKindAttr());
        return success();
    }
};

struct SplitReshape : OpConversionPattern<ReshapeOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(ReshapeOp op, ReshapeOp::Adaptor, ConversionPatternRewriter& rewriter) const override
    {
        auto [split, resultShape] = getSplitInfo(op);
        auto operandShape = getOperandSplitShape(op.getIn(), split.getValue());
        auto operand = splitTensor(rewriter.getRemappedValue(op.getIn()), split.getValue(), operandShape, rewriter);
        rewriter.replaceOpWithNewOp<ReshapeOp>(op, op.getType().clone(resultShape), operand);
        return success();
    }
};

struct SplitBroadcast : OpConversionPattern<BroadcastOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(
        BroadcastOp op, BroadcastOp::Adaptor, ConversionPatternRewriter& rewriter) const override
    {
        auto [split, resultShape] = getSplitInfo(op);
        auto operandShape = getOperandSplitShape(op.getOperand(), split.getValue());
        auto operand =
            splitTensor(rewriter.getRemappedValue(op.getOperand()), split.getValue(), operandShape, rewriter);
        rewriter.replaceOpWithNewOp<BroadcastOp>(op, op.getType().clone(resultShape), operand);
        return success();
    }
};

template <typename OpT>
struct ConvertOperands : OpConversionPattern<OpT> {
    using OpConversionPattern<OpT>::OpConversionPattern;

    LogicalResult matchAndRewrite(OpT op, typename OpT::Adaptor, ConversionPatternRewriter& rewriter) const override
    {
        SmallVector<Value, 8> operands;
        if (rewriter.getRemappedValues(op->getOperands(), operands).failed())
            return failure();
        auto newOp = rewriter.replaceOpWithNewOp<OpT>(op, op->getResultTypes(), operands, op->getAttrs());
        newOp->removeAttr(attr::needSplit);
        newOp->removeAttr(attr::splitShape);
        return success();
    }
};

struct ApplyCVStrategyPass : public asctile::impl::ApplyCVStrategyBase<ApplyCVStrategyPass> {
    void runOnOperation() override
    {
        auto funcOp = getOperation();
        MLIRContext* context = &getContext();
        ConversionTarget target(*context);
        target.markUnknownOpDynamicallyLegal([](Operation* op) { return !op->hasAttr(attr::needSplit); });
        RewritePatternSet patterns(context);
        patterns.add<
            SplitCopy, SplitStore, SplitElementwise, SplitReduce, SplitReshape, SplitBroadcast,
            ConvertOperands<DumpTensorOp>, ConvertOperands<InlineOp>, ConvertOperands<YieldOp>>(context);
        DenseSet<Operation*> unlegalizedOps;
        ConversionConfig config;
        config.unlegalizedOps = &unlegalizedOps;
        if (applyPartialConversion(funcOp, target, std::move(patterns)).failed()) {
            if (unlegalizedOps.empty())
                emitError(funcOp.getLoc(), "CV strategy application failed");
            for (auto* op : unlegalizedOps)
                op->emitOpError("prevents the surrounding CV strategy from application");
            signalPassFailure();
            return;
        }
        funcOp.walk([](CVStrategyOp op) {
            Block* body = op.getBody();
            auto* yieldOp = body->getTerminator();
            op->getBlock()->getOperations().splice(op->getIterator(), body->getOperations());
            op.replaceAllUsesWith(yieldOp->getOperands());
            yieldOp->erase();
            op->erase();
        });
    }
};

} // namespace

std::unique_ptr<Pass> mlir::asctile::createApplyCVStrategyPass() { return std::make_unique<ApplyCVStrategyPass>(); }
