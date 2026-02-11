/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ascir/Dialect/AscTile/IR/AscTile.h"
#include "ascir/Dialect/AscTile/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace asctile {
#define GEN_PASS_DEF_TRANSFORMMATHOPS
#include "ascir/Dialect/AscTile/Transforms/Passes.h.inc"
} // namespace asctile
} // namespace mlir

using namespace mlir;

namespace {

std::optional<TypedAttr> getSplatValue(arith::ConstantOp cstOp)
{
    if (!cstOp)
        return std::nullopt;
    auto dense = dyn_cast<DenseElementsAttr>(cstOp.getValue());
    if (!dense || !dense.isSplat())
        return std::nullopt;
    return dense.getSplatValue<TypedAttr>();
}

Value materializeSplatValue(OpBuilder& builder, Value cstTile)
{
    if (auto s = getSplatValue(cstTile.getDefiningOp<arith::ConstantOp>()))
        return builder.create<arith::ConstantOp>(cstTile.getLoc(), s.value());
    if (auto splat = cstTile.getDefiningOp<asctile::SplatOp>())
        return splat.getValue();
    return {};
}

template <typename ArithOp, typename TileOp>
struct ScalarizeArithOp : OpRewritePattern<ArithOp> {
    using OpRewritePattern<ArithOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(ArithOp op, PatternRewriter& rewriter) const override
    {
        if (!isa<asctile::TileType>(op.getType()))
            return failure();
        Value newLhs, newRhs;
        if (auto splat = materializeSplatValue(rewriter, op.getLhs())) {
            newLhs = op.getRhs();
            newRhs = splat;
        } else if (auto splat = materializeSplatValue(rewriter, op.getRhs())) {
            newLhs = op.getLhs();
            newRhs = splat;
        } else {
            return failure();
        }
        rewriter.replaceOpWithNewOp<TileOp>(op, op.getType(), newLhs, newRhs);
        return success();
    }
};

template <typename MaxOp>
struct MaxWithZeroToReluOp : OpRewritePattern<MaxOp> {
    using OpRewritePattern<MaxOp>::OpRewritePattern;

    virtual bool isValidType(Type type) const = 0;

    virtual bool isZero(Value value) const = 0;

    LogicalResult matchAndRewrite(MaxOp op, PatternRewriter& rewriter) const override
    {
        if (!isValidType(op.getType()))
            return failure();
        arith::ConstantOp cstOp;
        Value newLhs;
        if (auto defOp = op.getLhs().template getDefiningOp<arith::ConstantOp>()) {
            cstOp = defOp;
            newLhs = op.getRhs();
        } else if (auto defOp = op.getRhs().template getDefiningOp<arith::ConstantOp>()) {
            cstOp = defOp;
            newLhs = op.getLhs();
        }
        if (!cstOp || !newLhs) {
            return failure();
        }
        Value value = cstOp.getResult();
        if (isZero(value)) {
            rewriter.replaceOpWithNewOp<asctile::ReluOp>(op, op.getType(), newLhs);
            return success();
        }
        return failure();
    }
};

template <typename MaxOp>
struct MaxWithZeroToReluOpFloat : MaxWithZeroToReluOp<MaxOp> {
    using MaxWithZeroToReluOp<MaxOp>::MaxWithZeroToReluOp;

    bool isValidType(Type type) const override
    {
        if (auto vecType = dyn_cast<asctile::TileType>(type)) {
            return isa<Float16Type, Float32Type>(vecType.getElementType());
        }
        return false;
    }

    bool isZero(Value value) const override { return matchPattern(value, m_AnyZeroFloat()); }
};

template <typename MaxOp>
struct MaxWithZeroToReluOpInt : MaxWithZeroToReluOp<MaxOp> {
    using MaxWithZeroToReluOp<MaxOp>::MaxWithZeroToReluOp;

    bool isValidType(Type type) const override
    {
        if (auto vecType = dyn_cast<asctile::TileType>(type)) {
            return vecType.getElementType().isInteger(32);
        }
        return false;
    }

    bool isZero(Value value) const override { return matchPattern(value, m_Zero()); }
};

template <typename SubOp>
struct ScalarizeSub : OpRewritePattern<SubOp> {
    using OpRewritePattern<SubOp>::OpRewritePattern;

    static Value transformScalar(Value rhs, PatternRewriter& rewriter)
    {
        if constexpr (std::is_same_v<SubOp, arith::SubFOp>) {
            return rewriter.create<arith::NegFOp>(rhs.getLoc(), rhs);
        } else if constexpr (std::is_same_v<SubOp, arith::SubIOp>) {
            Value m1 = rewriter.create<arith::ConstantIntOp>(rhs.getLoc(), -1L, rhs.getType());
            return rewriter.create<arith::MulIOp>(rhs.getLoc(), rhs, m1);
        } else {
            llvm_unreachable("unexpected arith.sub* op");
        }
    }

    LogicalResult matchAndRewrite(SubOp op, PatternRewriter& rewriter) const override
    {
        if (!isa<asctile::TileType>(op.getType()))
            return failure();
        Value scalar = materializeSplatValue(rewriter, op.getRhs());
        if (!scalar)
            return failure();
        rewriter.replaceOpWithNewOp<asctile::AddsOp>(op, op.getType(), op.getLhs(), transformScalar(scalar, rewriter));
        return success();
    }
};

struct ScalarizeShL : OpRewritePattern<arith::ShLIOp> {
    using OpRewritePattern<arith::ShLIOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(arith::ShLIOp op, PatternRewriter& rewriter) const override
    {
        if (!isa<asctile::TileType>(op.getType()))
            return failure();

        Value scalar = materializeSplatValue(rewriter, op.getRhs());
        if (!scalar)
            return failure();

        rewriter.replaceOpWithNewOp<asctile::ShLSOp>(op, op.getType(), op.getLhs(), scalar);

        return success();
    }
};

struct ScalarizeShR : OpRewritePattern<arith::ShRSIOp> {
    using OpRewritePattern<arith::ShRSIOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(arith::ShRSIOp op, PatternRewriter& rewriter) const override
    {
        if (!isa<asctile::TileType>(op.getType()))
            return failure();

        Value scalar = materializeSplatValue(rewriter, op.getRhs());
        if (!scalar)
            return failure();

        rewriter.replaceOpWithNewOp<asctile::ShRSOp>(op, op.getType(), op.getLhs(), scalar);

        return success();
    }
};

class TransformMathOpsPass : public asctile::impl::TransformMathOpsBase<TransformMathOpsPass> {
public:
    TransformMathOpsPass() = default;

    void runOnOperation() override
    {
        auto op = getOperation();
        MLIRContext* context = op.getContext();
        RewritePatternSet patterns(context);
        patterns.add<
            //
            MaxWithZeroToReluOpFloat<arith::MaximumFOp>, MaxWithZeroToReluOpFloat<arith::MaxNumFOp>,
            MaxWithZeroToReluOpInt<arith::MaxSIOp>
            //
            >(context, /*benefit=*/2);
        patterns.add<
            //
            ScalarizeArithOp<arith::AddFOp, asctile::AddsOp>, ScalarizeArithOp<arith::AddIOp, asctile::AddsOp>,
            ScalarizeSub<arith::SubFOp>, ScalarizeSub<arith::SubIOp>, ScalarizeArithOp<arith::MulFOp, asctile::MulsOp>,
            ScalarizeArithOp<arith::MulIOp, asctile::MulsOp>, ScalarizeShL, ScalarizeShR
            //
            >(context, /*benefit=*/1);
        if (applyPatternsAndFoldGreedily(op, std::move(patterns)).failed())
            signalPassFailure();
    }
};
} // namespace

std::unique_ptr<Pass> mlir::asctile::createTransformMathOpsPass() { return std::make_unique<TransformMathOpsPass>(); }
