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
#include "ascir/Dialect/AscTile/IR/AscTile.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/Pass/Pass.h"

#include "Common.h"

namespace mlir {
namespace asclower {
#define GEN_PASS_DEF_REDRESSI1TILE
#include "ascir/Conversion/LowerToAsc/Passes.h.inc"
} // namespace asclower
} // namespace mlir

using namespace mlir;
using namespace mlir::asclower;

namespace {

struct RedressTypeConverter : public LoweringTypeConverter {
    RedressTypeConverter() : LoweringTypeConverter()
    {
        addConversion([](asctile::TileType type) {
            auto elType = type.getElementType();
            SmallVector<int64_t> shape(type.getShape());
            if (elType.isInteger(1)) {
                I1ReplacementType replType(type.getContext());
                auto numElements = static_cast<int64_t>(llvm::divideCeil(type.getNumElements(), replType.width));
                shape = {numElements};
                elType = replType.iType;
            }
            return asctile::TileType::get(shape, elType, type.getLoc());
        });
    }
};

struct RedressConversionTarget : public ConversionTarget {
    RedressConversionTarget(RedressTypeConverter &converter, MLIRContext *context) : ConversionTarget(*context)
    {
        addLegalDialect<arith::ArithDialect, memref::MemRefDialect, vector::VectorDialect>();
        addDynamicallyLegalOp<arith::ConstantOp, vector::BroadcastOp>([](Operation *op) {
            assert(op->getNumResults() == 1);
            if (auto type = dyn_cast<asctile::TileType>(op->getResult(0).getType())) {
                return !type.getElementType().isInteger(1);
            }
            return true;
        });
    }
};

struct RedressSplatConstant : ConvertOp<arith::ConstantOp> {
    using ConvertOp::converter;
    using ConvertOp::ConvertOp;

    LogicalResult convert(arith::ConstantOp op, ConvertRewriter &rewriter) const override
    {
        if (!isa_and_present<SplatElementsAttr>(op.getValue()))
            return failure();
        auto oldType = cast<asctile::TileType>(op.getType());
        assert(oldType.getElementType().isInteger(1));
        auto dense = dyn_cast<SplatElementsAttr>(op.getValue());
        I1ReplacementType replType(op.getContext());
        I1ReplacementType::UInt value = 0;
        if (!dense.getSplatValue<IntegerAttr>().getValue().isZero())
            value = replType.max();
        auto newType = cast<ShapedType>(converter().convertType(oldType));
        auto attr = SplatElementsAttr::get(newType, static_cast<I1ReplacementType::UInt>(value));
        rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, attr);
        return success();
    }
};

struct RedressDenseConstant : ConvertOp<arith::ConstantOp> {
    using ConvertOp::converter;
    using ConvertOp::ConvertOp;

    LogicalResult convert(arith::ConstantOp op, ConvertRewriter &rewriter) const override
    {
        auto dense = dyn_cast_if_present<DenseElementsAttr>(op.getValue());
        if (!dense || dense.isSplat())
            return failure();
        auto oldType = cast<ShapedType>(op.getType());
        Location loc = rewriter.getUnknownLoc();
        auto newType = cast<ShapedType>(converter().convertType(oldType));
        BitVector rawData(oldType.getNumElements(), false);
        for (auto [i, item] : llvm::enumerate(dense.getValues<BoolAttr>()))
            rawData[i] = item.getValue();
        std::vector<I1ReplacementType::Int> reinterp(newType.getNumElements(), 0U);
        std::memcpy(reinterp.data(), rawData.getData().data(), rawData.getMemorySize());
        rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, DenseElementsAttr::get(newType, ArrayRef(reinterp)));
        return success();
    }
};

struct RedressI1TilePass : public asclower::impl::RedressI1TileBase<RedressI1TilePass> {
    void runOnOperation() override
    {
        func::FuncOp funcOp = getOperation();
        RedressTypeConverter converter;
        MLIRContext *context = &getContext();
        RedressConversionTarget target(converter, context);
        RewritePatternSet patterns(context);
        patterns.insert<RedressSplatConstant, RedressDenseConstant>(converter, context);
        if (applyPartialConversion(funcOp, target, std::move(patterns)).failed())
            signalPassFailure();
    }
};

} // namespace

std::unique_ptr<Pass> mlir::asclower::createRedressI1TilePass()
{
    return std::make_unique<RedressI1TilePass>();
}
