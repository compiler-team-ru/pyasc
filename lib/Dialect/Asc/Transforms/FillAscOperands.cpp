/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under
 * the terms and conditions of CANN Open Software License Agreement Version 2.0
 * (the "License"). Please refer to the License for details. You may not use
 * this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
 * AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
 * FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
 * for the full text of the License.
 */

#include "ascir/Dialect/Asc/IR/Asc.h"
#include "ascir/Dialect/Asc/Transforms/Passes.h"
#include "ascir/Dialect/Asc/Utils/Attributes.h"
#include "ascir/Dialect/Asc/Utils/Utils.h"
#include "ascir/Dialect/EmitAsc/IR/EmitAsc.h"
#include "ascir/Dialect/Utils/ConstantOpBuilder.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"

namespace mlir {
namespace ascendc {
#define GEN_PASS_DEF_FILLASCOPERANDS
#include "ascir/Dialect/Asc/Transforms/Passes.h.inc"
} // namespace ascendc
} // namespace mlir

using namespace mlir;
using namespace ascir;

namespace {

template <typename T, typename... Types>
using AnyOfT = std::enable_if_t<llvm::is_one_of<T, Types...>::value, bool>;

constexpr int64_t dstBlkStride = 1;
constexpr int64_t src0BlkStride = 1;
constexpr int64_t src1BlkStride = 1;
constexpr int64_t dstRepStride = 8;
constexpr int64_t src0RepStride = 8;
constexpr int64_t src1RepStride = 8;

int64_t getRepeatTimes(ShapedType type)
{
    auto sizeType = ascendc::getElementTypeSize(type);
    assert((sizeType == 2 || sizeType == 4) && "Unsupported element type");
    auto numElemsPerRepeat = ascendc::repeatBlockSize / sizeType;
    return llvm::divideCeil(type.getNumElements(), numElemsPerRepeat);
}

std::pair<uint64_t, uint64_t> getMask(ShapedType type)
{
    auto sizeType = ascendc::getElementTypeSize(type);
    assert((sizeType == 2 || sizeType == 4) && "Unsupported element type");
    auto mask = llvm::maskTrailingOnes<uint64_t>(ascendc::bitmaskSize);
    if (sizeType == 2)
        return {mask, mask};
    return {0ul, mask};
}

template <class OpT>
void fillMask(OpBuilder &builder, Location loc, ConstantOpBuilder &consts, ShapedType srcType, OpT op)
{
    if (op->hasAttr(ascendc::attr::maskSet))
        return;
    auto [maskHVal, maskLVal] = getMask(srcType);
    auto mask = builder.create<emitasc::MaskOp>(loc, consts.i64(maskHVal), consts.i64(maskLVal));
    op.getMaskMutable().assign(mask);
}

template <typename OpType, AnyOfT<OpType, ascendc::UnaryL0Op, ascendc::CastL0Op, ascendc::CompareScalarL0Op> = true>
void fillOperation(OpType op)
{
    OpBuilder builder(op);
    ConstantOpBuilder consts(builder);
    auto srcType = dyn_cast<ShapedType>(op.getSrc().getType());
    auto loc = op.getLoc();
    fillMask(builder, loc, consts, srcType, op);
    auto repeatTimes = consts.i64(getRepeatTimes(srcType));
    op.getRepeatTimesMutable().assign(repeatTimes);
    auto dstBlkStrideVal = consts.i64(dstBlkStride);
    auto src0BlkStrideVal = consts.i64(src0BlkStride);
    auto dstRepStrideVal = consts.i64(dstRepStride);
    auto src0RepStrideVal = consts.i64(src0RepStride);
    auto repeatParams = builder.create<ascendc::ConstructOp>(
        loc, builder.getType<ascendc::UnaryRepeatParamsType>(),
        ValueRange {dstBlkStrideVal, src0BlkStrideVal, dstRepStrideVal, src0RepStrideVal});
    op.getRepeatParamsMutable().assign(repeatParams);
}

template <typename OpType, AnyOfT<OpType, ascendc::BinaryL0Op, ascendc::SelectL0Op, ascendc::CompareL0Op> = true>
void fillOperation(OpType op)
{
    OpBuilder builder(op);
    ConstantOpBuilder consts(builder);
    auto srcType = dyn_cast<ShapedType>(op.getSrc0().getType());
    auto loc = op.getLoc();
    fillMask(builder, loc, consts, srcType, op);
    auto repeatTimes = consts.i64(getRepeatTimes(srcType));
    op.getRepeatTimesMutable().assign(repeatTimes);
    auto dstBlkStrideVal = consts.i64(dstBlkStride);
    auto src0BlkStrideVal = consts.i64(src0BlkStride);
    auto src1BlkStrideVal = consts.i64(src1BlkStride);
    auto dstRepStrideVal = consts.i64(dstRepStride);
    auto src0RepStrideVal = consts.i64(src0RepStride);
    auto src1RepStrideVal = consts.i64(src1RepStride);
    auto repeatParams =
        builder.create<ascendc::ConstructOp>(loc, builder.getType<ascendc::BinaryRepeatParamsType>(),
                                             ValueRange {dstBlkStrideVal, src0BlkStrideVal, src1BlkStrideVal,
                                                         dstRepStrideVal, src0RepStrideVal, src1RepStrideVal});
    op.getRepeatParamsMutable().assign(repeatParams);
}

void fillOperation(ascendc::DuplicateL0Op op)
{
    OpBuilder builder(op);
    ConstantOpBuilder consts(builder);
    auto scalarType = dyn_cast<ShapedType>(op.getDst().getType());
    auto loc = op.getLoc();
    fillMask(builder, loc, consts, scalarType, op);
    auto repeatTimes = consts.i64(getRepeatTimes(scalarType));
    op.getRepeatTimesMutable().assign(repeatTimes);
}

void fillOperation(ascendc::VecScalarL0Op op)
{
    OpBuilder builder(op);
    ConstantOpBuilder consts(builder);
    auto srcType = dyn_cast<ShapedType>(op.getSrc().getType());
    auto loc = op.getLoc();
    auto elemType = dyn_cast<ShapedType>(op.getScalar().getType());

    auto scalarValue = consts.i64(0);
    op.getScalarMutable().assign(scalarValue);
    auto [maskHVal, maskLVal] = getMask(srcType);
    auto mask = builder.create<emitasc::MaskOp>(loc, consts.i64(maskHVal), consts.i64(maskLVal));
    op.getMaskMutable().assign(mask);
    auto repeatTimes = consts.i64(getRepeatTimes(srcType));
    op.getRepeatTimesMutable().assign(repeatTimes);
    auto dstBlkStrideVal = consts.i64(dstBlkStride);
    auto src0BlkStrideVal = consts.i64(src0BlkStride);
    auto dstRepStrideVal = consts.i64(dstRepStride);
    auto src0RepStrideVal = consts.i64(src0RepStride);
    auto repeatParams = builder.create<ascendc::ConstructOp>(
        loc, builder.getType<ascendc::UnaryRepeatParamsType>(),
        ValueRange {dstBlkStrideVal, src0BlkStrideVal, dstRepStrideVal, src0RepStrideVal});
    op.getRepeatParamsMutable().assign(repeatParams);
}

template <typename OpType, AnyOfT<OpType, ascendc::DuplicateL2Op, ascendc::UnaryL2Op, ascendc::BinaryL2Op,
                                  ascendc::VecScalarL2Op> = true>
void fillOperation(OpType op)
{
    OpBuilder builder(op);
    ConstantOpBuilder consts(builder);
    auto shapedType = dyn_cast<ShapedType>(op.getDst().getType()); // NOTE: Correct if dstType is same srcType
    auto numElems = shapedType.getNumElements();
    op.getCalCountMutable().assign(consts.i64(numElems));
}

struct FillAscOperandsPass : public ascendc::impl::FillAscOperandsBase<FillAscOperandsPass> {
    void runOnOperation() override
    {
        auto funcOp = getOperation();
        funcOp.walk([](Operation *op) {
            llvm::TypeSwitch<Operation *>(op)
                .Case<ascendc::UnaryL0Op, ascendc::CastL0Op, ascendc::CompareScalarL0Op, ascendc::DuplicateL0Op,
                      ascendc::BinaryL0Op, ascendc::VecScalarL0Op, ascendc::SelectL0Op, ascendc::CompareL0Op,
                      ascendc::DuplicateL2Op, ascendc::UnaryL2Op, ascendc::BinaryL2Op, ascendc::VecScalarL2Op>(
                    [](auto fillOp) { fillOperation(fillOp); })
                .Default([](Operation *) {});
        });
    }
};

} // namespace

namespace mlir {
namespace ascendc {
std::unique_ptr<Pass> createFillAscOperandsPass()
{
    return std::make_unique<FillAscOperandsPass>();
}
} // namespace ascendc
} // namespace mlir
