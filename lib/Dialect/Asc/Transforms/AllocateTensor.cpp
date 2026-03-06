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
#include "ascir/Dialect/Asc/Utils/Utils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"

#include <unordered_map>

namespace mlir {
namespace ascendc {
#define GEN_PASS_DEF_ALLOCATETENSOR
#include "ascir/Dialect/Asc/Transforms/Passes.h.inc"
} // namespace ascendc
} // namespace mlir

using namespace mlir;
using namespace mlir::ascendc;

namespace {

AddressSpace positionToAddressSpace(TPosition position)
{
    switch (position) {
    case TPosition::A1:
    case TPosition::B1:
        return AddressSpace::cbuf;
    case TPosition::A2:
        return AddressSpace::ca;
    case TPosition::B2:
        return AddressSpace::cb;
    case TPosition::CO1:
        return AddressSpace::cc;
    case TPosition::VECCALC:
    case TPosition::VECIN:
    case TPosition::VECOUT:
        return AddressSpace::ubuf;
    default:
        llvm_unreachable("unexpected TPosition value");
    }
}

class AllocateTensorPass : public ascendc::impl::AllocateTensorBase<AllocateTensorPass> {
    void runOnOperation() override
    {
        func::FuncOp funcOp = getOperation();
        if (funcOp.isDeclaration()) {
            return;
        }
        std::unordered_map<ascendc::AddressSpace, uint32_t> offsets;
        funcOp.walk<WalkOrder::PreOrder>([this, &offsets](LocalTensorAutoOp op) {
            auto type = op.getType();
            if (!type.hasStaticShape()) {
                op.emitOpError() << "must be static shape";
                signalPassFailure();
            }
            OpBuilder builder(op);
            uint32_t& addr = offsets[positionToAddressSpace(op.getPosition())];
            auto tensor =
                builder.create<LocalTensorV3Op>(op.getLoc(), type, op.getPosition(), addr, type.getNumElements());
            op->replaceAllUsesWith(tensor);
            op.erase();
            addr += llvm::alignTo<ubBlockSize>(getTypeSize(type));
        });
    }
};

} // namespace

namespace mlir {
namespace ascendc {
std::unique_ptr<Pass> createAllocateTensorPass() { return std::make_unique<AllocateTensorPass>(); }
} // namespace ascendc
} // namespace mlir
