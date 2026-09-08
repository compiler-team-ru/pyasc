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

#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"

#define GET_OP_CLASSES
#include "ascir/Dialect/AscVF/IR/AscVFOps.cpp.inc"

using namespace mlir;
using namespace mlir::ascvf;

namespace {
bool hasStaticShapes(Operation* op)
{
    return llvm::all_of(op->getOperandTypes(), [](Type type) {
        auto tensorType = dyn_cast<ascendc::LocalTensorType>(type);
        return !tensorType || tensorType.hasStaticShape();
    });
}
} // namespace

//===----------------------------------------------------------------------===//
// LoadOp
//===----------------------------------------------------------------------===//

LogicalResult LoadOp::verify()
{
    if (!getOperation()->getParentOfType<ascvf::VecScopeOp>()) {
        return emitOpError("must be inside ascvf.vec_scope block");
    }
    return success();
}

//===----------------------------------------------------------------------===//
// StoreOp
//===----------------------------------------------------------------------===//

LogicalResult StoreOp::verify()
{
    if (!getOperation()->getParentOfType<ascvf::VecScopeOp>()) {
        return emitOpError("must be inside ascvf.vec_scope block");
    }
    return success();
}

//===----------------------------------------------------------------------===//
// VFGroupOp
//===----------------------------------------------------------------------===//

LogicalResult VFGroupOp::verify()
{
    auto result = walk([&](Operation* op) {
        if (!hasStaticShapes(op)) {
            op->emitOpError("inside asvf.vf_group must have static shape");
            return WalkResult::interrupt();
        }
        return WalkResult::advance();
    });
    return failure(result.wasInterrupted());
}

//===----------------------------------------------------------------------===//
// VFForOp
//===----------------------------------------------------------------------===//

void VFForOp::build(OpBuilder& builder, OperationState& state, Value upperBound)
{
    OpBuilder::InsertionGuard guard(builder);
    state.addOperands(upperBound);
    Type type = builder.getIndexType();
    Region* bodyRegion = state.addRegion();
    Block* bodyBlock = builder.createBlock(bodyRegion);
    bodyBlock->addArgument(type, state.location);
    ensureTerminator(*bodyRegion, builder, state.location);
}

LogicalResult VFForOp::canonicalize(VFForOp op, PatternRewriter& rewriter)
{
    Block& block = op.getRegion().front();
    if (block.without_terminator().empty()) {
        rewriter.eraseOp(op);
        return success();
    }
    if (auto ub = getConstantIntValue(op.getUpperBound()); ub && ub.value() == 0) {
        rewriter.eraseOp(op);
        return success();
    }
    return failure();
}

LogicalResult VFForOp::verify()
{
    if (getBody()->getArguments().size() != 1)
        return emitOpError("block must have one argument");
    if (!getOperation()->getParentOfType<ascvf::VecScopeOp>()) {
        return emitOpError("must be inside ascvf.vec_scope block");
    }
    return success();
}

SmallVector<Region*> VFForOp::getLoopRegions()
{
    SmallVector<Region*> regions;
    regions.push_back(&getRegion());
    return regions;
}

//===----------------------------------------------------------------------===//
// AscVFDialect
//===----------------------------------------------------------------------===//

void AscVFDialect::registerOps()
{
    addOperations<
#define GET_OP_LIST
#include "ascir/Dialect/AscVF/IR/AscVFOps.cpp.inc"
        >();
}
