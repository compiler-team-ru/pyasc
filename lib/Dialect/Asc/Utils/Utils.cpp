/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ascir/Dialect/Asc/Utils/Utils.h"
#include "ascir/Dialect/Utils/Inlining.h"

#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Operation.h"

namespace mlir {

template <typename... T>
using AllowInline = ascir::AllowlistInlinerInterface<T...>;

namespace ascendc {

bool opPrecedes(Operation* lhs, Operation* rhs) { return lhs != rhs && lhs->isBeforeInBlock(rhs); }

bool opPrecedes(Operation* lhs, Operation* rhs, DominanceInfo& di)
{
    if (lhs == rhs) {
        return false;
    }
    Block* lhsBlk = lhs->getBlock();
    Block* rhsBlk = rhs->getBlock();
    if (lhsBlk == rhsBlk) {
        return lhs->isBeforeInBlock(rhs);
    }
    Block* dtr = di.findNearestCommonDominator(lhsBlk, rhsBlk);
    Operation* lhsAnc = dtr->findAncestorOpInBlock(*lhs);
    Operation* rhsAnc = dtr->findAncestorOpInBlock(*rhs);
    return lhsAnc->isBeforeInBlock(rhsAnc);
}

void registerInlinerInterfaces(DialectRegistry& registry)
{
    registry.addExtension(+[](MLIRContext* ctx, BuiltinDialect* dialect) {
        dialect->addInterface<AllowInline<UnrealizedConversionCastOp>>();
    });
    registry.addExtension(+[](MLIRContext* ctx, emitc::EmitCDialect* dialect) {
        dialect->addInterface<AllowInline<emitc::CastOp, emitc::ConstantOp>>();
    });
}

} // namespace ascendc
} // namespace mlir
