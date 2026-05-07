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
#include "ascir/Dialect/Asc/Utils/Attributes.h"
#include "ascir/Dialect/Utils/Inlining.h"

#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/TypeSwitch.h"

namespace mlir {

template <typename... T>
using AllowInline = ascir::AllowlistInlinerInterface<T...>;

namespace ascendc {

int64_t getTypeSize(Type type)
{
    if (auto shaped = dyn_cast<ShapedType>(type))
        return shaped.getNumElements() * getTypeSize(shaped.getElementType());
    return type.getIntOrFloatBitWidth() / CHAR_BIT;
}

int64_t getTypeSizeCubeBlockAlign(ShapedType type)
{
    int64_t size = 1;
    for (auto dim : type.getShape()) {
        size *= llvm::alignTo<cubeBlockSize>(dim);
    }
    return size * getElementTypeSize(type);
}

int64_t getElementTypeSize(ShapedType type) { return getTypeSize(type.getElementType()); }

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
    if (lhsAnc != rhsAnc)
        return lhsAnc->isBeforeInBlock(rhsAnc);
    if (lhs->isAncestor(rhs))
        return true;
    if (rhs->isAncestor(lhs))
        return false;
    if (auto ifOp = dyn_cast<scf::IfOp>(lhsAnc))
        return ifOp.thenBlock()->findAncestorOpInBlock(*lhs);
    if (auto whileOp = dyn_cast<scf::WhileOp>(lhsAnc))
        return whileOp.getBeforeBody()->findAncestorOpInBlock(*lhs);
    return di.properlyDominates(lhs, rhs);
}

void registerInlinerInterfaces(DialectRegistry& registry)
{
    registry.addExtension(+[](MLIRContext*, BuiltinDialect* dialect) {
        dialect->addInterface<AllowInline<UnrealizedConversionCastOp>>();
    });
    registry.addExtension(+[](MLIRContext*, emitc::EmitCDialect* dialect) {
        dialect->addInterface<AllowInline<emitc::CastOp, emitc::ConstantOp>>();
    });
}

ModuleOp getModule(Operation* op)
{
    if (isa<ModuleOp>(op))
        return cast<ModuleOp>(op);
    auto mod = op->getParentOfType<ModuleOp>();
    assert(mod && "operation must be within a module");
    return mod;
}

StringRef getCompilationArch(Operation* op)
{
    if (auto attr = getModule(op)->getAttrOfType<StringAttr>(ascendc::attr::compilationArch))
        return attr.getValue();
    return {};
}

StringRef getSocVersion(Operation* op)
{
    if (auto attr = getModule(op)->getAttrOfType<StringAttr>(ascendc::attr::socVersion))
        return attr.getValue();
    return {};
}

bool isTargetArchC310(Operation* op) { return getCompilationArch(op) == "c310"; }

} // namespace ascendc
} // namespace mlir
