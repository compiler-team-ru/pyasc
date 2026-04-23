/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ascir/Dialect/EmitAsc/IR/EmitAsc.h"

#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"

#define GET_OP_CLASSES
#include "ascir/Dialect/EmitAsc/IR/EmitAscOps.cpp.inc"

using namespace mlir;
using namespace mlir::emitasc;

//===----------------------------------------------------------------------===//
// PtrOffsetOp
//===----------------------------------------------------------------------===//

Value PtrOffsetOp::getViewSource() { return getBase(); }

OpFoldResult PtrOffsetOp::fold(FoldAdaptor adaptor)
{
    if (auto offset = getStaticOffset())
        return offset->isZero() ? getBase() : nullptr;
    if (auto offset = getDynamicOffset())
        return isConstantIntValue(offset, 0) ? getBase() : nullptr;
    return nullptr;
}

//===----------------------------------------------------------------------===//
// ReinterpretCastOp
//===----------------------------------------------------------------------===//

bool ReinterpretCastOp::areCastCompatible(TypeRange inputs, TypeRange outputs)
{
    return inputs.size() == 1U && outputs.size() == 1U;
}

//===----------------------------------------------------------------------===//
// VariableOp
//===----------------------------------------------------------------------===//

bool VariableOp::isStatic() { return getStaticInit().has_value(); }

OpFoldResult VariableOp::getInit(bool fold)
{
    auto dynamicInit = getDynamicInit();
    if (dynamicInit)
        return fold ? getAsOpFoldResult(dynamicInit) : dynamicInit;
    auto staticInit = getStaticInit();
    assert(staticInit.has_value() && "either static or dynamic init must exist");
    return staticInit.value();
}

//===----------------------------------------------------------------------===//
// EmitAscDialect
//===----------------------------------------------------------------------===//

void EmitAscDialect::registerOps()
{
    addOperations<
#define GET_OP_LIST
#include "ascir/Dialect/EmitAsc/IR/EmitAscOps.cpp.inc"
        >();
}
