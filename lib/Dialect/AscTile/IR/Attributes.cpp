/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ascir/Dialect/AscTile/IR/AscTile.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

#include "ascir/Dialect/AscTile/IR/AscTileEnums.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "ascir/Dialect/AscTile/IR/AscTileAttributes.cpp.inc"

using namespace mlir;
using namespace mlir::asctile;

//===----------------------------------------------------------------------===//
// GlobalTensorAttr
//===----------------------------------------------------------------------===//

LogicalResult GlobalTensorAttr::verifyEncoding(
    ArrayRef<int64_t> shape, Type, function_ref<InFlightDiagnostic()> emitError) const
{
    if (shape.empty())
        return emitError() << "global tensor must have at least one dimension";
    return success();
}

//===----------------------------------------------------------------------===//
// LocalTensorAttr
//===----------------------------------------------------------------------===//

LogicalResult LocalTensorAttr::verifyEncoding(
    ArrayRef<int64_t> shape, Type, function_ref<InFlightDiagnostic()> emitError) const
{
    if (shape.empty())
        return emitError() << "local tensor must have at least one dimension";
    if (ShapedType::isDynamicShape(shape))
        return emitError() << "local tensor must have static shape";
    return success();
}

//===----------------------------------------------------------------------===//
// AscTileDialect
//===----------------------------------------------------------------------===//

void AscTileDialect::registerAttributes()
{
    addAttributes<
#define GET_ATTRDEF_LIST
#include "ascir/Dialect/AscTile/IR/AscTileAttributes.cpp.inc"
        >();
}
