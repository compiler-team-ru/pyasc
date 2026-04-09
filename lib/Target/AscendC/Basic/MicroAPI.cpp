/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ascir/Target/Asc/Basic/MicroAPI.h"

using namespace mlir;
using namespace mlir::ascendc;

//===----------------------------------------------------------------------===//
// Other MicroAPI operations
//===----------------------------------------------------------------------===//

LogicalResult mlir::ascendc::printOperation(CodeEmitter &emitter, ascendc::DataCopyLoadOp op)
{
    auto &os = emitter.ostream();
    os << ascNamespace << "::" << op.getAPIName() << "(" << emitter.getOrCreateName(op.getDstReg()) << ", "
       << emitter.getOrCreateName(op.getSrc()) << ")";
    return success();
}

LogicalResult mlir::ascendc::printOperation(CodeEmitter &emitter, ascendc::DataCopyStoreOp op)
{
    auto &os = emitter.ostream();
    os << ascNamespace << "::" << op.getAPIName() << "(" << emitter.getOrCreateName(op.getDst()) << ", "
       << emitter.getOrCreateName(op.getSrcReg()) << ", " << emitter.getOrCreateName(op.getMaskReg()) << ")";
    return success();
}

LogicalResult mlir::ascendc::printOperation(CodeEmitter &emitter, ascendc::UpdateMaskOp op)
{
    auto &os = emitter.ostream();
    FAIL_OR(emitter.emitVariableDeclaration(op->getResult(0), false));
    os << " = ";
    os << ascNamespace << "::" << op.getAPIName() << "<";
    FAIL_OR(emitter.emitType(op.getLoc(), op.getType()));
    os << ">(*" << emitter.getOrCreateName(op.getCount()) << ")";
    return success();
}

LogicalResult mlir::ascendc::printOperation(CodeEmitter &emitter, ascendc::RegTensorOp op)
{
    FAIL_OR(emitter.emitVariableDeclaration(op->getResult(0), false));
    return success();
}

LogicalResult mlir::ascendc::printOperation(CodeEmitter &emitter, ascendc::GetVecLenOp op)
{
    FAIL_OR(emitter.emitVariableDeclaration(op->getResult(0), false));
    auto &os = emitter.ostream();
    os << " = " << ascNamespace << "::" << op.getAPIName() << "()";
    return success();
}
