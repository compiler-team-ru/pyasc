/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASCIR_TARGET_ASC_BASIC_MICRO_API_H
#define ASCIR_TARGET_ASC_BASIC_MICRO_API_H

#include "ascir/Target/Asc/Common.h"

namespace mlir {
namespace ascendc {

//===----------------------------------------------------------------------===//
// Binary MicroAPI operations
//===----------------------------------------------------------------------===//

template <typename BinaryMicroOp>
LogicalResultForT<
    BinaryMicroOp, ascendc::AddMicroOp, ascendc::AndMicroOp, ascendc::DivMicroOp, ascendc::FusedAbsSubMicroOp,
    ascendc::FusedExpSubMicroOp, ascendc::FusedMulDstAddMicroOp, ascendc::SubMicroOp, ascendc::MaxMicroOp,
    ascendc::MinMicroOp, ascendc::MulMicroOp, ascendc::MulAddDstMicroOp, ascendc::OrMicroOp, ascendc::PreluMicroOp,
    ascendc::XorMicroOp>
printOperation(CodeEmitter& emitter, BinaryMicroOp op)
{
    auto& os = emitter.ostream();
    os << ascNamespace << "::" << op.getAPIName() << "(" << emitter.getOrCreateName(op.getDstReg()) << ", "
       << emitter.getOrCreateName(op.getSrc0Reg()) << ", " << emitter.getOrCreateName(op.getSrc1Reg()) << ", "
       << emitter.getOrCreateName(op.getMaskReg()) << ")";
    return success();
}

//===----------------------------------------------------------------------===//
// Unary MicroAPI operations
//===----------------------------------------------------------------------===//

template <typename UnaryMicroOp>
LogicalResultForT<
    UnaryMicroOp, ascendc::AbsMicroOp, ascendc::ExpMicroOp, ascendc::LnMicroOp, ascendc::LogMicroOp,
    ascendc::Log10MicroOp, ascendc::MaskNotMicroOp, ascendc::NegMicroOp, ascendc::NotMicroOp, ascendc::ReluMicroOp,
    ascendc::SqrtMicroOp>
printOperation(CodeEmitter& emitter, UnaryMicroOp op)
{
    auto& os = emitter.ostream();
    os << ascNamespace << "::" << op.getAPIName() << "(" << emitter.getOrCreateName(op.getDstReg()) << ", "
       << emitter.getOrCreateName(op.getSrcReg()) << ", " << emitter.getOrCreateName(op.getMaskReg()) << ")";
    return success();
}

//===----------------------------------------------------------------------===//
// Other MicroAPI operations
//===----------------------------------------------------------------------===//

LogicalResult printOperation(CodeEmitter& emitter, ascendc::DataCopyLoadOp op);

LogicalResult printOperation(CodeEmitter& emitter, ascendc::DataCopyStoreOp op);

LogicalResult printOperation(CodeEmitter& emitter, ascendc::UpdateMaskOp op);

LogicalResult printOperation(CodeEmitter& emitter, ascendc::RegTensorOp op);

} // namespace ascendc
} // namespace mlir

#endif // ASCIR_TARGET_ASC_BASIC_MICRO_API_H
