/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASC_INTERFACES_TD
#define ASC_INTERFACES_TD

include "mlir/IR/OpBase.td"

def AscendC_CalCount : AnyTypeOf<[I8, I16, I32, I64]>;

class GetMethod<string fieldName, string methodName, string returnType = "::mlir::Value">
    : InterfaceMethod<"Obtain `" # fieldName # "` parameter", returnType, methodName>;

class GetMutableMethod<string fieldName, string methodName>
    : InterfaceMethod<"Obtain the mutable `" # fieldName # "` operand", "::mlir::OpOperand&", methodName>;

class AscendC_InterfaceMethods {
  // Method list should be kept sorted by name.
  InterfaceMethod getAPIName = InterfaceMethod<"Obtain Ascend C library name", "::llvm::StringRef", "getAPIName">;
  InterfaceMethod getCalCount = GetMethod<"calCount", "getCalCount">;
  InterfaceMethod getCalCountMutable = GetMutableMethod<"calCount", "getCalCountMutable">;
  InterfaceMethod getComment = InterfaceMethod<"Obtain describing comment", "::llvm::StringRef", "getComment">;
  InterfaceMethod getDst = GetMethod<"dst", "getDst">;
  InterfaceMethod getDstBlkStride = GetMethod<"dstBlkStride", "getDstBlkStride">;
  InterfaceMethod getDstBlkStrideMutable = GetMutableMethod<"dstBlkStride", "getDstBlkStrideMutable">;
  InterfaceMethod getDstRepStride = GetMethod<"dstRepStride", "getDstRepStride">;
  InterfaceMethod getDstRepStrideMutable = GetMutableMethod<"dstRepStride", "getDstRepStrideMutable">;
  InterfaceMethod getMask = GetMethod<"mask", "getMask">;
  InterfaceMethod getMaskMutable = GetMutableMethod<"mask", "getMaskMutable">;
  InterfaceMethod getParams = GetMethod<"params", "getParams">;
  InterfaceMethod getPattern = GetMethod<"pattern", "getPattern", "::mlir::ascendc::ReducePattern">;
  InterfaceMethod getRepeatParams = GetMethod<"repeatParams", "getRepeatParams">;
  InterfaceMethod getRepeatParamsMutable = GetMutableMethod<"repeatParams", "getRepeatParamsMutable">;
  InterfaceMethod getRepeatTimes = GetMethod<"repeatTimes", "getRepeatTimes">;
  InterfaceMethod getRepeatTimesMutable = GetMutableMethod<"repeatTimes", "getRepeatTimesMutable">;
  InterfaceMethod getIsSetMask = GetMethod<"isSetMask", "getIsSetMask", "bool">;
  InterfaceMethod getScalar = GetMethod<"scalar", "getScalar">;
  InterfaceMethod getScalarMutable = GetMutableMethod<"scalar", "getScalarMutable">;
  InterfaceMethod getSharedTmpBuffer = GetMethod<"sharedTmpBuffer", "getSharedTmpBuffer">;
  InterfaceMethod getSrc = GetMethod<"src", "getSrc">;
  InterfaceMethod getSrcBlkStride = GetMethod<"srcBlkStride", "getSrcBlkStride">;
  InterfaceMethod getSrcRepStride = GetMethod<"srcRepStride", "getSrcRepStride">;
  InterfaceMethod getSrc0 = GetMethod<"src0", "getSrc0">;
  InterfaceMethod getSrc0BlkStride = GetMethod<"src0BlkStride", "getSrc0BlkStride">;
  InterfaceMethod getSrc0RepStride = GetMethod<"src0RepStride", "getSrc0RepStride">;
  InterfaceMethod getSrc1 = GetMethod<"src1", "getSrc1">;
  InterfaceMethod getSrc1BlkStride = GetMethod<"src1BlkStride", "getSrc1BlkStride">;
  InterfaceMethod getSrc1RepStride = GetMethod<"src1RepStride", "getSrc1RepStride">;
  InterfaceMethod getSrcShape = GetMethod<"srcShape", "getSrcShape", "mlir::OperandRange">;
  InterfaceMethod getIsReuseSource = GetMethod<"isReuseSource", "getIsReuseSource">;
  InterfaceMethod getIsExhaustedSuspension = GetMethod<"isExhaustedSuspension",
    "getIsExhaustedSuspension", "bool">;
  InterfaceMethod getIsFullSort = GetMethod<"isFullSort",
    "getIsFullSort", "bool">;
  InterfaceMethod getVdeq = GetMethod<"vdeq", "getVdeq">;
  InterfaceMethod getVdeqInfo = GetMethod<"vdeqInfo", "getVdeqInfo">;
}

class AscendC_OpInterface<string name, list<Interface> baseInterfaces = []>
    : OpInterface<name, baseInterfaces>, AscendC_InterfaceMethods {
  let cppNamespace = "::mlir::ascendc";
}

//===----------------------------------------------------------------------===//
// Ascend C API interfaces
//===----------------------------------------------------------------------===//

def APIOpInterface : AscendC_OpInterface<"APIOp"> {
  let description = "Base interface for operations representing Ascend C API";
  let methods = [getAPIName, getComment];
}

def OpWithDstInterface : AscendC_OpInterface<"OpWithDst"> {
  let description = "Operation with `dst` operand (typically tensor)";
  let methods = [getDst];
}

def OpWithSrcInterface : AscendC_OpInterface<"OpWithSrc"> {
  let description = "Operation with source tensor operands";
  let methods = [InterfaceMethod<"Get source tensors", "::llvm::SmallVector<::mlir::Value>", "getSrcTensors">];
}

def VectorOpInterface : AscendC_OpInterface<"VectorOp", [APIOpInterface]> {
  let description = "Base interface for vector operations";
}

def DataCopyOpInterface :
    AscendC_OpInterface<"DataCopyOp", [APIOpInterface, OpWithDstInterface]> {
  InterfaceMethod getDirection = InterfaceMethod<"Get copy direction",
    "::mlir::ascendc::CopyDirection", "getDirection", (ins), "", [{
    auto dstType = $_op.getDst().getType();
    auto srcType = $_op.getSrc().getType();
    if (isa<GlobalTensorType>(srcType)) {
      if (isa<GlobalTensorType>(dstType))
        return CopyDirection::gm_gm;
      if (isa<LocalTensorType>(dstType))
        return CopyDirection::gm_ubuf;
    } else if (isa<LocalTensorType>(srcType)) {
      if (isa<GlobalTensorType>(dstType))
        return CopyDirection::ubuf_gm;
      if (isa<LocalTensorType>(dstType))
        return CopyDirection::ubuf_ubuf;
    }
    return CopyDirection::Unknown;
  }]>;
  InterfaceMethod setSrc = InterfaceMethod<"Set src operand", "void", "setSrc",
                                           (ins "Value":$src), "", [{
    $_op.getSrcMutable().assign(src);
  }]>;

  let description = "Data copy operation";
  let methods = [getSrc, getDirection, setSrc];
}

//===----------------------------------------------------------------------===//
// Vector unary operations
//===----------------------------------------------------------------------===//

def UnaryOpInterface : AscendC_OpInterface<"UnaryOp", [
  VectorOpInterface, OpWithDstInterface, OpWithSrcInterface
]> {
  let description = "Vector unary operation";
  let methods = [getSrc];
}

def UnaryL0OpInterface : AscendC_OpInterface<"UnaryL0Op", [UnaryOpInterface]> {
  let description = "Vector unary operation (L0 API)";
  let methods = [
    getMask, getRepeatTimes, getRepeatParams,
    getMaskMutable, getRepeatTimesMutable, getRepeatParamsMutable,
  ];
}

def UnaryL2OpInterface : AscendC_OpInterface<"UnaryL2Op", [UnaryOpInterface]> {
  let description = "Vector unary operation (L2 API)";
  let methods = [getCalCount, getCalCountMutable];
}

//===----------------------------------------------------------------------===//
// Vector binary operations
//===----------------------------------------------------------------------===//

def BinaryOpInterface : AscendC_OpInterface<"BinaryOp", [
  VectorOpInterface, OpWithDstInterface, OpWithSrcInterface
]> {
  let description = "Vector binary operation";
  let methods = [getSrc0, getSrc1];
}

def BinaryL0OpInterface : AscendC_OpInterface<"BinaryL0Op", [BinaryOpInterface]> {
  let description = "Vector binary operation (L0 API)";
  let methods = [
    getMask, getRepeatTimes, getRepeatParams,
    getMaskMutable, getRepeatTimesMutable, getRepeatParamsMutable,
  ];
}

def BinaryL2OpInterface : AscendC_OpInterface<"BinaryL2Op", [BinaryOpInterface]> {
  let description = "Vector binary operation (L2 API)";
  let methods = [getCalCount, getCalCountMutable];
}

def BinaryL3OpInterface : AscendC_OpInterface<"BinaryL3Op", [BinaryOpInterface]> {
  let description = "Vector binary operation (L3 API)";
}

//===----------------------------------------------------------------------===//
// Vector-scalar operations
//===----------------------------------------------------------------------===//

def VecScalarOpInterface : AscendC_OpInterface<"VecScalarOp", [
  VectorOpInterface, OpWithDstInterface, OpWithSrcInterface
]> {
  let description = "Vector-scalar operation";
  let methods = [getSrc, getScalar, getScalarMutable];
}

def VecScalarL0OpInterface : AscendC_OpInterface<"VecScalarL0Op", [VecScalarOpInterface]> {
  let description = "Vector-scalar operation (L0 API)";
  let methods = [getMask, getRepeatTimes, getRepeatParams,
                 getMaskMutable, getRepeatTimesMutable, getRepeatParamsMutable];
}

def VecScalarL2OpInterface : AscendC_OpInterface<"VecScalarL2Op", [VecScalarOpInterface]> {
  let description = "Vector-scalar operation (L2 API)";
  let methods = [getCalCount, getCalCountMutable];
}

//===----------------------------------------------------------------------===//
// Math library operations
//===----------------------------------------------------------------------===//

def MathLibraryOpInterface : AscendC_OpInterface<"MathLibraryOp", [
  VectorOpInterface, OpWithDstInterface, OpWithSrcInterface
]> {
  let description = "Math library operation";
}

def UnaryMathOpInterface : AscendC_OpInterface<"UnaryMathOp", [MathLibraryOpInterface]> {
  let description = "Unary math library operation";
  let methods = [getSrc, getSharedTmpBuffer, getCalCount, getIsReuseSource];
}

def BinaryMathOpInterface : AscendC_OpInterface<"BinaryMathOp", [MathLibraryOpInterface]> {
  let description = "Binary math library operation";
  let methods = [getSrc0, getSrc1, getSharedTmpBuffer, getCalCount, getIsReuseSource];
}

//===----------------------------------------------------------------------===//
// Data Conversion operations
//===----------------------------------------------------------------------===//

def DataConversionOpInterface
    : AscendC_OpInterface<"DataConversionOp", [APIOpInterface]> {
  let description = "Data conversion operation";
}

//===----------------------------------------------------------------------===//
// Reduction operations
//===----------------------------------------------------------------------===//

def ReduceOpInterface : AscendC_OpInterface<"ReduceOp", [
  VectorOpInterface, OpWithDstInterface, OpWithSrcInterface
]> {
    let methods = [getPattern, getSharedTmpBuffer, getSrcShape, getSrc];
}

#endif // ASC_INTERFACES_TD
