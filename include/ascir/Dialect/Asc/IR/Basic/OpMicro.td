/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASC_BASIC_OP_MICROAPI_TD
#define ASC_BASIC_OP_MICROAPI_TD

include "Base.td"
include "Core/Attributes.td"
include "Core/Interfaces.td"
include "Core/Types.td"

include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Binary MicroAPI operations
//===----------------------------------------------------------------------===//

def AddMicroOp : BinaryMicroOp<"add", "Add">;
def AndMicroOp : BinaryMicroOp<"and", "And">;
def DivMicroOp : BinaryMicroOp<"div", "Div">;
def FusedAbsSubMicroOp : BinaryMicroOp<"fused_abs_sub", "FusedAbsSub">;
def FusedExpSubMicroOp : BinaryMicroOp<"fused_exp_sub", "FusedExpSub">;
def FusedMulDstAddMicroOp : BinaryMicroOp<"fused_mul_dst_add", "FusedMulDstAdd">;
def SubMicroOp : BinaryMicroOp<"sub", "Sub">;
def MaxMicroOp : BinaryMicroOp<"max", "Max">;
def MinMicroOp : BinaryMicroOp<"min", "Min">;
def MulMicroOp : BinaryMicroOp<"mul", "Mul">;
def MulAddDstMicroOp : BinaryMicroOp<"mul_add_dst", "MulAddDst">;
def OrMicroOp : BinaryMicroOp<"or", "Or">;
def PreluMicroOp : BinaryMicroOp<"prelu", "Prelu">;
def XorMicroOp : BinaryMicroOp<"xor", "Xor">;

//===----------------------------------------------------------------------===//
// Unary MicroAPI operations
//===----------------------------------------------------------------------===//

def AbsMicroOp : UnaryMicroOp<"abs", "Abs">;
def ExpMicroOp : UnaryMicroOp<"exp", "Exp">;
def LnMicroOp : UnaryMicroOp<"ln", "Ln">;
def LogMicroOp : UnaryMicroOp<"log", "Log">;
def Log10MicroOp : UnaryMicroOp<"log10", "Log10">;
def MaskNotMicroOp : UnaryMicroOp<"mask_not", "MaskNot">;
def NegMicroOp : UnaryMicroOp<"neg", "Neg">;
def NotMicroOp : UnaryMicroOp<"not", "Not">;
def ReluMicroOp : UnaryMicroOp<"relu", "Relu">;
def SqrtMicroOp : UnaryMicroOp<"sqrt", "Sqrt">;

//===----------------------------------------------------------------------===//
// Reduce MicroAPI operations
//===----------------------------------------------------------------------===//

def ReduceMaxMicroOp : MicroAPIOp<"reduce_max_micro", "ReduceMax"> {
  let arguments = (ins AscendC_RegTensor:$dstReg, AscendC_RegTensor:$srcReg, AscendC_MaskReg:$maskReg);
}

def ReduceSumMicroOp : MicroAPIOp<"reduce_sum_micro", "ReduceSum"> {
  let arguments = (ins AscendC_RegTensor:$dstReg, AscendC_RegTensor:$srcReg, AscendC_MaskReg:$maskReg);
}

def ReduceMinMicroOp : MicroAPIOp<"reduce_min_micro", "ReduceMin"> {
  let arguments = (ins AscendC_RegTensor:$dstReg, AscendC_RegTensor:$srcReg, AscendC_MaskReg:$maskReg);
}

def DuplicateMicroOp : MicroAPIOp<"duplicate_micro", "Duplicate"> {
  let arguments = (ins AscendC_RegTensor:$dstReg, AscendC_RegTensor:$srcReg, AscendC_MaskReg:$maskReg);
}

def AscendC_DuplicateScalarMicroOp : AscendC_Op<"duplicate", []> {
  let arguments = (ins AscendC_RegTensor:$dstReg, AnyType:$scalar);
  let assemblyFormat = "$dstReg `,` $scalar attr-dict `:` type($dstReg) `,` type($scalar)";
}

//===----------------------------------------------------------------------===//
// Other MicroAPI operations
//===----------------------------------------------------------------------===//

def MicroAPI_DataCopyLoadOp : MicroAPIOp<"data_copy_vld_micro", "DataCopy"> {
  let description = "Perform copying between tensors (UB to RegTensor)";
  let arguments = (ins AscendC_RegTensor:$dstReg,
                       AnyRankedOrUnrankedMemRef:$src);
}

def MicroAPI_DataCopyStoreOp : MicroAPIOp<"data_copy_vst_micro", "DataCopy"> {
  let description = "Perform copying between tensors (RegTensor to UB)";
  let arguments = (ins AnyRankedOrUnrankedMemRef:$dst,
                       AscendC_RegTensor:$srcReg,
                       AscendC_MaskReg:$maskReg);
}

def MicroAPI_UpdateMaskOp : MicroAPIOp<"update_mask", "UpdateMask"> {
  let summary = "Update reg mask";
  let comment = "The mask is updated on every iteration to match the total count";
  let arguments = (ins AnyRankedOrUnrankedMemRef:$count, TypeAttr:$type);
  let results = (outs AscendC_MaskReg:$maskReg);
  let assemblyFormat = [{
    $type `,` $count attr-dict `:` type($count)
  }];
}

def MicroAPI_RegTensorOp : MicroAPIOp<"reg_tensor", "RegTensor"> {
  let summary = "Create RegTensor";
  let results = (outs AscendC_RegTensor:$result);
  let assemblyFormat = [{
    attr-dict `:` type($result)
  }];
}

def MicroAPI_MaskRegOp : MicroAPIOp<"mask_reg", "MaskReg"> {
  let summary = "Create RegTensor";
  let results = (outs AscendC_MaskReg:$result);
  let assemblyFormat = [{
    attr-dict `:` type($result)
  }];
}

def MicroAPI_CreateMaskOp : MicroAPIOp<"create_mask", "CreateMask", [Pure]> {
  let summary = "Create RegTensor";
  let arguments = (ins TypeAttr:$dtype, AscendC_MaskPattern:$mask);
  let results = (outs AscendC_MaskReg:$result);
  let assemblyFormat = [{
    $dtype `,` $mask attr-dict `:` type($result)
  }];
}

#endif // ASC_BASIC_OP_MICROAPI_TD
