// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: ascir-opt -asclower-math -canonicalize %s | FileCheck %s

// CHECK-LABEL: func.func @lower_log(%arg0: tensor<777xf32, #asctile.local<UB>>) -> tensor<777xf32, #asctile.local<UB>> {
// CHECK:       %0 = builtin.unrealized_conversion_cast %arg0 : tensor<777xf32, #asctile.local<UB>> to !ascendc.local_tensor<777xf32>
// CHECK-NEXT:  %1 = ascendc.local_tensor_auto veccalc() : <777xf32>
// CHECK-NEXT:  %2 = builtin.unrealized_conversion_cast %1 : !ascendc.local_tensor<777xf32> to tensor<777xf32, #asctile.local<UB>>
// CHECK-NEXT:  ascendc.ln_l2 %1, %0, %c777_i64 : !ascendc.local_tensor<777xf32>, !ascendc.local_tensor<777xf32>, i64
// CHECK-NEXT:  return %2 : tensor<777xf32, #asctile.local<UB>>
// CHECK-NEXT:}
func.func @lower_log(%arg0: tensor<777xf32, #asctile.local<UB>>) -> tensor<777xf32, #asctile.local<UB>> {
  %0 = math.log %arg0 : tensor<777xf32, #asctile.local<UB>>
  return %0 : tensor<777xf32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @lower_log2(%arg0: tensor<777xf32, #asctile.local<UB>>) -> tensor<777xf32, #asctile.local<UB>> {
// CHECK:       %0 = builtin.unrealized_conversion_cast %arg0 : tensor<777xf32, #asctile.local<UB>> to !ascendc.local_tensor<777xf32>
// CHECK-NEXT:  %1 = ascendc.local_tensor_auto veccalc() : <777xf32>
// CHECK-NEXT:  %2 = builtin.unrealized_conversion_cast %1 : !ascendc.local_tensor<777xf32> to tensor<777xf32, #asctile.local<UB>>
// CHECK-NEXT:  ascendc.log2 %1, %0, %c777_i64, %false {operandSegmentSizes = array<i32: 1, 1, 0, 1, 1>} : !ascendc.local_tensor<777xf32>, !ascendc.local_tensor<777xf32>, i64, i1
// CHECK-NEXT:  return %2 : tensor<777xf32, #asctile.local<UB>>
// CHECK-NEXT:}
func.func @lower_log2(%arg0: tensor<777xf32, #asctile.local<UB>>) -> tensor<777xf32, #asctile.local<UB>> {
  %0 = math.log2 %arg0 : tensor<777xf32, #asctile.local<UB>>
  return %0 : tensor<777xf32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @lower_erf(%arg0: tensor<777xf32, #asctile.local<UB>>) -> tensor<777xf32, #asctile.local<UB>> {
// CHECK:       %0 = builtin.unrealized_conversion_cast %arg0 : tensor<777xf32, #asctile.local<UB>> to !ascendc.local_tensor<777xf32>
// CHECK-NEXT:  %1 = ascendc.local_tensor_auto veccalc() : <777xf32>
// CHECK-NEXT:  %2 = builtin.unrealized_conversion_cast %1 : !ascendc.local_tensor<777xf32> to tensor<777xf32, #asctile.local<UB>>
// CHECK-NEXT:  ascendc.erf %1, %0, %c777_i64, %false {operandSegmentSizes = array<i32: 1, 1, 0, 1, 1>} : !ascendc.local_tensor<777xf32>, !ascendc.local_tensor<777xf32>, i64, i1
// CHECK-NEXT:  return %2 : tensor<777xf32, #asctile.local<UB>>
// CHECK-NEXT:}
func.func @lower_erf(%arg0: tensor<777xf32, #asctile.local<UB>>) -> tensor<777xf32, #asctile.local<UB>> {
  %0 = math.erf %arg0 : tensor<777xf32, #asctile.local<UB>>
  return %0 : tensor<777xf32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @lower_asin(%arg0: tensor<777xf32, #asctile.local<UB>>) -> tensor<777xf32, #asctile.local<UB>> {
// CHECK:       %0 = builtin.unrealized_conversion_cast %arg0 : tensor<777xf32, #asctile.local<UB>> to !ascendc.local_tensor<777xf32>
// CHECK-NEXT:  %1 = ascendc.local_tensor_auto veccalc() : <777xf32>
// CHECK-NEXT:  %2 = builtin.unrealized_conversion_cast %1 : !ascendc.local_tensor<777xf32> to tensor<777xf32, #asctile.local<UB>>
// CHECK-NEXT:  ascendc.asin %1, %0, %c777_i64, %false {operandSegmentSizes = array<i32: 1, 1, 0, 1, 1>} : !ascendc.local_tensor<777xf32>, !ascendc.local_tensor<777xf32>, i64, i1
// CHECK-NEXT:  return %2 : tensor<777xf32, #asctile.local<UB>>
// CHECK-NEXT:}
func.func @lower_asin(%arg0: tensor<777xf32, #asctile.local<UB>>) -> tensor<777xf32, #asctile.local<UB>> {
  %0 = math.asin %arg0 : tensor<777xf32, #asctile.local<UB>>
  return %0 : tensor<777xf32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @lower_exp(%arg0: tensor<777xf32, #asctile.local<UB>>) -> tensor<777xf32, #asctile.local<UB>> {
// CHECK:       %0 = builtin.unrealized_conversion_cast %arg0 : tensor<777xf32, #asctile.local<UB>> to !ascendc.local_tensor<777xf32>
// CHECK-NEXT:  %1 = ascendc.local_tensor_auto veccalc() : <777xf32>
// CHECK-NEXT:  %2 = builtin.unrealized_conversion_cast %1 : !ascendc.local_tensor<777xf32> to tensor<777xf32, #asctile.local<UB>>
// CHECK-NEXT:  ascendc.exp_l2 %1, %0, %c777_i64 : !ascendc.local_tensor<777xf32>, !ascendc.local_tensor<777xf32>, i64
// CHECK-NEXT:  return %2 : tensor<777xf32, #asctile.local<UB>>
// CHECK-NEXT:}
func.func @lower_exp(%arg0: tensor<777xf32, #asctile.local<UB>>) -> tensor<777xf32, #asctile.local<UB>> {
  %0 = math.exp %arg0 : tensor<777xf32, #asctile.local<UB>>
  return %0 : tensor<777xf32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @lower_cos(%arg0: tensor<777xf32, #asctile.local<UB>>) -> tensor<777xf32, #asctile.local<UB>> {
// CHECK:       %0 = builtin.unrealized_conversion_cast %arg0 : tensor<777xf32, #asctile.local<UB>> to !ascendc.local_tensor<777xf32>
// CHECK-NEXT:  %1 = ascendc.local_tensor_auto veccalc() : <777xf32>
// CHECK-NEXT:  %2 = builtin.unrealized_conversion_cast %1 : !ascendc.local_tensor<777xf32> to tensor<777xf32, #asctile.local<UB>>
// CHECK-NEXT:  ascendc.cos %1, %0, %c777_i64, %false {operandSegmentSizes = array<i32: 1, 1, 0, 1, 1>} : !ascendc.local_tensor<777xf32>, !ascendc.local_tensor<777xf32>, i64, i1
// CHECK-NEXT:  return %2 : tensor<777xf32, #asctile.local<UB>>
// CHECK-NEXT:}
func.func @lower_cos(%arg0: tensor<777xf32, #asctile.local<UB>>) -> tensor<777xf32, #asctile.local<UB>> {
  %0 = math.cos %arg0 : tensor<777xf32, #asctile.local<UB>>
  return %0 : tensor<777xf32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @lower_sin(%arg0: tensor<777xf32, #asctile.local<UB>>) -> tensor<777xf32, #asctile.local<UB>> {
// CHECK:       %0 = builtin.unrealized_conversion_cast %arg0 : tensor<777xf32, #asctile.local<UB>> to !ascendc.local_tensor<777xf32>
// CHECK-NEXT:  %1 = ascendc.local_tensor_auto veccalc() : <777xf32>
// CHECK-NEXT:  %2 = builtin.unrealized_conversion_cast %1 : !ascendc.local_tensor<777xf32> to tensor<777xf32, #asctile.local<UB>>
// CHECK-NEXT:  ascendc.sin %1, %0, %c777_i64, %false {operandSegmentSizes = array<i32: 1, 1, 0, 1, 1>} : !ascendc.local_tensor<777xf32>, !ascendc.local_tensor<777xf32>, i64, i1
// CHECK-NEXT:  return %2 : tensor<777xf32, #asctile.local<UB>>
// CHECK-NEXT:}
func.func @lower_sin(%arg0: tensor<777xf32, #asctile.local<UB>>) -> tensor<777xf32, #asctile.local<UB>> {
  %0 = math.sin %arg0 : tensor<777xf32, #asctile.local<UB>>
  return %0 : tensor<777xf32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @lower_sqrt(%arg0: tensor<777xf32, #asctile.local<UB>>) -> tensor<777xf32, #asctile.local<UB>> {
// CHECK:       %0 = builtin.unrealized_conversion_cast %arg0 : tensor<777xf32, #asctile.local<UB>> to !ascendc.local_tensor<777xf32>
// CHECK-NEXT:  %1 = ascendc.local_tensor_auto veccalc() : <777xf32>
// CHECK-NEXT:  %2 = builtin.unrealized_conversion_cast %1 : !ascendc.local_tensor<777xf32> to tensor<777xf32, #asctile.local<UB>>
// CHECK-NEXT:  ascendc.sqrt_l2 %1, %0, %c777_i64 : !ascendc.local_tensor<777xf32>, !ascendc.local_tensor<777xf32>, i64
// CHECK-NEXT:  return %2 : tensor<777xf32, #asctile.local<UB>>
// CHECK-NEXT:}
func.func @lower_sqrt(%arg0: tensor<777xf32, #asctile.local<UB>>) -> tensor<777xf32, #asctile.local<UB>> {
  %0 = math.sqrt %arg0 : tensor<777xf32, #asctile.local<UB>>
  return %0 : tensor<777xf32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @lower_absf(%arg0: tensor<777xf32, #asctile.local<UB>>) -> tensor<777xf32, #asctile.local<UB>> {
// CHECK:       %0 = builtin.unrealized_conversion_cast %arg0 : tensor<777xf32, #asctile.local<UB>> to !ascendc.local_tensor<777xf32>
// CHECK-NEXT:  %1 = ascendc.local_tensor_auto veccalc() : <777xf32>
// CHECK-NEXT:  %2 = builtin.unrealized_conversion_cast %1 : !ascendc.local_tensor<777xf32> to tensor<777xf32, #asctile.local<UB>>
// CHECK-NEXT:  ascendc.abs_l2 %1, %0, %c777_i64 : !ascendc.local_tensor<777xf32>, !ascendc.local_tensor<777xf32>, i64
// CHECK-NEXT:  return %2 : tensor<777xf32, #asctile.local<UB>>
// CHECK-NEXT:}
func.func @lower_absf(%arg0: tensor<777xf32, #asctile.local<UB>>) -> tensor<777xf32, #asctile.local<UB>> {
  %0 = math.absf %arg0 : tensor<777xf32, #asctile.local<UB>>
  return %0 : tensor<777xf32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @lower_ceil(%arg0: tensor<777xf32, #asctile.local<UB>>) -> tensor<777xf32, #asctile.local<UB>> {
// CHECK:       %0 = builtin.unrealized_conversion_cast %arg0 : tensor<777xf32, #asctile.local<UB>> to !ascendc.local_tensor<777xf32>
// CHECK-NEXT:  %1 = ascendc.local_tensor_auto veccalc() : <777xf32>
// CHECK-NEXT:  %2 = builtin.unrealized_conversion_cast %1 : !ascendc.local_tensor<777xf32> to tensor<777xf32, #asctile.local<UB>>
// CHECK-NEXT:  ascendc.ceil %1, %0, %c777_i64, %false {operandSegmentSizes = array<i32: 1, 1, 0, 1, 1>} : !ascendc.local_tensor<777xf32>, !ascendc.local_tensor<777xf32>, i64, i1
// CHECK-NEXT:  return %2 : tensor<777xf32, #asctile.local<UB>>
// CHECK-NEXT:}
func.func @lower_ceil(%arg0: tensor<777xf32, #asctile.local<UB>>) -> tensor<777xf32, #asctile.local<UB>> {
  %0 = math.ceil %arg0 : tensor<777xf32, #asctile.local<UB>>
  return %0 : tensor<777xf32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @lower_floor(%arg0: tensor<777xf32, #asctile.local<UB>>) -> tensor<777xf32, #asctile.local<UB>> {
// CHECK:       %0 = builtin.unrealized_conversion_cast %arg0 : tensor<777xf32, #asctile.local<UB>> to !ascendc.local_tensor<777xf32>
// CHECK-NEXT:  %1 = ascendc.local_tensor_auto veccalc() : <777xf32>
// CHECK-NEXT:  %2 = builtin.unrealized_conversion_cast %1 : !ascendc.local_tensor<777xf32> to tensor<777xf32, #asctile.local<UB>>
// CHECK-NEXT:  ascendc.floor %1, %0, %c777_i64, %false {operandSegmentSizes = array<i32: 1, 1, 0, 1, 1>} : !ascendc.local_tensor<777xf32>, !ascendc.local_tensor<777xf32>, i64, i1
// CHECK-NEXT:  return %2 : tensor<777xf32, #asctile.local<UB>>
// CHECK-NEXT:}
func.func @lower_floor(%arg0: tensor<777xf32, #asctile.local<UB>>) -> tensor<777xf32, #asctile.local<UB>> {
  %0 = math.floor %arg0 : tensor<777xf32, #asctile.local<UB>>
  return %0 : tensor<777xf32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @lower_round(%arg0: tensor<777xf32, #asctile.local<UB>>) -> tensor<777xf32, #asctile.local<UB>> {
// CHECK:       %0 = builtin.unrealized_conversion_cast %arg0 : tensor<777xf32, #asctile.local<UB>> to !ascendc.local_tensor<777xf32>
// CHECK-NEXT:  %1 = ascendc.local_tensor_auto veccalc() : <777xf32>
// CHECK-NEXT:  %2 = builtin.unrealized_conversion_cast %1 : !ascendc.local_tensor<777xf32> to tensor<777xf32, #asctile.local<UB>>
// CHECK-NEXT:  ascendc.round %1, %0, %c777_i64, %false {operandSegmentSizes = array<i32: 1, 1, 0, 1, 1>} : !ascendc.local_tensor<777xf32>, !ascendc.local_tensor<777xf32>, i64, i1
// CHECK-NEXT:  return %2 : tensor<777xf32, #asctile.local<UB>>
// CHECK-NEXT:}
func.func @lower_round(%arg0: tensor<777xf32, #asctile.local<UB>>) -> tensor<777xf32, #asctile.local<UB>> {
  %0 = math.round %arg0 : tensor<777xf32, #asctile.local<UB>>
  return %0 : tensor<777xf32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @lower_acos(%arg0: tensor<777xf32, #asctile.local<UB>>) -> tensor<777xf32, #asctile.local<UB>> {
// CHECK:       %0 = builtin.unrealized_conversion_cast %arg0 : tensor<777xf32, #asctile.local<UB>> to !ascendc.local_tensor<777xf32>
// CHECK-NEXT:  %1 = ascendc.local_tensor_auto veccalc() : <777xf32>
// CHECK-NEXT:  %2 = builtin.unrealized_conversion_cast %1 : !ascendc.local_tensor<777xf32> to tensor<777xf32, #asctile.local<UB>>
// CHECK-NEXT:  ascendc.acos %1, %0, %c777_i64, %false {operandSegmentSizes = array<i32: 1, 1, 0, 1, 1>} : !ascendc.local_tensor<777xf32>, !ascendc.local_tensor<777xf32>, i64, i1
// CHECK-NEXT:  return %2 : tensor<777xf32, #asctile.local<UB>>
// CHECK-NEXT:}
func.func @lower_acos(%arg0: tensor<777xf32, #asctile.local<UB>>) -> tensor<777xf32, #asctile.local<UB>> {
  %0 = math.acos %arg0 : tensor<777xf32, #asctile.local<UB>>
  return %0 : tensor<777xf32, #asctile.local<UB>>
}
