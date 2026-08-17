// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: ascir-opt -asclower-realize-conversion-cast %s | FileCheck %s

// CHECK-LABEL: func.func @direct_bit_cast(%arg0: !ascendc.local_tensor<32xf32>) -> !ascendc.local_tensor<16x2xi32> {
// CHECK-NEXT:  %0 = ascendc.reinterpret_cast %arg0 : !ascendc.local_tensor<32xf32> to !ascendc.local_tensor<16x2xi32>
// CHECK-NEXT:  return %0 : !ascendc.local_tensor<16x2xi32>
// CHECK-NEXT:}
func.func @direct_bit_cast(%arg0: !ascendc.local_tensor<32xf32>) -> !ascendc.local_tensor<16x2xi32> {
  %0 = builtin.unrealized_conversion_cast %arg0 : !ascendc.local_tensor<32xf32> to !ascendc.local_tensor<16x2xi32>
  return %0 : !ascendc.local_tensor<16x2xi32>
}

// CHECK-LABEL: func.func @indirect_sign_cast(%arg0: !ascendc.local_tensor<32xi32>) -> !ascendc.local_tensor<32xui32> {
// CHECK-NEXT:  %0 = ascendc.reinterpret_cast %arg0 : !ascendc.local_tensor<32xi32> to !ascendc.local_tensor<32xui32>
// CHECK-NEXT:  return %0 : !ascendc.local_tensor<32xui32>
// CHECK-NEXT:}
func.func @indirect_sign_cast(%arg0: !ascendc.local_tensor<32xi32>) -> !ascendc.local_tensor<32xui32> {
  %0 = builtin.unrealized_conversion_cast %arg0 : !ascendc.local_tensor<32xi32> to vector<32xi32>
  %1 = builtin.unrealized_conversion_cast %0 : vector<32xi32> to !ascendc.local_tensor<32xui32>
  return %1 : !ascendc.local_tensor<32xui32>
}

// CHECK-LABEL: func.func @noop_three_cast(%arg0: !ascendc.local_tensor<32xi32>) -> !ascendc.local_tensor<32xi32> {
// CHECK-NEXT:  return %arg0 : !ascendc.local_tensor<32xi32>
// CHECK-NEXT:}
func.func @noop_three_cast(%arg0: !ascendc.local_tensor<32xi32>) -> !ascendc.local_tensor<32xi32> {
  %0 = builtin.unrealized_conversion_cast %arg0 : !ascendc.local_tensor<32xi32> to vector<32xi32>
  %1 = builtin.unrealized_conversion_cast %0 : vector<32xi32> to memref<32xf32, 21>
  %2 = builtin.unrealized_conversion_cast %1 : memref<32xf32, 21> to !ascendc.local_tensor<32xi32>
  return %2 : !ascendc.local_tensor<32xi32>
}
