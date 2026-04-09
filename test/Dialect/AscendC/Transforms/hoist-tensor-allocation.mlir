// Copyright (c) 2025 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: ascir-opt -ascendc-hoist-ub-allocation=exclude-in-out %s | FileCheck %s

// CHECK-LABEL: func.func @hoist_ub_allocation_for
// CHECK:       %0 = ascendc.local_tensor_auto veccalc() : <64xf32>
// CHECK-NEXT:  scf.for %arg3 = %arg0 to %arg1 step %arg0 : i32 {
// CHECK-NEXT:    %1 = ascendc.local_tensor_auto vecin() input : <64xf32>
// CHECK-NEXT:    %2 = ascendc.local_tensor_auto vecout() output : <64xf32>
// CHECK-NEXT:    ascendc.add_l2 %2, %1, %0, %arg1 : !ascendc.local_tensor<64xf32>, !ascendc.local_tensor<64xf32>, !ascendc.local_tensor<64xf32>, i32
func.func @hoist_ub_allocation_for(%arg0 : i32, %arg1: i32, %arg2: i64) {
    scf.for %arg3 = %arg0 to %arg1 step %arg0 : i32 {
        %1 = ascendc.local_tensor_auto vecin() input : <64xf32>
        %2 = ascendc.local_tensor_auto veccalc() : <64xf32>
        %3 = ascendc.local_tensor_auto vecout() output : <64xf32>
        ascendc.add_l2 %3, %1, %2, %arg1 : !ascendc.local_tensor<64xf32>, !ascendc.local_tensor<64xf32>, !ascendc.local_tensor<64xf32>, i32
    }
    return
}

// CHECK-LABEL: func.func @hoist_ub_allocation_nested_for
// CHECK:       %0 = ascendc.local_tensor_auto veccalc() : <64xf32>
// CHECK-NEXT:  scf.for %arg4 = %arg0 to %arg1 step %arg0 : i32 {
// CHECK-NEXT:    scf.for %arg5 = %arg0 to %arg2 step %arg0 : i32 {
// CHECK-NEXT:      %1 = ascendc.local_tensor_auto vecin() input : <64xf32>
// CHECK-NEXT:      %2 = ascendc.local_tensor_auto vecout() output : <64xf32>
// CHECK-NEXT:      ascendc.add_l2 %2, %1, %0, %arg1 : !ascendc.local_tensor<64xf32>, !ascendc.local_tensor<64xf32>, !ascendc.local_tensor<64xf32>, i32
func.func @hoist_ub_allocation_nested_for(%arg0 : i32, %arg1: i32, %arg2: i32, %arg3: i64) {
    scf.for %arg4 = %arg0 to %arg1 step %arg0 : i32 {
        scf.for %arg5 = %arg0 to %arg2 step %arg0 : i32 {
            %1 = ascendc.local_tensor_auto vecin() input : <64xf32>
            %2 = ascendc.local_tensor_auto veccalc() : <64xf32>
            %3 = ascendc.local_tensor_auto vecout() output : <64xf32>
            ascendc.add_l2 %3, %1, %2, %arg1 : !ascendc.local_tensor<64xf32>, !ascendc.local_tensor<64xf32>, !ascendc.local_tensor<64xf32>, i32
        }
    }
    return
}
