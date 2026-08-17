// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: ascir-opt %s | ascir-opt
// RUN: ascir-opt %s --mlir-print-op-generic | ascir-opt

module {
  func.func @softmax_kernel(%arg0: memref<*xf32, 22>, %arg1: memref<*xf32, 22>, %arg2: i32, %arg3: i32) attributes {ascendc.aicore, ascendc.global} {
    %cst = arith.constant 0.000000e+00 : f32
    %c0_i32 = arith.constant 0 : i32
    %0 = asctile.tensor %arg0(%arg2, %arg3) : memref<*xf32, 22>, tensor<?x?xf32, #asctile.global>
    %1 = asctile.tensor %arg1(%arg2, %arg3) : memref<*xf32, 22>, tensor<?x?xf32, #asctile.global>
    %2 = ascendc.get_block_idx : i32
    %3 = ascendc.get_block_num : i32
    scf.for %arg4 = %2 to %arg2 step %3  : i32 {
      %4 = asctile.load %0[%arg4, %c0_i32], %cst : tensor<?x?xf32, #asctile.global>, tensor<1x1024xf32, #asctile.local<UB>>
      %5 = asctile.reduce_as_1d <max> %4 : tensor<1x1024xf32, #asctile.local<UB>>, f32
      %6 = tensor.splat %5 : tensor<1x1024xf32, #asctile.local<UB>>
      %7 = arith.subf %4, %6 : tensor<1x1024xf32, #asctile.local<UB>>
      %8 = math.exp %7 : tensor<1x1024xf32, #asctile.local<UB>>
      %9 = asctile.reduce_as_1d <sum> %8 : tensor<1x1024xf32, #asctile.local<UB>>, f32
      %10 = tensor.splat %9 : tensor<1x1024xf32, #asctile.local<UB>>
      %11 = arith.divf %8, %10 : tensor<1x1024xf32, #asctile.local<UB>>
      asctile.store %11, %1[%arg4, %c0_i32] : tensor<1x1024xf32, #asctile.local<UB>>, tensor<?x?xf32, #asctile.global>
    } {asctile.unroll_factor = 1 : index}
    return
  }
}
