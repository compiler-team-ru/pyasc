// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: ascir-opt -asctile-transform-math-ops %s | FileCheck %s

// CHECK-LABEL: func.func public @addf_scalar_rhs(%arg0: tensor<32xf32, #asctile.local<UB>>) -> tensor<32xf32, #asctile.local<UB>> {
// CHECK-NEXT:  %cst = arith.constant 1.57079637 : f32
// CHECK-NEXT:  %0 = asctile.adds %arg0, %cst : tensor<32xf32, #asctile.local<UB>>
// CHECK-NEXT:  return %0 : tensor<32xf32, #asctile.local<UB>>
// CHECK-NEXT:}
func.func public @addf_scalar_rhs(%arg0: tensor<32xf32, #asctile.local<UB>>) -> tensor<32xf32, #asctile.local<UB>> {
  %cst = arith.constant dense<1.57079637> : tensor<32xf32, #asctile.local<UB>>
  %0 = arith.addf %arg0, %cst : tensor<32xf32, #asctile.local<UB>>
  return %0 : tensor<32xf32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func public @addf_scalar_lhs(%arg0: tensor<32xf32, #asctile.local<UB>>) -> tensor<32xf32, #asctile.local<UB>> {
// CHECK-NEXT:  %cst = arith.constant 1.57079637 : f32
// CHECK-NEXT:  %0 = asctile.adds %arg0, %cst : tensor<32xf32, #asctile.local<UB>>
// CHECK-NEXT:  return %0 : tensor<32xf32, #asctile.local<UB>>
// CHECK-NEXT:}
func.func public @addf_scalar_lhs(%arg0: tensor<32xf32, #asctile.local<UB>>) -> tensor<32xf32, #asctile.local<UB>> {
  %cst = arith.constant dense<1.57079637> : tensor<32xf32, #asctile.local<UB>>
  %0 = arith.addf %cst, %arg0 : tensor<32xf32, #asctile.local<UB>>
  return %0 : tensor<32xf32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func public @subf_scalar_rhs(%arg0: tensor<32xf32, #asctile.local<UB>>) -> tensor<32xf32, #asctile.local<UB>> {
// CHECK-NEXT:  %cst = arith.constant 1.57079637 : f32
// CHECK-NEXT:  %0 = asctile.subs %arg0, %cst : tensor<32xf32, #asctile.local<UB>>
// CHECK-NEXT:  return %0 : tensor<32xf32, #asctile.local<UB>>
// CHECK-NEXT:}
func.func public @subf_scalar_rhs(%arg0: tensor<32xf32, #asctile.local<UB>>) -> tensor<32xf32, #asctile.local<UB>> {
  %cst = arith.constant dense<1.57079637> : tensor<32xf32, #asctile.local<UB>>
  %0 = arith.subf %arg0, %cst : tensor<32xf32, #asctile.local<UB>>
  return %0 : tensor<32xf32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func public @subf_scalar_lhs_no_scalarization(%arg0: tensor<32xf32, #asctile.local<UB>>) -> tensor<32xf32, #asctile.local<UB>> {
// CHECK-NEXT:  %cst = arith.constant dense<1.57079637> : tensor<32xf32, #asctile.local<UB>>
// CHECK-NEXT:  %0 = arith.subf %cst, %arg0 : tensor<32xf32, #asctile.local<UB>>
// CHECK-NEXT:  return %0 : tensor<32xf32, #asctile.local<UB>>
// CHECK-NEXT:}
func.func public @subf_scalar_lhs_no_scalarization(%arg0: tensor<32xf32, #asctile.local<UB>>) -> tensor<32xf32, #asctile.local<UB>> {
  %cst = arith.constant dense<1.57079637> : tensor<32xf32, #asctile.local<UB>>
  %0 = arith.subf %cst, %arg0 : tensor<32xf32, #asctile.local<UB>>
  return %0 : tensor<32xf32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func public @mulf_scalar_rhs(%arg0: tensor<32xf32, #asctile.local<UB>>) -> tensor<32xf32, #asctile.local<UB>> {
// CHECK-NEXT:  %cst = arith.constant 1.57079637 : f32
// CHECK-NEXT:  %0 = asctile.muls %arg0, %cst : tensor<32xf32, #asctile.local<UB>>
// CHECK-NEXT:  return %0 : tensor<32xf32, #asctile.local<UB>>
// CHECK-NEXT:}
func.func public @mulf_scalar_rhs(%arg0: tensor<32xf32, #asctile.local<UB>>) -> tensor<32xf32, #asctile.local<UB>> {
  %cst = arith.constant dense<1.57079637> : tensor<32xf32, #asctile.local<UB>>
  %0 = arith.mulf %arg0, %cst : tensor<32xf32, #asctile.local<UB>>
  return %0 : tensor<32xf32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func public @mulf_scalar_lhs(%arg0: tensor<32xf32, #asctile.local<UB>>) -> tensor<32xf32, #asctile.local<UB>> {
// CHECK-NEXT:  %cst = arith.constant 1.57079637 : f32
// CHECK-NEXT:  %0 = asctile.muls %arg0, %cst : tensor<32xf32, #asctile.local<UB>>
// CHECK-NEXT:  return %0 : tensor<32xf32, #asctile.local<UB>>
// CHECK-NEXT:}
func.func public @mulf_scalar_lhs(%arg0: tensor<32xf32, #asctile.local<UB>>) -> tensor<32xf32, #asctile.local<UB>> {
  %cst = arith.constant dense<1.57079637> : tensor<32xf32, #asctile.local<UB>>
  %0 = arith.mulf %cst, %arg0 : tensor<32xf32, #asctile.local<UB>>
  return %0 : tensor<32xf32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func public @divf_scalar_rhs(%arg0: tensor<32xf32, #asctile.local<UB>>) -> tensor<32xf32, #asctile.local<UB>> {
// CHECK-NEXT:  %cst = arith.constant 1.57079637 : f32
// CHECK-NEXT:  %0 = asctile.divs %arg0, %cst : tensor<32xf32, #asctile.local<UB>>
// CHECK-NEXT:  return %0 : tensor<32xf32, #asctile.local<UB>>
// CHECK-NEXT:}
func.func public @divf_scalar_rhs(%arg0: tensor<32xf32, #asctile.local<UB>>) -> tensor<32xf32, #asctile.local<UB>> {
  %cst = arith.constant dense<1.57079637> : tensor<32xf32, #asctile.local<UB>>
  %0 = arith.divf %arg0, %cst : tensor<32xf32, #asctile.local<UB>>
  return %0 : tensor<32xf32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func public @divf_scalar_lhs_no_scalarization(%arg0: tensor<32xf32, #asctile.local<UB>>) -> tensor<32xf32, #asctile.local<UB>> {
// CHECK-NEXT:  %cst = arith.constant dense<1.57079637> : tensor<32xf32, #asctile.local<UB>>
// CHECK-NEXT:  %0 = arith.divf %cst, %arg0 : tensor<32xf32, #asctile.local<UB>>
// CHECK-NEXT:  return %0 : tensor<32xf32, #asctile.local<UB>>
// CHECK-NEXT:}
func.func public @divf_scalar_lhs_no_scalarization(%arg0: tensor<32xf32, #asctile.local<UB>>) -> tensor<32xf32, #asctile.local<UB>> {
  %cst = arith.constant dense<1.57079637> : tensor<32xf32, #asctile.local<UB>>
  %0 = arith.divf %cst, %arg0 : tensor<32xf32, #asctile.local<UB>>
  return %0 : tensor<32xf32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func public @addi_scalar_rhs(%arg0: tensor<32xi32, #asctile.local<UB>>) -> tensor<32xi32, #asctile.local<UB>> {
// CHECK-NEXT:  %c514_i32 = arith.constant 514 : i32
// CHECK-NEXT:  %0 = asctile.adds %arg0, %c514_i32 : tensor<32xi32, #asctile.local<UB>>
// CHECK-NEXT:  return %0 : tensor<32xi32, #asctile.local<UB>>
// CHECK-NEXT:}
func.func public @addi_scalar_rhs(%arg0: tensor<32xi32, #asctile.local<UB>>) -> tensor<32xi32, #asctile.local<UB>> {
  %cst = arith.constant dense<514> : tensor<32xi32, #asctile.local<UB>>
  %0 = arith.addi %arg0, %cst : tensor<32xi32, #asctile.local<UB>>
  return %0 : tensor<32xi32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func public @addi_scalar_lhs(%arg0: tensor<32xi32, #asctile.local<UB>>) -> tensor<32xi32, #asctile.local<UB>> {
// CHECK-NEXT:  %c514_i32 = arith.constant 514 : i32
// CHECK-NEXT:  %0 = asctile.adds %arg0, %c514_i32 : tensor<32xi32, #asctile.local<UB>>
// CHECK-NEXT:  return %0 : tensor<32xi32, #asctile.local<UB>>
// CHECK-NEXT:}
func.func public @addi_scalar_lhs(%arg0: tensor<32xi32, #asctile.local<UB>>) -> tensor<32xi32, #asctile.local<UB>> {
  %cst = arith.constant dense<514> : tensor<32xi32, #asctile.local<UB>>
  %0 = arith.addi %cst, %arg0 : tensor<32xi32, #asctile.local<UB>>
  return %0 : tensor<32xi32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func public @subi_scalar_rhs(%arg0: tensor<32xi32, #asctile.local<UB>>) -> tensor<32xi32, #asctile.local<UB>> {
// CHECK-NEXT:  %c514_i32 = arith.constant 514 : i32
// CHECK-NEXT:  %0 = asctile.subs %arg0, %c514_i32 : tensor<32xi32, #asctile.local<UB>>
// CHECK-NEXT:  return %0 : tensor<32xi32, #asctile.local<UB>>
// CHECK-NEXT:}
func.func public @subi_scalar_rhs(%arg0: tensor<32xi32, #asctile.local<UB>>) -> tensor<32xi32, #asctile.local<UB>> {
  %cst = arith.constant dense<514> : tensor<32xi32, #asctile.local<UB>>
  %0 = arith.subi %arg0, %cst : tensor<32xi32, #asctile.local<UB>>
  return %0 : tensor<32xi32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func public @subi_scalar_lhs_no_scalarization(%arg0: tensor<32xi32, #asctile.local<UB>>) -> tensor<32xi32, #asctile.local<UB>> {
// CHECK-NEXT:  %cst = arith.constant dense<514> : tensor<32xi32, #asctile.local<UB>>
// CHECK-NEXT:  %0 = arith.subi %cst, %arg0 : tensor<32xi32, #asctile.local<UB>>
// CHECK-NEXT:  return %0 : tensor<32xi32, #asctile.local<UB>>
// CHECK-NEXT:}
func.func public @subi_scalar_lhs_no_scalarization(%arg0: tensor<32xi32, #asctile.local<UB>>) -> tensor<32xi32, #asctile.local<UB>> {
  %cst = arith.constant dense<514> : tensor<32xi32, #asctile.local<UB>>
  %0 = arith.subi %cst, %arg0 : tensor<32xi32, #asctile.local<UB>>
  return %0 : tensor<32xi32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func public @muli_scalar_rhs(%arg0: tensor<32xi32, #asctile.local<UB>>) -> tensor<32xi32, #asctile.local<UB>> {
// CHECK-NEXT:  %c514_i32 = arith.constant 514 : i32
// CHECK-NEXT:  %0 = asctile.muls %arg0, %c514_i32 : tensor<32xi32, #asctile.local<UB>>
// CHECK-NEXT:  return %0 : tensor<32xi32, #asctile.local<UB>>
// CHECK-NEXT:}
func.func public @muli_scalar_rhs(%arg0: tensor<32xi32, #asctile.local<UB>>) -> tensor<32xi32, #asctile.local<UB>> {
  %cst = arith.constant dense<514> : tensor<32xi32, #asctile.local<UB>>
  %0 = arith.muli %arg0, %cst : tensor<32xi32, #asctile.local<UB>>
  return %0 : tensor<32xi32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func public @muli_scalar_lhs(%arg0: tensor<32xi32, #asctile.local<UB>>) -> tensor<32xi32, #asctile.local<UB>> {
// CHECK-NEXT:  %c514_i32 = arith.constant 514 : i32
// CHECK-NEXT:  %0 = asctile.muls %arg0, %c514_i32 : tensor<32xi32, #asctile.local<UB>>
// CHECK-NEXT:  return %0 : tensor<32xi32, #asctile.local<UB>>
// CHECK-NEXT:}
func.func public @muli_scalar_lhs(%arg0: tensor<32xi32, #asctile.local<UB>>) -> tensor<32xi32, #asctile.local<UB>> {
  %cst = arith.constant dense<514> : tensor<32xi32, #asctile.local<UB>>
  %0 = arith.muli %cst, %arg0 : tensor<32xi32, #asctile.local<UB>>
  return %0 : tensor<32xi32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func public @no_scalarization_if_no_constant_float(%arg0: tensor<32xf32, #asctile.local<UB>>, %arg1: tensor<32xf32, #asctile.local<UB>>) -> tensor<32xf32, #asctile.local<UB>> {
// CHECK-NEXT:  %0 = arith.addf %arg0, %arg1 : tensor<32xf32, #asctile.local<UB>>
// CHECK-NEXT:  %1 = arith.mulf %arg0, %0 : tensor<32xf32, #asctile.local<UB>>
// CHECK-NEXT:  %2 = arith.subf %1, %arg1 : tensor<32xf32, #asctile.local<UB>>
// CHECK-NEXT:  return %2 : tensor<32xf32, #asctile.local<UB>>
// CHECK-NEXT:}
func.func public @no_scalarization_if_no_constant_float(%arg0: tensor<32xf32, #asctile.local<UB>>, %arg1: tensor<32xf32, #asctile.local<UB>>) -> tensor<32xf32, #asctile.local<UB>> {
  %0 = arith.addf %arg0, %arg1 : tensor<32xf32, #asctile.local<UB>>
  %1 = arith.mulf %arg0, %0 : tensor<32xf32, #asctile.local<UB>>
  %2 = arith.subf %1, %arg1 : tensor<32xf32, #asctile.local<UB>>
  return %2 : tensor<32xf32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func public @no_scalarization_if_no_constant_int(%arg0: tensor<32xi32, #asctile.local<UB>>, %arg1: tensor<32xi32, #asctile.local<UB>>) -> tensor<32xi32, #asctile.local<UB>> {
// CHECK-NEXT:  %0 = arith.addi %arg0, %arg1 : tensor<32xi32, #asctile.local<UB>>
// CHECK-NEXT:  %1 = arith.muli %0, %arg1 : tensor<32xi32, #asctile.local<UB>>
// CHECK-NEXT:  %2 = arith.subi %arg0, %1 : tensor<32xi32, #asctile.local<UB>>
// CHECK-NEXT:  return %2 : tensor<32xi32, #asctile.local<UB>>
// CHECK-NEXT:}
func.func public @no_scalarization_if_no_constant_int(%arg0: tensor<32xi32, #asctile.local<UB>>, %arg1: tensor<32xi32, #asctile.local<UB>>) -> tensor<32xi32, #asctile.local<UB>> {
  %0 = arith.addi %arg0, %arg1 : tensor<32xi32, #asctile.local<UB>>
  %1 = arith.muli %0, %arg1 : tensor<32xi32, #asctile.local<UB>>
  %2 = arith.subi %arg0, %1 : tensor<32xi32, #asctile.local<UB>>
  return %2 : tensor<32xi32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func public @no_scalarization_if_no_splat_float(%arg0: tensor<3xf32, #asctile.local<UB>>) -> tensor<3xf32, #asctile.local<UB>> {
// CHECK-NEXT:  %cst = arith.constant dense<[5.140000e+02, 4.150000e+02, 1.450000e+02]> : tensor<3xf32, #asctile.local<UB>>
// CHECK-NEXT:  %0 = arith.addf %arg0, %cst : tensor<3xf32, #asctile.local<UB>>
// CHECK-NEXT:  return %0 : tensor<3xf32, #asctile.local<UB>>
// CHECK-NEXT:}
func.func public @no_scalarization_if_no_splat_float(%arg0: tensor<3xf32, #asctile.local<UB>>) -> tensor<3xf32, #asctile.local<UB>> {
  %cst = arith.constant dense<[514.0, 415.0, 145.0]> : tensor<3xf32, #asctile.local<UB>>
  %0 = arith.addf %arg0, %cst : tensor<3xf32, #asctile.local<UB>>
  return %0 : tensor<3xf32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func public @no_scalarization_if_no_splat_int(%arg0: tensor<3xi32, #asctile.local<UB>>) -> tensor<3xi32, #asctile.local<UB>> {
// CHECK-NEXT:  %cst = arith.constant dense<[514, 415, 145]> : tensor<3xi32, #asctile.local<UB>>
// CHECK-NEXT:  %0 = arith.addi %arg0, %cst : tensor<3xi32, #asctile.local<UB>>
// CHECK-NEXT:  return %0 : tensor<3xi32, #asctile.local<UB>>
// CHECK-NEXT:}
func.func public @no_scalarization_if_no_splat_int(%arg0: tensor<3xi32, #asctile.local<UB>>) -> tensor<3xi32, #asctile.local<UB>> {
  %cst = arith.constant dense<[514, 415, 145]> : tensor<3xi32, #asctile.local<UB>>
  %0 = arith.addi %arg0, %cst : tensor<3xi32, #asctile.local<UB>>
  return %0 : tensor<3xi32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func public @shli_constant(%arg0: tensor<32xi32, #asctile.local<UB>>) -> tensor<32xi32, #asctile.local<UB>> {
// CHECK-NEXT:  %c2_i32 = arith.constant 2 : i32
// CHECK-NEXT:  %0 = asctile.shls %arg0, %c2_i32 : tensor<32xi32, #asctile.local<UB>>
// CHECK-NEXT:  return %0 : tensor<32xi32, #asctile.local<UB>>
// CHECK-NEXT:}
func.func public @shli_constant(%arg0: tensor<32xi32, #asctile.local<UB>>) -> tensor<32xi32, #asctile.local<UB>> {
  %cst = arith.constant dense<2> : tensor<32xi32, #asctile.local<UB>>
  %0 = arith.shli %arg0, %cst : tensor<32xi32, #asctile.local<UB>>
  return %0 : tensor<32xi32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func public @shli_splat(%arg0: tensor<32xi32, #asctile.local<UB>>, %arg1: i32) -> tensor<32xi32, #asctile.local<UB>> {
// CHECK-NEXT:  %0 = asctile.shls %arg0, %arg1 : tensor<32xi32, #asctile.local<UB>>
// CHECK-NEXT:  return %0 : tensor<32xi32, #asctile.local<UB>>
// CHECK-NEXT:}
func.func public @shli_splat(%arg0: tensor<32xi32, #asctile.local<UB>>, %arg1: i32) -> tensor<32xi32, #asctile.local<UB>> {
  %0 = tensor.splat %arg1 : tensor<32xi32, #asctile.local<UB>>
  %1 = arith.shli %arg0, %0 : tensor<32xi32, #asctile.local<UB>>
  return %1 : tensor<32xi32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func public @shrsi_constant(%arg0: tensor<32xi32, #asctile.local<UB>>) -> tensor<32xi32, #asctile.local<UB>> {
// CHECK-NEXT:  %c2_i32 = arith.constant 2 : i32
// CHECK-NEXT:  %0 = asctile.shrs %arg0, %c2_i32 : tensor<32xi32, #asctile.local<UB>>
// CHECK-NEXT:  return %0 : tensor<32xi32, #asctile.local<UB>>
// CHECK-NEXT:}
func.func public @shrsi_constant(%arg0: tensor<32xi32, #asctile.local<UB>>) -> tensor<32xi32, #asctile.local<UB>> {
  %cst = arith.constant dense<2> : tensor<32xi32, #asctile.local<UB>>
  %0 = arith.shrsi %arg0, %cst : tensor<32xi32, #asctile.local<UB>>
  return %0 : tensor<32xi32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func public @shrsi_splat(%arg0: tensor<32xi32, #asctile.local<UB>>, %arg1: i32) -> tensor<32xi32, #asctile.local<UB>> {
// CHECK-NEXT:  %0 = asctile.shrs %arg0, %arg1 : tensor<32xi32, #asctile.local<UB>>
// CHECK-NEXT:  return %0 : tensor<32xi32, #asctile.local<UB>>
// CHECK-NEXT:}
func.func public @shrsi_splat(%arg0: tensor<32xi32, #asctile.local<UB>>, %arg1: i32) -> tensor<32xi32, #asctile.local<UB>> {
  %0 = tensor.splat %arg1 : tensor<32xi32, #asctile.local<UB>>
  %1 = arith.shrsi %arg0, %0 : tensor<32xi32, #asctile.local<UB>>
  return %1 : tensor<32xi32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func public @max_with_zero_lhs_f32(%arg0: tensor<32xf32, #asctile.local<UB>>) -> tensor<32xf32, #asctile.local<UB>> {
// CHECK-NEXT:  %0 = asctile.relu %arg0 : tensor<32xf32, #asctile.local<UB>>
// CHECK-NEXT:  return %0 : tensor<32xf32, #asctile.local<UB>>
// CHECK-NEXT:}
func.func public @max_with_zero_lhs_f32(%arg0: tensor<32xf32, #asctile.local<UB>>) -> tensor<32xf32, #asctile.local<UB>> {
  %cst = arith.constant dense<0.0> : tensor<32xf32, #asctile.local<UB>>
  %0 = arith.maximumf %cst, %arg0 : tensor<32xf32, #asctile.local<UB>>
  return %0 : tensor<32xf32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func public @max_with_zero_rhs_f32(%arg0: tensor<32xf32, #asctile.local<UB>>) -> tensor<32xf32, #asctile.local<UB>> {
// CHECK-NEXT:  %0 = asctile.relu %arg0 : tensor<32xf32, #asctile.local<UB>>
// CHECK-NEXT:  return %0 : tensor<32xf32, #asctile.local<UB>>
// CHECK-NEXT:}
func.func public @max_with_zero_rhs_f32(%arg0: tensor<32xf32, #asctile.local<UB>>) -> tensor<32xf32, #asctile.local<UB>> {
  %cst = arith.constant 0.0 : f32
  %0 = tensor.splat %cst : tensor<32xf32, #asctile.local<UB>>
  %1 = arith.maximumf %arg0, %0 : tensor<32xf32, #asctile.local<UB>>
  return %1 : tensor<32xf32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func public @max_with_zero_lhs_i32(%arg0: tensor<32xi32, #asctile.local<UB>>) -> tensor<32xi32, #asctile.local<UB>> {
// CHECK-NEXT:  %0 = asctile.relu %arg0 : tensor<32xi32, #asctile.local<UB>>
// CHECK-NEXT:  return %0 : tensor<32xi32, #asctile.local<UB>>
// CHECK-NEXT:}
func.func public @max_with_zero_lhs_i32(%arg0: tensor<32xi32, #asctile.local<UB>>) -> tensor<32xi32, #asctile.local<UB>> {
  %cst = arith.constant 0 : i32
  %0 = tensor.splat %cst : tensor<32xi32, #asctile.local<UB>>
  %1 = arith.maxsi %0, %arg0 : tensor<32xi32, #asctile.local<UB>>
  return %1 : tensor<32xi32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func public @max_with_zero_rhs_i32(%arg0: tensor<32xi32, #asctile.local<UB>>) -> tensor<32xi32, #asctile.local<UB>> {
// CHECK-NEXT:  %0 = asctile.relu %arg0 : tensor<32xi32, #asctile.local<UB>>
// CHECK-NEXT:  return %0 : tensor<32xi32, #asctile.local<UB>>
// CHECK-NEXT:}
func.func public @max_with_zero_rhs_i32(%arg0: tensor<32xi32, #asctile.local<UB>>) -> tensor<32xi32, #asctile.local<UB>> {
  %cst = arith.constant dense<0> : tensor<32xi32, #asctile.local<UB>>
  %0 = arith.maxsi %arg0, %cst : tensor<32xi32, #asctile.local<UB>>
  return %0 : tensor<32xi32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func public @max_with_non_zero_lhs_f32(%arg0: tensor<32xf32, #asctile.local<UB>>) -> tensor<32xf32, #asctile.local<UB>> {
// CHECK-NEXT:  %cst = arith.constant 1.000000e+00 : f32
// CHECK-NEXT:  %0 = asctile.maxs %arg0, %cst : tensor<32xf32, #asctile.local<UB>>
// CHECK-NEXT:  return %0 : tensor<32xf32, #asctile.local<UB>>
// CHECK-NEXT:}
func.func public @max_with_non_zero_lhs_f32(%arg0: tensor<32xf32, #asctile.local<UB>>) -> tensor<32xf32, #asctile.local<UB>> {
  %cst = arith.constant dense<1.0> : tensor<32xf32, #asctile.local<UB>>
  %0 = arith.maximumf %cst, %arg0 : tensor<32xf32, #asctile.local<UB>>
  return %0 : tensor<32xf32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func public @max_with_non_zero_rhs_f32(%arg0: tensor<32xf32, #asctile.local<UB>>) -> tensor<32xf32, #asctile.local<UB>> {
// CHECK-NEXT:  %cst = arith.constant 1.000000e+00 : f32
// CHECK-NEXT:  %0 = asctile.maxs %arg0, %cst : tensor<32xf32, #asctile.local<UB>>
// CHECK-NEXT:  return %0 : tensor<32xf32, #asctile.local<UB>>
// CHECK-NEXT:}
func.func public @max_with_non_zero_rhs_f32(%arg0: tensor<32xf32, #asctile.local<UB>>) -> tensor<32xf32, #asctile.local<UB>> {
  %cst = arith.constant dense<1.0> : tensor<32xf32, #asctile.local<UB>>
  %0 = arith.maximumf %arg0, %cst : tensor<32xf32, #asctile.local<UB>>
  return %0 : tensor<32xf32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func public @max_with_non_zero_lhs_i32(%arg0: tensor<32xi32, #asctile.local<UB>>) -> tensor<32xi32, #asctile.local<UB>> {
// CHECK-NEXT:  %c1_i32 = arith.constant 1 : i32
// CHECK-NEXT:  %0 = asctile.maxs %arg0, %c1_i32 : tensor<32xi32, #asctile.local<UB>>
// CHECK-NEXT:  return %0 : tensor<32xi32, #asctile.local<UB>>
// CHECK-NEXT:}
func.func public @max_with_non_zero_lhs_i32(%arg0: tensor<32xi32, #asctile.local<UB>>) -> tensor<32xi32, #asctile.local<UB>> {
  %cst = arith.constant dense<1> : tensor<32xi32, #asctile.local<UB>>
  %0 = arith.maxsi %cst, %arg0 : tensor<32xi32, #asctile.local<UB>>
  return %0 : tensor<32xi32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func public @max_with_non_zero_rhs_i32(%arg0: tensor<32xi32, #asctile.local<UB>>) -> tensor<32xi32, #asctile.local<UB>> {
// CHECK-NEXT:  %c1_i32 = arith.constant 1 : i32
// CHECK-NEXT:  %0 = asctile.maxs %arg0, %c1_i32 : tensor<32xi32, #asctile.local<UB>>
// CHECK-NEXT:  return %0 : tensor<32xi32, #asctile.local<UB>>
// CHECK-NEXT:}
func.func public @max_with_non_zero_rhs_i32(%arg0: tensor<32xi32, #asctile.local<UB>>) -> tensor<32xi32, #asctile.local<UB>> {
  %cst = arith.constant dense<1> : tensor<32xi32, #asctile.local<UB>>
  %0 = arith.maxsi %arg0, %cst : tensor<32xi32, #asctile.local<UB>>
  return %0 : tensor<32xi32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func public @cmp_lt_scalar_rhs(%arg0: tensor<32xf32, #asctile.local<UB>>) -> tensor<32xi1, #asctile.local<UB>> {
// CHECK-NEXT:  %cst = arith.constant 1.57070315 : f32
// CHECK-NEXT:  %0 = asctile.cmps LT %arg0, %cst : tensor<32xf32, #asctile.local<UB>>
// CHECK-NEXT:  return %0 : tensor<32xi1, #asctile.local<UB>>
// CHECK-NEXT:}
func.func public @cmp_lt_scalar_rhs(%arg0: tensor<32xf32, #asctile.local<UB>>) -> tensor<32xi1, #asctile.local<UB>> {
  %cst = arith.constant dense<1.57070313> : tensor<32xf32, #asctile.local<UB>>
  %0 = arith.cmpf olt, %arg0, %cst : tensor<32xf32, #asctile.local<UB>>
  return %0 : tensor<32xi1, #asctile.local<UB>>
}

// CHECK-LABEL: func.func public @cmp_gt_scalar_rhs(%arg0: tensor<32xf32, #asctile.local<UB>>) -> tensor<32xi1, #asctile.local<UB>> {
// CHECK-NEXT:  %cst = arith.constant 1.57070315 : f32
// CHECK-NEXT:  %0 = asctile.cmps GT %arg0, %cst : tensor<32xf32, #asctile.local<UB>>
// CHECK-NEXT:  return %0 : tensor<32xi1, #asctile.local<UB>>
// CHECK-NEXT:}
func.func public @cmp_gt_scalar_rhs(%arg0: tensor<32xf32, #asctile.local<UB>>) -> tensor<32xi1, #asctile.local<UB>> {
  %cst = arith.constant dense<1.57070313> : tensor<32xf32, #asctile.local<UB>>
  %0 = arith.cmpf ogt, %arg0, %cst : tensor<32xf32, #asctile.local<UB>>
  return %0 : tensor<32xi1, #asctile.local<UB>>
}

// CHECK-LABEL: func.func public @cmp_eq_scalar_rhs(%arg0: tensor<32xf32, #asctile.local<UB>>) -> tensor<32xi1, #asctile.local<UB>> {
// CHECK-NEXT:  %cst = arith.constant 1.57070315 : f32
// CHECK-NEXT:  %0 = asctile.cmps EQ %arg0, %cst : tensor<32xf32, #asctile.local<UB>>
// CHECK-NEXT:  return %0 : tensor<32xi1, #asctile.local<UB>>
// CHECK-NEXT:}
func.func public @cmp_eq_scalar_rhs(%arg0: tensor<32xf32, #asctile.local<UB>>) -> tensor<32xi1, #asctile.local<UB>> {
  %cst = arith.constant dense<1.57070313> : tensor<32xf32, #asctile.local<UB>>
  %0 = arith.cmpf oeq, %arg0, %cst : tensor<32xf32, #asctile.local<UB>>
  return %0 : tensor<32xi1, #asctile.local<UB>>
}

// CHECK-LABEL: func.func public @cmp_ne_scalar_rhs(%arg0: tensor<32xf32, #asctile.local<UB>>) -> tensor<32xi1, #asctile.local<UB>> {
// CHECK-NEXT:  %cst = arith.constant 1.57070315 : f32
// CHECK-NEXT:  %0 = asctile.cmps NE %arg0, %cst : tensor<32xf32, #asctile.local<UB>>
// CHECK-NEXT:  return %0 : tensor<32xi1, #asctile.local<UB>>
// CHECK-NEXT:}
func.func public @cmp_ne_scalar_rhs(%arg0: tensor<32xf32, #asctile.local<UB>>) -> tensor<32xi1, #asctile.local<UB>> {
  %cst = arith.constant dense<1.57070313> : tensor<32xf32, #asctile.local<UB>>
  %0 = arith.cmpf one, %arg0, %cst : tensor<32xf32, #asctile.local<UB>>
  return %0 : tensor<32xi1, #asctile.local<UB>>
}

// CHECK-LABEL: func.func public @cmp_le_scalar_rhs(%arg0: tensor<32xf32, #asctile.local<UB>>) -> tensor<32xi1, #asctile.local<UB>> {
// CHECK-NEXT:  %cst = arith.constant 1.57070315 : f32
// CHECK-NEXT:  %0 = asctile.cmps LE %arg0, %cst : tensor<32xf32, #asctile.local<UB>>
// CHECK-NEXT:  return %0 : tensor<32xi1, #asctile.local<UB>>
// CHECK-NEXT:}
func.func public @cmp_le_scalar_rhs(%arg0: tensor<32xf32, #asctile.local<UB>>) -> tensor<32xi1, #asctile.local<UB>> {
  %cst = arith.constant dense<1.57070313> : tensor<32xf32, #asctile.local<UB>>
  %0 = arith.cmpf ole, %arg0, %cst : tensor<32xf32, #asctile.local<UB>>
  return %0 : tensor<32xi1, #asctile.local<UB>>
}

// CHECK-LABEL: func.func public @cmp_ge_scalar_rhs(%arg0: tensor<32xf32, #asctile.local<UB>>) -> tensor<32xi1, #asctile.local<UB>> {
// CHECK-NEXT:  %cst = arith.constant 1.57070315 : f32
// CHECK-NEXT:  %0 = asctile.cmps GE %arg0, %cst : tensor<32xf32, #asctile.local<UB>>
// CHECK-NEXT:  return %0 : tensor<32xi1, #asctile.local<UB>>
// CHECK-NEXT:}
func.func public @cmp_ge_scalar_rhs(%arg0: tensor<32xf32, #asctile.local<UB>>) -> tensor<32xi1, #asctile.local<UB>> {
  %cst = arith.constant dense<1.57070313> : tensor<32xf32, #asctile.local<UB>>
  %0 = arith.cmpf oge, %arg0, %cst : tensor<32xf32, #asctile.local<UB>>
  return %0 : tensor<32xi1, #asctile.local<UB>>
}

// CHECK-LABEL: func.func public @cmp_lt_scalar_lhs(%arg0: tensor<32xf32, #asctile.local<UB>>) -> tensor<32xi1, #asctile.local<UB>> {
// CHECK-NEXT:  %cst = arith.constant 1.57070315 : f32
// CHECK-NEXT:  %0 = asctile.cmps GT %arg0, %cst : tensor<32xf32, #asctile.local<UB>>
// CHECK-NEXT:  return %0 : tensor<32xi1, #asctile.local<UB>>
// CHECK-NEXT:}
func.func public @cmp_lt_scalar_lhs(%arg0: tensor<32xf32, #asctile.local<UB>>) -> tensor<32xi1, #asctile.local<UB>> {
  %cst = arith.constant dense<1.57070313> : tensor<32xf32, #asctile.local<UB>>
  %0 = arith.cmpf olt, %cst, %arg0 : tensor<32xf32, #asctile.local<UB>>
  return %0 : tensor<32xi1, #asctile.local<UB>>
}

// CHECK-LABEL: func.func public @cmp_gt_scalar_lhs(%arg0: tensor<32xf32, #asctile.local<UB>>) -> tensor<32xi1, #asctile.local<UB>> {
// CHECK-NEXT:  %cst = arith.constant 1.57070315 : f32
// CHECK-NEXT:  %0 = asctile.cmps LT %arg0, %cst : tensor<32xf32, #asctile.local<UB>>
// CHECK-NEXT:  return %0 : tensor<32xi1, #asctile.local<UB>>
// CHECK-NEXT:}
func.func public @cmp_gt_scalar_lhs(%arg0: tensor<32xf32, #asctile.local<UB>>) -> tensor<32xi1, #asctile.local<UB>> {
  %cst = arith.constant dense<1.57070313> : tensor<32xf32, #asctile.local<UB>>
  %0 = arith.cmpf ogt, %cst, %arg0 : tensor<32xf32, #asctile.local<UB>>
  return %0 : tensor<32xi1, #asctile.local<UB>>
}

// CHECK-LABEL: func.func public @cmp_eq_scalar_lhs(%arg0: tensor<32xf32, #asctile.local<UB>>) -> tensor<32xi1, #asctile.local<UB>> {
// CHECK-NEXT:  %cst = arith.constant 1.57070315 : f32
// CHECK-NEXT:  %0 = asctile.cmps EQ %arg0, %cst : tensor<32xf32, #asctile.local<UB>>
// CHECK-NEXT:  return %0 : tensor<32xi1, #asctile.local<UB>>
// CHECK-NEXT:}
func.func public @cmp_eq_scalar_lhs(%arg0: tensor<32xf32, #asctile.local<UB>>) -> tensor<32xi1, #asctile.local<UB>> {
  %cst = arith.constant dense<1.57070313> : tensor<32xf32, #asctile.local<UB>>
  %0 = arith.cmpf oeq, %cst, %arg0 : tensor<32xf32, #asctile.local<UB>>
  return %0 : tensor<32xi1, #asctile.local<UB>>
}

// CHECK-LABEL: func.func public @cmp_ne_scalar_lhs(%arg0: tensor<32xf32, #asctile.local<UB>>) -> tensor<32xi1, #asctile.local<UB>> {
// CHECK-NEXT:  %cst = arith.constant 1.57070315 : f32
// CHECK-NEXT:  %0 = asctile.cmps NE %arg0, %cst : tensor<32xf32, #asctile.local<UB>>
// CHECK-NEXT:  return %0 : tensor<32xi1, #asctile.local<UB>>
// CHECK-NEXT:}
func.func public @cmp_ne_scalar_lhs(%arg0: tensor<32xf32, #asctile.local<UB>>) -> tensor<32xi1, #asctile.local<UB>> {
  %cst = arith.constant dense<1.57070313> : tensor<32xf32, #asctile.local<UB>>
  %0 = arith.cmpf one, %cst, %arg0 : tensor<32xf32, #asctile.local<UB>>
  return %0 : tensor<32xi1, #asctile.local<UB>>
}

// CHECK-LABEL: func.func public @cmp_le_scalar_lhs(%arg0: tensor<32xf32, #asctile.local<UB>>) -> tensor<32xi1, #asctile.local<UB>> {
// CHECK-NEXT:  %cst = arith.constant 1.57070315 : f32
// CHECK-NEXT:  %0 = asctile.cmps GE %arg0, %cst : tensor<32xf32, #asctile.local<UB>>
// CHECK-NEXT:  return %0 : tensor<32xi1, #asctile.local<UB>>
// CHECK-NEXT:}
func.func public @cmp_le_scalar_lhs(%arg0: tensor<32xf32, #asctile.local<UB>>) -> tensor<32xi1, #asctile.local<UB>> {
  %cst = arith.constant dense<1.57070313> : tensor<32xf32, #asctile.local<UB>>
  %0 = arith.cmpf ole, %cst, %arg0 : tensor<32xf32, #asctile.local<UB>>
  return %0 : tensor<32xi1, #asctile.local<UB>>
}

// CHECK-LABEL: func.func public @cmp_ge_scalar_lhs(%arg0: tensor<32xf32, #asctile.local<UB>>) -> tensor<32xi1, #asctile.local<UB>> {
// CHECK-NEXT:  %cst = arith.constant 1.57070315 : f32
// CHECK-NEXT:  %0 = asctile.cmps LE %arg0, %cst : tensor<32xf32, #asctile.local<UB>>
// CHECK-NEXT:  return %0 : tensor<32xi1, #asctile.local<UB>>
// CHECK-NEXT:}
func.func public @cmp_ge_scalar_lhs(%arg0: tensor<32xf32, #asctile.local<UB>>) -> tensor<32xi1, #asctile.local<UB>> {
  %cst = arith.constant dense<1.57070313> : tensor<32xf32, #asctile.local<UB>>
  %0 = arith.cmpf oge, %cst, %arg0 : tensor<32xf32, #asctile.local<UB>>
  return %0 : tensor<32xi1, #asctile.local<UB>>
}

// CHECK-LABEL: func.func public @cmp_lt_scalar_rhs_splat(%arg0: tensor<32xf32, #asctile.local<UB>>, %arg1: f32) -> tensor<32xi1, #asctile.local<UB>> {
// CHECK-NEXT:  %0 = asctile.cmps LT %arg0, %arg1 : tensor<32xf32, #asctile.local<UB>>
// CHECK-NEXT:  return %0 : tensor<32xi1, #asctile.local<UB>>
// CHECK-NEXT:}
func.func public @cmp_lt_scalar_rhs_splat(%arg0: tensor<32xf32, #asctile.local<UB>>, %arg1: f32) -> tensor<32xi1, #asctile.local<UB>> {
  %splat = tensor.splat %arg1 : tensor<32xf32, #asctile.local<UB>>
  %0 = arith.cmpf olt, %arg0, %splat : tensor<32xf32, #asctile.local<UB>>
  return %0 : tensor<32xi1, #asctile.local<UB>>
}

// CHECK-LABEL: func.func public @cmp_gt_scalar_lhs_splat(%arg0: tensor<32xf32, #asctile.local<UB>>, %arg1: f32) -> tensor<32xi1, #asctile.local<UB>> {
// CHECK-NEXT:  %0 = asctile.cmps LT %arg0, %arg1 : tensor<32xf32, #asctile.local<UB>>
// CHECK-NEXT:  return %0 : tensor<32xi1, #asctile.local<UB>>
// CHECK-NEXT:}
func.func public @cmp_gt_scalar_lhs_splat(%arg0: tensor<32xf32, #asctile.local<UB>>, %arg1: f32) -> tensor<32xi1, #asctile.local<UB>> {
  %splat = tensor.splat %arg1 : tensor<32xf32, #asctile.local<UB>>
  %0 = arith.cmpf ogt, %splat, %arg0 : tensor<32xf32, #asctile.local<UB>>
  return %0 : tensor<32xi1, #asctile.local<UB>>
}

// CHECK-LABEL: func.func public @cmp_no_scalarization(%arg0: tensor<32xf32, #asctile.local<UB>>, %arg1: tensor<32xf32, #asctile.local<UB>>) -> tensor<32xi1, #asctile.local<UB>> {
// CHECK-NEXT:  %0 = arith.cmpf olt, %arg0, %arg1 : tensor<32xf32, #asctile.local<UB>>
// CHECK-NEXT:  return %0 : tensor<32xi1, #asctile.local<UB>>
// CHECK-NEXT:}
func.func public @cmp_no_scalarization(%arg0: tensor<32xf32, #asctile.local<UB>>, %arg1: tensor<32xf32, #asctile.local<UB>>) -> tensor<32xi1, #asctile.local<UB>> {
  %0 = arith.cmpf olt, %arg0, %arg1 : tensor<32xf32, #asctile.local<UB>>
  return %0 : tensor<32xi1, #asctile.local<UB>>
}

// CHECK-LABEL: func.func public @leaky_relu_cmp_mulf_f32(%arg0: tensor<32xf32, #asctile.local<UB>>) -> tensor<32xf32, #asctile.local<UB>> {
// CHECK-NEXT:  %cst = arith.constant 0.00999999977 : f32
// CHECK-NEXT:  %0 = asctile.leaky_relu %arg0, %cst : tensor<32xf32, #asctile.local<UB>>
// CHECK-NEXT:  return %0 : tensor<32xf32, #asctile.local<UB>>
// CHECK-NEXT:}
func.func public @leaky_relu_cmp_mulf_f32(%arg0: tensor<32xf32, #asctile.local<UB>>) -> tensor<32xf32, #asctile.local<UB>> {
  %zero = arith.constant dense<0.0> : tensor<32xf32, #asctile.local<UB>>
  %alpha = arith.constant dense<0.01> : tensor<32xf32, #asctile.local<UB>>
  %cmp = arith.cmpf oge, %arg0, %zero : tensor<32xf32, #asctile.local<UB>>
  %mul = arith.mulf %arg0, %alpha : tensor<32xf32, #asctile.local<UB>>
  %result = arith.select %cmp, %arg0, %mul :  tensor<32xi1, #asctile.local<UB>>, tensor<32xf32, #asctile.local<UB>>
  return %result : tensor<32xf32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func public @leaky_relu_cmp_mulf_f16(%arg0: tensor<32xf16, #asctile.local<UB>>) -> tensor<32xf16, #asctile.local<UB>> {
// CHECK-NEXT:  %cst = arith.constant 1.000210e-02 : f16
// CHECK-NEXT:  %0 = asctile.leaky_relu %arg0, %cst : tensor<32xf16, #asctile.local<UB>>
// CHECK-NEXT:  return %0 : tensor<32xf16, #asctile.local<UB>>
// CHECK-NEXT:}
func.func public @leaky_relu_cmp_mulf_f16(%arg0: tensor<32xf16, #asctile.local<UB>>) -> tensor<32xf16, #asctile.local<UB>> {
  %zero = arith.constant dense<0.0> : tensor<32xf16, #asctile.local<UB>>
  %alpha = arith.constant dense<0.01> : tensor<32xf16, #asctile.local<UB>>
  %cmp = arith.cmpf oge, %arg0, %zero : tensor<32xf16, #asctile.local<UB>>
  %mul = arith.mulf %arg0, %alpha : tensor<32xf16, #asctile.local<UB>>
  %result = arith.select %cmp, %arg0, %mul : tensor<32xi1, #asctile.local<UB>>, tensor<32xf16, #asctile.local<UB>>
  return %result : tensor<32xf16, #asctile.local<UB>>
}

// CHECK-LABEL: func.func public @leaky_relu_inverted_lt_f32(%arg0: tensor<32xf32, #asctile.local<UB>>) -> tensor<32xf32, #asctile.local<UB>> {
// CHECK-NEXT:  %cst = arith.constant 0.00999999977 : f32
// CHECK-NEXT:  %0 = asctile.leaky_relu %arg0, %cst : tensor<32xf32, #asctile.local<UB>>
// CHECK-NEXT:  return %0 : tensor<32xf32, #asctile.local<UB>>
// CHECK-NEXT:}
func.func public @leaky_relu_inverted_lt_f32(%arg0: tensor<32xf32, #asctile.local<UB>>) -> tensor<32xf32, #asctile.local<UB>> {
  %zero = arith.constant dense<0.0> : tensor<32xf32, #asctile.local<UB>>
  %alpha = arith.constant dense<0.01> : tensor<32xf32, #asctile.local<UB>>
  %cmp = arith.cmpf olt, %arg0, %zero : tensor<32xf32, #asctile.local<UB>>
  %mul = arith.mulf %arg0, %alpha : tensor<32xf32, #asctile.local<UB>>
  %result = arith.select %cmp, %mul, %arg0 : tensor<32xi1, #asctile.local<UB>>, tensor<32xf32, #asctile.local<UB>>
  return %result : tensor<32xf32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func public @leaky_relu_cmps_muls_f32(%arg0: tensor<32xf32, #asctile.local<UB>>, %arg1: f32) -> tensor<32xf32, #asctile.local<UB>> {
// CHECK-NEXT:  %0 = asctile.leaky_relu %arg0, %arg1 : tensor<32xf32, #asctile.local<UB>>
// CHECK-NEXT:  return %0 : tensor<32xf32, #asctile.local<UB>>
// CHECK-NEXT:}
func.func public @leaky_relu_cmps_muls_f32(%arg0: tensor<32xf32, #asctile.local<UB>>, %arg1: f32) -> tensor<32xf32, #asctile.local<UB>> {
  %zero_scalar = arith.constant 0.0 : f32
  %cmp = asctile.cmps GE %arg0, %zero_scalar : tensor<32xf32, #asctile.local<UB>>
  %mul = asctile.muls %arg0, %arg1 : tensor<32xf32, #asctile.local<UB>>
  %result = arith.select %cmp, %arg0, %mul : tensor<32xi1, #asctile.local<UB>>, tensor<32xf32, #asctile.local<UB>>
  return %result : tensor<32xf32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func public @no_leaky_relu_i32(%arg0: tensor<32xi32, #asctile.local<UB>>) -> tensor<32xi32, #asctile.local<UB>> {
// CHECK:       %0 = asctile.cmps GE %arg0, %c0_i32 : tensor<32xi32, #asctile.local<UB>>
// CHECK-NEXT:  %1 = asctile.muls %arg0, %c-1_i32 : tensor<32xi32, #asctile.local<UB>>
// CHECK-NEXT:  %2 = arith.select %0, %arg0, %1 : tensor<32xi1, #asctile.local<UB>>, tensor<32xi32, #asctile.local<UB>>
// CHECK-NEXT:  return %2 : tensor<32xi32, #asctile.local<UB>>
// CHECK-NEXT:}
func.func public @no_leaky_relu_i32(%arg0: tensor<32xi32, #asctile.local<UB>>) -> tensor<32xi32, #asctile.local<UB>> {
  %zero = arith.constant dense<0> : tensor<32xi32, #asctile.local<UB>>
  %alpha = arith.constant dense<-1> : tensor<32xi32, #asctile.local<UB>>
  %cmp = arith.cmpi sge, %arg0, %zero : tensor<32xi32, #asctile.local<UB>>
  %mul = arith.muli %arg0, %alpha : tensor<32xi32, #asctile.local<UB>>
  %result = arith.select %cmp, %arg0, %mul : tensor<32xi1, #asctile.local<UB>>, tensor<32xi32, #asctile.local<UB>>
  return %result : tensor<32xi32, #asctile.local<UB>>
}
