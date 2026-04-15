// RUN: ascir-opt -asctile-transform-math-ops %s | FileCheck %s

// CHECK-LABEL: func.func public @addf_scalar_rhs(%arg0: !asctile.tile<32xf32, UB>) -> !asctile.tile<32xf32, UB> {
// CHECK-NEXT:  %cst = arith.constant 1.57079637 : f32
// CHECK-NEXT:  %0 = asctile.adds %arg0, %cst : !asctile.tile<32xf32, UB>
// CHECK-NEXT:  return %0 : !asctile.tile<32xf32, UB>
// CHECK-NEXT:}
func.func public @addf_scalar_rhs(%arg0: !asctile.tile<32xf32, UB>) -> !asctile.tile<32xf32, UB> {
  %cst = arith.constant dense<1.57079637> : !asctile.tile<32xf32, UB>
  %0 = arith.addf %arg0, %cst : !asctile.tile<32xf32, UB>
  return %0 : !asctile.tile<32xf32, UB>
}

// CHECK-LABEL: func.func public @addf_scalar_lhs(%arg0: !asctile.tile<32xf32, UB>) -> !asctile.tile<32xf32, UB> {
// CHECK-NEXT:  %cst = arith.constant 1.57079637 : f32
// CHECK-NEXT:  %0 = asctile.adds %arg0, %cst : !asctile.tile<32xf32, UB>
// CHECK-NEXT:  return %0 : !asctile.tile<32xf32, UB>
// CHECK-NEXT:}
func.func public @addf_scalar_lhs(%arg0: !asctile.tile<32xf32, UB>) -> !asctile.tile<32xf32, UB> {
  %cst = arith.constant dense<1.57079637> : !asctile.tile<32xf32, UB>
  %0 = arith.addf %cst, %arg0 : !asctile.tile<32xf32, UB>
  return %0 : !asctile.tile<32xf32, UB>
}

// CHECK-LABEL: func.func public @subf_scalar_rhs(%arg0: !asctile.tile<32xf32, UB>) -> !asctile.tile<32xf32, UB> {
// CHECK-NEXT:  %cst = arith.constant 1.57079637 : f32
// CHECK-NEXT:  %0 = asctile.subs %arg0, %cst : !asctile.tile<32xf32, UB>
// CHECK-NEXT:  return %0 : !asctile.tile<32xf32, UB>
// CHECK-NEXT:}
func.func public @subf_scalar_rhs(%arg0: !asctile.tile<32xf32, UB>) -> !asctile.tile<32xf32, UB> {
  %cst = arith.constant dense<1.57079637> : !asctile.tile<32xf32, UB>
  %0 = arith.subf %arg0, %cst : !asctile.tile<32xf32, UB>
  return %0 : !asctile.tile<32xf32, UB>
}

// CHECK-LABEL: func.func public @subf_scalar_lhs_no_scalarization(%arg0: !asctile.tile<32xf32, UB>) -> !asctile.tile<32xf32, UB> {
// CHECK-NEXT:  %cst = arith.constant dense<1.57079637> : !asctile.tile<32xf32, UB>
// CHECK-NEXT:  %0 = arith.subf %cst, %arg0 : !asctile.tile<32xf32, UB>
// CHECK-NEXT:  return %0 : !asctile.tile<32xf32, UB>
// CHECK-NEXT:}
func.func public @subf_scalar_lhs_no_scalarization(%arg0: !asctile.tile<32xf32, UB>) -> !asctile.tile<32xf32, UB> {
  %cst = arith.constant dense<1.57079637> : !asctile.tile<32xf32, UB>
  %0 = arith.subf %cst, %arg0 : !asctile.tile<32xf32, UB>
  return %0 : !asctile.tile<32xf32, UB>
}

// CHECK-LABEL: func.func public @mulf_scalar_rhs(%arg0: !asctile.tile<32xf32, UB>) -> !asctile.tile<32xf32, UB> {
// CHECK-NEXT:  %cst = arith.constant 1.57079637 : f32
// CHECK-NEXT:  %0 = asctile.muls %arg0, %cst : !asctile.tile<32xf32, UB>
// CHECK-NEXT:  return %0 : !asctile.tile<32xf32, UB>
// CHECK-NEXT:}
func.func public @mulf_scalar_rhs(%arg0: !asctile.tile<32xf32, UB>) -> !asctile.tile<32xf32, UB> {
  %cst = arith.constant dense<1.57079637> : !asctile.tile<32xf32, UB>
  %0 = arith.mulf %arg0, %cst : !asctile.tile<32xf32, UB>
  return %0 : !asctile.tile<32xf32, UB>
}

// CHECK-LABEL: func.func public @mulf_scalar_lhs(%arg0: !asctile.tile<32xf32, UB>) -> !asctile.tile<32xf32, UB> {
// CHECK-NEXT:  %cst = arith.constant 1.57079637 : f32
// CHECK-NEXT:  %0 = asctile.muls %arg0, %cst : !asctile.tile<32xf32, UB>
// CHECK-NEXT:  return %0 : !asctile.tile<32xf32, UB>
// CHECK-NEXT:}
func.func public @mulf_scalar_lhs(%arg0: !asctile.tile<32xf32, UB>) -> !asctile.tile<32xf32, UB> {
  %cst = arith.constant dense<1.57079637> : !asctile.tile<32xf32, UB>
  %0 = arith.mulf %cst, %arg0 : !asctile.tile<32xf32, UB>
  return %0 : !asctile.tile<32xf32, UB>
}

// CHECK-LABEL: func.func public @divf_scalar_rhs(%arg0: !asctile.tile<32xf32, UB>) -> !asctile.tile<32xf32, UB> {
// CHECK-NEXT:  %cst = arith.constant 1.57079637 : f32
// CHECK-NEXT:  %0 = asctile.divs %arg0, %cst : !asctile.tile<32xf32, UB>
// CHECK-NEXT:  return %0 : !asctile.tile<32xf32, UB>
// CHECK-NEXT:}
func.func public @divf_scalar_rhs(%arg0: !asctile.tile<32xf32, UB>) -> !asctile.tile<32xf32, UB> {
  %cst = arith.constant dense<1.57079637> : !asctile.tile<32xf32, UB>
  %0 = arith.divf %arg0, %cst : !asctile.tile<32xf32, UB>
  return %0 : !asctile.tile<32xf32, UB>
}

// CHECK-LABEL: func.func public @divf_scalar_lhs_no_scalarization(%arg0: !asctile.tile<32xf32, UB>) -> !asctile.tile<32xf32, UB> {
// CHECK-NEXT:  %cst = arith.constant dense<1.57079637> : !asctile.tile<32xf32, UB>
// CHECK-NEXT:  %0 = arith.divf %cst, %arg0 : !asctile.tile<32xf32, UB>
// CHECK-NEXT:  return %0 : !asctile.tile<32xf32, UB>
// CHECK-NEXT:}
func.func public @divf_scalar_lhs_no_scalarization(%arg0: !asctile.tile<32xf32, UB>) -> !asctile.tile<32xf32, UB> {
  %cst = arith.constant dense<1.57079637> : !asctile.tile<32xf32, UB>
  %0 = arith.divf %cst, %arg0 : !asctile.tile<32xf32, UB>
  return %0 : !asctile.tile<32xf32, UB>
}

// CHECK-LABEL: func.func public @addi_scalar_rhs(%arg0: !asctile.tile<32xi32, UB>) -> !asctile.tile<32xi32, UB> {
// CHECK-NEXT:  %c514_i32 = arith.constant 514 : i32
// CHECK-NEXT:  %0 = asctile.adds %arg0, %c514_i32 : !asctile.tile<32xi32, UB>
// CHECK-NEXT:  return %0 : !asctile.tile<32xi32, UB>
// CHECK-NEXT:}
func.func public @addi_scalar_rhs(%arg0: !asctile.tile<32xi32, UB>) -> !asctile.tile<32xi32, UB> {
  %cst = arith.constant dense<514> : !asctile.tile<32xi32, UB>
  %0 = arith.addi %arg0, %cst : !asctile.tile<32xi32, UB>
  return %0 : !asctile.tile<32xi32, UB>
}

// CHECK-LABEL: func.func public @addi_scalar_lhs(%arg0: !asctile.tile<32xi32, UB>) -> !asctile.tile<32xi32, UB> {
// CHECK-NEXT:  %c514_i32 = arith.constant 514 : i32
// CHECK-NEXT:  %0 = asctile.adds %arg0, %c514_i32 : !asctile.tile<32xi32, UB>
// CHECK-NEXT:  return %0 : !asctile.tile<32xi32, UB>
// CHECK-NEXT:}
func.func public @addi_scalar_lhs(%arg0: !asctile.tile<32xi32, UB>) -> !asctile.tile<32xi32, UB> {
  %cst = arith.constant dense<514> : !asctile.tile<32xi32, UB>
  %0 = arith.addi %cst, %arg0 : !asctile.tile<32xi32, UB>
  return %0 : !asctile.tile<32xi32, UB>
}

// CHECK-LABEL: func.func public @subi_scalar_rhs(%arg0: !asctile.tile<32xi32, UB>) -> !asctile.tile<32xi32, UB> {
// CHECK-NEXT:  %c514_i32 = arith.constant 514 : i32
// CHECK-NEXT:  %0 = asctile.subs %arg0, %c514_i32 : !asctile.tile<32xi32, UB>
// CHECK-NEXT:  return %0 : !asctile.tile<32xi32, UB>
// CHECK-NEXT:}
func.func public @subi_scalar_rhs(%arg0: !asctile.tile<32xi32, UB>) -> !asctile.tile<32xi32, UB> {
  %cst = arith.constant dense<514> : !asctile.tile<32xi32, UB>
  %0 = arith.subi %arg0, %cst : !asctile.tile<32xi32, UB>
  return %0 : !asctile.tile<32xi32, UB>
}

// CHECK-LABEL: func.func public @subi_scalar_lhs_no_scalarization(%arg0: !asctile.tile<32xi32, UB>) -> !asctile.tile<32xi32, UB> {
// CHECK-NEXT:  %cst = arith.constant dense<514> : !asctile.tile<32xi32, UB>
// CHECK-NEXT:  %0 = arith.subi %cst, %arg0 : !asctile.tile<32xi32, UB>
// CHECK-NEXT:  return %0 : !asctile.tile<32xi32, UB>
// CHECK-NEXT:}
func.func public @subi_scalar_lhs_no_scalarization(%arg0: !asctile.tile<32xi32, UB>) -> !asctile.tile<32xi32, UB> {
  %cst = arith.constant dense<514> : !asctile.tile<32xi32, UB>
  %0 = arith.subi %cst, %arg0 : !asctile.tile<32xi32, UB>
  return %0 : !asctile.tile<32xi32, UB>
}

// CHECK-LABEL: func.func public @muli_scalar_rhs(%arg0: !asctile.tile<32xi32, UB>) -> !asctile.tile<32xi32, UB> {
// CHECK-NEXT:  %c514_i32 = arith.constant 514 : i32
// CHECK-NEXT:  %0 = asctile.muls %arg0, %c514_i32 : !asctile.tile<32xi32, UB>
// CHECK-NEXT:  return %0 : !asctile.tile<32xi32, UB>
// CHECK-NEXT:}
func.func public @muli_scalar_rhs(%arg0: !asctile.tile<32xi32, UB>) -> !asctile.tile<32xi32, UB> {
  %cst = arith.constant dense<514> : !asctile.tile<32xi32, UB>
  %0 = arith.muli %arg0, %cst : !asctile.tile<32xi32, UB>
  return %0 : !asctile.tile<32xi32, UB>
}

// CHECK-LABEL: func.func public @muli_scalar_lhs(%arg0: !asctile.tile<32xi32, UB>) -> !asctile.tile<32xi32, UB> {
// CHECK-NEXT:  %c514_i32 = arith.constant 514 : i32
// CHECK-NEXT:  %0 = asctile.muls %arg0, %c514_i32 : !asctile.tile<32xi32, UB>
// CHECK-NEXT:  return %0 : !asctile.tile<32xi32, UB>
// CHECK-NEXT:}
func.func public @muli_scalar_lhs(%arg0: !asctile.tile<32xi32, UB>) -> !asctile.tile<32xi32, UB> {
  %cst = arith.constant dense<514> : !asctile.tile<32xi32, UB>
  %0 = arith.muli %cst, %arg0 : !asctile.tile<32xi32, UB>
  return %0 : !asctile.tile<32xi32, UB>
}

// CHECK-LABEL: func.func public @divsi_scalar_rhs(%arg0: !asctile.tile<32xi32, UB>) -> !asctile.tile<32xi32, UB> {
// CHECK-NEXT:  %c514_i32 = arith.constant 514 : i32
// CHECK-NEXT:  %0 = asctile.divs %arg0, %c514_i32 : !asctile.tile<32xi32, UB>
// CHECK-NEXT:  return %0 : !asctile.tile<32xi32, UB>
// CHECK-NEXT:}
func.func public @divsi_scalar_rhs(%arg0: !asctile.tile<32xi32, UB>) -> !asctile.tile<32xi32, UB> {
  %cst = arith.constant dense<514> : !asctile.tile<32xi32, UB>
  %0 = arith.divsi %arg0, %cst : !asctile.tile<32xi32, UB>
  return %0 : !asctile.tile<32xi32, UB>
}

// CHECK-LABEL: func.func public @divsi_scalar_lhs_no_scalarization(%arg0: !asctile.tile<32xi32, UB>) -> !asctile.tile<32xi32, UB> {
// CHECK-NEXT:  %cst = arith.constant dense<514> : !asctile.tile<32xi32, UB>
// CHECK-NEXT:  %0 = arith.divsi %cst, %arg0 : !asctile.tile<32xi32, UB>
// CHECK-NEXT:  return %0 : !asctile.tile<32xi32, UB>
// CHECK-NEXT:}
func.func public @divsi_scalar_lhs_no_scalarization(%arg0: !asctile.tile<32xi32, UB>) -> !asctile.tile<32xi32, UB> {
  %cst = arith.constant dense<514> : !asctile.tile<32xi32, UB>
  %0 = arith.divsi %cst, %arg0 : !asctile.tile<32xi32, UB>
  return %0 : !asctile.tile<32xi32, UB>
}

// CHECK-LABEL: func.func public @no_scalarization_if_no_constant_float(%arg0: !asctile.tile<32xf32, UB>, %arg1: !asctile.tile<32xf32, UB>) -> !asctile.tile<32xf32, UB> {
// CHECK-NEXT:  %0 = arith.addf %arg0, %arg1 : !asctile.tile<32xf32, UB>
// CHECK-NEXT:  %1 = arith.mulf %arg0, %0 : !asctile.tile<32xf32, UB>
// CHECK-NEXT:  %2 = arith.subf %1, %arg1 : !asctile.tile<32xf32, UB>
// CHECK-NEXT:  return %2 : !asctile.tile<32xf32, UB>
// CHECK-NEXT:}
func.func public @no_scalarization_if_no_constant_float(%arg0: !asctile.tile<32xf32, UB>, %arg1: !asctile.tile<32xf32, UB>) -> !asctile.tile<32xf32, UB> {
  %0 = arith.addf %arg0, %arg1 : !asctile.tile<32xf32, UB>
  %1 = arith.mulf %arg0, %0 : !asctile.tile<32xf32, UB>
  %2 = arith.subf %1, %arg1 : !asctile.tile<32xf32, UB>
  return %2 : !asctile.tile<32xf32, UB>
}

// CHECK-LABEL: func.func public @no_scalarization_if_no_constant_int(%arg0: !asctile.tile<32xi32, UB>, %arg1: !asctile.tile<32xi32, UB>) -> !asctile.tile<32xi32, UB> {
// CHECK-NEXT:  %0 = arith.addi %arg0, %arg1 : !asctile.tile<32xi32, UB>
// CHECK-NEXT:  %1 = arith.muli %0, %arg1 : !asctile.tile<32xi32, UB>
// CHECK-NEXT:  %2 = arith.subi %arg0, %1 : !asctile.tile<32xi32, UB>
// CHECK-NEXT:  return %2 : !asctile.tile<32xi32, UB>
// CHECK-NEXT:}
func.func public @no_scalarization_if_no_constant_int(%arg0: !asctile.tile<32xi32, UB>, %arg1: !asctile.tile<32xi32, UB>) -> !asctile.tile<32xi32, UB> {
  %0 = arith.addi %arg0, %arg1 : !asctile.tile<32xi32, UB>
  %1 = arith.muli %0, %arg1 : !asctile.tile<32xi32, UB>
  %2 = arith.subi %arg0, %1 : !asctile.tile<32xi32, UB>
  return %2 : !asctile.tile<32xi32, UB>
}

// CHECK-LABEL: func.func public @no_scalarization_if_no_splat_float(%arg0: !asctile.tile<3xf32, UB>) -> !asctile.tile<3xf32, UB> {
// CHECK-NEXT:  %cst = arith.constant dense<[5.140000e+02, 4.150000e+02, 1.450000e+02]> : !asctile.tile<3xf32, UB>
// CHECK-NEXT:  %0 = arith.addf %arg0, %cst : !asctile.tile<3xf32, UB>
// CHECK-NEXT:  return %0 : !asctile.tile<3xf32, UB>
// CHECK-NEXT:}
func.func public @no_scalarization_if_no_splat_float(%arg0: !asctile.tile<3xf32, UB>) -> !asctile.tile<3xf32, UB> {
  %cst = arith.constant dense<[514.0, 415.0, 145.0]> : !asctile.tile<3xf32, UB>
  %0 = arith.addf %arg0, %cst : !asctile.tile<3xf32, UB>
  return %0 : !asctile.tile<3xf32, UB>
}

// CHECK-LABEL: func.func public @no_scalarization_if_no_splat_int(%arg0: !asctile.tile<3xi32, UB>) -> !asctile.tile<3xi32, UB> {
// CHECK-NEXT:  %cst = arith.constant dense<[514, 415, 145]> : !asctile.tile<3xi32, UB>
// CHECK-NEXT:  %0 = arith.addi %arg0, %cst : !asctile.tile<3xi32, UB>
// CHECK-NEXT:  return %0 : !asctile.tile<3xi32, UB>
// CHECK-NEXT:}
func.func public @no_scalarization_if_no_splat_int(%arg0: !asctile.tile<3xi32, UB>) -> !asctile.tile<3xi32, UB> {
  %cst = arith.constant dense<[514, 415, 145]> : !asctile.tile<3xi32, UB>
  %0 = arith.addi %arg0, %cst : !asctile.tile<3xi32, UB>
  return %0 : !asctile.tile<3xi32, UB>
}

// CHECK-LABEL: func.func public @shli_constant(%arg0: !asctile.tile<32xi32, UB>) -> !asctile.tile<32xi32, UB> {
// CHECK-NEXT:  %c2_i32 = arith.constant 2 : i32
// CHECK-NEXT:  %0 = asctile.shls %arg0, %c2_i32 : !asctile.tile<32xi32, UB>
// CHECK-NEXT:  return %0 : !asctile.tile<32xi32, UB>
// CHECK-NEXT:}
func.func public @shli_constant(%arg0: !asctile.tile<32xi32, UB>) -> !asctile.tile<32xi32, UB> {
  %cst = arith.constant dense<2> : !asctile.tile<32xi32, UB>
  %0 = arith.shli %arg0, %cst : !asctile.tile<32xi32, UB>
  return %0 : !asctile.tile<32xi32, UB>
}

// CHECK-LABEL: func.func public @shli_splat(%arg0: !asctile.tile<32xi32, UB>, %arg1: i32) -> !asctile.tile<32xi32, UB> {
// CHECK-NEXT:  %0 = asctile.shls %arg0, %arg1 : !asctile.tile<32xi32, UB>
// CHECK-NEXT:  return %0 : !asctile.tile<32xi32, UB>
// CHECK-NEXT:}
func.func public @shli_splat(%arg0: !asctile.tile<32xi32, UB>, %arg1: i32) -> !asctile.tile<32xi32, UB> {
  %0 = asctile.splat %arg1 : !asctile.tile<32xi32, UB>
  %1 = arith.shli %arg0, %0 : !asctile.tile<32xi32, UB>
  return %1 : !asctile.tile<32xi32, UB>
}

// CHECK-LABEL: func.func public @shrsi_constant(%arg0: !asctile.tile<32xi32, UB>) -> !asctile.tile<32xi32, UB> {
// CHECK-NEXT:  %c2_i32 = arith.constant 2 : i32
// CHECK-NEXT:  %0 = asctile.shrs %arg0, %c2_i32 : !asctile.tile<32xi32, UB>
// CHECK-NEXT:  return %0 : !asctile.tile<32xi32, UB>
// CHECK-NEXT:}
func.func public @shrsi_constant(%arg0: !asctile.tile<32xi32, UB>) -> !asctile.tile<32xi32, UB> {
  %cst = arith.constant dense<2> : !asctile.tile<32xi32, UB>
  %0 = arith.shrsi %arg0, %cst : !asctile.tile<32xi32, UB>
  return %0 : !asctile.tile<32xi32, UB>
}

// CHECK-LABEL: func.func public @shrsi_splat(%arg0: !asctile.tile<32xi32, UB>, %arg1: i32) -> !asctile.tile<32xi32, UB> {
// CHECK-NEXT:  %0 = asctile.shrs %arg0, %arg1 : !asctile.tile<32xi32, UB>
// CHECK-NEXT:  return %0 : !asctile.tile<32xi32, UB>
// CHECK-NEXT:}
func.func public @shrsi_splat(%arg0: !asctile.tile<32xi32, UB>, %arg1: i32) -> !asctile.tile<32xi32, UB> {
  %0 = asctile.splat %arg1 : !asctile.tile<32xi32, UB>
  %1 = arith.shrsi %arg0, %0 : !asctile.tile<32xi32, UB>
  return %1 : !asctile.tile<32xi32, UB>
}

// CHECK-LABEL: func.func public @max_with_zero_lhs_f32(%arg0: !asctile.tile<32xf32, UB>) -> !asctile.tile<32xf32, UB> {
// CHECK-NEXT:  %0 = asctile.relu %arg0 : !asctile.tile<32xf32, UB>
// CHECK-NEXT:  return %0 : !asctile.tile<32xf32, UB>
// CHECK-NEXT:}
func.func public @max_with_zero_lhs_f32(%arg0: !asctile.tile<32xf32, UB>) -> !asctile.tile<32xf32, UB> {
  %cst = arith.constant dense<0.0> : !asctile.tile<32xf32, UB>
  %0 = arith.maximumf %cst, %arg0 : !asctile.tile<32xf32, UB>
  return %0 : !asctile.tile<32xf32, UB>
}

// CHECK-LABEL: func.func public @max_with_zero_rhs_f32(%arg0: !asctile.tile<32xf32, UB>) -> !asctile.tile<32xf32, UB> {
// CHECK-NEXT:  %0 = asctile.relu %arg0 : !asctile.tile<32xf32, UB>
// CHECK-NEXT:  return %0 : !asctile.tile<32xf32, UB>
// CHECK-NEXT:}
func.func public @max_with_zero_rhs_f32(%arg0: !asctile.tile<32xf32, UB>) -> !asctile.tile<32xf32, UB> {
  %cst = arith.constant 0.0 : f32
  %0 = asctile.splat %cst : !asctile.tile<32xf32, UB>
  %1 = arith.maximumf %arg0, %0 : !asctile.tile<32xf32, UB>
  return %1 : !asctile.tile<32xf32, UB>
}

// CHECK-LABEL: func.func public @max_with_zero_lhs_i32(%arg0: !asctile.tile<32xi32, UB>) -> !asctile.tile<32xi32, UB> {
// CHECK-NEXT:  %0 = asctile.relu %arg0 : !asctile.tile<32xi32, UB>
// CHECK-NEXT:  return %0 : !asctile.tile<32xi32, UB>
// CHECK-NEXT:}
func.func public @max_with_zero_lhs_i32(%arg0: !asctile.tile<32xi32, UB>) -> !asctile.tile<32xi32, UB> {
  %cst = arith.constant 0 : i32
  %0 = asctile.splat %cst : !asctile.tile<32xi32, UB>
  %1 = arith.maxsi %0, %arg0 : !asctile.tile<32xi32, UB>
  return %1 : !asctile.tile<32xi32, UB>
}

// CHECK-LABEL: func.func public @max_with_zero_rhs_i32(%arg0: !asctile.tile<32xi32, UB>) -> !asctile.tile<32xi32, UB> {
// CHECK-NEXT:  %0 = asctile.relu %arg0 : !asctile.tile<32xi32, UB>
// CHECK-NEXT:  return %0 : !asctile.tile<32xi32, UB>
// CHECK-NEXT:}
func.func public @max_with_zero_rhs_i32(%arg0: !asctile.tile<32xi32, UB>) -> !asctile.tile<32xi32, UB> {
  %cst = arith.constant dense<0> : !asctile.tile<32xi32, UB>
  %0 = arith.maxsi %arg0, %cst : !asctile.tile<32xi32, UB>
  return %0 : !asctile.tile<32xi32, UB>
}

// CHECK-LABEL: func.func public @max_with_non_zero_lhs_f32(%arg0: !asctile.tile<32xf32, UB>) -> !asctile.tile<32xf32, UB> {
// CHECK-NEXT:  %cst = arith.constant 1.000000e+00 : f32
// CHECK-NEXT:  %0 = asctile.maxs %arg0, %cst : !asctile.tile<32xf32, UB>
// CHECK-NEXT:  return %0 : !asctile.tile<32xf32, UB>
// CHECK-NEXT:}
func.func public @max_with_non_zero_lhs_f32(%arg0: !asctile.tile<32xf32, UB>) -> !asctile.tile<32xf32, UB> {
  %cst = arith.constant dense<1.0> : !asctile.tile<32xf32, UB>
  %0 = arith.maximumf %cst, %arg0 : !asctile.tile<32xf32, UB>
  return %0 : !asctile.tile<32xf32, UB>
}

// CHECK-LABEL: func.func public @max_with_non_zero_rhs_f32(%arg0: !asctile.tile<32xf32, UB>) -> !asctile.tile<32xf32, UB> {
// CHECK-NEXT:  %cst = arith.constant 1.000000e+00 : f32
// CHECK-NEXT:  %0 = asctile.maxs %arg0, %cst : !asctile.tile<32xf32, UB>
// CHECK-NEXT:  return %0 : !asctile.tile<32xf32, UB>
// CHECK-NEXT:}
func.func public @max_with_non_zero_rhs_f32(%arg0: !asctile.tile<32xf32, UB>) -> !asctile.tile<32xf32, UB> {
  %cst = arith.constant dense<1.0> : !asctile.tile<32xf32, UB>
  %0 = arith.maximumf %arg0, %cst : !asctile.tile<32xf32, UB>
  return %0 : !asctile.tile<32xf32, UB>
}

// CHECK-LABEL: func.func public @max_with_non_zero_lhs_i32(%arg0: !asctile.tile<32xi32, UB>) -> !asctile.tile<32xi32, UB> {
// CHECK-NEXT:  %c1_i32 = arith.constant 1 : i32
// CHECK-NEXT:  %0 = asctile.maxs %arg0, %c1_i32 : !asctile.tile<32xi32, UB>
// CHECK-NEXT:  return %0 : !asctile.tile<32xi32, UB>
// CHECK-NEXT:}
func.func public @max_with_non_zero_lhs_i32(%arg0: !asctile.tile<32xi32, UB>) -> !asctile.tile<32xi32, UB> {
  %cst = arith.constant dense<1> : !asctile.tile<32xi32, UB>
  %0 = arith.maxsi %cst, %arg0 : !asctile.tile<32xi32, UB>
  return %0 : !asctile.tile<32xi32, UB>
}

// CHECK-LABEL: func.func public @max_with_non_zero_rhs_i32(%arg0: !asctile.tile<32xi32, UB>) -> !asctile.tile<32xi32, UB> {
// CHECK-NEXT:  %c1_i32 = arith.constant 1 : i32
// CHECK-NEXT:  %0 = asctile.maxs %arg0, %c1_i32 : !asctile.tile<32xi32, UB>
// CHECK-NEXT:  return %0 : !asctile.tile<32xi32, UB>
// CHECK-NEXT:}
func.func public @max_with_non_zero_rhs_i32(%arg0: !asctile.tile<32xi32, UB>) -> !asctile.tile<32xi32, UB> {
  %cst = arith.constant dense<1> : !asctile.tile<32xi32, UB>
  %0 = arith.maxsi %arg0, %cst : !asctile.tile<32xi32, UB>
  return %0 : !asctile.tile<32xi32, UB>
}

// CHECK-LABEL: func.func public @cmp_lt_scalar_rhs(%arg0: !asctile.tile<32xf32, UB>) -> !asctile.tile<32xi1, UB> {
// CHECK-NEXT:  %cst = arith.constant 1.57070315 : f32
// CHECK-NEXT:  %0 = asctile.cmps LT %arg0, %cst : !asctile.tile<32xf32, UB>
// CHECK-NEXT:  return %0 : !asctile.tile<32xi1, UB>
// CHECK-NEXT:}
func.func public @cmp_lt_scalar_rhs(%arg0: !asctile.tile<32xf32, UB>) -> !asctile.tile<32xi1, UB> {
  %cst = arith.constant dense<1.57070313> : !asctile.tile<32xf32, UB>
  %0 = asctile.cmp LT %arg0, %cst : !asctile.tile<32xf32, UB>
  return %0 : !asctile.tile<32xi1, UB>
}

// CHECK-LABEL: func.func public @cmp_gt_scalar_rhs(%arg0: !asctile.tile<32xf32, UB>) -> !asctile.tile<32xi1, UB> {
// CHECK-NEXT:  %cst = arith.constant 1.57070315 : f32
// CHECK-NEXT:  %0 = asctile.cmps GT %arg0, %cst : !asctile.tile<32xf32, UB>
// CHECK-NEXT:  return %0 : !asctile.tile<32xi1, UB>
// CHECK-NEXT:}
func.func public @cmp_gt_scalar_rhs(%arg0: !asctile.tile<32xf32, UB>) -> !asctile.tile<32xi1, UB> {
  %cst = arith.constant dense<1.57070313> : !asctile.tile<32xf32, UB>
  %0 = asctile.cmp GT %arg0, %cst : !asctile.tile<32xf32, UB>
  return %0 : !asctile.tile<32xi1, UB>
}

// CHECK-LABEL: func.func public @cmp_eq_scalar_rhs(%arg0: !asctile.tile<32xf32, UB>) -> !asctile.tile<32xi1, UB> {
// CHECK-NEXT:  %cst = arith.constant 1.57070315 : f32
// CHECK-NEXT:  %0 = asctile.cmps EQ %arg0, %cst : !asctile.tile<32xf32, UB>
// CHECK-NEXT:  return %0 : !asctile.tile<32xi1, UB>
// CHECK-NEXT:}
func.func public @cmp_eq_scalar_rhs(%arg0: !asctile.tile<32xf32, UB>) -> !asctile.tile<32xi1, UB> {
  %cst = arith.constant dense<1.57070313> : !asctile.tile<32xf32, UB>
  %0 = asctile.cmp EQ %arg0, %cst : !asctile.tile<32xf32, UB>
  return %0 : !asctile.tile<32xi1, UB>
}

// CHECK-LABEL: func.func public @cmp_ne_scalar_rhs(%arg0: !asctile.tile<32xf32, UB>) -> !asctile.tile<32xi1, UB> {
// CHECK-NEXT:  %cst = arith.constant 1.57070315 : f32
// CHECK-NEXT:  %0 = asctile.cmps NE %arg0, %cst : !asctile.tile<32xf32, UB>
// CHECK-NEXT:  return %0 : !asctile.tile<32xi1, UB>
// CHECK-NEXT:}
func.func public @cmp_ne_scalar_rhs(%arg0: !asctile.tile<32xf32, UB>) -> !asctile.tile<32xi1, UB> {
  %cst = arith.constant dense<1.57070313> : !asctile.tile<32xf32, UB>
  %0 = asctile.cmp NE %arg0, %cst : !asctile.tile<32xf32, UB>
  return %0 : !asctile.tile<32xi1, UB>
}

// CHECK-LABEL: func.func public @cmp_le_scalar_rhs(%arg0: !asctile.tile<32xf32, UB>) -> !asctile.tile<32xi1, UB> {
// CHECK-NEXT:  %cst = arith.constant 1.57070315 : f32
// CHECK-NEXT:  %0 = asctile.cmps LE %arg0, %cst : !asctile.tile<32xf32, UB>
// CHECK-NEXT:  return %0 : !asctile.tile<32xi1, UB>
// CHECK-NEXT:}
func.func public @cmp_le_scalar_rhs(%arg0: !asctile.tile<32xf32, UB>) -> !asctile.tile<32xi1, UB> {
  %cst = arith.constant dense<1.57070313> : !asctile.tile<32xf32, UB>
  %0 = asctile.cmp LE %arg0, %cst : !asctile.tile<32xf32, UB>
  return %0 : !asctile.tile<32xi1, UB>
}

// CHECK-LABEL: func.func public @cmp_ge_scalar_rhs(%arg0: !asctile.tile<32xf32, UB>) -> !asctile.tile<32xi1, UB> {
// CHECK-NEXT:  %cst = arith.constant 1.57070315 : f32
// CHECK-NEXT:  %0 = asctile.cmps GE %arg0, %cst : !asctile.tile<32xf32, UB>
// CHECK-NEXT:  return %0 : !asctile.tile<32xi1, UB>
// CHECK-NEXT:}
func.func public @cmp_ge_scalar_rhs(%arg0: !asctile.tile<32xf32, UB>) -> !asctile.tile<32xi1, UB> {
  %cst = arith.constant dense<1.57070313> : !asctile.tile<32xf32, UB>
  %0 = asctile.cmp GE %arg0, %cst : !asctile.tile<32xf32, UB>
  return %0 : !asctile.tile<32xi1, UB>
}

// CHECK-LABEL: func.func public @cmp_lt_scalar_lhs(%arg0: !asctile.tile<32xf32, UB>) -> !asctile.tile<32xi1, UB> {
// CHECK-NEXT:  %cst = arith.constant 1.57070315 : f32
// CHECK-NEXT:  %0 = asctile.cmps GT %arg0, %cst : !asctile.tile<32xf32, UB>
// CHECK-NEXT:  return %0 : !asctile.tile<32xi1, UB>
// CHECK-NEXT:}
func.func public @cmp_lt_scalar_lhs(%arg0: !asctile.tile<32xf32, UB>) -> !asctile.tile<32xi1, UB> {
  %cst = arith.constant dense<1.57070313> : !asctile.tile<32xf32, UB>
  %0 = asctile.cmp LT %cst, %arg0 : !asctile.tile<32xf32, UB>
  return %0 : !asctile.tile<32xi1, UB>
}

// CHECK-LABEL: func.func public @cmp_gt_scalar_lhs(%arg0: !asctile.tile<32xf32, UB>) -> !asctile.tile<32xi1, UB> {
// CHECK-NEXT:  %cst = arith.constant 1.57070315 : f32
// CHECK-NEXT:  %0 = asctile.cmps LT %arg0, %cst : !asctile.tile<32xf32, UB>
// CHECK-NEXT:  return %0 : !asctile.tile<32xi1, UB>
// CHECK-NEXT:}
func.func public @cmp_gt_scalar_lhs(%arg0: !asctile.tile<32xf32, UB>) -> !asctile.tile<32xi1, UB> {
  %cst = arith.constant dense<1.57070313> : !asctile.tile<32xf32, UB>
  %0 = asctile.cmp GT %cst, %arg0 : !asctile.tile<32xf32, UB>
  return %0 : !asctile.tile<32xi1, UB>
}

// CHECK-LABEL: func.func public @cmp_eq_scalar_lhs(%arg0: !asctile.tile<32xf32, UB>) -> !asctile.tile<32xi1, UB> {
// CHECK-NEXT:  %cst = arith.constant 1.57070315 : f32
// CHECK-NEXT:  %0 = asctile.cmps EQ %arg0, %cst : !asctile.tile<32xf32, UB>
// CHECK-NEXT:  return %0 : !asctile.tile<32xi1, UB>
// CHECK-NEXT:}
func.func public @cmp_eq_scalar_lhs(%arg0: !asctile.tile<32xf32, UB>) -> !asctile.tile<32xi1, UB> {
  %cst = arith.constant dense<1.57070313> : !asctile.tile<32xf32, UB>
  %0 = asctile.cmp EQ %cst, %arg0 : !asctile.tile<32xf32, UB>
  return %0 : !asctile.tile<32xi1, UB>
}

// CHECK-LABEL: func.func public @cmp_ne_scalar_lhs(%arg0: !asctile.tile<32xf32, UB>) -> !asctile.tile<32xi1, UB> {
// CHECK-NEXT:  %cst = arith.constant 1.57070315 : f32
// CHECK-NEXT:  %0 = asctile.cmps NE %arg0, %cst : !asctile.tile<32xf32, UB>
// CHECK-NEXT:  return %0 : !asctile.tile<32xi1, UB>
// CHECK-NEXT:}
func.func public @cmp_ne_scalar_lhs(%arg0: !asctile.tile<32xf32, UB>) -> !asctile.tile<32xi1, UB> {
  %cst = arith.constant dense<1.57070313> : !asctile.tile<32xf32, UB>
  %0 = asctile.cmp NE %cst, %arg0 : !asctile.tile<32xf32, UB>
  return %0 : !asctile.tile<32xi1, UB>
}

// CHECK-LABEL: func.func public @cmp_le_scalar_lhs(%arg0: !asctile.tile<32xf32, UB>) -> !asctile.tile<32xi1, UB> {
// CHECK-NEXT:  %cst = arith.constant 1.57070315 : f32
// CHECK-NEXT:  %0 = asctile.cmps GE %arg0, %cst : !asctile.tile<32xf32, UB>
// CHECK-NEXT:  return %0 : !asctile.tile<32xi1, UB>
// CHECK-NEXT:}
func.func public @cmp_le_scalar_lhs(%arg0: !asctile.tile<32xf32, UB>) -> !asctile.tile<32xi1, UB> {
  %cst = arith.constant dense<1.57070313> : !asctile.tile<32xf32, UB>
  %0 = asctile.cmp LE %cst, %arg0 : !asctile.tile<32xf32, UB>
  return %0 : !asctile.tile<32xi1, UB>
}

// CHECK-LABEL: func.func public @cmp_ge_scalar_lhs(%arg0: !asctile.tile<32xf32, UB>) -> !asctile.tile<32xi1, UB> {
// CHECK-NEXT:  %cst = arith.constant 1.57070315 : f32
// CHECK-NEXT:  %0 = asctile.cmps LE %arg0, %cst : !asctile.tile<32xf32, UB>
// CHECK-NEXT:  return %0 : !asctile.tile<32xi1, UB>
// CHECK-NEXT:}
func.func public @cmp_ge_scalar_lhs(%arg0: !asctile.tile<32xf32, UB>) -> !asctile.tile<32xi1, UB> {
  %cst = arith.constant dense<1.57070313> : !asctile.tile<32xf32, UB>
  %0 = asctile.cmp GE %cst, %arg0 : !asctile.tile<32xf32, UB>
  return %0 : !asctile.tile<32xi1, UB>
}

// CHECK-LABEL: func.func public @cmp_lt_scalar_rhs_splat(%arg0: !asctile.tile<32xf32, UB>, %arg1: f32) -> !asctile.tile<32xi1, UB> {
// CHECK-NEXT:  %0 = asctile.cmps LT %arg0, %arg1 : !asctile.tile<32xf32, UB>
// CHECK-NEXT:  return %0 : !asctile.tile<32xi1, UB>
// CHECK-NEXT:}
func.func public @cmp_lt_scalar_rhs_splat(%arg0: !asctile.tile<32xf32, UB>, %arg1: f32) -> !asctile.tile<32xi1, UB> {
  %splat = asctile.splat %arg1 : !asctile.tile<32xf32, UB>
  %0 = asctile.cmp LT %arg0, %splat : !asctile.tile<32xf32, UB>
  return %0 : !asctile.tile<32xi1, UB>
}

// CHECK-LABEL: func.func public @cmp_gt_scalar_lhs_splat(%arg0: !asctile.tile<32xf32, UB>, %arg1: f32) -> !asctile.tile<32xi1, UB> {
// CHECK-NEXT:  %0 = asctile.cmps LT %arg0, %arg1 : !asctile.tile<32xf32, UB>
// CHECK-NEXT:  return %0 : !asctile.tile<32xi1, UB>
// CHECK-NEXT:}
func.func public @cmp_gt_scalar_lhs_splat(%arg0: !asctile.tile<32xf32, UB>, %arg1: f32) -> !asctile.tile<32xi1, UB> {
  %splat = asctile.splat %arg1 : !asctile.tile<32xf32, UB>
  %0 = asctile.cmp GT %splat, %arg0 : !asctile.tile<32xf32, UB>
  return %0 : !asctile.tile<32xi1, UB>
}

// CHECK-LABEL: func.func public @cmp_no_scalarization(%arg0: !asctile.tile<32xf32, UB>, %arg1: !asctile.tile<32xf32, UB>) -> !asctile.tile<32xi1, UB> {
// CHECK-NEXT:  %0 = asctile.cmp LT %arg0, %arg1 : !asctile.tile<32xf32, UB>
// CHECK-NEXT:  return %0 : !asctile.tile<32xi1, UB>
// CHECK-NEXT:}
func.func public @cmp_no_scalarization(%arg0: !asctile.tile<32xf32, UB>, %arg1: !asctile.tile<32xf32, UB>) -> !asctile.tile<32xi1, UB> {
  %0 = asctile.cmp LT %arg0, %arg1 : !asctile.tile<32xf32, UB>
  return %0 : !asctile.tile<32xi1, UB>
}
