// RUN: ascir-opt --asclower-expand-math %s | FileCheck %s

// CHECK-LABEL: func.func @expand_math_rsqrt(%arg0: !asctile.tile<16xf32, UB>) -> !asctile.tile<16xf32, UB> {
// CHECK-NEXT:  %cst = arith.constant dense<1.000000e+00> : !asctile.tile<16xf32, UB>
// CHECK-NEXT:  %0 = math.sqrt %arg0 : !asctile.tile<16xf32, UB>
// CHECK-NEXT:  %1 = arith.divf %cst, %0 : !asctile.tile<16xf32, UB>
// CHECK-NEXT:  return %1 : !asctile.tile<16xf32, UB>
// CHECK-NEXT:}
func.func @expand_math_rsqrt(%arg0: !asctile.tile<16xf32, UB>) -> !asctile.tile<16xf32, UB> {
  %0 = math.rsqrt %arg0 : !asctile.tile<16xf32, UB>
  return %0 : !asctile.tile<16xf32, UB>
}

// CHECK-LABEL: func.func @expand_math_exp2(%arg0: !asctile.tile<16xf32, UB>) -> !asctile.tile<16xf32, UB> {
// CHECK-NEXT:  %cst = arith.constant dense<0.693147182> : !asctile.tile<16xf32, UB>
// CHECK-NEXT:  %0 = arith.mulf %arg0, %cst : !asctile.tile<16xf32, UB>
// CHECK-NEXT:  %1 = math.exp %0 : !asctile.tile<16xf32, UB>
// CHECK-NEXT:  return %1 : !asctile.tile<16xf32, UB>
// CHECK-NEXT:}
func.func @expand_math_exp2(%arg0: !asctile.tile<16xf32, UB>) -> !asctile.tile<16xf32, UB> {
  %0 = math.exp2 %arg0 : !asctile.tile<16xf32, UB>
  return %0 : !asctile.tile<16xf32, UB>
}
