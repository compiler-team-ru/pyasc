// RUN: ascir-opt --asclower-expand-math %s | FileCheck %s

// CHECK-LABEL: func.func @expand_math_rsqrt(%arg0: tensor<16xf32, #asctile.local<UB>>) -> tensor<16xf32, #asctile.local<UB>> {
// CHECK-NEXT:  %cst = arith.constant dense<1.000000e+00> : tensor<16xf32, #asctile.local<UB>>
// CHECK-NEXT:  %0 = math.sqrt %arg0 : tensor<16xf32, #asctile.local<UB>>
// CHECK-NEXT:  %1 = arith.divf %cst, %0 : tensor<16xf32, #asctile.local<UB>>
// CHECK-NEXT:  return %1 : tensor<16xf32, #asctile.local<UB>>
// CHECK-NEXT:}
func.func @expand_math_rsqrt(%arg0: tensor<16xf32, #asctile.local<UB>>) -> tensor<16xf32, #asctile.local<UB>> {
  %0 = math.rsqrt %arg0 : tensor<16xf32, #asctile.local<UB>>
  return %0 : tensor<16xf32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @expand_math_exp2(%arg0: tensor<16xf32, #asctile.local<UB>>) -> tensor<16xf32, #asctile.local<UB>> {
// CHECK-NEXT:  %cst = arith.constant dense<0.693147182> : tensor<16xf32, #asctile.local<UB>>
// CHECK-NEXT:  %0 = arith.mulf %arg0, %cst : tensor<16xf32, #asctile.local<UB>>
// CHECK-NEXT:  %1 = math.exp %0 : tensor<16xf32, #asctile.local<UB>>
// CHECK-NEXT:  return %1 : tensor<16xf32, #asctile.local<UB>>
// CHECK-NEXT:}
func.func @expand_math_exp2(%arg0: tensor<16xf32, #asctile.local<UB>>) -> tensor<16xf32, #asctile.local<UB>> {
  %0 = math.exp2 %arg0 : tensor<16xf32, #asctile.local<UB>>
  return %0 : tensor<16xf32, #asctile.local<UB>>
}
