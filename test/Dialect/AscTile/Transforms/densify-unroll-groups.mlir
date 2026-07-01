// RUN: ascir-opt -asctile-densify-unroll-groups %s | FileCheck %s

// CHECK-LABEL: func.func @densify_if_in_same_block(%arg0: tensor<32xf32, #asctile.global>, %arg1: i32, %arg2: i32) {
// CHECK-NEXT:  %0 = asctile.load %arg0[%arg1] : tensor<32xf32, #asctile.global>, tensor<32xf32, #asctile.local<UB>>
// CHECK-NEXT:  %1 = asctile.load %arg0[%arg2] : tensor<32xf32, #asctile.global>, tensor<32xf32, #asctile.local<UB>>
// CHECK-NEXT:  %2 = asctile.relu %0 : tensor<32xf32, #asctile.local<UB>>
// CHECK-NEXT:  %3 = asctile.relu %1 : tensor<32xf32, #asctile.local<UB>>
// CHECK-NEXT:  asctile.store %0, %arg0[%arg1] : tensor<32xf32, #asctile.local<UB>>, tensor<32xf32, #asctile.global>
// CHECK-NEXT:  asctile.store %1, %arg0[%arg2] : tensor<32xf32, #asctile.local<UB>>, tensor<32xf32, #asctile.global>
// CHECK-NEXT:  return
// CHECK-NEXT:}
func.func @densify_if_in_same_block(%arg0: tensor<32xf32, #asctile.global>, %arg1: i32, %arg2: i32) {
  %0 = asctile.load %arg0[%arg1] {asctile.unroll_group = 0 : i64} : tensor<32xf32, #asctile.global>, tensor<32xf32, #asctile.local<UB>>
  %1 = asctile.relu %0 : tensor<32xf32, #asctile.local<UB>>
  %2 = asctile.load %arg0[%arg2] {asctile.unroll_group = 0 : i64} : tensor<32xf32, #asctile.global>, tensor<32xf32, #asctile.local<UB>>
  asctile.store %0, %arg0[%arg1] {asctile.unroll_group = 1 : i64} : tensor<32xf32, #asctile.local<UB>>, tensor<32xf32, #asctile.global>
  %3 = asctile.relu %2 : tensor<32xf32, #asctile.local<UB>>
  asctile.store %2, %arg0[%arg2] {asctile.unroll_group = 1 : i64} : tensor<32xf32, #asctile.local<UB>>, tensor<32xf32, #asctile.global>
  return
}
