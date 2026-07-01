// RUN: ascir-opt -asctile-unscalarize-reduction %s | FileCheck %s

// CHECK-LABEL: func.func @unscalarize_multiple_users(%arg0: tensor<16xf32, #asctile.local<UB>>) -> (tensor<16xf32, #asctile.local<UB>>, tensor<16xf32, #asctile.local<UB>>) {
// CHECK-NEXT:  %0 = asctile.reduce_as_1d <sum> %arg0 : tensor<16xf32, #asctile.local<UB>>, tensor<1xf32, #asctile.local<UB>>
// CHECK-NEXT:  %1 = asctile.broadcast %0 : tensor<1xf32, #asctile.local<UB>> to tensor<16xf32, #asctile.local<UB>>
// CHECK-NEXT:  %2 = arith.addf %arg0, %1 : tensor<16xf32, #asctile.local<UB>>
// CHECK-NEXT:  %3 = asctile.broadcast %0 : tensor<1xf32, #asctile.local<UB>> to tensor<16xf32, #asctile.local<UB>>
// CHECK-NEXT:  %4 = arith.subf %arg0, %3 : tensor<16xf32, #asctile.local<UB>>
// CHECK-NEXT:  return %2, %4 : tensor<16xf32, #asctile.local<UB>>, tensor<16xf32, #asctile.local<UB>>
// CHECK-NEXT:}
func.func @unscalarize_multiple_users(%arg0: tensor<16xf32, #asctile.local<UB>>) -> (tensor<16xf32, #asctile.local<UB>>, tensor<16xf32, #asctile.local<UB>>) {
  %0 = asctile.reduce_as_1d <sum> %arg0 : tensor<16xf32, #asctile.local<UB>>, f32
  %1 = asctile.adds %arg0, %0 : tensor<16xf32, #asctile.local<UB>>
  %2 = asctile.subs %arg0, %0 : tensor<16xf32, #asctile.local<UB>>
  return %1, %2 : tensor<16xf32, #asctile.local<UB>>, tensor<16xf32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @unscalarize_chain_users(%arg0: tensor<16xf32, #asctile.local<UB>>) -> tensor<16xf32, #asctile.local<UB>> {
// CHECK-NEXT:  %0 = asctile.reduce_as_1d <sum> %arg0 : tensor<16xf32, #asctile.local<UB>>, tensor<1xf32, #asctile.local<UB>>
// CHECK-NEXT:  %1 = asctile.broadcast %0 : tensor<1xf32, #asctile.local<UB>> to tensor<16xf32, #asctile.local<UB>>
// CHECK-NEXT:  %2 = arith.addf %arg0, %1 : tensor<16xf32, #asctile.local<UB>>
// CHECK-NEXT:  %3 = asctile.reduce_as_1d <min> %2 : tensor<16xf32, #asctile.local<UB>>, tensor<1xf32, #asctile.local<UB>>
// CHECK-NEXT:  %4 = asctile.broadcast %3 : tensor<1xf32, #asctile.local<UB>> to tensor<16xf32, #asctile.local<UB>>
// CHECK-NEXT:  %5 = arith.subf %arg0, %4 : tensor<16xf32, #asctile.local<UB>>
// CHECK-NEXT:  return %5 : tensor<16xf32, #asctile.local<UB>>
// CHECK-NEXT:}
func.func @unscalarize_chain_users(%arg0: tensor<16xf32, #asctile.local<UB>>) -> tensor<16xf32, #asctile.local<UB>> {
  %0 = asctile.reduce_as_1d <sum> %arg0 : tensor<16xf32, #asctile.local<UB>>, f32
  %1 = asctile.adds %arg0, %0 : tensor<16xf32, #asctile.local<UB>>
  %2 = asctile.reduce_as_1d <min> %1 : tensor<16xf32, #asctile.local<UB>>, f32
  %3 = asctile.subs %arg0, %2 : tensor<16xf32, #asctile.local<UB>>
  return %3 : tensor<16xf32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @skip_if_any_untransformable_users(%arg0: tensor<16xf32, #asctile.local<UB>>) -> (tensor<16xf32, #asctile.local<UB>>, f32) {
// CHECK-NEXT:  %0 = asctile.reduce_as_1d <sum> %arg0 : tensor<16xf32, #asctile.local<UB>>, f32
// CHECK-NEXT:  %1 = asctile.adds %arg0, %0 : tensor<16xf32, #asctile.local<UB>>
// CHECK-NEXT:  %2 = arith.addf %0, %0 : f32
// CHECK-NEXT:  return %1, %2 : tensor<16xf32, #asctile.local<UB>>, f32
// CHECK-NEXT:}
func.func @skip_if_any_untransformable_users(%arg0: tensor<16xf32, #asctile.local<UB>>) -> (tensor<16xf32, #asctile.local<UB>>, f32) {
  %0 = asctile.reduce_as_1d <sum> %arg0 : tensor<16xf32, #asctile.local<UB>>, f32
  %1 = asctile.adds %arg0, %0 : tensor<16xf32, #asctile.local<UB>>
  %2 = arith.addf %0, %0 : f32
  return %1, %2 : tensor<16xf32, #asctile.local<UB>>, f32
}

// CHECK-LABEL: func.func @skip_reduce_standalone(%arg0: tensor<16xf32, #asctile.local<UB>>) -> f32 {
// CHECK-NEXT:  %0 = asctile.reduce_as_1d <sum> %arg0 : tensor<16xf32, #asctile.local<UB>>, f32
// CHECK-NEXT:  return %0 : f32
// CHECK-NEXT:}
func.func @skip_reduce_standalone(%arg0: tensor<16xf32, #asctile.local<UB>>) -> f32 {
  %0 = asctile.reduce_as_1d <sum> %arg0 : tensor<16xf32, #asctile.local<UB>>, f32
  return %0 : f32
}
