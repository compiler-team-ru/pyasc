// RUN: ascir-opt -asctile-mark-matmul-acc-with-bias %s | FileCheck %s

// CHECK-LABEL: func.func @mark_with_bias(%arg0: tensor<16x16xf16, #asctile.local<L0A>>, %arg1: tensor<16x16xf16, #asctile.local<L0B>>, %arg2: tensor<16xf32, #asctile.local<BT>>) -> tensor<16x16xf32, #asctile.local<L0C>> {
// CHECK:       %0 = asctile.accumulator %arg2 : tensor<16x16xf32, #asctile.local<L0C>>, tensor<16xf32, #asctile.local<BT>>
// CHECK-NEXT:  asctile.matmul_acc %0, %arg0, %arg1 {asctile.has_bias} : tensor<16x16xf32, #asctile.local<L0C>>, tensor<16x16xf16, #asctile.local<L0A>>, tensor<16x16xf16, #asctile.local<L0B>>
// CHECK-NEXT:  return %0 : tensor<16x16xf32, #asctile.local<L0C>>
// CHECK-NEXT:}
func.func @mark_with_bias(%arg0: tensor<16x16xf16, #asctile.local<L0A>>, %arg1: tensor<16x16xf16, #asctile.local<L0B>>, %bias: tensor<16xf32, #asctile.local<BT>>) -> tensor<16x16xf32, #asctile.local<L0C>> {
  %acc = asctile.accumulator %bias : tensor<16x16xf32, #asctile.local<L0C>>, tensor<16xf32, #asctile.local<BT>>
  asctile.matmul_acc %acc, %arg0, %arg1 : tensor<16x16xf32, #asctile.local<L0C>>, tensor<16x16xf16, #asctile.local<L0A>>, tensor<16x16xf16, #asctile.local<L0B>>
  return %acc : tensor<16x16xf32, #asctile.local<L0C>>
}

// CHECK-LABEL: func.func @no_mark_without_bias(%arg0: tensor<16x16xf16, #asctile.local<L0A>>, %arg1: tensor<16x16xf16, #asctile.local<L0B>>) -> tensor<16x16xf32, #asctile.local<L0C>> {
// CHECK:       %0 = asctile.accumulator : tensor<16x16xf32, #asctile.local<L0C>>
// CHECK-NEXT:  asctile.matmul_acc %0, %arg0, %arg1 : tensor<16x16xf32, #asctile.local<L0C>>, tensor<16x16xf16, #asctile.local<L0A>>, tensor<16x16xf16, #asctile.local<L0B>>
// CHECK-NEXT:  return %0 : tensor<16x16xf32, #asctile.local<L0C>>
// CHECK-NEXT:}
func.func @no_mark_without_bias(%arg0: tensor<16x16xf16, #asctile.local<L0A>>, %arg1: tensor<16x16xf16, #asctile.local<L0B>>) -> tensor<16x16xf32, #asctile.local<L0C>> {
  %acc = asctile.accumulator : tensor<16x16xf32, #asctile.local<L0C>>
  asctile.matmul_acc %acc, %arg0, %arg1 : tensor<16x16xf32, #asctile.local<L0C>>, tensor<16x16xf16, #asctile.local<L0A>>, tensor<16x16xf16, #asctile.local<L0B>>
  return %acc : tensor<16x16xf32, #asctile.local<L0C>>
}