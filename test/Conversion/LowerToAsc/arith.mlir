// RUN: ascir-opt -asclower-arith -canonicalize %s | FileCheck %s

// CHECK-LABEL: func.func @lower_splat_constant() -> tensor<3xf32, #asctile.local<UB>> {
// CHECK:       %cst = arith.constant 8.000000e+00 : f32
// CHECK:       %0 = ascendc.local_tensor_auto veccalc() : <3xf32>
// CHECK-NEXT:  %1 = builtin.unrealized_conversion_cast %0 : !ascendc.local_tensor<3xf32> to tensor<3xf32, #asctile.local<UB>>
// CHECK-NEXT:  ascendc.duplicate_l2 %0, %cst, %c3_i64 : !ascendc.local_tensor<3xf32>, f32, i64
// CHECK-NEXT:  return %1 : tensor<3xf32, #asctile.local<UB>>
// CHECK-NEXT:}
func.func @lower_splat_constant() -> tensor<3xf32, #asctile.local<UB>> {
  %0 = arith.constant dense<8.0> : tensor<3xf32, #asctile.local<UB>>
  return %0 : tensor<3xf32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @lower_dense_constant() -> tensor<3xf32, #asctile.local<UB>> {
// CHECK:       %cst = arith.constant 3.000000e+01 : f32
// CHECK:       %cst_0 = arith.constant 2.000000e+00 : f32
// CHECK:       %cst_1 = arith.constant 1.500000e+00 : f32
// CHECK-NEXT:  %0 = ascendc.local_tensor_auto veccalc() : <3xf32>
// CHECK-NEXT:  %1 = builtin.unrealized_conversion_cast %0 : !ascendc.local_tensor<3xf32> to tensor<3xf32, #asctile.local<UB>>
// CHECK-NEXT:  ascendc.local_tensor.set_value %0, %c0_i32, %cst_1 : !ascendc.local_tensor<3xf32>, i32, f32
// CHECK-NEXT:  ascendc.local_tensor.set_value %0, %c1_i32, %cst_0 : !ascendc.local_tensor<3xf32>, i32, f32
// CHECK-NEXT:  ascendc.local_tensor.set_value %0, %c2_i32, %cst : !ascendc.local_tensor<3xf32>, i32, f32
// CHECK-NEXT:  return %1 : tensor<3xf32, #asctile.local<UB>>
// CHECK-NEXT:}
func.func @lower_dense_constant() -> tensor<3xf32, #asctile.local<UB>> {
  %0 = arith.constant dense<[1.5, 2.0, 30.0]> : tensor<3xf32, #asctile.local<UB>>
  return %0 : tensor<3xf32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @lower_negf(%arg0: tensor<16xf32, #asctile.local<UB>>) -> tensor<16xf32, #asctile.local<UB>> {
// CHECK:       %0 = builtin.unrealized_conversion_cast %arg0 : tensor<16xf32, #asctile.local<UB>> to !ascendc.local_tensor<16xf32>
// CHECK-NEXT:  %1 = ascendc.local_tensor_auto veccalc() : <16xf32>
// CHECK-NEXT:  %2 = builtin.unrealized_conversion_cast %1 : !ascendc.local_tensor<16xf32> to tensor<16xf32, #asctile.local<UB>>
// CHECK-NEXT:  ascendc.muls_l2 %1, %0, %cst, %c16_i64 : !ascendc.local_tensor<16xf32>, !ascendc.local_tensor<16xf32>, f32, i64
// CHECK-NEXT:  return %2 : tensor<16xf32, #asctile.local<UB>>
// CHECK-NEXT:}
func.func @lower_negf(%arg0: tensor<16xf32, #asctile.local<UB>>) -> tensor<16xf32, #asctile.local<UB>> {
  %0 = arith.negf %arg0 : tensor<16xf32, #asctile.local<UB>>
  return %0 : tensor<16xf32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @lower_select(%arg0: tensor<32xi1, #asctile.local<UB>>, %arg1: tensor<32xf32, #asctile.local<UB>>, %arg2: tensor<32xf32, #asctile.local<UB>>) -> tensor<32xf32, #asctile.local<UB>> {
// CHECK:       %0 = builtin.unrealized_conversion_cast %arg2 : tensor<32xf32, #asctile.local<UB>> to !ascendc.local_tensor<32xf32>
// CHECK-NEXT:  %1 = builtin.unrealized_conversion_cast %arg1 : tensor<32xf32, #asctile.local<UB>> to !ascendc.local_tensor<32xf32>
// CHECK-NEXT:  %2 = builtin.unrealized_conversion_cast %arg0 : tensor<32xi1, #asctile.local<UB>> to !ascendc.local_tensor<4xi8>
// CHECK-NEXT:  %3 = ascendc.local_tensor_auto veccalc() : <32xf32>
// CHECK-NEXT:  %4 = builtin.unrealized_conversion_cast %3 : !ascendc.local_tensor<32xf32> to tensor<32xf32, #asctile.local<UB>>
// CHECK-NEXT:  %5 = builtin.unrealized_conversion_cast %2 : !ascendc.local_tensor<4xi8> to !ascendc.local_tensor<4xui8>
// CHECK-NEXT:  %6 = ascendc.construct !ascendc.binary_repeat_params()
// CHECK-NEXT:  ascendc.select_l0 %3, %5, %1, %0, %c0_i64, %c0_i64, %6 {selMode = 2 : i32} : !ascendc.local_tensor<32xf32>, !ascendc.local_tensor<4xui8>, !ascendc.local_tensor<32xf32>, !ascendc.local_tensor<32xf32>, i64, i64, !ascendc.binary_repeat_params
// CHECK-NEXT:  return %4 : tensor<32xf32, #asctile.local<UB>>
// CHECK-NEXT:}
func.func @lower_select(%arg0: tensor<32xi1, #asctile.local<UB>>, %arg1: tensor<32xf32, #asctile.local<UB>>, %arg2: tensor<32xf32, #asctile.local<UB>>) -> tensor<32xf32, #asctile.local<UB>> {
  %0 = arith.select %arg0, %arg1, %arg2 : tensor<32xi1, #asctile.local<UB>>, tensor<32xf32, #asctile.local<UB>>
  return %0 : tensor<32xf32, #asctile.local<UB>>
}
