// RUN: ascir-opt -asclower-arith %s | FileCheck %s

// CHECK-LABEL: func.func @lower_splat_constant() -> !asctile.tile<3xf32, UB> {
// CHECK-NEXT:    %cst = arith.constant 8.000000e+00 : f32
// CHECK-NEXT:    %0 = ascendc.local_tensor_auto veccalc() : <3xf32>
// CHECK-NEXT:    %1 = builtin.unrealized_conversion_cast %0 : !ascendc.local_tensor<3xf32> to !asctile.tile<3xf32, UB>
// CHECK-NEXT:    %c3_i64 = arith.constant 3 : i64
// CHECK-NEXT:    ascendc.duplicate_l2 %0, %cst, %c3_i64 : !ascendc.local_tensor<3xf32>, f32, i64
// CHECK-NEXT:    return %1 : !asctile.tile<3xf32, UB>
// CHECK-NEXT: }
func.func @lower_splat_constant() -> !asctile.tile<3xf32, UB> {
  %0 = arith.constant dense<8.0> : !asctile.tile<3xf32, UB>
  return %0 : !asctile.tile<3xf32, UB>
}

// CHECK-LABEL: func.func @lower_dense_constant() -> !asctile.tile<3xf32, UB> {
// CHECK-NEXT:    %0 = ascendc.local_tensor_auto veccalc() : <3xf32>
// CHECK-NEXT:    %1 = builtin.unrealized_conversion_cast %0 : !ascendc.local_tensor<3xf32> to !asctile.tile<3xf32, UB>
// CHECK-NEXT:    %cst = arith.constant 1.500000e+00 : f32
// CHECK-NEXT:    %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:    ascendc.local_tensor.set_value %0, %c0_i32, %cst : !ascendc.local_tensor<3xf32>, i32, f32
// CHECK-NEXT:    %cst_0 = arith.constant 2.000000e+00 : f32
// CHECK-NEXT:    %c1_i32 = arith.constant 1 : i32
// CHECK-NEXT:    ascendc.local_tensor.set_value %0, %c1_i32, %cst_0 : !ascendc.local_tensor<3xf32>, i32, f32
// CHECK-NEXT:    %cst_1 = arith.constant 3.000000e+01 : f32
// CHECK-NEXT:    %c2_i32 = arith.constant 2 : i32
// CHECK-NEXT:    ascendc.local_tensor.set_value %0, %c2_i32, %cst_1 : !ascendc.local_tensor<3xf32>, i32, f32
// CHECK-NEXT:    return %1 : !asctile.tile<3xf32, UB>
// CHECK-NEXT:  }
func.func @lower_dense_constant() -> !asctile.tile<3xf32, UB> {
  %0 = arith.constant dense<[1.5, 2.0, 30.0]> : !asctile.tile<3xf32, UB>
  return %0 : !asctile.tile<3xf32, UB>
}

// CHECK-LABEL: func.func @lower_negf(%arg0: !asctile.tile<16xf32, UB>) -> !asctile.tile<16xf32, UB> {
// CHECK-NEXT:    %0 = builtin.unrealized_conversion_cast %arg0 : !asctile.tile<16xf32, UB> to !ascendc.local_tensor<16xf32>
// CHECK-NEXT:    %1 = ascendc.local_tensor_auto veccalc() : <16xf32>
// CHECK-NEXT:    %2 = builtin.unrealized_conversion_cast %1 : !ascendc.local_tensor<16xf32> to !asctile.tile<16xf32, UB>
// CHECK-NEXT:    %cst = arith.constant -1.000000e+00 : f32
// CHECK-NEXT:    %c16_i64 = arith.constant 16 : i64
// CHECK-NEXT:    ascendc.muls_l2 %1, %0, %cst, %c16_i64 : !ascendc.local_tensor<16xf32>, !ascendc.local_tensor<16xf32>, f32, i64
// CHECK-NEXT:    return %2 : !asctile.tile<16xf32, UB>
// CHECK-NEXT: }
func.func @lower_negf(%arg0: !asctile.tile<16xf32, UB>) -> !asctile.tile<16xf32, UB> {
  %0 = arith.negf %arg0 : !asctile.tile<16xf32, UB>
  return %0 : !asctile.tile<16xf32, UB>
}
