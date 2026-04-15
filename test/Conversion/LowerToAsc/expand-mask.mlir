// RUN: ascir-opt --asclower-expand-mask --canonicalize %s | FileCheck %s

// CHECK-LABEL: func.func @lower_bitwise_mask(%arg0: i64, %arg1: i64, %arg2: i32, %arg3: !ascendc.local_tensor<16xf32>, %arg4: !ascendc.local_tensor<16xf32>, %arg5: !ascendc.local_tensor<16xf32>) -> i64 {
// CHECK:       %0 = ascendc.construct !ascendc.binary_repeat_params()
// CHECK-NEXT:  %1 = arith.sitofp %arg2 : i32 to f32
// CHECK-NEXT:  ascendc.duplicate_l2 %arg3, %1, %c0_i32 : !ascendc.local_tensor<16xf32>, f32, i32
// CHECK-NEXT:  %2 = emitasc.mask %arg0, %arg1
// CHECK-NEXT:  ascendc.min_l0 %arg3, %arg4, %arg5, %2, %c0_i64, %0 {asc.mask_set} : !ascendc.local_tensor<16xf32>, !ascendc.local_tensor<16xf32>, !ascendc.local_tensor<16xf32>, !emitasc.mask, i64, !ascendc.binary_repeat_params
// CHECK-NEXT:  %3 = ascendc.construct !ascendc.unary_repeat_params()
// CHECK-NEXT:  %4 = arith.sitofp %arg2 : i32 to f32
// CHECK-NEXT:  ascendc.duplicate_l2 %arg3, %4, %c0_i32 : !ascendc.local_tensor<16xf32>, f32, i32
// CHECK-NEXT:  %5 = emitasc.mask %arg0, %arg1
// CHECK-NEXT:  ascendc.abs_l0 %arg3, %arg4, %5, %c0_i64, %3 {asc.mask_set} : !ascendc.local_tensor<16xf32>, !ascendc.local_tensor<16xf32>, !emitasc.mask, i64, !ascendc.unary_repeat_params
// CHECK-NEXT:  return %arg0 : i64
// CHECK-NEXT:}
func.func @lower_bitwise_mask(%arg0: i64, %arg1: i64, %arg2: i32, %arg3: !ascendc.local_tensor<16xf32>, %arg4: !ascendc.local_tensor<16xf32>, %arg5: !ascendc.local_tensor<16xf32>) -> i64 {
  asctile.bitwise_mask %arg0, %arg1, %arg2 : i32 {
    %c16_i64 = arith.constant 16 : i64
    ascendc.min_l2 %arg3, %arg4, %arg5, %c16_i64 : !ascendc.local_tensor<16xf32>, !ascendc.local_tensor<16xf32>, !ascendc.local_tensor<16xf32>, i64
    ascendc.abs_l2 %arg3, %arg4, %c16_i64 : !ascendc.local_tensor<16xf32>, !ascendc.local_tensor<16xf32>, i64
  }
  return %arg0 : i64
}

// CHECK-LABEL: func.func @lower_count_mask(%arg0: i64, %arg1: i32, %arg2: i32, %arg3: !ascendc.local_tensor<16xf32>, %arg4: !ascendc.local_tensor<16xf32>, %arg5: !ascendc.local_tensor<16xf32>) -> i32 {
// CHECK:       %0 = ascendc.construct !ascendc.binary_repeat_params()
// CHECK-NEXT:  %1 = arith.sitofp %arg1 : i32 to f32
// CHECK-NEXT:  ascendc.duplicate_l2 %arg3, %1, %c0_i32 : !ascendc.local_tensor<16xf32>, f32, i32
// CHECK-NEXT:  ascendc.min_l0 %arg3, %arg4, %arg5, %arg0, %c0_i64, %0 {asc.mask_set} : !ascendc.local_tensor<16xf32>, !ascendc.local_tensor<16xf32>, !ascendc.local_tensor<16xf32>, i64, i64, !ascendc.binary_repeat_params
// CHECK-NEXT:  %2 = ascendc.construct !ascendc.unary_repeat_params()
// CHECK-NEXT:  %3 = arith.sitofp %arg1 : i32 to f32
// CHECK-NEXT:  ascendc.duplicate_l2 %arg3, %3, %c0_i32 : !ascendc.local_tensor<16xf32>, f32, i32
// CHECK-NEXT:  ascendc.abs_l0 %arg3, %arg4, %arg0, %c0_i64, %2 {asc.mask_set} : !ascendc.local_tensor<16xf32>, !ascendc.local_tensor<16xf32>, i64, i64, !ascendc.unary_repeat_params
// CHECK-NEXT:  return %arg1 : i32
// CHECK-NEXT:}
func.func @lower_count_mask(%arg0: i64, %arg1: i32, %arg2: i32, %arg3: !ascendc.local_tensor<16xf32>, %arg4: !ascendc.local_tensor<16xf32>, %arg5: !ascendc.local_tensor<16xf32>) -> i32 {
  asctile.count_mask %arg0, %arg1 : i32 {
    %c16_i64 = arith.constant 16 : i64
    ascendc.min_l2 %arg3, %arg4, %arg5, %c16_i64 : !ascendc.local_tensor<16xf32>, !ascendc.local_tensor<16xf32>, !ascendc.local_tensor<16xf32>, i64
    ascendc.abs_l2 %arg3, %arg4, %c16_i64 : !ascendc.local_tensor<16xf32>, !ascendc.local_tensor<16xf32>, i64
  }
  return %arg1 : i32
}
