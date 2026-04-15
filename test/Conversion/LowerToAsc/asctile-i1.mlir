// RUN: ascir-opt --asclower-asctile-i1 %s | FileCheck %s

// CHECK-LABEL: func.func @lower_cmp(%arg0: !asctile.tile<16xf32, UB>, %arg1: !asctile.tile<16xf32, UB>) -> !asctile.tile<16xi1, UB> {
// CHECK-NEXT:  %0 = builtin.unrealized_conversion_cast %arg1 : !asctile.tile<16xf32, UB> to !ascendc.local_tensor<16xf32>
// CHECK-NEXT:  %1 = builtin.unrealized_conversion_cast %arg0 : !asctile.tile<16xf32, UB> to !ascendc.local_tensor<16xf32>
// CHECK-NEXT:  %c0_i64 = arith.constant 0 : i64
// CHECK-NEXT:  %2 = ascendc.local_tensor_auto veccalc() : <2xi8>
// CHECK-NEXT:  %3 = builtin.unrealized_conversion_cast %2 : !ascendc.local_tensor<2xi8> to !ascendc.local_tensor<2xui8>
// CHECK-NEXT:  %4 = ascendc.reinterpret_cast %3 : !ascendc.local_tensor<2xui8> to !ascendc.local_tensor<2xui8>
// CHECK-NEXT:  %5 = builtin.unrealized_conversion_cast %4 : !ascendc.local_tensor<2xui8> to !asctile.tile<16xi1, UB>
// CHECK-NEXT:  %6 = ascendc.construct !ascendc.binary_repeat_params()
// CHECK-NEXT:  ascendc.compare_l0 %4, %1, %0, %c0_i64, %c0_i64, %6 {cmpMode = 0 : i64}
// CHECK-NEXT:  return
// CHECK-NEXT:}
func.func @lower_cmp(%arg0: !asctile.tile<16xf32, UB>, %arg1: !asctile.tile<16xf32, UB>) -> !asctile.tile<16xi1, UB> {
  %0 = asctile.cmp "LT" %arg0, %arg1 : !asctile.tile<16xf32, UB>
  return %0 : !asctile.tile<16xi1, UB>
}

// CHECK-LABEL: func.func @lower_cmps(%arg0: !asctile.tile<16xf32, UB>, %arg1: f32) -> !asctile.tile<16xi1, UB> {
// CHECK-NEXT:  %0 = builtin.unrealized_conversion_cast %arg0 : !asctile.tile<16xf32, UB> to !ascendc.local_tensor<16xf32>
// CHECK-NEXT:  %1 = ascendc.local_tensor_auto veccalc() : <2xi8>
// CHECK-NEXT:  %2 = builtin.unrealized_conversion_cast %1 : !ascendc.local_tensor<2xi8> to !ascendc.local_tensor<2xui8>
// CHECK-NEXT:  %3 = ascendc.reinterpret_cast %2 : !ascendc.local_tensor<2xui8> to !ascendc.local_tensor<2xui8>
// CHECK-NEXT:  %4 = builtin.unrealized_conversion_cast %3 : !ascendc.local_tensor<2xui8> to !asctile.tile<16xi1, UB>
// CHECK-NEXT:  %c0_i64 = arith.constant 0 : i64
// CHECK-NEXT:  %5 = ascendc.construct !ascendc.unary_repeat_params()
// CHECK-NEXT:  ascendc.compare_scalar_l0 %3, %0, %arg1, %c0_i64, %c0_i64, %5 {cmpMode = 0 : i64}
// CHECK-NEXT:  return
// CHECK-NEXT:}
func.func @lower_cmps(%arg0: !asctile.tile<16xf32, UB>, %arg1: f32) -> !asctile.tile<16xi1, UB> {
  %0 = asctile.cmps "LT" %arg0, %arg1 : !asctile.tile<16xf32, UB>
  return %0 : !asctile.tile<16xi1, UB>
}

// CHECK-LABEL: func.func @lower_select(%arg0: !asctile.tile<32xi1, UB>, %arg1: !asctile.tile<32xf32, UB>, %arg2: !asctile.tile<32xf32, UB>) -> !asctile.tile<32xf32, UB> {
// CHECK-NEXT:  %0 = builtin.unrealized_conversion_cast %arg2 : !asctile.tile<32xf32, UB> to !ascendc.local_tensor<32xf32>
// CHECK-NEXT:  %1 = builtin.unrealized_conversion_cast %arg1 : !asctile.tile<32xf32, UB> to !ascendc.local_tensor<32xf32>
// CHECK-NEXT:  %2 = builtin.unrealized_conversion_cast %arg0 : !asctile.tile<32xi1, UB> to !ascendc.local_tensor<4xi8>
// CHECK-NEXT:  %3 = ascendc.local_tensor_auto veccalc() : <32xf32>
// CHECK-NEXT:  %4 = builtin.unrealized_conversion_cast %3 : !ascendc.local_tensor<32xf32> to !asctile.tile<32xf32, UB>
// CHECK-NEXT:  %5 = builtin.unrealized_conversion_cast %2 : !ascendc.local_tensor<4xi8> to !ascendc.local_tensor<4xui8>
// CHECK-NEXT:  %6 = ascendc.reinterpret_cast %5 : !ascendc.local_tensor<4xui8> to !ascendc.local_tensor<4xui8>
// CHECK-NEXT:  %c0_i64 = arith.constant 0 : i64
// CHECK-NEXT:  %7 = ascendc.construct !ascendc.binary_repeat_params()
// CHECK-NEXT:  ascendc.select_l0 %3, %6, %1, %0, %c0_i64, %c0_i64, %7 {selMode = 2 : i32} : !ascendc.local_tensor<32xf32>, !ascendc.local_tensor<4xui8>, !ascendc.local_tensor<32xf32>, !ascendc.local_tensor<32xf32>, i64, i64, !ascendc.binary_repeat_params
// CHECK-NEXT:  return %4 : !asctile.tile<32xf32, UB>
// CHECK-NEXT:}
func.func @lower_select(%arg0: !asctile.tile<32xi1, UB>, %arg1: !asctile.tile<32xf32, UB>, %arg2: !asctile.tile<32xf32, UB>) -> !asctile.tile<32xf32, UB> {
  %0 = asctile.select %arg0, %arg1, %arg2 : !asctile.tile<32xf32, UB>
  return %0 : !asctile.tile<32xf32, UB>
}
