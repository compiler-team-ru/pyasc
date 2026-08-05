// RUN: ascir-opt --asclower-asctile-i1 %s | FileCheck %s

// CHECK-LABEL: func.func @lower_cmps(%arg0: tensor<16xf32, #asctile.local<UB>>, %arg1: f32) -> tensor<16xi1, #asctile.local<UB>> {
// CHECK-NEXT:  %0 = builtin.unrealized_conversion_cast %arg0 : tensor<16xf32, #asctile.local<UB>> to !ascendc.local_tensor<16xf32>
// CHECK-NEXT:  %1 = ascendc.local_tensor_auto veccalc() : <2xi8>
// CHECK-NEXT:  %2 = builtin.unrealized_conversion_cast %1 : !ascendc.local_tensor<2xi8> to !ascendc.local_tensor<2xui8>
// CHECK-NEXT:  %3 = ascendc.reinterpret_cast %2 : !ascendc.local_tensor<2xui8> to !ascendc.local_tensor<2xui8>
// CHECK-NEXT:  %4 = builtin.unrealized_conversion_cast %3 : !ascendc.local_tensor<2xui8> to tensor<16xi1, #asctile.local<UB>>
// CHECK-NEXT:  %c0_i64 = arith.constant 0 : i64
// CHECK-NEXT:  %5 = ascendc.construct !ascendc.unary_repeat_params()
// CHECK-NEXT:  ascendc.compare_scalar_l0 %3, %0, %arg1, %c0_i64, %c0_i64, %5 {cmpMode = 0 : i64}
// CHECK-NEXT:  return
// CHECK-NEXT:}
func.func @lower_cmps(%arg0: tensor<16xf32, #asctile.local<UB>>, %arg1: f32) -> tensor<16xi1, #asctile.local<UB>> {
  %0 = asctile.cmps "LT" %arg0, %arg1 : tensor<16xf32, #asctile.local<UB>>
  return %0 : tensor<16xi1, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @lower_cmps_i8(%arg0: tensor<16xi8, #asctile.local<UB>>, %arg1: i8) -> tensor<16xi1, #asctile.local<UB>> {
// CHECK-NEXT:  %0 = builtin.unrealized_conversion_cast %arg0 : tensor<16xi8, #asctile.local<UB>> to !ascendc.local_tensor<16xi8>
// CHECK-NEXT:  %1 = ascendc.local_tensor_auto veccalc() : <16xf16>
// CHECK-NEXT:  %c16_i64 = arith.constant 16 : i64
// CHECK-NEXT:  ascendc.cast_l2 %1, %0, %c16_i64 {roundMode = 0 : i32} : !ascendc.local_tensor<16xf16>, !ascendc.local_tensor<16xi8>, i64
// CHECK-NEXT:  %2 = arith.sitofp %arg1 : i8 to f16
// CHECK-NEXT:  %3 = ascendc.local_tensor_auto veccalc() : <2xi8>
// CHECK-NEXT:  %4 = builtin.unrealized_conversion_cast %3 : !ascendc.local_tensor<2xi8> to !ascendc.local_tensor<2xui8>
// CHECK-NEXT:  %5 = ascendc.reinterpret_cast %4 : !ascendc.local_tensor<2xui8> to !ascendc.local_tensor<2xui8>
// CHECK-NEXT:  %6 = builtin.unrealized_conversion_cast %5 : !ascendc.local_tensor<2xui8> to tensor<16xi1, #asctile.local<UB>>
// CHECK-NEXT:  %c0_i64 = arith.constant 0 : i64
// CHECK-NEXT:  %7 = ascendc.construct !ascendc.unary_repeat_params()
// CHECK-NEXT:  ascendc.compare_scalar_l0 %5, %1, %2, %c0_i64, %c0_i64, %7 {cmpMode = 5 : i64} : !ascendc.local_tensor<2xui8>, !ascendc.local_tensor<16xf16>, f16, i64, i64, !ascendc.unary_repeat_params
// CHECK-NEXT:  return %6 : tensor<16xi1, #asctile.local<UB>>
// CHECK-NEXT:}
func.func @lower_cmps_i8(%arg0: tensor<16xi8, #asctile.local<UB>>, %arg1: i8) -> tensor<16xi1, #asctile.local<UB>> {
  %0 = asctile.cmps "NE" %arg0, %arg1 : tensor<16xi8, #asctile.local<UB>>
  return %0 : tensor<16xi1, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @lower_cmps_i16(%arg0: tensor<16xi16, #asctile.local<UB>>, %arg1: i16) -> tensor<16xi1, #asctile.local<UB>> {
// CHECK-NEXT:  %0 = builtin.unrealized_conversion_cast %arg0 : tensor<16xi16, #asctile.local<UB>> to !ascendc.local_tensor<16xi16>
// CHECK-NEXT:  %1 = ascendc.local_tensor_auto veccalc() : <16xf16>
// CHECK-NEXT:  %c16_i64 = arith.constant 16 : i64
// CHECK-NEXT:  ascendc.cast_l2 %1, %0, %c16_i64 {roundMode = 0 : i32} : !ascendc.local_tensor<16xf16>, !ascendc.local_tensor<16xi16>, i64
// CHECK-NEXT:  %2 = arith.sitofp %arg1 : i16 to f16
// CHECK-NEXT:  %3 = ascendc.local_tensor_auto veccalc() : <2xi8>
// CHECK-NEXT:  %4 = builtin.unrealized_conversion_cast %3 : !ascendc.local_tensor<2xi8> to !ascendc.local_tensor<2xui8>
// CHECK-NEXT:  %5 = ascendc.reinterpret_cast %4 : !ascendc.local_tensor<2xui8> to !ascendc.local_tensor<2xui8>
// CHECK-NEXT:  %6 = builtin.unrealized_conversion_cast %5 : !ascendc.local_tensor<2xui8> to tensor<16xi1, #asctile.local<UB>>
// CHECK-NEXT:  %c0_i64 = arith.constant 0 : i64
// CHECK-NEXT:  %7 = ascendc.construct !ascendc.unary_repeat_params()
// CHECK-NEXT:  ascendc.compare_scalar_l0 %5, %1, %2, %c0_i64, %c0_i64, %7 {cmpMode = 5 : i64} : !ascendc.local_tensor<2xui8>, !ascendc.local_tensor<16xf16>, f16, i64, i64, !ascendc.unary_repeat_params
// CHECK-NEXT:  return %6 : tensor<16xi1, #asctile.local<UB>>
// CHECK-NEXT:}
func.func @lower_cmps_i16(%arg0: tensor<16xi16, #asctile.local<UB>>, %arg1: i16) -> tensor<16xi1, #asctile.local<UB>> {
  %0 = asctile.cmps "NE" %arg0, %arg1 : tensor<16xi16, #asctile.local<UB>>
  return %0 : tensor<16xi1, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @lower_cmps_i32(%arg0: tensor<16xi32, #asctile.local<UB>>, %arg1: i32) -> tensor<16xi1, #asctile.local<UB>> {
// CHECK-NEXT:  %0 = builtin.unrealized_conversion_cast %arg0 : tensor<16xi32, #asctile.local<UB>> to !ascendc.local_tensor<16xi32>
// CHECK-NEXT:  %1 = ascendc.local_tensor_auto veccalc() : <16xf32>
// CHECK-NEXT:  %c16_i64 = arith.constant 16 : i64
// CHECK-NEXT:  ascendc.cast_l2 %1, %0, %c16_i64 {roundMode = 0 : i32} : !ascendc.local_tensor<16xf32>, !ascendc.local_tensor<16xi32>, i64
// CHECK-NEXT:  %2 = arith.sitofp %arg1 : i32 to f32
// CHECK-NEXT:  %3 = ascendc.local_tensor_auto veccalc() : <2xi8>
// CHECK-NEXT:  %4 = builtin.unrealized_conversion_cast %3 : !ascendc.local_tensor<2xi8> to !ascendc.local_tensor<2xui8>
// CHECK-NEXT:  %5 = ascendc.reinterpret_cast %4 : !ascendc.local_tensor<2xui8> to !ascendc.local_tensor<2xui8>
// CHECK-NEXT:  %6 = builtin.unrealized_conversion_cast %5 : !ascendc.local_tensor<2xui8> to tensor<16xi1, #asctile.local<UB>>
// CHECK-NEXT:  %c0_i64 = arith.constant 0 : i64
// CHECK-NEXT:  %7 = ascendc.construct !ascendc.unary_repeat_params()
// CHECK-NEXT:  ascendc.compare_scalar_l0 %5, %1, %2, %c0_i64, %c0_i64, %7 {cmpMode = 5 : i64} : !ascendc.local_tensor<2xui8>, !ascendc.local_tensor<16xf32>, f32, i64, i64, !ascendc.unary_repeat_params
// CHECK-NEXT:  return %6 : tensor<16xi1, #asctile.local<UB>>
// CHECK-NEXT:}
func.func @lower_cmps_i32(%arg0: tensor<16xi32, #asctile.local<UB>>, %arg1: i32) -> tensor<16xi1, #asctile.local<UB>> {
  %0 = asctile.cmps "NE" %arg0, %arg1 : tensor<16xi32, #asctile.local<UB>>
  return %0 : tensor<16xi1, #asctile.local<UB>>
}
