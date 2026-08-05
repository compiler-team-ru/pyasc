// RUN: ascir-opt -asclower-arith-binary -canonicalize -split-input-file %s | FileCheck %s

// CHECK-LABEL: func.func @lower_addf(%arg0: tensor<16xf32, #asctile.local<UB>>, %arg1: tensor<16xf32, #asctile.local<UB>>) -> tensor<16xf32, #asctile.local<UB>> {
// CHECK:       %0 = builtin.unrealized_conversion_cast %arg1 : tensor<16xf32, #asctile.local<UB>> to !ascendc.local_tensor<16xf32>
// CHECK-NEXT:  %1 = builtin.unrealized_conversion_cast %arg0 : tensor<16xf32, #asctile.local<UB>> to !ascendc.local_tensor<16xf32>
// CHECK-NEXT:  %2 = ascendc.local_tensor_auto veccalc() : <16xf32>
// CHECK-NEXT:  %3 = builtin.unrealized_conversion_cast %2 : !ascendc.local_tensor<16xf32> to tensor<16xf32, #asctile.local<UB>>
// CHECK-NEXT:  ascendc.add_l2 %2, %1, %0, %c16_i64 : !ascendc.local_tensor<16xf32>, !ascendc.local_tensor<16xf32>, !ascendc.local_tensor<16xf32>, i64
// CHECK-NEXT:  return %3 : tensor<16xf32, #asctile.local<UB>>
// CHECK-NEXT:}
func.func @lower_addf(%arg0: tensor<16xf32, #asctile.local<UB>>, %arg1: tensor<16xf32, #asctile.local<UB>>) -> tensor<16xf32, #asctile.local<UB>> {
  %0 = arith.addf %arg0, %arg1 : tensor<16xf32, #asctile.local<UB>>
  return %0 : tensor<16xf32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @lower_addi(%arg0: tensor<16xi32, #asctile.local<UB>>, %arg1: tensor<16xi32, #asctile.local<UB>>) -> tensor<16xi32, #asctile.local<UB>> {
// CHECK:       %0 = builtin.unrealized_conversion_cast %arg1 : tensor<16xi32, #asctile.local<UB>> to !ascendc.local_tensor<16xi32>
// CHECK-NEXT:  %1 = builtin.unrealized_conversion_cast %arg0 : tensor<16xi32, #asctile.local<UB>> to !ascendc.local_tensor<16xi32>
// CHECK-NEXT:  %2 = ascendc.local_tensor_auto veccalc() : <16xi32>
// CHECK-NEXT:  %3 = builtin.unrealized_conversion_cast %2 : !ascendc.local_tensor<16xi32> to tensor<16xi32, #asctile.local<UB>>
// CHECK-NEXT:  ascendc.add_l2 %2, %1, %0, %c16_i64 : !ascendc.local_tensor<16xi32>, !ascendc.local_tensor<16xi32>, !ascendc.local_tensor<16xi32>, i64
// CHECK-NEXT:  return %3 : tensor<16xi32, #asctile.local<UB>>
// CHECK-NEXT:}
func.func @lower_addi(%arg0: tensor<16xi32, #asctile.local<UB>>, %arg1: tensor<16xi32, #asctile.local<UB>>) -> tensor<16xi32, #asctile.local<UB>> {
  %0 = arith.addi %arg0, %arg1 : tensor<16xi32, #asctile.local<UB>>
  return %0 : tensor<16xi32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @lower_subf(%arg0: tensor<16xf32, #asctile.local<UB>>, %arg1: tensor<16xf32, #asctile.local<UB>>) -> tensor<16xf32, #asctile.local<UB>> {
// CHECK:       %0 = builtin.unrealized_conversion_cast %arg1 : tensor<16xf32, #asctile.local<UB>> to !ascendc.local_tensor<16xf32>
// CHECK-NEXT:  %1 = builtin.unrealized_conversion_cast %arg0 : tensor<16xf32, #asctile.local<UB>> to !ascendc.local_tensor<16xf32>
// CHECK-NEXT:  %2 = ascendc.local_tensor_auto veccalc() : <16xf32>
// CHECK-NEXT:  %3 = builtin.unrealized_conversion_cast %2 : !ascendc.local_tensor<16xf32> to tensor<16xf32, #asctile.local<UB>>
// CHECK-NEXT:  ascendc.sub_l2 %2, %1, %0, %c16_i64 : !ascendc.local_tensor<16xf32>, !ascendc.local_tensor<16xf32>, !ascendc.local_tensor<16xf32>, i64
// CHECK-NEXT:  return %3 : tensor<16xf32, #asctile.local<UB>>
// CHECK-NEXT:}
func.func @lower_subf(%arg0: tensor<16xf32, #asctile.local<UB>>, %arg1: tensor<16xf32, #asctile.local<UB>>) -> tensor<16xf32, #asctile.local<UB>> {
  %0 = arith.subf %arg0, %arg1 : tensor<16xf32, #asctile.local<UB>>
  return %0 : tensor<16xf32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @lower_subi(%arg0: tensor<16xi32, #asctile.local<UB>>, %arg1: tensor<16xi32, #asctile.local<UB>>) -> tensor<16xi32, #asctile.local<UB>> {
// CHECK:       %0 = builtin.unrealized_conversion_cast %arg1 : tensor<16xi32, #asctile.local<UB>> to !ascendc.local_tensor<16xi32>
// CHECK-NEXT:  %1 = builtin.unrealized_conversion_cast %arg0 : tensor<16xi32, #asctile.local<UB>> to !ascendc.local_tensor<16xi32>
// CHECK-NEXT:  %2 = ascendc.local_tensor_auto veccalc() : <16xi32>
// CHECK-NEXT:  %3 = builtin.unrealized_conversion_cast %2 : !ascendc.local_tensor<16xi32> to tensor<16xi32, #asctile.local<UB>>
// CHECK-NEXT:  ascendc.sub_l2 %2, %1, %0, %c16_i64 : !ascendc.local_tensor<16xi32>, !ascendc.local_tensor<16xi32>, !ascendc.local_tensor<16xi32>, i64
// CHECK-NEXT:  return %3 : tensor<16xi32, #asctile.local<UB>>
// CHECK-NEXT:}
func.func @lower_subi(%arg0: tensor<16xi32, #asctile.local<UB>>, %arg1: tensor<16xi32, #asctile.local<UB>>) -> tensor<16xi32, #asctile.local<UB>> {
  %0 = arith.subi %arg0, %arg1 : tensor<16xi32, #asctile.local<UB>>
  return %0 : tensor<16xi32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @lower_mulf(%arg0: tensor<16xf32, #asctile.local<UB>>, %arg1: tensor<16xf32, #asctile.local<UB>>) -> tensor<16xf32, #asctile.local<UB>> {
// CHECK:       %0 = builtin.unrealized_conversion_cast %arg1 : tensor<16xf32, #asctile.local<UB>> to !ascendc.local_tensor<16xf32>
// CHECK-NEXT:  %1 = builtin.unrealized_conversion_cast %arg0 : tensor<16xf32, #asctile.local<UB>> to !ascendc.local_tensor<16xf32>
// CHECK-NEXT:  %2 = ascendc.local_tensor_auto veccalc() : <16xf32>
// CHECK-NEXT:  %3 = builtin.unrealized_conversion_cast %2 : !ascendc.local_tensor<16xf32> to tensor<16xf32, #asctile.local<UB>>
// CHECK-NEXT:  ascendc.mul_l2 %2, %1, %0, %c16_i64 : !ascendc.local_tensor<16xf32>, !ascendc.local_tensor<16xf32>, !ascendc.local_tensor<16xf32>, i64
// CHECK-NEXT:  return %3 : tensor<16xf32, #asctile.local<UB>>
// CHECK-NEXT:}
func.func @lower_mulf(%arg0: tensor<16xf32, #asctile.local<UB>>, %arg1: tensor<16xf32, #asctile.local<UB>>) -> tensor<16xf32, #asctile.local<UB>> {
  %0 = arith.mulf %arg0, %arg1 : tensor<16xf32, #asctile.local<UB>>
  return %0 : tensor<16xf32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @lower_muli(%arg0: tensor<16xi32, #asctile.local<UB>>, %arg1: tensor<16xi32, #asctile.local<UB>>) -> tensor<16xi32, #asctile.local<UB>> {
// CHECK:       %0 = builtin.unrealized_conversion_cast %arg1 : tensor<16xi32, #asctile.local<UB>> to !ascendc.local_tensor<16xi32>
// CHECK-NEXT:  %1 = builtin.unrealized_conversion_cast %arg0 : tensor<16xi32, #asctile.local<UB>> to !ascendc.local_tensor<16xi32>
// CHECK-NEXT:  %2 = ascendc.local_tensor_auto veccalc() : <16xi32>
// CHECK-NEXT:  %3 = builtin.unrealized_conversion_cast %2 : !ascendc.local_tensor<16xi32> to tensor<16xi32, #asctile.local<UB>>
// CHECK-NEXT:  ascendc.mul_l2 %2, %1, %0, %c16_i64 : !ascendc.local_tensor<16xi32>, !ascendc.local_tensor<16xi32>, !ascendc.local_tensor<16xi32>, i64
// CHECK-NEXT:  return %3 : tensor<16xi32, #asctile.local<UB>>
// CHECK-NEXT:}
func.func @lower_muli(%arg0: tensor<16xi32, #asctile.local<UB>>, %arg1: tensor<16xi32, #asctile.local<UB>>) -> tensor<16xi32, #asctile.local<UB>> {
  %0 = arith.muli %arg0, %arg1 : tensor<16xi32, #asctile.local<UB>>
  return %0 : tensor<16xi32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @lower_divf(%arg0: tensor<16xf32, #asctile.local<UB>>, %arg1: tensor<16xf32, #asctile.local<UB>>) -> tensor<16xf32, #asctile.local<UB>> {
// CHECK:       %0 = builtin.unrealized_conversion_cast %arg1 : tensor<16xf32, #asctile.local<UB>> to !ascendc.local_tensor<16xf32>
// CHECK-NEXT:  %1 = builtin.unrealized_conversion_cast %arg0 : tensor<16xf32, #asctile.local<UB>> to !ascendc.local_tensor<16xf32>
// CHECK-NEXT:  %2 = ascendc.local_tensor_auto veccalc() : <16xf32>
// CHECK-NEXT:  %3 = builtin.unrealized_conversion_cast %2 : !ascendc.local_tensor<16xf32> to tensor<16xf32, #asctile.local<UB>>
// CHECK-NEXT:  ascendc.div_l2 %2, %1, %0, %c16_i64 : !ascendc.local_tensor<16xf32>, !ascendc.local_tensor<16xf32>, !ascendc.local_tensor<16xf32>, i64
// CHECK-NEXT:  return %3 : tensor<16xf32, #asctile.local<UB>>
// CHECK-NEXT:}
func.func @lower_divf(%arg0: tensor<16xf32, #asctile.local<UB>>, %arg1: tensor<16xf32, #asctile.local<UB>>) -> tensor<16xf32, #asctile.local<UB>> {
  %0 = arith.divf %arg0, %arg1 : tensor<16xf32, #asctile.local<UB>>
  return %0 : tensor<16xf32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @lower_maximumf(%arg0: tensor<16xf32, #asctile.local<UB>>, %arg1: tensor<16xf32, #asctile.local<UB>>) -> tensor<16xf32, #asctile.local<UB>> {
// CHECK:       %0 = builtin.unrealized_conversion_cast %arg1 : tensor<16xf32, #asctile.local<UB>> to !ascendc.local_tensor<16xf32>
// CHECK-NEXT:  %1 = builtin.unrealized_conversion_cast %arg0 : tensor<16xf32, #asctile.local<UB>> to !ascendc.local_tensor<16xf32>
// CHECK-NEXT:  %2 = ascendc.local_tensor_auto veccalc() : <16xf32>
// CHECK-NEXT:  %3 = builtin.unrealized_conversion_cast %2 : !ascendc.local_tensor<16xf32> to tensor<16xf32, #asctile.local<UB>>
// CHECK-NEXT:  ascendc.max_l2 %2, %1, %0, %c16_i64 : !ascendc.local_tensor<16xf32>, !ascendc.local_tensor<16xf32>, !ascendc.local_tensor<16xf32>, i64
// CHECK-NEXT:  return %3 : tensor<16xf32, #asctile.local<UB>>
// CHECK-NEXT:}
func.func @lower_maximumf(%arg0: tensor<16xf32, #asctile.local<UB>>, %arg1: tensor<16xf32, #asctile.local<UB>>) -> tensor<16xf32, #asctile.local<UB>> {
  %0 = arith.maximumf %arg0, %arg1 : tensor<16xf32, #asctile.local<UB>>
  return %0 : tensor<16xf32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @lower_minimumf(%arg0: tensor<16xf32, #asctile.local<UB>>, %arg1: tensor<16xf32, #asctile.local<UB>>) -> tensor<16xf32, #asctile.local<UB>> {
// CHECK:       %0 = builtin.unrealized_conversion_cast %arg1 : tensor<16xf32, #asctile.local<UB>> to !ascendc.local_tensor<16xf32>
// CHECK-NEXT:  %1 = builtin.unrealized_conversion_cast %arg0 : tensor<16xf32, #asctile.local<UB>> to !ascendc.local_tensor<16xf32>
// CHECK-NEXT:  %2 = ascendc.local_tensor_auto veccalc() : <16xf32>
// CHECK-NEXT:  %3 = builtin.unrealized_conversion_cast %2 : !ascendc.local_tensor<16xf32> to tensor<16xf32, #asctile.local<UB>>
// CHECK-NEXT:  ascendc.min_l2 %2, %1, %0, %c16_i64 : !ascendc.local_tensor<16xf32>, !ascendc.local_tensor<16xf32>, !ascendc.local_tensor<16xf32>, i64
// CHECK-NEXT:  return %3 : tensor<16xf32, #asctile.local<UB>>
// CHECK-NEXT:}
func.func @lower_minimumf(%arg0: tensor<16xf32, #asctile.local<UB>>, %arg1: tensor<16xf32, #asctile.local<UB>>) -> tensor<16xf32, #asctile.local<UB>> {
  %0 = arith.minimumf %arg0, %arg1 : tensor<16xf32, #asctile.local<UB>>
  return %0 : tensor<16xf32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @lower_maxnumf(%arg0: tensor<16xf32, #asctile.local<UB>>, %arg1: tensor<16xf32, #asctile.local<UB>>) -> tensor<16xf32, #asctile.local<UB>> {
// CHECK:       %0 = builtin.unrealized_conversion_cast %arg1 : tensor<16xf32, #asctile.local<UB>> to !ascendc.local_tensor<16xf32>
// CHECK-NEXT:  %1 = builtin.unrealized_conversion_cast %arg0 : tensor<16xf32, #asctile.local<UB>> to !ascendc.local_tensor<16xf32>
// CHECK-NEXT:  %2 = ascendc.local_tensor_auto veccalc() : <16xf32>
// CHECK-NEXT:  %3 = builtin.unrealized_conversion_cast %2 : !ascendc.local_tensor<16xf32> to tensor<16xf32, #asctile.local<UB>>
// CHECK-NEXT:  ascendc.max_l2 %2, %1, %0, %c16_i64 : !ascendc.local_tensor<16xf32>, !ascendc.local_tensor<16xf32>, !ascendc.local_tensor<16xf32>, i64
// CHECK-NEXT:  return %3 : tensor<16xf32, #asctile.local<UB>>
// CHECK-NEXT:}
func.func @lower_maxnumf(%arg0: tensor<16xf32, #asctile.local<UB>>, %arg1: tensor<16xf32, #asctile.local<UB>>) -> tensor<16xf32, #asctile.local<UB>> {
  %0 = arith.maxnumf %arg0, %arg1 : tensor<16xf32, #asctile.local<UB>>
  return %0 : tensor<16xf32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @lower_minnumf(%arg0: tensor<16xf32, #asctile.local<UB>>, %arg1: tensor<16xf32, #asctile.local<UB>>) -> tensor<16xf32, #asctile.local<UB>> {
// CHECK:       %0 = builtin.unrealized_conversion_cast %arg1 : tensor<16xf32, #asctile.local<UB>> to !ascendc.local_tensor<16xf32>
// CHECK-NEXT:  %1 = builtin.unrealized_conversion_cast %arg0 : tensor<16xf32, #asctile.local<UB>> to !ascendc.local_tensor<16xf32>
// CHECK-NEXT:  %2 = ascendc.local_tensor_auto veccalc() : <16xf32>
// CHECK-NEXT:  %3 = builtin.unrealized_conversion_cast %2 : !ascendc.local_tensor<16xf32> to tensor<16xf32, #asctile.local<UB>>
// CHECK-NEXT:  ascendc.min_l2 %2, %1, %0, %c16_i64 : !ascendc.local_tensor<16xf32>, !ascendc.local_tensor<16xf32>, !ascendc.local_tensor<16xf32>, i64
// CHECK-NEXT:  return %3 : tensor<16xf32, #asctile.local<UB>>
// CHECK-NEXT:}
func.func @lower_minnumf(%arg0: tensor<16xf32, #asctile.local<UB>>, %arg1: tensor<16xf32, #asctile.local<UB>>) -> tensor<16xf32, #asctile.local<UB>> {
  %0 = arith.minnumf %arg0, %arg1 : tensor<16xf32, #asctile.local<UB>>
  return %0 : tensor<16xf32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @lower_andi(%arg0: tensor<16xi32, #asctile.local<UB>>, %arg1: tensor<16xi32, #asctile.local<UB>>) -> tensor<16xi32, #asctile.local<UB>> {
// CHECK:       %0 = builtin.unrealized_conversion_cast %arg1 : tensor<16xi32, #asctile.local<UB>> to !ascendc.local_tensor<16xi32>
// CHECK-NEXT:  %1 = builtin.unrealized_conversion_cast %arg0 : tensor<16xi32, #asctile.local<UB>> to !ascendc.local_tensor<16xi32>
// CHECK-NEXT:  %2 = ascendc.local_tensor_auto veccalc() : <16xi32>
// CHECK-NEXT:  %3 = builtin.unrealized_conversion_cast %2 : !ascendc.local_tensor<16xi32> to tensor<16xi32, #asctile.local<UB>>
// CHECK-NEXT:  ascendc.add_l2 %2, %1, %0, %c16_i64 : !ascendc.local_tensor<16xi32>, !ascendc.local_tensor<16xi32>, !ascendc.local_tensor<16xi32>, i64
// CHECK-NEXT:  return %3 : tensor<16xi32, #asctile.local<UB>>
// CHECK-NEXT:}
func.func @lower_andi(%arg0: tensor<16xi32, #asctile.local<UB>>, %arg1: tensor<16xi32, #asctile.local<UB>>) -> tensor<16xi32, #asctile.local<UB>> {
  %0 = arith.addi %arg0, %arg1 : tensor<16xi32, #asctile.local<UB>>
  return %0 : tensor<16xi32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @lower_ori(%arg0: tensor<16xi32, #asctile.local<UB>>, %arg1: tensor<16xi32, #asctile.local<UB>>) -> tensor<16xi32, #asctile.local<UB>> {
// CHECK:       %0 = builtin.unrealized_conversion_cast %arg1 : tensor<16xi32, #asctile.local<UB>> to !ascendc.local_tensor<16xi32>
// CHECK-NEXT:  %1 = builtin.unrealized_conversion_cast %arg0 : tensor<16xi32, #asctile.local<UB>> to !ascendc.local_tensor<16xi32>
// CHECK-NEXT:  %2 = ascendc.local_tensor_auto veccalc() : <16xi32>
// CHECK-NEXT:  %3 = builtin.unrealized_conversion_cast %2 : !ascendc.local_tensor<16xi32> to tensor<16xi32, #asctile.local<UB>>
// CHECK-NEXT:  ascendc.or_l2 %2, %1, %0, %c16_i64 : !ascendc.local_tensor<16xi32>, !ascendc.local_tensor<16xi32>, !ascendc.local_tensor<16xi32>, i64
// CHECK-NEXT:  return %3 : tensor<16xi32, #asctile.local<UB>>
// CHECK-NEXT:}
func.func @lower_ori(%arg0: tensor<16xi32, #asctile.local<UB>>, %arg1: tensor<16xi32, #asctile.local<UB>>) -> tensor<16xi32, #asctile.local<UB>> {
  %0 = arith.ori %arg0, %arg1 : tensor<16xi32, #asctile.local<UB>>
  return %0 : tensor<16xi32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @lower_cmpf(%arg0: tensor<16xf32, #asctile.local<UB>>, %arg1: tensor<16xf32, #asctile.local<UB>>) -> tensor<16xi1, #asctile.local<UB>> {
// CHECK:       %0 = builtin.unrealized_conversion_cast %arg1 : tensor<16xf32, #asctile.local<UB>> to !ascendc.local_tensor<16xf32>
// CHECK-NEXT:  %1 = builtin.unrealized_conversion_cast %arg0 : tensor<16xf32, #asctile.local<UB>> to !ascendc.local_tensor<16xf32>
// CHECK-NEXT:  %2 = ascendc.local_tensor_auto veccalc() : <2xi8>
// CHECK-NEXT:  %3 = builtin.unrealized_conversion_cast %2 : !ascendc.local_tensor<2xi8> to !ascendc.local_tensor<2xui8>
// CHECK-NEXT:  %4 = builtin.unrealized_conversion_cast %3 : !ascendc.local_tensor<2xui8> to tensor<16xi1, #asctile.local<UB>>
// CHECK-NEXT:  %5 = ascendc.construct !ascendc.binary_repeat_params()
// CHECK-NEXT:  ascendc.compare_l0 %3, %1, %0, %c0_i64, %c0_i64, %5 {cmpMode = 0 : i64} : !ascendc.local_tensor<2xui8>, !ascendc.local_tensor<16xf32>, !ascendc.local_tensor<16xf32>, i64, i64, !ascendc.binary_repeat_params
// CHECK-NEXT:  return %4 : tensor<16xi1, #asctile.local<UB>>
// CHECK-NEXT:}
func.func @lower_cmpf(%arg0: tensor<16xf32, #asctile.local<UB>>, %arg1: tensor<16xf32, #asctile.local<UB>>) -> tensor<16xi1, #asctile.local<UB>> {
  %0 = arith.cmpf olt, %arg0, %arg1 : tensor<16xf32, #asctile.local<UB>>
  return %0 : tensor<16xi1, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @lower_cmpi(%arg0: tensor<16xi32, #asctile.local<UB>>, %arg1: tensor<16xi32, #asctile.local<UB>>) -> tensor<16xi1, #asctile.local<UB>> {
// CHECK:       %0 = builtin.unrealized_conversion_cast %arg1 : tensor<16xi32, #asctile.local<UB>> to !ascendc.local_tensor<16xi32>
// CHECK-NEXT:  %1 = builtin.unrealized_conversion_cast %arg0 : tensor<16xi32, #asctile.local<UB>> to !ascendc.local_tensor<16xi32>
// CHECK-NEXT:  %2 = ascendc.local_tensor_auto veccalc() : <16xf32>
// CHECK-NEXT:  %3 = ascendc.local_tensor_auto veccalc() : <16xf32>
// CHECK-NEXT:  ascendc.cast_l2 %2, %1, %c0_i64 {roundMode = 0 : i32} : !ascendc.local_tensor<16xf32>, !ascendc.local_tensor<16xi32>, i64
// CHECK-NEXT:  ascendc.cast_l2 %3, %0, %c0_i64 {roundMode = 0 : i32} : !ascendc.local_tensor<16xf32>, !ascendc.local_tensor<16xi32>, i64
// CHECK-NEXT:  %4 = ascendc.local_tensor_auto veccalc() : <2xi8>
// CHECK-NEXT:  %5 = builtin.unrealized_conversion_cast %4 : !ascendc.local_tensor<2xi8> to !ascendc.local_tensor<2xui8>
// CHECK-NEXT:  %6 = builtin.unrealized_conversion_cast %5 : !ascendc.local_tensor<2xui8> to tensor<16xi1, #asctile.local<UB>>
// CHECK-NEXT:  %7 = ascendc.construct !ascendc.binary_repeat_params()
// CHECK-NEXT:  ascendc.compare_l0 %5, %2, %3, %c0_i64, %c0_i64, %7 {cmpMode = 1 : i64} : !ascendc.local_tensor<2xui8>, !ascendc.local_tensor<16xf32>, !ascendc.local_tensor<16xf32>, i64, i64, !ascendc.binary_repeat_params
// CHECK-NEXT:  return %6 : tensor<16xi1, #asctile.local<UB>>
// CHECK-NEXT:}
func.func @lower_cmpi(%arg0: tensor<16xi32, #asctile.local<UB>>, %arg1: tensor<16xi32, #asctile.local<UB>>) -> tensor<16xi1, #asctile.local<UB>> {
  %0 = arith.cmpi sgt, %arg0, %arg1 : tensor<16xi32, #asctile.local<UB>>
  return %0 : tensor<16xi1, #asctile.local<UB>>
}

// -----

module attributes {asc.compilation_arch = "c310"} {
// CHECK-LABEL: func.func @lower_cmpi_c310(%arg0: tensor<16xi32, #asctile.local<UB>>, %arg1: tensor<16xi32, #asctile.local<UB>>) -> tensor<16xi1, #asctile.local<UB>> {
// CHECK:       %0 = builtin.unrealized_conversion_cast %arg1 : tensor<16xi32, #asctile.local<UB>> to !ascendc.local_tensor<16xi32>
// CHECK-NEXT:  %1 = builtin.unrealized_conversion_cast %arg0 : tensor<16xi32, #asctile.local<UB>> to !ascendc.local_tensor<16xi32>
// CHECK-NEXT:  %2 = ascendc.local_tensor_auto veccalc() : <2xi8>
// CHECK-NEXT:  %3 = builtin.unrealized_conversion_cast %2 : !ascendc.local_tensor<2xi8> to !ascendc.local_tensor<2xui8>
// CHECK-NEXT:  %4 = builtin.unrealized_conversion_cast %3 : !ascendc.local_tensor<2xui8> to tensor<16xi1, #asctile.local<UB>>
// CHECK-NEXT:  %5 = ascendc.construct !ascendc.binary_repeat_params()
// CHECK-NEXT:  ascendc.compare_l0 %3, %1, %0, %c0_i64, %c0_i64, %5 {cmpMode = 1 : i64} : !ascendc.local_tensor<2xui8>, !ascendc.local_tensor<16xi32>, !ascendc.local_tensor<16xi32>, i64, i64, !ascendc.binary_repeat_params
// CHECK-NEXT:  return %4 : tensor<16xi1, #asctile.local<UB>>
// CHECK-NEXT:}
  func.func @lower_cmpi_c310(%arg0: tensor<16xi32, #asctile.local<UB>>, %arg1: tensor<16xi32, #asctile.local<UB>>) -> tensor<16xi1, #asctile.local<UB>> {
    %0 = arith.cmpi sgt, %arg0, %arg1 : tensor<16xi32, #asctile.local<UB>>
    return %0 : tensor<16xi1, #asctile.local<UB>>
  }
}
