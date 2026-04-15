// RUN: ascir-opt --asclower-arith-binary %s | FileCheck %s

// CHECK-LABEL: func.func @lower_addf(%arg0: !asctile.tile<16xf32, UB>, %arg1: !asctile.tile<16xf32, UB>) -> !asctile.tile<16xf32, UB> {
// CHECK-NEXT:  %0 = builtin.unrealized_conversion_cast %arg1 : !asctile.tile<16xf32, UB> to !ascendc.local_tensor<16xf32>
// CHECK-NEXT:  %1 = builtin.unrealized_conversion_cast %arg0 : !asctile.tile<16xf32, UB> to !ascendc.local_tensor<16xf32>
// CHECK-NEXT:  %2 = ascendc.local_tensor_auto veccalc() : <16xf32>
// CHECK-NEXT:  %3 = builtin.unrealized_conversion_cast %2 : !ascendc.local_tensor<16xf32> to !asctile.tile<16xf32, UB>
// CHECK-NEXT:  %c16_i64 = arith.constant 16 : i64
// CHECK-NEXT:  ascendc.add_l2 %2, %1, %0, %c16_i64 : !ascendc.local_tensor<16xf32>, !ascendc.local_tensor<16xf32>, !ascendc.local_tensor<16xf32>, i64
// CHECK-NEXT:  return %3 : !asctile.tile<16xf32, UB>
// CHECK-NEXT:}
func.func @lower_addf(%arg0: !asctile.tile<16xf32, UB>, %arg1: !asctile.tile<16xf32, UB>) -> !asctile.tile<16xf32, UB> {
  %0 = arith.addf %arg0, %arg1 : !asctile.tile<16xf32, UB>
  return %0 : !asctile.tile<16xf32, UB>
}

// CHECK-LABEL: func.func @lower_addi(%arg0: !asctile.tile<16xi32, UB>, %arg1: !asctile.tile<16xi32, UB>) -> !asctile.tile<16xi32, UB> {
// CHECK-NEXT:  %0 = builtin.unrealized_conversion_cast %arg1 : !asctile.tile<16xi32, UB> to !ascendc.local_tensor<16xi32>
// CHECK-NEXT:  %1 = builtin.unrealized_conversion_cast %arg0 : !asctile.tile<16xi32, UB> to !ascendc.local_tensor<16xi32>
// CHECK-NEXT:  %2 = ascendc.local_tensor_auto veccalc() : <16xi32>
// CHECK-NEXT:  %3 = builtin.unrealized_conversion_cast %2 : !ascendc.local_tensor<16xi32> to !asctile.tile<16xi32, UB>
// CHECK-NEXT:  %c16_i64 = arith.constant 16 : i64
// CHECK-NEXT:  ascendc.add_l2 %2, %1, %0, %c16_i64 : !ascendc.local_tensor<16xi32>, !ascendc.local_tensor<16xi32>, !ascendc.local_tensor<16xi32>, i64
// CHECK-NEXT:  return %3 : !asctile.tile<16xi32, UB>
// CHECK-NEXT:}
func.func @lower_addi(%arg0: !asctile.tile<16xi32, UB>, %arg1: !asctile.tile<16xi32, UB>) -> !asctile.tile<16xi32, UB> {
  %0 = arith.addi %arg0, %arg1 : !asctile.tile<16xi32, UB>
  return %0 : !asctile.tile<16xi32, UB>
}

// CHECK-LABEL: func.func @lower_subf(%arg0: !asctile.tile<16xf32, UB>, %arg1: !asctile.tile<16xf32, UB>) -> !asctile.tile<16xf32, UB> {
// CHECK-NEXT:  %0 = builtin.unrealized_conversion_cast %arg1 : !asctile.tile<16xf32, UB> to !ascendc.local_tensor<16xf32>
// CHECK-NEXT:  %1 = builtin.unrealized_conversion_cast %arg0 : !asctile.tile<16xf32, UB> to !ascendc.local_tensor<16xf32>
// CHECK-NEXT:  %2 = ascendc.local_tensor_auto veccalc() : <16xf32>
// CHECK-NEXT:  %3 = builtin.unrealized_conversion_cast %2 : !ascendc.local_tensor<16xf32> to !asctile.tile<16xf32, UB>
// CHECK-NEXT:  %c16_i64 = arith.constant 16 : i64
// CHECK-NEXT:  ascendc.sub_l2 %2, %1, %0, %c16_i64 : !ascendc.local_tensor<16xf32>, !ascendc.local_tensor<16xf32>, !ascendc.local_tensor<16xf32>, i64
// CHECK-NEXT:  return %3 : !asctile.tile<16xf32, UB>
// CHECK-NEXT:}
func.func @lower_subf(%arg0: !asctile.tile<16xf32, UB>, %arg1: !asctile.tile<16xf32, UB>) -> !asctile.tile<16xf32, UB> {
  %0 = arith.subf %arg0, %arg1 : !asctile.tile<16xf32, UB>
  return %0 : !asctile.tile<16xf32, UB>
}

// CHECK-LABEL: func.func @lower_subi(%arg0: !asctile.tile<16xi32, UB>, %arg1: !asctile.tile<16xi32, UB>) -> !asctile.tile<16xi32, UB> {
// CHECK-NEXT:  %0 = builtin.unrealized_conversion_cast %arg1 : !asctile.tile<16xi32, UB> to !ascendc.local_tensor<16xi32>
// CHECK-NEXT:  %1 = builtin.unrealized_conversion_cast %arg0 : !asctile.tile<16xi32, UB> to !ascendc.local_tensor<16xi32>
// CHECK-NEXT:  %2 = ascendc.local_tensor_auto veccalc() : <16xi32>
// CHECK-NEXT:  %3 = builtin.unrealized_conversion_cast %2 : !ascendc.local_tensor<16xi32> to !asctile.tile<16xi32, UB>
// CHECK-NEXT:  %c16_i64 = arith.constant 16 : i64
// CHECK-NEXT:  ascendc.sub_l2 %2, %1, %0, %c16_i64 : !ascendc.local_tensor<16xi32>, !ascendc.local_tensor<16xi32>, !ascendc.local_tensor<16xi32>, i64
// CHECK-NEXT:  return %3 : !asctile.tile<16xi32, UB>
// CHECK-NEXT:}
func.func @lower_subi(%arg0: !asctile.tile<16xi32, UB>, %arg1: !asctile.tile<16xi32, UB>) -> !asctile.tile<16xi32, UB> {
  %0 = arith.subi %arg0, %arg1 : !asctile.tile<16xi32, UB>
  return %0 : !asctile.tile<16xi32, UB>
}

// CHECK-LABEL: func.func @lower_mulf(%arg0: !asctile.tile<16xf32, UB>, %arg1: !asctile.tile<16xf32, UB>) -> !asctile.tile<16xf32, UB> {
// CHECK-NEXT:  %0 = builtin.unrealized_conversion_cast %arg1 : !asctile.tile<16xf32, UB> to !ascendc.local_tensor<16xf32>
// CHECK-NEXT:  %1 = builtin.unrealized_conversion_cast %arg0 : !asctile.tile<16xf32, UB> to !ascendc.local_tensor<16xf32>
// CHECK-NEXT:  %2 = ascendc.local_tensor_auto veccalc() : <16xf32>
// CHECK-NEXT:  %3 = builtin.unrealized_conversion_cast %2 : !ascendc.local_tensor<16xf32> to !asctile.tile<16xf32, UB>
// CHECK-NEXT:  %c16_i64 = arith.constant 16 : i64
// CHECK-NEXT:  ascendc.mul_l2 %2, %1, %0, %c16_i64 : !ascendc.local_tensor<16xf32>, !ascendc.local_tensor<16xf32>, !ascendc.local_tensor<16xf32>, i64
// CHECK-NEXT:  return %3 : !asctile.tile<16xf32, UB>
// CHECK-NEXT:}
func.func @lower_mulf(%arg0: !asctile.tile<16xf32, UB>, %arg1: !asctile.tile<16xf32, UB>) -> !asctile.tile<16xf32, UB> {
  %0 = arith.mulf %arg0, %arg1 : !asctile.tile<16xf32, UB>
  return %0 : !asctile.tile<16xf32, UB>
}

// CHECK-LABEL: func.func @lower_muli(%arg0: !asctile.tile<16xi32, UB>, %arg1: !asctile.tile<16xi32, UB>) -> !asctile.tile<16xi32, UB> {
// CHECK-NEXT:  %0 = builtin.unrealized_conversion_cast %arg1 : !asctile.tile<16xi32, UB> to !ascendc.local_tensor<16xi32>
// CHECK-NEXT:  %1 = builtin.unrealized_conversion_cast %arg0 : !asctile.tile<16xi32, UB> to !ascendc.local_tensor<16xi32>
// CHECK-NEXT:  %2 = ascendc.local_tensor_auto veccalc() : <16xi32>
// CHECK-NEXT:  %3 = builtin.unrealized_conversion_cast %2 : !ascendc.local_tensor<16xi32> to !asctile.tile<16xi32, UB>
// CHECK-NEXT:  %c16_i64 = arith.constant 16 : i64
// CHECK-NEXT:  ascendc.mul_l2 %2, %1, %0, %c16_i64 : !ascendc.local_tensor<16xi32>, !ascendc.local_tensor<16xi32>, !ascendc.local_tensor<16xi32>, i64
// CHECK-NEXT:  return %3 : !asctile.tile<16xi32, UB>
// CHECK-NEXT:}
func.func @lower_muli(%arg0: !asctile.tile<16xi32, UB>, %arg1: !asctile.tile<16xi32, UB>) -> !asctile.tile<16xi32, UB> {
  %0 = arith.muli %arg0, %arg1 : !asctile.tile<16xi32, UB>
  return %0 : !asctile.tile<16xi32, UB>
}

// CHECK-LABEL: func.func @lower_divf(%arg0: !asctile.tile<16xf32, UB>, %arg1: !asctile.tile<16xf32, UB>) -> !asctile.tile<16xf32, UB> {
// CHECK-NEXT:  %0 = builtin.unrealized_conversion_cast %arg1 : !asctile.tile<16xf32, UB> to !ascendc.local_tensor<16xf32>
// CHECK-NEXT:  %1 = builtin.unrealized_conversion_cast %arg0 : !asctile.tile<16xf32, UB> to !ascendc.local_tensor<16xf32>
// CHECK-NEXT:  %2 = ascendc.local_tensor_auto veccalc() : <16xf32>
// CHECK-NEXT:  %3 = builtin.unrealized_conversion_cast %2 : !ascendc.local_tensor<16xf32> to !asctile.tile<16xf32, UB>
// CHECK-NEXT:  %c16_i64 = arith.constant 16 : i64
// CHECK-NEXT:  ascendc.div_l2 %2, %1, %0, %c16_i64 : !ascendc.local_tensor<16xf32>, !ascendc.local_tensor<16xf32>, !ascendc.local_tensor<16xf32>, i64
// CHECK-NEXT:  return %3 : !asctile.tile<16xf32, UB>
// CHECK-NEXT:}
func.func @lower_divf(%arg0: !asctile.tile<16xf32, UB>, %arg1: !asctile.tile<16xf32, UB>) -> !asctile.tile<16xf32, UB> {
  %0 = arith.divf %arg0, %arg1 : !asctile.tile<16xf32, UB>
  return %0 : !asctile.tile<16xf32, UB>
}

// CHECK-LABEL: func.func @lower_maximumf(%arg0: !asctile.tile<16xf32, UB>, %arg1: !asctile.tile<16xf32, UB>) -> !asctile.tile<16xf32, UB> {
// CHECK-NEXT:  %0 = builtin.unrealized_conversion_cast %arg1 : !asctile.tile<16xf32, UB> to !ascendc.local_tensor<16xf32>
// CHECK-NEXT:  %1 = builtin.unrealized_conversion_cast %arg0 : !asctile.tile<16xf32, UB> to !ascendc.local_tensor<16xf32>
// CHECK-NEXT:  %2 = ascendc.local_tensor_auto veccalc() : <16xf32>
// CHECK-NEXT:  %3 = builtin.unrealized_conversion_cast %2 : !ascendc.local_tensor<16xf32> to !asctile.tile<16xf32, UB>
// CHECK-NEXT:  %c16_i64 = arith.constant 16 : i64
// CHECK-NEXT:  ascendc.max_l2 %2, %1, %0, %c16_i64 : !ascendc.local_tensor<16xf32>, !ascendc.local_tensor<16xf32>, !ascendc.local_tensor<16xf32>, i64
// CHECK-NEXT:  return %3 : !asctile.tile<16xf32, UB>
// CHECK-NEXT:}
func.func @lower_maximumf(%arg0: !asctile.tile<16xf32, UB>, %arg1: !asctile.tile<16xf32, UB>) -> !asctile.tile<16xf32, UB> {
  %0 = arith.maximumf %arg0, %arg1 : !asctile.tile<16xf32, UB>
  return %0 : !asctile.tile<16xf32, UB>
}

// CHECK-LABEL: func.func @lower_minimumf(%arg0: !asctile.tile<16xf32, UB>, %arg1: !asctile.tile<16xf32, UB>) -> !asctile.tile<16xf32, UB> {
// CHECK-NEXT:  %0 = builtin.unrealized_conversion_cast %arg1 : !asctile.tile<16xf32, UB> to !ascendc.local_tensor<16xf32>
// CHECK-NEXT:  %1 = builtin.unrealized_conversion_cast %arg0 : !asctile.tile<16xf32, UB> to !ascendc.local_tensor<16xf32>
// CHECK-NEXT:  %2 = ascendc.local_tensor_auto veccalc() : <16xf32>
// CHECK-NEXT:  %3 = builtin.unrealized_conversion_cast %2 : !ascendc.local_tensor<16xf32> to !asctile.tile<16xf32, UB>
// CHECK-NEXT:  %c16_i64 = arith.constant 16 : i64
// CHECK-NEXT:  ascendc.min_l2 %2, %1, %0, %c16_i64 : !ascendc.local_tensor<16xf32>, !ascendc.local_tensor<16xf32>, !ascendc.local_tensor<16xf32>, i64
// CHECK-NEXT:  return %3 : !asctile.tile<16xf32, UB>
// CHECK-NEXT:}
func.func @lower_minimumf(%arg0: !asctile.tile<16xf32, UB>, %arg1: !asctile.tile<16xf32, UB>) -> !asctile.tile<16xf32, UB> {
  %0 = arith.minimumf %arg0, %arg1 : !asctile.tile<16xf32, UB>
  return %0 : !asctile.tile<16xf32, UB>
}

// CHECK-LABEL: func.func @lower_maxnumf(%arg0: !asctile.tile<16xf32, UB>, %arg1: !asctile.tile<16xf32, UB>) -> !asctile.tile<16xf32, UB> {
// CHECK-NEXT:  %0 = builtin.unrealized_conversion_cast %arg1 : !asctile.tile<16xf32, UB> to !ascendc.local_tensor<16xf32>
// CHECK-NEXT:  %1 = builtin.unrealized_conversion_cast %arg0 : !asctile.tile<16xf32, UB> to !ascendc.local_tensor<16xf32>
// CHECK-NEXT:  %2 = ascendc.local_tensor_auto veccalc() : <16xf32>
// CHECK-NEXT:  %3 = builtin.unrealized_conversion_cast %2 : !ascendc.local_tensor<16xf32> to !asctile.tile<16xf32, UB>
// CHECK-NEXT:  %c16_i64 = arith.constant 16 : i64
// CHECK-NEXT:  ascendc.max_l2 %2, %1, %0, %c16_i64 : !ascendc.local_tensor<16xf32>, !ascendc.local_tensor<16xf32>, !ascendc.local_tensor<16xf32>, i64
// CHECK-NEXT:  return %3 : !asctile.tile<16xf32, UB>
// CHECK-NEXT:}
func.func @lower_maxnumf(%arg0: !asctile.tile<16xf32, UB>, %arg1: !asctile.tile<16xf32, UB>) -> !asctile.tile<16xf32, UB> {
  %0 = arith.maxnumf %arg0, %arg1 : !asctile.tile<16xf32, UB>
  return %0 : !asctile.tile<16xf32, UB>
}

// CHECK-LABEL: func.func @lower_minnumf(%arg0: !asctile.tile<16xf32, UB>, %arg1: !asctile.tile<16xf32, UB>) -> !asctile.tile<16xf32, UB> {
// CHECK-NEXT:  %0 = builtin.unrealized_conversion_cast %arg1 : !asctile.tile<16xf32, UB> to !ascendc.local_tensor<16xf32>
// CHECK-NEXT:  %1 = builtin.unrealized_conversion_cast %arg0 : !asctile.tile<16xf32, UB> to !ascendc.local_tensor<16xf32>
// CHECK-NEXT:  %2 = ascendc.local_tensor_auto veccalc() : <16xf32>
// CHECK-NEXT:  %3 = builtin.unrealized_conversion_cast %2 : !ascendc.local_tensor<16xf32> to !asctile.tile<16xf32, UB>
// CHECK-NEXT:  %c16_i64 = arith.constant 16 : i64
// CHECK-NEXT:  ascendc.min_l2 %2, %1, %0, %c16_i64 : !ascendc.local_tensor<16xf32>, !ascendc.local_tensor<16xf32>, !ascendc.local_tensor<16xf32>, i64
// CHECK-NEXT:  return %3 : !asctile.tile<16xf32, UB>
// CHECK-NEXT:}
func.func @lower_minnumf(%arg0: !asctile.tile<16xf32, UB>, %arg1: !asctile.tile<16xf32, UB>) -> !asctile.tile<16xf32, UB> {
  %0 = arith.minnumf %arg0, %arg1 : !asctile.tile<16xf32, UB>
  return %0 : !asctile.tile<16xf32, UB>
}

// CHECK-LABEL: func.func @lower_andi(%arg0: !asctile.tile<16xi32, UB>, %arg1: !asctile.tile<16xi32, UB>) -> !asctile.tile<16xi32, UB> {
// CHECK-NEXT:  %0 = builtin.unrealized_conversion_cast %arg1 : !asctile.tile<16xi32, UB> to !ascendc.local_tensor<16xi32>
// CHECK-NEXT:  %1 = builtin.unrealized_conversion_cast %arg0 : !asctile.tile<16xi32, UB> to !ascendc.local_tensor<16xi32>
// CHECK-NEXT:  %2 = ascendc.local_tensor_auto veccalc() : <16xi32>
// CHECK-NEXT:  %3 = builtin.unrealized_conversion_cast %2 : !ascendc.local_tensor<16xi32> to !asctile.tile<16xi32, UB>
// CHECK-NEXT:  %c16_i64 = arith.constant 16 : i64
// CHECK-NEXT:  ascendc.add_l2 %2, %1, %0, %c16_i64 : !ascendc.local_tensor<16xi32>, !ascendc.local_tensor<16xi32>, !ascendc.local_tensor<16xi32>, i64
// CHECK-NEXT:  return %3 : !asctile.tile<16xi32, UB>
// CHECK-NEXT:}
func.func @lower_andi(%arg0: !asctile.tile<16xi32, UB>, %arg1: !asctile.tile<16xi32, UB>) -> !asctile.tile<16xi32, UB> {
  %0 = arith.addi %arg0, %arg1 : !asctile.tile<16xi32, UB>
  return %0 : !asctile.tile<16xi32, UB>
}

// CHECK-LABEL: func.func @lower_ori(%arg0: !asctile.tile<16xi32, UB>, %arg1: !asctile.tile<16xi32, UB>) -> !asctile.tile<16xi32, UB> {
// CHECK-NEXT:  %0 = builtin.unrealized_conversion_cast %arg1 : !asctile.tile<16xi32, UB> to !ascendc.local_tensor<16xi32>
// CHECK-NEXT:  %1 = builtin.unrealized_conversion_cast %arg0 : !asctile.tile<16xi32, UB> to !ascendc.local_tensor<16xi32>
// CHECK-NEXT:  %2 = ascendc.local_tensor_auto veccalc() : <16xi32>
// CHECK-NEXT:  %3 = builtin.unrealized_conversion_cast %1 : !ascendc.local_tensor<16xi32> to !ascendc.local_tensor<64xi8>
// CHECK-NEXT:  %4 = ascendc.reinterpret_cast %3 : !ascendc.local_tensor<64xi8> to !ascendc.local_tensor<64xi8>
// CHECK-NEXT:  %5 = builtin.unrealized_conversion_cast %0 : !ascendc.local_tensor<16xi32> to !ascendc.local_tensor<64xi8>
// CHECK-NEXT:  %6 = ascendc.reinterpret_cast %5 : !ascendc.local_tensor<64xi8> to !ascendc.local_tensor<64xi8>
// CHECK-NEXT:  %7 = builtin.unrealized_conversion_cast %2 : !ascendc.local_tensor<16xi32> to !ascendc.local_tensor<64xi8>
// CHECK-NEXT:  %8 = ascendc.reinterpret_cast %7 : !ascendc.local_tensor<64xi8> to !ascendc.local_tensor<64xi8>
// CHECK-NEXT:  %c1_i64 = arith.constant 1 : i64
// CHECK-NEXT:  ascendc.or_l2 %8, %4, %6, %c1_i64 : !ascendc.local_tensor<64xi8>, !ascendc.local_tensor<64xi8>, !ascendc.local_tensor<64xi8>, i64
// CHECK-NEXT:  %9 = builtin.unrealized_conversion_cast %8 : !ascendc.local_tensor<64xi8> to !ascendc.local_tensor<16xi32>
// CHECK-NEXT:  %10 = ascendc.reinterpret_cast %9 : !ascendc.local_tensor<16xi32> to !ascendc.local_tensor<16xi32>
// CHECK-NEXT:  %11 = builtin.unrealized_conversion_cast %10 : !ascendc.local_tensor<16xi32> to !asctile.tile<16xi32, UB>
// CHECK-NEXT:  return %11 : !asctile.tile<16xi32, UB>
// CHECK-NEXT:}
func.func @lower_ori(%arg0: !asctile.tile<16xi32, UB>, %arg1: !asctile.tile<16xi32, UB>) -> !asctile.tile<16xi32, UB> {
  %0 = arith.ori %arg0, %arg1 : !asctile.tile<16xi32, UB>
  return %0 : !asctile.tile<16xi32, UB>
}
