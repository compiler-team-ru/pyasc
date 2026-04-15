// RUN: ascir-opt --asclower-math %s | FileCheck %s

// CHECK-LABEL: func.func @lower_log(%arg0: !asctile.tile<777xf32, UB>) -> !asctile.tile<777xf32, UB> {
// CHECK-NEXT:    %0 = builtin.unrealized_conversion_cast %arg0 : !asctile.tile<777xf32, UB> to !ascendc.local_tensor<777xf32>
// CHECK-NEXT:    %1 = ascendc.local_tensor_auto veccalc() : <777xf32>
// CHECK-NEXT:    %2 = builtin.unrealized_conversion_cast %1 : !ascendc.local_tensor<777xf32> to !asctile.tile<777xf32, UB>
// CHECK-NEXT:    %c777_i64 = arith.constant 777 : i64
// CHECK-NEXT:    %false = arith.constant false
// CHECK-NEXT: ascendc.log %1, %0, %c777_i64, %false {operandSegmentSizes = array<i32: 1, 1, 0, 1, 1>}
func.func @lower_log(%arg0: !asctile.tile<777xf32, UB>) -> !asctile.tile<777xf32, UB> {
  %0 = math.log %arg0 : !asctile.tile<777xf32, UB>
  return %0 : !asctile.tile<777xf32, UB>
}

// CHECK-LABEL: func.func @lower_log2(%arg0: !asctile.tile<777xf32, UB>) -> !asctile.tile<777xf32, UB> {
// CHECK-NEXT:    %0 = builtin.unrealized_conversion_cast %arg0 : !asctile.tile<777xf32, UB> to !ascendc.local_tensor<777xf32>
// CHECK-NEXT:    %1 = ascendc.local_tensor_auto veccalc() : <777xf32>
// CHECK-NEXT:    %2 = builtin.unrealized_conversion_cast %1 : !ascendc.local_tensor<777xf32> to !asctile.tile<777xf32, UB>
// CHECK-NEXT:    %c777_i64 = arith.constant 777 : i64
// CHECK-NEXT:    %false = arith.constant false
// CHECK-NEXT: ascendc.log2 %1, %0, %c777_i64, %false {operandSegmentSizes = array<i32: 1, 1, 0, 1, 1>}
func.func @lower_log2(%arg0: !asctile.tile<777xf32, UB>) -> !asctile.tile<777xf32, UB> {
  %0 = math.log2 %arg0 : !asctile.tile<777xf32, UB>
  return %0 : !asctile.tile<777xf32, UB>
}

// CHECK-LABEL: func.func @lower_erf(%arg0: !asctile.tile<777xf32, UB>) -> !asctile.tile<777xf32, UB> {
// CHECK-NEXT:    %0 = builtin.unrealized_conversion_cast %arg0 : !asctile.tile<777xf32, UB> to !ascendc.local_tensor<777xf32>
// CHECK-NEXT:    %1 = ascendc.local_tensor_auto veccalc() : <777xf32>
// CHECK-NEXT:    %2 = builtin.unrealized_conversion_cast %1 : !ascendc.local_tensor<777xf32> to !asctile.tile<777xf32, UB>
// CHECK-NEXT:    %c777_i64 = arith.constant 777 : i64
// CHECK-NEXT:    %false = arith.constant false
// CHECK-NEXT: ascendc.erf %1, %0, %c777_i64, %false {operandSegmentSizes = array<i32: 1, 1, 0, 1, 1>}
func.func @lower_erf(%arg0: !asctile.tile<777xf32, UB>) -> !asctile.tile<777xf32, UB> {
  %0 = math.erf %arg0 : !asctile.tile<777xf32, UB>
  return %0 : !asctile.tile<777xf32, UB>
}

// CHECK-LABEL: func.func @lower_asin(%arg0: !asctile.tile<777xf32, UB>) -> !asctile.tile<777xf32, UB> {
// CHECK-NEXT:    %0 = builtin.unrealized_conversion_cast %arg0 : !asctile.tile<777xf32, UB> to !ascendc.local_tensor<777xf32>
// CHECK-NEXT:    %1 = ascendc.local_tensor_auto veccalc() : <777xf32>
// CHECK-NEXT:    %2 = builtin.unrealized_conversion_cast %1 : !ascendc.local_tensor<777xf32> to !asctile.tile<777xf32, UB>
// CHECK-NEXT:    %c777_i64 = arith.constant 777 : i64
// CHECK-NEXT:    %false = arith.constant false
// CHECK-NEXT: ascendc.asin %1, %0, %c777_i64, %false {operandSegmentSizes = array<i32: 1, 1, 0, 1, 1>}
func.func @lower_asin(%arg0: !asctile.tile<777xf32, UB>) -> !asctile.tile<777xf32, UB> {
  %0 = math.asin %arg0 : !asctile.tile<777xf32, UB>
  return %0 : !asctile.tile<777xf32, UB>
}

// CHECK-LABEL: func.func @lower_exp(%arg0: !asctile.tile<777xf32, UB>) -> !asctile.tile<777xf32, UB> {
// CHECK-NEXT:    %0 = builtin.unrealized_conversion_cast %arg0 : !asctile.tile<777xf32, UB> to !ascendc.local_tensor<777xf32>
// CHECK-NEXT:    %1 = ascendc.local_tensor_auto veccalc() : <777xf32>
// CHECK-NEXT:    %2 = builtin.unrealized_conversion_cast %1 : !ascendc.local_tensor<777xf32> to !asctile.tile<777xf32, UB>
// CHECK-NEXT:    %c777_i64 = arith.constant 777 : i64
// CHECK-NEXT:    ascendc.exp_l2 %1, %0, %c777_i64 : !ascendc.local_tensor<777xf32>, !ascendc.local_tensor<777xf32>, i64
// CHECK-NEXT:    return %2 : !asctile.tile<777xf32, UB>
// CHECK-NEXT: }
func.func @lower_exp(%arg0: !asctile.tile<777xf32, UB>) -> !asctile.tile<777xf32, UB> {
  %0 = math.exp %arg0 : !asctile.tile<777xf32, UB>
  return %0 : !asctile.tile<777xf32, UB>
}

// CHECK-LABEL: func.func @lower_cos(%arg0: !asctile.tile<777xf32, UB>) -> !asctile.tile<777xf32, UB> {
// CHECK-NEXT:    %0 = builtin.unrealized_conversion_cast %arg0 : !asctile.tile<777xf32, UB> to !ascendc.local_tensor<777xf32>
// CHECK-NEXT:    %1 = ascendc.local_tensor_auto veccalc() : <777xf32>
// CHECK-NEXT:    %2 = builtin.unrealized_conversion_cast %1 : !ascendc.local_tensor<777xf32> to !asctile.tile<777xf32, UB>
// CHECK-NEXT:    %c777_i64 = arith.constant 777 : i64
// CHECK-NEXT:    %false = arith.constant false
// CHECK-NEXT: ascendc.cos %1, %0, %c777_i64, %false {operandSegmentSizes = array<i32: 1, 1, 0, 1, 1>}
func.func @lower_cos(%arg0: !asctile.tile<777xf32, UB>) -> !asctile.tile<777xf32, UB> {
  %0 = math.cos %arg0 : !asctile.tile<777xf32, UB>
  return %0 : !asctile.tile<777xf32, UB>
}

// CHECK-LABEL: func.func @lower_sin(%arg0: !asctile.tile<777xf32, UB>) -> !asctile.tile<777xf32, UB> {
// CHECK-NEXT:    %0 = builtin.unrealized_conversion_cast %arg0 : !asctile.tile<777xf32, UB> to !ascendc.local_tensor<777xf32>
// CHECK-NEXT:    %1 = ascendc.local_tensor_auto veccalc() : <777xf32>
// CHECK-NEXT:    %2 = builtin.unrealized_conversion_cast %1 : !ascendc.local_tensor<777xf32> to !asctile.tile<777xf32, UB>
// CHECK-NEXT:    %c777_i64 = arith.constant 777 : i64
// CHECK-NEXT:    %false = arith.constant false
// CHECK-NEXT: ascendc.sin %1, %0, %c777_i64, %false {operandSegmentSizes = array<i32: 1, 1, 0, 1, 1>}
func.func @lower_sin(%arg0: !asctile.tile<777xf32, UB>) -> !asctile.tile<777xf32, UB> {
  %0 = math.sin %arg0 : !asctile.tile<777xf32, UB>
  return %0 : !asctile.tile<777xf32, UB>
}

// CHECK-LABEL: func.func @lower_sqrt(%arg0: !asctile.tile<777xf32, UB>) -> !asctile.tile<777xf32, UB> {
// CHECK-NEXT:    %0 = builtin.unrealized_conversion_cast %arg0 : !asctile.tile<777xf32, UB> to !ascendc.local_tensor<777xf32>
// CHECK-NEXT:    %1 = ascendc.local_tensor_auto veccalc() : <777xf32>
// CHECK-NEXT:    %2 = builtin.unrealized_conversion_cast %1 : !ascendc.local_tensor<777xf32> to !asctile.tile<777xf32, UB>
// CHECK-NEXT:    %c777_i64 = arith.constant 777 : i64
// CHECK-NEXT:    ascendc.sqrt_l2 %1, %0, %c777_i64 : !ascendc.local_tensor<777xf32>, !ascendc.local_tensor<777xf32>, i64
// CHECK-NEXT:    return %2 : !asctile.tile<777xf32, UB>
// CHECK-NEXT: }
func.func @lower_sqrt(%arg0: !asctile.tile<777xf32, UB>) -> !asctile.tile<777xf32, UB> {
  %0 = math.sqrt %arg0 : !asctile.tile<777xf32, UB>
  return %0 : !asctile.tile<777xf32, UB>
}

// CHECK-LABEL: func.func @lower_absf(%arg0: !asctile.tile<777xf32, UB>) -> !asctile.tile<777xf32, UB> {
// CHECK-NEXT:    %0 = builtin.unrealized_conversion_cast %arg0 : !asctile.tile<777xf32, UB> to !ascendc.local_tensor<777xf32>
// CHECK-NEXT:    %1 = ascendc.local_tensor_auto veccalc() : <777xf32>
// CHECK-NEXT:    %2 = builtin.unrealized_conversion_cast %1 : !ascendc.local_tensor<777xf32> to !asctile.tile<777xf32, UB>
// CHECK-NEXT:    %c777_i64 = arith.constant 777 : i64
// CHECK-NEXT:    ascendc.abs_l2 %1, %0, %c777_i64 : !ascendc.local_tensor<777xf32>, !ascendc.local_tensor<777xf32>, i64
// CHECK-NEXT:    return %2 : !asctile.tile<777xf32, UB>
// CHECK-NEXT: }
func.func @lower_absf(%arg0: !asctile.tile<777xf32, UB>) -> !asctile.tile<777xf32, UB> {
  %0 = math.absf %arg0 : !asctile.tile<777xf32, UB>
  return %0 : !asctile.tile<777xf32, UB>
}

// CHECK-LABEL: func.func @lower_ceil(%arg0: !asctile.tile<777xf32, UB>) -> !asctile.tile<777xf32, UB> {
// CHECK-NEXT:    %0 = builtin.unrealized_conversion_cast %arg0 : !asctile.tile<777xf32, UB> to !ascendc.local_tensor<777xf32>
// CHECK-NEXT:    %1 = ascendc.local_tensor_auto veccalc() : <777xf32>
// CHECK-NEXT:    %2 = builtin.unrealized_conversion_cast %1 : !ascendc.local_tensor<777xf32> to !asctile.tile<777xf32, UB>
// CHECK-NEXT:    %c777_i64 = arith.constant 777 : i64
// CHECK-NEXT:    %false = arith.constant false
// CHECK-NEXT: ascendc.ceil %1, %0, %c777_i64, %false {operandSegmentSizes = array<i32: 1, 1, 0, 1, 1>}
func.func @lower_ceil(%arg0: !asctile.tile<777xf32, UB>) -> !asctile.tile<777xf32, UB> {
  %0 = math.ceil %arg0 : !asctile.tile<777xf32, UB>
  return %0 : !asctile.tile<777xf32, UB>
}

// CHECK-LABEL: func.func @lower_floor(%arg0: !asctile.tile<777xf32, UB>) -> !asctile.tile<777xf32, UB> {
// CHECK-NEXT:    %0 = builtin.unrealized_conversion_cast %arg0 : !asctile.tile<777xf32, UB> to !ascendc.local_tensor<777xf32>
// CHECK-NEXT:    %1 = ascendc.local_tensor_auto veccalc() : <777xf32>
// CHECK-NEXT:    %2 = builtin.unrealized_conversion_cast %1 : !ascendc.local_tensor<777xf32> to !asctile.tile<777xf32, UB>
// CHECK-NEXT:    %c777_i64 = arith.constant 777 : i64
// CHECK-NEXT:    %false = arith.constant false
// CHECK-NEXT: ascendc.floor %1, %0, %c777_i64, %false {operandSegmentSizes = array<i32: 1, 1, 0, 1, 1>}
func.func @lower_floor(%arg0: !asctile.tile<777xf32, UB>) -> !asctile.tile<777xf32, UB> {
  %0 = math.floor %arg0 : !asctile.tile<777xf32, UB>
  return %0 : !asctile.tile<777xf32, UB>
}

// CHECK-LABEL: func.func @lower_round(%arg0: !asctile.tile<777xf32, UB>) -> !asctile.tile<777xf32, UB> {
// CHECK-NEXT:    %0 = builtin.unrealized_conversion_cast %arg0 : !asctile.tile<777xf32, UB> to !ascendc.local_tensor<777xf32>
// CHECK-NEXT:    %1 = ascendc.local_tensor_auto veccalc() : <777xf32>
// CHECK-NEXT:    %2 = builtin.unrealized_conversion_cast %1 : !ascendc.local_tensor<777xf32> to !asctile.tile<777xf32, UB>
// CHECK-NEXT:    %c777_i64 = arith.constant 777 : i64
// CHECK-NEXT:    %false = arith.constant false
// CHECK-NEXT: ascendc.round %1, %0, %c777_i64, %false {operandSegmentSizes = array<i32: 1, 1, 0, 1, 1>}
func.func @lower_round(%arg0: !asctile.tile<777xf32, UB>) -> !asctile.tile<777xf32, UB> {
  %0 = math.round %arg0 : !asctile.tile<777xf32, UB>
  return %0 : !asctile.tile<777xf32, UB>
}

// CHECK-LABEL: func.func @lower_acos(%arg0: !asctile.tile<777xf32, UB>) -> !asctile.tile<777xf32, UB> {
// CHECK-NEXT:    %0 = builtin.unrealized_conversion_cast %arg0 : !asctile.tile<777xf32, UB> to !ascendc.local_tensor<777xf32>
// CHECK-NEXT:    %1 = ascendc.local_tensor_auto veccalc() : <777xf32>
// CHECK-NEXT:    %2 = builtin.unrealized_conversion_cast %1 : !ascendc.local_tensor<777xf32> to !asctile.tile<777xf32, UB>
// CHECK-NEXT:    %c777_i64 = arith.constant 777 : i64
// CHECK-NEXT:    %false = arith.constant false
// CHECK-NEXT: ascendc.acos %1, %0, %c777_i64, %false {operandSegmentSizes = array<i32: 1, 1, 0, 1, 1>}
func.func @lower_acos(%arg0: !asctile.tile<777xf32, UB>) -> !asctile.tile<777xf32, UB> {
  %0 = math.acos %arg0 : !asctile.tile<777xf32, UB>
  return %0 : !asctile.tile<777xf32, UB>
}
