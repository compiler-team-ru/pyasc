// RUN: ascir-opt --asclower-scf %s | FileCheck %s

// CHECK-LABEL: func.func @lower_if(%arg0: i1, %arg1: !asctile.tile<16xf32, UB>, %arg2: !asctile.tile<16xf32, UB>) -> !asctile.tile<16xf32, UB> {
// CHECK-NEXT:  %0 = scf.if %arg0 -> (!ascendc.local_tensor<16xf32>) {
// CHECK-NEXT:    %2 = arith.addf %arg1, %arg2 : !asctile.tile<16xf32, UB>
// CHECK-NEXT:    %3 = builtin.unrealized_conversion_cast %2 : !asctile.tile<16xf32, UB> to !ascendc.local_tensor<16xf32>
// CHECK-NEXT:    scf.yield %3 : !ascendc.local_tensor<16xf32>
// CHECK-NEXT:  } else {
// CHECK-NEXT:    %2 = arith.subf %arg1, %arg2 : !asctile.tile<16xf32, UB>
// CHECK-NEXT:    %3 = builtin.unrealized_conversion_cast %2 : !asctile.tile<16xf32, UB> to !ascendc.local_tensor<16xf32>
// CHECK-NEXT:    scf.yield %3 : !ascendc.local_tensor<16xf32>
// CHECK-NEXT:  }
// CHECK-NEXT:  %1 = builtin.unrealized_conversion_cast %0 : !ascendc.local_tensor<16xf32> to !asctile.tile<16xf32, UB>
// CHECK-NEXT:  return %1 : !asctile.tile<16xf32, UB>
// CHECK-NEXT:}
func.func @lower_if(%arg0: i1, %arg1: !asctile.tile<16xf32, UB>, %arg2: !asctile.tile<16xf32, UB>) -> !asctile.tile<16xf32, UB> {
  %0 = scf.if %arg0 -> !asctile.tile<16xf32, UB> {
    %1 = arith.addf %arg1, %arg2 : !asctile.tile<16xf32, UB>
    scf.yield %1 : !asctile.tile<16xf32, UB>
  } else {
    %1 = arith.subf %arg1, %arg2 : !asctile.tile<16xf32, UB>
    scf.yield %1 : !asctile.tile<16xf32, UB>
  }
  return %0 : !asctile.tile<16xf32, UB>
}

// CHECK-LABEL: func.func @lower_for(%arg0: !asctile.tile<16xf32, UB>, %arg1: index, %arg2: index, %arg3: index) -> !asctile.tile<16xf32, UB> {
// CHECK-NEXT:  %0 = builtin.unrealized_conversion_cast %arg0 : !asctile.tile<16xf32, UB> to !ascendc.local_tensor<16xf32>
// CHECK-NEXT:  %1 = scf.for %arg4 = %arg1 to %arg2 step %arg3 iter_args(%arg5 = %0) -> (!ascendc.local_tensor<16xf32>) {
// CHECK-NEXT:    %3 = builtin.unrealized_conversion_cast %arg5 : !ascendc.local_tensor<16xf32> to !asctile.tile<16xf32, UB>
// CHECK-NEXT:    %4 = arith.addf %3, %arg0 : !asctile.tile<16xf32, UB>
// CHECK-NEXT:    %5 = builtin.unrealized_conversion_cast %4 : !asctile.tile<16xf32, UB> to !ascendc.local_tensor<16xf32>
// CHECK-NEXT:    scf.yield %5 : !ascendc.local_tensor<16xf32>
// CHECK-NEXT:  }
// CHECK-NEXT:  %2 = builtin.unrealized_conversion_cast %1 : !ascendc.local_tensor<16xf32> to !asctile.tile<16xf32, UB>
// CHECK-NEXT:  return %2 : !asctile.tile<16xf32, UB>
// CHECK-NEXT:}
func.func @lower_for(%arg0: !asctile.tile<16xf32, UB>, %arg1: index, %arg2: index, %arg3: index) -> !asctile.tile<16xf32, UB> {
  %0 = scf.for %arg4 = %arg1 to %arg2 step %arg3 iter_args(%arg5 = %arg0) -> !asctile.tile<16xf32, UB> {
    %1 = arith.addf %arg5, %arg0 : !asctile.tile<16xf32, UB>
    scf.yield %1 : !asctile.tile<16xf32, UB>
  }
  return %0 : !asctile.tile<16xf32, UB>
}

// CHECK-LABEL: func.func @lower_while(%arg0: !asctile.tile<16xf32, UB>, %arg1: !asctile.tile<16xf32, UB>) -> !asctile.tile<16xf32, UB> {
// CHECK-NEXT:  %0 = builtin.unrealized_conversion_cast %arg0 : !asctile.tile<16xf32, UB> to !ascendc.local_tensor<16xf32>
// CHECK-NEXT:  %1 = scf.while (%arg2 = %0) : (!ascendc.local_tensor<16xf32>) -> !ascendc.local_tensor<16xf32> {
// CHECK-NEXT:    %false = arith.constant false
// CHECK-NEXT:    scf.condition(%false) %arg2 : !ascendc.local_tensor<16xf32>
// CHECK-NEXT:  } do {
// CHECK-NEXT:  ^bb0(%arg2: !ascendc.local_tensor<16xf32>):
// CHECK-NEXT:    %3 = builtin.unrealized_conversion_cast %arg2 : !ascendc.local_tensor<16xf32> to !asctile.tile<16xf32, UB>
// CHECK-NEXT:    %4 = arith.addf %3, %3 : !asctile.tile<16xf32, UB>
// CHECK-NEXT:    %5 = builtin.unrealized_conversion_cast %4 : !asctile.tile<16xf32, UB> to !ascendc.local_tensor<16xf32>
// CHECK-NEXT:    scf.yield %5 : !ascendc.local_tensor<16xf32>
// CHECK-NEXT:  }
// CHECK-NEXT:  %2 = builtin.unrealized_conversion_cast %1 : !ascendc.local_tensor<16xf32> to !asctile.tile<16xf32, UB>
// CHECK-NEXT:  return %2 : !asctile.tile<16xf32, UB>
// CHECK-NEXT:}
func.func @lower_while(%arg0: !asctile.tile<16xf32, UB>, %arg1: !asctile.tile<16xf32, UB>) -> !asctile.tile<16xf32, UB> {
  %0 = scf.while (%arg2 = %arg0) : (!asctile.tile<16xf32, UB>) -> !asctile.tile<16xf32, UB> {
    %false = arith.constant false
    scf.condition(%false) %arg2 : !asctile.tile<16xf32, UB>
  } do {
  ^bb0(%arg2: !asctile.tile<16xf32, UB>):
    %1 = arith.addf %arg2, %arg2 : !asctile.tile<16xf32, UB>
    scf.yield %1 : !asctile.tile<16xf32, UB>
  }
  return %0 : !asctile.tile<16xf32, UB>
}
