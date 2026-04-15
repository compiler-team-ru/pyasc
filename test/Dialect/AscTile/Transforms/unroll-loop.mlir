// RUN: ascir-opt -asctile-unroll-loop -canonicalize -cse %s | FileCheck %s

// CHECK-LABEL: func.func @unroll_static_loop(%arg0: !asctile.tensor<32xi32>) {
// CHECK:       scf.for %arg1 = %c0 to %c32 step %c2 {
// CHECK-NEXT:    %0 = arith.index_cast %arg1 : index to i32
// CHECK-NEXT:    asctile.set_value %0, %arg0[%c0_i32] : i32, !asctile.tensor<32xi32>
// CHECK-NEXT:    %1 = arith.addi %arg1, %c1 : index
// CHECK-NEXT:    %2 = arith.index_cast %1 : index to i32
// CHECK-NEXT:    asctile.set_value %2, %arg0[%c0_i32] : i32, !asctile.tensor<32xi32>
// CHECK-NEXT:  }
// CHECK-NEXT:  return
// CHECK-NEXT:}
func.func @unroll_static_loop(%arg0: !asctile.tensor<32xi32>) {
  %c0 = arith.constant 0 : index
  %c0_i32 = arith.constant 0 : i32
  %c1 = arith.constant 1 : index
  %c32 = arith.constant 32 : index
  scf.for %arg1 = %c0 to %c32 step %c1 {
    %3 = arith.index_cast %arg1 : index to i32
    asctile.set_value %3, %arg0[%c0_i32] : i32, !asctile.tensor<32xi32>
    scf.yield
  } {asctile.unroll_factor = 2}
  return
}

// CHECK-LABEL: func.func @unroll_dynamic_loop(%arg0: !asctile.tensor<32xi32>, %arg1: index) {
// CHECK:       %0 = arith.remsi %arg1, %c3 : index
// CHECK-NEXT:  %1 = arith.subi %arg1, %0 : index
// CHECK-NEXT:  scf.for %arg2 = %c0 to %1 step %c3 {
// CHECK-NEXT:    %2 = arith.index_cast %arg2 : index to i32
// CHECK-NEXT:    asctile.set_value %2, %arg0[%c0_i32] : i32, !asctile.tensor<32xi32>
// CHECK-NEXT:    %3 = arith.addi %arg2, %c1 : index
// CHECK-NEXT:    %4 = arith.index_cast %3 : index to i32
// CHECK-NEXT:    asctile.set_value %4, %arg0[%c0_i32] : i32, !asctile.tensor<32xi32>
// CHECK-NEXT:    %5 = arith.addi %arg2, %c2 : index
// CHECK-NEXT:    %6 = arith.index_cast %5 : index to i32
// CHECK-NEXT:    asctile.set_value %6, %arg0[%c0_i32] : i32, !asctile.tensor<32xi32>
// CHECK-NEXT:  }
// CHECK-NEXT:  scf.for %arg2 = %1 to %arg1 step %c1 {
// CHECK-NEXT:    %2 = arith.index_cast %arg2 : index to i32
// CHECK-NEXT:    asctile.set_value %2, %arg0[%c0_i32] : i32, !asctile.tensor<32xi32>
// CHECK-NEXT:  }
// CHECK-NEXT:  return
// CHECK-NEXT:}
func.func @unroll_dynamic_loop(%arg0: !asctile.tensor<32xi32>, %arg1: index) {
  %c0 = arith.constant 0 : index
  %c0_i32 = arith.constant 0 : i32
  %c1 = arith.constant 1 : index
  scf.for %arg2 = %c0 to %arg1 step %c1 {
    %3 = arith.index_cast %arg2 : index to i32
    asctile.set_value %3, %arg0[%c0_i32] : i32, !asctile.tensor<32xi32>
    scf.yield
  } {asctile.unroll_factor = 3}
  return
}
