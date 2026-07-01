// RUN: ascir-opt -asctile-unroll-loop -canonicalize -cse %s | FileCheck %s --check-prefixes=CHECK,NOANN
// RUN: ascir-opt -asctile-unroll-loop=annotate -canonicalize -cse %s | FileCheck %s --check-prefixes=CHECK,ANNOT

// CHECK-LABEL: func.func @unroll_static_loop(%arg0: tensor<32xi32, #asctile.global>) {
// CHECK:       scf.for %arg1 = %c0 to %c32 step %c2 {
// NOANN-NEXT:    %0 = arith.index_cast %arg1 : index to i32
// ANNOT-NEXT:    %0 = arith.index_cast %arg1 {asctile.unroll_iter = 0 : i64} : index to i32
// NOANN-NEXT:    asctile.set_value %0, %arg0[%c0_i32] : i32, tensor<32xi32, #asctile.global>
// ANNOT-NEXT:    asctile.set_value %0, %arg0[%c0_i32] {asctile.unroll_iter = 0 : i64} : i32, tensor<32xi32, #asctile.global>
// CHECK-NEXT:    %1 = arith.addi %arg1, %c1 : index
// NOANN-NEXT:    %2 = arith.index_cast %1 : index to i32
// ANNOT-NEXT:    %2 = arith.index_cast %1 {asctile.unroll_iter = 1 : i64} : index to i32
// NOANN-NEXT:    asctile.set_value %2, %arg0[%c0_i32] : i32, tensor<32xi32, #asctile.global>
// ANNOT-NEXT:    asctile.set_value %2, %arg0[%c0_i32] {asctile.unroll_iter = 1 : i64} : i32, tensor<32xi32, #asctile.global>
// CHECK-NEXT:  }
// CHECK-NEXT:  return
// CHECK-NEXT:}
func.func @unroll_static_loop(%arg0: tensor<32xi32, #asctile.global>) {
  %c0 = arith.constant 0 : index
  %c0_i32 = arith.constant 0 : i32
  %c1 = arith.constant 1 : index
  %c32 = arith.constant 32 : index
  scf.for %arg1 = %c0 to %c32 step %c1 {
    %3 = arith.index_cast %arg1 : index to i32
    asctile.set_value %3, %arg0[%c0_i32] : i32, tensor<32xi32, #asctile.global>
    scf.yield
  } {asctile.unroll_factor = 2}
  return
}

// CHECK-LABEL: func.func @unroll_dynamic_loop(%arg0: tensor<32xi32, #asctile.global>, %arg1: index) {
// CHECK:       %0 = arith.remsi %arg1, %c3 : index
// CHECK-NEXT:  %1 = arith.subi %arg1, %0 : index
// CHECK-NEXT:  scf.for %arg2 = %c0 to %1 step %c3 {
// NOANN-NEXT:    %2 = arith.index_cast %arg2 : index to i32
// ANNOT-NEXT:    %2 = arith.index_cast %arg2 {asctile.unroll_iter = 0 : i64} : index to i32
// NOANN-NEXT:    asctile.set_value %2, %arg0[%c0_i32] : i32, tensor<32xi32, #asctile.global>
// ANNOT-NEXT:    asctile.set_value %2, %arg0[%c0_i32] {asctile.unroll_iter = 0 : i64} : i32, tensor<32xi32, #asctile.global>
// CHECK-NEXT:    %3 = arith.addi %arg2, %c1 : index
// NOANN-NEXT:    %4 = arith.index_cast %3 : index to i32
// ANNOT-NEXT:    %4 = arith.index_cast %3 {asctile.unroll_iter = 1 : i64} : index to i32
// NOANN-NEXT:    asctile.set_value %4, %arg0[%c0_i32] : i32, tensor<32xi32, #asctile.global>
// ANNOT-NEXT:    asctile.set_value %4, %arg0[%c0_i32] {asctile.unroll_iter = 1 : i64} : i32, tensor<32xi32, #asctile.global>
// CHECK-NEXT:    %5 = arith.addi %arg2, %c2 : index
// NOANN-NEXT:    %6 = arith.index_cast %5 : index to i32
// ANNOT-NEXT:    %6 = arith.index_cast %5 {asctile.unroll_iter = 2 : i64} : index to i32
// NOANN-NEXT:    asctile.set_value %6, %arg0[%c0_i32] : i32, tensor<32xi32, #asctile.global>
// ANNOT-NEXT:    asctile.set_value %6, %arg0[%c0_i32] {asctile.unroll_iter = 2 : i64} : i32, tensor<32xi32, #asctile.global>
// CHECK-NEXT:  }
// CHECK-NEXT:  scf.for %arg2 = %1 to %arg1 step %c1 {
// CHECK-NEXT:    %2 = arith.index_cast %arg2 : index to i32
// CHECK-NEXT:    asctile.set_value %2, %arg0[%c0_i32] : i32, tensor<32xi32, #asctile.global>
// CHECK-NEXT:  }
// CHECK-NEXT:  return
// CHECK-NEXT:}
func.func @unroll_dynamic_loop(%arg0: tensor<32xi32, #asctile.global>, %arg1: index) {
  %c0 = arith.constant 0 : index
  %c0_i32 = arith.constant 0 : i32
  %c1 = arith.constant 1 : index
  scf.for %arg2 = %c0 to %arg1 step %c1 {
    %3 = arith.index_cast %arg2 : index to i32
    asctile.set_value %3, %arg0[%c0_i32] : i32, tensor<32xi32, #asctile.global>
    scf.yield
  } {asctile.unroll_factor = 3}
  return
}
