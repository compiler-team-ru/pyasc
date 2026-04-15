// RUN: ascir-opt -asctile-promote-pure-ops %s | FileCheck %s

// CHECK-LABEL: func.func @promote(%arg0: f32, %arg1: f32) {
// CHECK-DAG:   %c1 = arith.constant 1 : index
// CHECK-DAG:   %c32 = arith.constant 32 : index
// CHECK-DAG:   %c0 = arith.constant 0 : index
// CHECK-NEXT:  scf.for %arg2 = %c0 to %c32 step %c1 {
// CHECK-NEXT:    %cst = arith.constant 7.000000e+00 : f32
// CHECK-NEXT:    %0 = arith.addf %arg0, %cst : f32
// CHECK-NEXT:    %1 = arith.mulf %arg0, %arg1 : f32
// CHECK-NEXT:    %2 = arith.subf %0, %1 : f32
// CHECK-NEXT:    %3 = arith.addf %arg0, %arg1 : f32
// CHECK-NEXT:    %4 = arith.addf %arg0, %3 : f32
// CHECK-NEXT:    gpu.barrier
// CHECK-NEXT:    gpu.barrier
// CHECK-NEXT:  }
// CHECK-NEXT:  return
// CHECK-NEXT:}
func.func @promote(%arg0: f32, %arg1: f32) {
  %c0 = arith.constant 0 : index
  %c32 = arith.constant 32 : index
  %c1 = arith.constant 1 : index
  scf.for %arg2 = %c0 to %c32 step %c1 {
    %0 = arith.addf %arg0, %arg1 : f32
    %1 = arith.addf %arg0, %0 : f32
    gpu.barrier
    %2 = arith.mulf %arg0, %arg1 : f32
    %3 = arith.constant 7.000000e+00 : f32
    %4 = arith.addf %arg0, %3 : f32
    %5 = arith.subf %4, %2 : f32
    gpu.barrier
  }
  return
}
