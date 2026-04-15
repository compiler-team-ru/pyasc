// RUN: ascir-opt -asctile-tag-unroll-groups %s | FileCheck %s --check-prefixes=CHECK,LARGE
// RUN: ascir-opt -asctile-tag-unroll-groups=small-groups %s | FileCheck %s --check-prefixes=CHECK,SMALL

// CHECK-LABEL: func.func @tag_load_inside_loop_with_factor(%arg0: !asctile.tensor<32xf32>) {
// CHECK:       %0 = asctile.load %arg0[%c0_i32] : !asctile.tensor<32xf32>, !asctile.tile<16xf32, UB>
// CHECK-NEXT:  scf.for %arg1 = %c0_i32 to %c32_i32 step %c1_i32  : i32 {
// LARGE-NEXT:    %1 = asctile.load %arg0[%c0_i32] {asctile.unroll_group = 0 : i64} : !asctile.tensor<32xf32>, !asctile.tile<16xf32, UB>
// LARGE-NEXT:    %2 = asctile.load %arg0[%c0_i32] {asctile.unroll_group = 0 : i64} : !asctile.tensor<32xf32>, !asctile.tile<16xf32, UB>
// LARGE-NEXT:    asctile.store %1, %arg0[%c0_i32] {asctile.unroll_group = 1 : i64} : !asctile.tile<16xf32, UB>, !asctile.tensor<32xf32>
// SMALL-NEXT:    %1 = asctile.load %arg0[%c0_i32] {asctile.unroll_group = 0 : i64} : !asctile.tensor<32xf32>, !asctile.tile<16xf32, UB>
// SMALL-NEXT:    %2 = asctile.load %arg0[%c0_i32] {asctile.unroll_group = 1 : i64} : !asctile.tensor<32xf32>, !asctile.tile<16xf32, UB>
// SMALL-NEXT:    asctile.store %1, %arg0[%c0_i32] {asctile.unroll_group = 2 : i64} : !asctile.tile<16xf32, UB>, !asctile.tensor<32xf32>
// CHECK-NEXT:  } {asctile.unroll_factor = 4 : i64}
// CHECK-NEXT:  return
// CHECK-NEXT:}
func.func @tag_load_inside_loop_with_factor(%arg0: !asctile.tensor<32xf32>) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c32_i32 = arith.constant 32 : i32
  %0 = asctile.load %arg0[%c0_i32] : !asctile.tensor<32xf32>, !asctile.tile<16xf32, UB>
  scf.for %arg1 = %c0_i32 to %c32_i32 step %c1_i32 : i32 {
    %1 = asctile.load %arg0[%c0_i32] : !asctile.tensor<32xf32>, !asctile.tile<16xf32, UB>
    %2 = asctile.load %arg0[%c0_i32] : !asctile.tensor<32xf32>, !asctile.tile<16xf32, UB>
    asctile.store %1, %arg0[%c0_i32] : !asctile.tile<16xf32, UB>, !asctile.tensor<32xf32>
    scf.yield
  } {asctile.unroll_factor = 4}
  return
}

// CHECK-LABEL: func.func @skip_load_inside_loop(%arg0: !asctile.tensor<32xf32>) {
// CHECK:       %0 = asctile.load %arg0[%c0_i32] : !asctile.tensor<32xf32>, !asctile.tile<16xf32, UB>
// CHECK-NEXT:  scf.for %arg1 = %c0_i32 to %c32_i32 step %c1_i32  : i32 {
// CHECK-NEXT:    %1 = asctile.load %arg0[%c0_i32] : !asctile.tensor<32xf32>, !asctile.tile<16xf32, UB>
// CHECK-NEXT:    %2 = asctile.load %arg0[%c0_i32] : !asctile.tensor<32xf32>, !asctile.tile<16xf32, UB>
// CHECK-NEXT:    asctile.store %1, %arg0[%c0_i32] : !asctile.tile<16xf32, UB>, !asctile.tensor<32xf32>
// CHECK-NEXT:  }
// CHECK-NEXT:  return
// CHECK-NEXT:}
func.func @skip_load_inside_loop(%arg0: !asctile.tensor<32xf32>) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c32_i32 = arith.constant 32 : i32
  %0 = asctile.load %arg0[%c0_i32] : !asctile.tensor<32xf32>, !asctile.tile<16xf32, UB>
  scf.for %arg1 = %c0_i32 to %c32_i32 step %c1_i32 : i32 {
    %1 = asctile.load %arg0[%c0_i32] : !asctile.tensor<32xf32>, !asctile.tile<16xf32, UB>
    %2 = asctile.load %arg0[%c0_i32] : !asctile.tensor<32xf32>, !asctile.tile<16xf32, UB>
    asctile.store %1, %arg0[%c0_i32] : !asctile.tile<16xf32, UB>, !asctile.tensor<32xf32>
    scf.yield
  }
  return
}
