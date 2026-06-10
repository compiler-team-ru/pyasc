// RUN: ascir-opt -asctile-vector-transpose-to-load %s | FileCheck %s

// CHECK-LABEL: func.func public @transpose_to_load_2d
// CHECK:      asctile.load
// CHECK-SAME: asctile.transpose_dims = array<i32: 1, 0>
// CHECK-SAME: !asctile.tensor<16x32xf32>, !asctile.tile<32x16xf32, UB>
// CHECK-NOT:  asctile.transpose
// CHECK:      return [[RET:.*]] !asctile.tile<32x16xf32, UB>
  func.func public @transpose_to_load_2d(%arg0: memref<*xf32, 22>, %arg1: i32, %arg2: i32, %arg3: f32, %arg4: i32) -> !asctile.tile<32x16xf32, UB> {
    %0 = asctile.tensor %arg0() : memref<*xf32, 22>, !asctile.tensor<16x32xf32>
    %cst = arith.constant 0.000000e+00 : f32
    %1 = asctile.load %0[%arg1, %arg2], %arg3, (%arg4, %arg4) : !asctile.tensor<16x32xf32>, !asctile.tile<16x32xf32, UB>
    %2 = asctile.transpose %1, [1 : i32, 0 : i32] : !asctile.tile<16x32xf32, UB> to !asctile.tile<32x16xf32, UB>
    return %2 : !asctile.tile<32x16xf32, UB>
  }


// CHECK-LABEL: func.func public @transpose_to_load_3d
// CHECK:      asctile.load
// CHECK-SAME: asctile.transpose_dims = array<i32: 2, 1, 0>
// CHECK-SAME: !asctile.tensor<128x16x32xf32>, !asctile.tile<32x16x128xf32, UB>
// CHECK-NOT:  asctile.transpose
// CHECK:      return [[RET:.*]] !asctile.tile<32x16x128xf32, UB>
  func.func public @transpose_to_load_3d(%arg0: memref<*xf32, 22>, %arg1: i32, %arg2: i32, %arg3: f32, %arg4: i32) -> !asctile.tile<32x16x128xf32, UB> {
    %0 = asctile.tensor %arg0() : memref<*xf32, 22>, !asctile.tensor<128x16x32xf32>
    %cst = arith.constant 0.000000e+00 : f32
    %1 = asctile.load %0[%arg1, %arg2, %arg4], %arg3, (%arg4, %arg4, %arg4) : !asctile.tensor<128x16x32xf32>, !asctile.tile<128x16x32xf32, UB>
    %2 = asctile.transpose %1, [2 : i32, 1 : i32, 0: i32] : !asctile.tile<128x16x32xf32, UB> to !asctile.tile<32x16x128xf32, UB>
    return %2 : !asctile.tile<32x16x128xf32, UB>
  }

// CHECK-LABEL: func.func public @transpose_to_load_4d
// CHECK:      asctile.load
// CHECK-SAME: asctile.transpose_dims = array<i32: 3, 2, 1, 0>
// CHECK-SAME: !asctile.tensor<256x128x16x32xf32>, !asctile.tile<32x16x128x256xf32, UB>
// CHECK-NOT:  asctile.transpose
// CHECK:      return [[RET:.*]] !asctile.tile<32x16x128x256xf32, UB>
  func.func public @transpose_to_load_4d(%arg0: memref<*xf32, 22>, %arg1: i32, %arg2: i32, %arg3: f32, %arg4: i32) -> !asctile.tile<32x16x128x256xf32, UB> {
    %0 = asctile.tensor %arg0() : memref<*xf32, 22>, !asctile.tensor<256x128x16x32xf32>
    %cst = arith.constant 0.000000e+00 : f32
    %1 = asctile.load %0[%arg1, %arg2, %arg4, %arg4], %arg3, (%arg4, %arg4, %arg4, %arg4) : !asctile.tensor<256x128x16x32xf32>, !asctile.tile<256x128x16x32xf32, UB>
    %2 = asctile.transpose %1, [3 : i32, 2 : i32, 1 : i32, 0: i32] : !asctile.tile<256x128x16x32xf32, UB> to !asctile.tile<32x16x128x256xf32, UB>
    return %2 : !asctile.tile<32x16x128x256xf32, UB>
  }
