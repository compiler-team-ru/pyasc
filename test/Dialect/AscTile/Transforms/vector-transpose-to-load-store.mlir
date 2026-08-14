// RUN: ascir-opt -asctile-vector-transpose-to-load-store %s | FileCheck %s

// CHECK-LABEL: func.func public @transpose_to_load_2d
// CHECK:      asctile.load
// CHECK-SAME: asctile.transpose_dims = array<i32: 1, 0>
// CHECK-SAME: tensor<16x32xf32, #asctile.global>, tensor<32x16xf32, #asctile.local<UB>>
// CHECK-NOT:  asctile.transpose
// CHECK:      return [[RET:.*]] tensor<32x16xf32, #asctile.local<UB>>
  func.func public @transpose_to_load_2d(%arg0: memref<*xf32, 22>, %arg1: i32, %arg2: i32, %arg3: f32, %arg4: i32) -> tensor<32x16xf32, #asctile.local<UB>> {
    %0 = asctile.tensor %arg0() : memref<*xf32, 22>, tensor<16x32xf32, #asctile.global>
    %cst = arith.constant 0.000000e+00 : f32
    %1 = asctile.load %0[%arg1, %arg2], %arg3, (%arg4, %arg4) : tensor<16x32xf32, #asctile.global>, tensor<16x32xf32, #asctile.local<UB>>
    %2 = asctile.transpose %1, [1 : i32, 0 : i32] : tensor<16x32xf32, #asctile.local<UB>> to tensor<32x16xf32, #asctile.local<UB>>
    return %2 : tensor<32x16xf32, #asctile.local<UB>>
  }


// CHECK-LABEL: func.func public @transpose_to_load_3d
// CHECK:      asctile.load
// CHECK-SAME: asctile.transpose_dims = array<i32: 2, 1, 0>
// CHECK-SAME: tensor<128x16x32xf32, #asctile.global>, tensor<32x16x128xf32, #asctile.local<UB>>
// CHECK-NOT:  asctile.transpose
// CHECK:      return [[RET:.*]] tensor<32x16x128xf32, #asctile.local<UB>>
  func.func public @transpose_to_load_3d(%arg0: memref<*xf32, 22>, %arg1: i32, %arg2: i32, %arg3: f32, %arg4: i32) -> tensor<32x16x128xf32, #asctile.local<UB>> {
    %0 = asctile.tensor %arg0() : memref<*xf32, 22>, tensor<128x16x32xf32, #asctile.global>
    %cst = arith.constant 0.000000e+00 : f32
    %1 = asctile.load %0[%arg1, %arg2, %arg4], %arg3, (%arg4, %arg4, %arg4) : tensor<128x16x32xf32, #asctile.global>, tensor<128x16x32xf32, #asctile.local<UB>>
    %2 = asctile.transpose %1, [2 : i32, 1 : i32, 0: i32] : tensor<128x16x32xf32, #asctile.local<UB>> to tensor<32x16x128xf32, #asctile.local<UB>>
    return %2 : tensor<32x16x128xf32, #asctile.local<UB>>
  }

// CHECK-LABEL: func.func public @transpose_to_load_4d
// CHECK:      asctile.load
// CHECK-SAME: asctile.transpose_dims = array<i32: 3, 2, 1, 0>
// CHECK-SAME: tensor<256x128x16x32xf32, #asctile.global>, tensor<32x16x128x256xf32, #asctile.local<UB>>
// CHECK-NOT:  asctile.transpose
// CHECK:      return [[RET:.*]] tensor<32x16x128x256xf32, #asctile.local<UB>>
  func.func public @transpose_to_load_4d(%arg0: memref<*xf32, 22>, %arg1: i32, %arg2: i32, %arg3: f32, %arg4: i32) -> tensor<32x16x128x256xf32, #asctile.local<UB>> {
    %0 = asctile.tensor %arg0() : memref<*xf32, 22>, tensor<256x128x16x32xf32, #asctile.global>
    %cst = arith.constant 0.000000e+00 : f32
    %1 = asctile.load %0[%arg1, %arg2, %arg4, %arg4], %arg3, (%arg4, %arg4, %arg4, %arg4) : tensor<256x128x16x32xf32, #asctile.global>, tensor<256x128x16x32xf32, #asctile.local<UB>>
    %2 = asctile.transpose %1, [3 : i32, 2 : i32, 1 : i32, 0: i32] : tensor<256x128x16x32xf32, #asctile.local<UB>> to tensor<32x16x128x256xf32, #asctile.local<UB>>
    return %2 : tensor<32x16x128x256xf32, #asctile.local<UB>>
  }

// CHECK-LABEL: store_good_3d
// CHECK: asctile.store
// CHECK-SAME: {asctile.transpose_dims = array<i32: 1, 0, 2>}
func.func public @store_good_3d(%arg0 : tensor<16x8x32xf32, #asctile.local<UB>>, %arg1 : tensor<32x32x32xf32, #asctile.global>) {
    %0 = arith.constant 0 : i32
    %1 = asctile.transpose %arg0, [1 : i32, 0 : i32, 2 : i32]: tensor<16x8x32xf32, #asctile.local<UB>> to tensor<8x16x32xf32, #asctile.local<UB>>
    asctile.store %1, %arg1[%0, %0, %0]: tensor<8x16x32xf32, #asctile.local<UB>>, tensor<32x32x32xf32, #asctile.global>
    return
}

// CHECK-LABEL: transpose_to_store_4d
// CHECK: asctile.store
// CHECK-SAME: {asctile.transpose_dims = array<i32: 0, 2, 1, 3>}
func.func public @transpose_to_store_4d(%arg0 : tensor<2x16x8x32xf32, #asctile.local<UB>>, %arg1 : tensor<32x32x32x32xf32, #asctile.global>) {
    %0 = arith.constant 0 : i32
    %1 = asctile.transpose %arg0, [0 : i32, 2 : i32, 1 : i32, 3 : i32]: tensor<2x16x8x32xf32, #asctile.local<UB>> to tensor<2x8x16x32xf32, #asctile.local<UB>>
    asctile.store %1, %arg1[%0, %0, %0, %0]: tensor<2x8x16x32xf32, #asctile.local<UB>>, tensor<32x32x32x32xf32, #asctile.global>
    return
}

// CHECK-LABEL: store_to_store_skip
// CHECK: asctile.transpose
// CHECK-NOT: {asctile.transpose_dims = array<i32: 2, 0, 1>}
// CHECK: asctile.store
func.func public @store_to_store_skip(%arg0 : tensor<16x8x32xf32, #asctile.local<UB>>, %arg1 : tensor<32x32x32xf32, #asctile.global>) {
    %0 = arith.constant 0 : i32
    %1 = asctile.transpose %arg0, [2 : i32, 0 : i32, 1 : i32]: tensor<16x8x32xf32, #asctile.local<UB>> to tensor<8x16x32xf32, #asctile.local<UB>>
    asctile.store %1, %arg1[%0, %0, %0]: tensor<8x16x32xf32, #asctile.local<UB>>, tensor<32x32x32xf32, #asctile.global>
    return
}

// CHECK-LABEL: store_to_store_skip_2d
// CHECK: asctile.transpose
// CHECK-NOT: {asctile.transpose_dims = array<i32: 1, 0>}
// CHECK: asctile.store
func.func public @store_to_store_skip_2d(%arg0 : tensor<16x32xf32, #asctile.local<UB>>, %arg1 : tensor<32x32xf32, #asctile.global>) {
    %0 = arith.constant 0 : i32
    %1 = asctile.transpose %arg0, [1 : i32, 0 : i32]: tensor<16x32xf32, #asctile.local<UB>> to tensor<32x16xf32, #asctile.local<UB>>
    asctile.store %1, %arg1[%0, %0, %0]: tensor<32x16xf32, #asctile.local<UB>>, tensor<32x32xf32, #asctile.global>
    return
}
