// RUN: ascir-opt --asclower-atomic %s | FileCheck %s

// CHECK-LABEL: func.func @lower_atomic_add(%arg0: !asctile.tile<16xf32, UB>, %arg1: !asctile.tensor<16xf32>, %arg2: i32) {
// CHECK-NEXT:  ascendc.set_atomic_add  {dtype = f32} :
// CHECK-NEXT:  asctile.store %arg0, %arg1[%arg2] : !asctile.tile<16xf32, UB>, !asctile.tensor<16xf32>
// CHECK-NEXT:  ascendc.set_atomic_none  :
// CHECK-NEXT:  return
// CHECK-NEXT:}
func.func @lower_atomic_add(%arg0: !asctile.tile<16xf32, UB>, %arg1: !asctile.tensor<16xf32>, %arg2: i32) {
  asctile.atomic_rmw <Add> %arg0, %arg1 [%arg2] : !asctile.tile<16xf32, UB>, !asctile.tensor<16xf32>
  return
}

// CHECK-LABEL: func.func @lower_atomic_max(%arg0: !asctile.tile<16xf32, UB>, %arg1: !asctile.tensor<16xf32>, %arg2: i32) {
// CHECK-NEXT:  ascendc.set_atomic_max  {dtype = f32} :
// CHECK-NEXT:  asctile.store %arg0, %arg1[%arg2] : !asctile.tile<16xf32, UB>, !asctile.tensor<16xf32>
// CHECK-NEXT:  ascendc.set_atomic_none  :
// CHECK-NEXT:  return
// CHECK-NEXT:}
func.func @lower_atomic_max(%arg0: !asctile.tile<16xf32, UB>, %arg1: !asctile.tensor<16xf32>, %arg2: i32) {
  asctile.atomic_rmw <Max> %arg0, %arg1 [%arg2] : !asctile.tile<16xf32, UB>, !asctile.tensor<16xf32>
  return
}

// CHECK-LABEL: func.func @lower_atomic_min(%arg0: !asctile.tile<16xf32, UB>, %arg1: !asctile.tensor<16xf32>, %arg2: i32) {
// CHECK-NEXT:  ascendc.set_atomic_min  {dtype = f32} :
// CHECK-NEXT:  asctile.store %arg0, %arg1[%arg2] : !asctile.tile<16xf32, UB>, !asctile.tensor<16xf32>
// CHECK-NEXT:  ascendc.set_atomic_none  :
// CHECK-NEXT:  return
// CHECK-NEXT:}
func.func @lower_atomic_min(%arg0: !asctile.tile<16xf32, UB>, %arg1: !asctile.tensor<16xf32>, %arg2: i32) {
  asctile.atomic_rmw <Min> %arg0, %arg1 [%arg2] : !asctile.tile<16xf32, UB>, !asctile.tensor<16xf32>
  return
}
