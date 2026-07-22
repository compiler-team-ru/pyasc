// RUN: ascir-opt --asctile-transform-store-fixpipe %s | FileCheck %s

// CHECK-LABEL: func.func @copy_from_l0c(%arg0: tensor<16x16xf32, #asctile.local<L0C>>) -> tensor<16x16xf32, #asctile.local<UB>> {
// CHECK:      %0 = asctile.copy_fixpipe %arg0[%c0_i32, %c0_i32] : tensor<16x16xf32, #asctile.local<L0C>>, tensor<16x16xf32, #asctile.local<UB>>
// CHECK-NEXT: return %0 : tensor<16x16xf32, #asctile.local<UB>>
// CHECK-NEXT:}
func.func @copy_from_l0c(%arg0: tensor<16x16xf32, #asctile.local<L0C>>) -> tensor<16x16xf32, #asctile.local<UB>> {
  %c0 = arith.constant 0 : i32
  %0 = asctile.copy %arg0[%c0, %c0] : tensor<16x16xf32, #asctile.local<L0C>>, tensor<16x16xf32, #asctile.local<UB>>
  return %0 : tensor<16x16xf32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @store_from_l0c(%arg0: tensor<16x16xf32, #asctile.local<L0C>>, %arg1: tensor<16x16xf32, #asctile.global>) {
// CHECK:      asctile.store_fixpipe %arg0, %arg1[%c0_i32, %c0_i32] : tensor<16x16xf32, #asctile.local<L0C>>, tensor<16x16xf32, #asctile.global>
// CHECK-NEXT: return
// CHECK-NEXT:}
func.func @store_from_l0c(%arg0: tensor<16x16xf32, #asctile.local<L0C>>, %arg1: tensor<16x16xf32, #asctile.global>) {
  %c0 = arith.constant 0 : i32
  asctile.store %arg0, %arg1[%c0, %c0] : tensor<16x16xf32, #asctile.local<L0C>>, tensor<16x16xf32, #asctile.global>
  return
}

// CHECK-LABEL: func.func @store_with_relu(%arg0: tensor<16x16xf32, #asctile.local<L0C>>, %arg1: tensor<16x16xf32, #asctile.global>) {
// CHECK:      asctile.store_fixpipe %arg0, %arg1[%c0_i32, %c0_i32] {relu} : tensor<16x16xf32, #asctile.local<L0C>>, tensor<16x16xf32, #asctile.global>
// CHECK-NEXT: return
// CHECK-NEXT:}
func.func @store_with_relu(%arg0: tensor<16x16xf32, #asctile.local<L0C>>, %arg1: tensor<16x16xf32, #asctile.global>) {
  %c0 = arith.constant 0 : i32
  %relu = asctile.relu %arg0 : tensor<16x16xf32, #asctile.local<L0C>>
  asctile.store %relu, %arg1[%c0, %c0] : tensor<16x16xf32, #asctile.local<L0C>>, tensor<16x16xf32, #asctile.global>
  return
}

// CHECK-LABEL: func.func @store_with_cast(%arg0: tensor<16x16xf32, #asctile.local<L0C>>, %arg1: tensor<16x16xi8, #asctile.global>) {
// CHECK:      asctile.store_fixpipe %arg0, %arg1[%c0_i32, %c0_i32] {quantize} : tensor<16x16xf32, #asctile.local<L0C>>, tensor<16x16xi8, #asctile.global>
// CHECK-NEXT: return
// CHECK-NEXT:}
func.func @store_with_cast(%arg0: tensor<16x16xf32, #asctile.local<L0C>>, %arg1: tensor<16x16xi8, #asctile.global>) {
  %c0 = arith.constant 0 : i32
  %cast = asctile.cast <default> %arg0 : tensor<16x16xf32, #asctile.local<L0C>> to tensor<16x16xi8, #asctile.local<L0C>>
  asctile.store %cast, %arg1[%c0, %c0] : tensor<16x16xi8, #asctile.local<L0C>>, tensor<16x16xi8, #asctile.global>
  return
}

// CHECK-LABEL: func.func @copy_with_relu(%arg0: tensor<16x16xf32, #asctile.local<L0C>>) -> tensor<16x16xf32, #asctile.local<UB>> {
// CHECK:      %0 = asctile.copy_fixpipe %arg0[%c0_i32, %c0_i32] {relu} : tensor<16x16xf32, #asctile.local<L0C>>, tensor<16x16xf32, #asctile.local<UB>>
// CHECK-NEXT: return %0 : tensor<16x16xf32, #asctile.local<UB>>
// CHECK-NEXT:}
func.func @copy_with_relu(%arg0: tensor<16x16xf32, #asctile.local<L0C>>) -> tensor<16x16xf32, #asctile.local<UB>> {
  %c0 = arith.constant 0 : i32
  %relu = asctile.relu %arg0 : tensor<16x16xf32, #asctile.local<L0C>>
  %0 = asctile.copy %relu[%c0, %c0] : tensor<16x16xf32, #asctile.local<L0C>>, tensor<16x16xf32, #asctile.local<UB>>
  return %0 : tensor<16x16xf32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @copy_with_cast(%arg0: tensor<16x16xf32, #asctile.local<L0C>>) -> tensor<16x16xi8, #asctile.local<UB>> {
// CHECK:      %0 = asctile.copy_fixpipe %arg0[%c0_i32, %c0_i32] {quantize} : tensor<16x16xf32, #asctile.local<L0C>>, tensor<16x16xi8, #asctile.local<UB>>
// CHECK-NEXT: return %0 : tensor<16x16xi8, #asctile.local<UB>>
// CHECK-NEXT:}
func.func @copy_with_cast(%arg0: tensor<16x16xf32, #asctile.local<L0C>>) -> tensor<16x16xi8, #asctile.local<UB>> {
  %c0 = arith.constant 0 : i32
  %cast = asctile.cast <default> %arg0 : tensor<16x16xf32, #asctile.local<L0C>> to tensor<16x16xi8, #asctile.local<L0C>>
  %0 = asctile.copy %cast[%c0, %c0] : tensor<16x16xi8, #asctile.local<L0C>>, tensor<16x16xi8, #asctile.local<UB>>
  return %0 : tensor<16x16xi8, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @non_l0c_copy(%arg0: tensor<16x16xf32, #asctile.local<UB>>) -> tensor<16x16xf32, #asctile.local<UB>> {
// CHECK:      %0 = asctile.copy %arg0[%c0_i32, %c0_i32] : tensor<16x16xf32, #asctile.local<UB>>, tensor<16x16xf32, #asctile.local<UB>>
// CHECK-NEXT: return %0 : tensor<16x16xf32, #asctile.local<UB>>
// CHECK-NEXT:}
func.func @non_l0c_copy(%arg0: tensor<16x16xf32, #asctile.local<UB>>) -> tensor<16x16xf32, #asctile.local<UB>> {
  %c0 = arith.constant 0 : i32
  %0 = asctile.copy %arg0[%c0, %c0] : tensor<16x16xf32, #asctile.local<UB>>, tensor<16x16xf32, #asctile.local<UB>>
  return %0 : tensor<16x16xf32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @non_l0c_store(%arg0: tensor<16x16xf32, #asctile.local<UB>>, %arg1: tensor<16x16xf32, #asctile.global>) {
// CHECK:      asctile.store %arg0, %arg1[%c0_i32, %c0_i32] : tensor<16x16xf32, #asctile.local<UB>>, tensor<16x16xf32, #asctile.global>
// CHECK-NEXT: return
// CHECK-NEXT:}
func.func @non_l0c_store(%arg0: tensor<16x16xf32, #asctile.local<UB>>, %arg1: tensor<16x16xf32, #asctile.global>) {
  %c0 = arith.constant 0 : i32
  asctile.store %arg0, %arg1[%c0, %c0] : tensor<16x16xf32, #asctile.local<UB>>, tensor<16x16xf32, #asctile.global>
  return
}
