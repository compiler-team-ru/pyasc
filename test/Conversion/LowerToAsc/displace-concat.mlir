// RUN: ascir-opt -asclower-displace-concat -canonicalize -allow-unregistered-dialect %s | FileCheck %s

// CHECK-LABEL: func.func @fold_concat(%arg0: tensor<16xf32, #asctile.local<UB>>) -> tensor<16xf32, #asctile.local<UB>> {
// CHECK-NEXT:  return %arg0 : tensor<16xf32, #asctile.local<UB>>
// CHECK-NEXT:}
func.func @fold_concat(%arg0: tensor<16xf32, #asctile.local<UB>>) -> tensor<16xf32, #asctile.local<UB>> {
  %0 = tensor.concat dim(0) %arg0 : (tensor<16xf32, #asctile.local<UB>>) -> tensor<16xf32, #asctile.local<UB>>
  return %0 : tensor<16xf32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @fully_displace_concat() -> tensor<8x16xf32, #asctile.local<UB>> {
// CHECK:       %0 = ascendc.local_tensor_auto veccalc() : <8x16xf32>
// CHECK-NEXT:  %1 = ascendc.local_tensor.subindex %0[%c0_i64] : !ascendc.local_tensor<8x16xf32>, i64, !ascendc.local_tensor<3x16xf32>
// CHECK-NEXT:  %2 = ascendc.local_tensor.subindex %0[%c48_i64] : !ascendc.local_tensor<8x16xf32>, i64, !ascendc.local_tensor<5x16xf32>
// CHECK-NEXT:  %3 = builtin.unrealized_conversion_cast %0 : !ascendc.local_tensor<8x16xf32> to tensor<8x16xf32, #asctile.local<UB>>
// CHECK-NEXT:  "tensor_user"(%1) : (!ascendc.local_tensor<3x16xf32>) -> ()
// CHECK-NEXT:  %4 = builtin.unrealized_conversion_cast %1 : !ascendc.local_tensor<3x16xf32> to tensor<3x16xf32, #asctile.local<UB>>
// CHECK-NEXT:  "tile_user"(%4) : (tensor<3x16xf32, #asctile.local<UB>>) -> ()
// CHECK-NEXT:  "tensor_user"(%2) : (!ascendc.local_tensor<5x16xf32>) -> ()
// CHECK-NEXT:  %5 = builtin.unrealized_conversion_cast %2 : !ascendc.local_tensor<5x16xf32> to tensor<5x16xf32, #asctile.local<UB>>
// CHECK-NEXT:  "tile_user"(%5) : (tensor<5x16xf32, #asctile.local<UB>>) -> ()
// CHECK-NEXT:  return %3 : tensor<8x16xf32, #asctile.local<UB>>
// CHECK-NEXT:}
func.func @fully_displace_concat() -> tensor<8x16xf32, #asctile.local<UB>> {
  %0 = ascendc.local_tensor_auto veccalc() : <3x16xf32>
  "tensor_user"(%0) : (!ascendc.local_tensor<3x16xf32>) -> ()
  %1 = builtin.unrealized_conversion_cast %0 : !ascendc.local_tensor<3x16xf32> to tensor<3x16xf32, #asctile.local<UB>>
  "tile_user"(%1) : (tensor<3x16xf32, #asctile.local<UB>>) -> ()
  %2 = ascendc.local_tensor_auto veccalc() : <5x16xf32>
  "tensor_user"(%2) : (!ascendc.local_tensor<5x16xf32>) -> ()
  %3 = builtin.unrealized_conversion_cast %2 : !ascendc.local_tensor<5x16xf32> to tensor<5x16xf32, #asctile.local<UB>>
  "tile_user"(%3) : (tensor<5x16xf32, #asctile.local<UB>>) -> ()
  %4 = tensor.concat dim(0) %1, %3 : (tensor<3x16xf32, #asctile.local<UB>>, tensor<5x16xf32, #asctile.local<UB>>) -> tensor<8x16xf32, #asctile.local<UB>>
  return %4 : tensor<8x16xf32, #asctile.local<UB>>
}

// CHECK-LABEL: func.func @convert_concat_fallback(%arg0: tensor<10xf32, #asctile.local<UB>>, %arg1: tensor<6xf32, #asctile.local<UB>>) -> tensor<16xf32, #asctile.local<UB>> {
// CHECK:       %0 = builtin.unrealized_conversion_cast %arg1 : tensor<6xf32, #asctile.local<UB>> to !ascendc.local_tensor<6xf32>
// CHECK-NEXT:  %1 = builtin.unrealized_conversion_cast %arg0 : tensor<10xf32, #asctile.local<UB>> to !ascendc.local_tensor<10xf32>
// CHECK-NEXT:  %2 = ascendc.local_tensor_auto veccalc() : <16xf32>
// CHECK-NEXT:  %3 = builtin.unrealized_conversion_cast %2 : !ascendc.local_tensor<16xf32> to tensor<16xf32, #asctile.local<UB>>
// CHECK-NEXT:  %4 = ascendc.local_tensor.subindex %2[%c0_i64] : !ascendc.local_tensor<16xf32>, i64, !ascendc.local_tensor<10xf32>
// CHECK-NEXT:  ascendc.data_copy_l2 %4, %1, %c10_i64 : !ascendc.local_tensor<10xf32>, !ascendc.local_tensor<10xf32>, i64
// CHECK-NEXT:  %5 = ascendc.local_tensor.subindex %2[%c10_i64] : !ascendc.local_tensor<16xf32>, i64, !ascendc.local_tensor<6xf32>
// CHECK-NEXT:  ascendc.data_copy_l2 %5, %0, %c6_i64 : !ascendc.local_tensor<6xf32>, !ascendc.local_tensor<6xf32>, i64
// CHECK-NEXT:  return %3 : tensor<16xf32, #asctile.local<UB>>
func.func @convert_concat_fallback(%arg0: tensor<10xf32, #asctile.local<UB>>, %arg1: tensor<6xf32, #asctile.local<UB>>) -> tensor<16xf32, #asctile.local<UB>> {
  %0 = tensor.concat dim(0) %arg0, %arg1 : (tensor<10xf32, #asctile.local<UB>>, tensor<6xf32, #asctile.local<UB>>) -> tensor<16xf32, #asctile.local<UB>>
  return %0 : tensor<16xf32, #asctile.local<UB>>
}
