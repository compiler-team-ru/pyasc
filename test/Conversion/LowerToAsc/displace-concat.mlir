// RUN: ascir-opt -asclower-displace-concat -canonicalize -allow-unregistered-dialect %s | FileCheck %s

// CHECK-LABEL: func.func @fold_concat(%arg0: !asctile.tile<16xf32, UB>) -> !asctile.tile<16xf32, UB> {
// CHECK-NEXT:  return %arg0 : !asctile.tile<16xf32, UB>
// CHECK-NEXT:}
func.func @fold_concat(%arg0: !asctile.tile<16xf32, UB>) -> !asctile.tile<16xf32, UB> {
  %0 = asctile.concat %arg0 : !asctile.tile<16xf32, UB> -> !asctile.tile<16xf32, UB>
  return %0 : !asctile.tile<16xf32, UB>
}

// CHECK-LABEL: func.func @fully_displace_concat() -> !asctile.tile<8x16xf32, UB> {
// CHECK:       %0 = ascendc.local_tensor_auto veccalc() : <8x16xf32>
// CHECK-NEXT:  %1 = ascendc.local_tensor.subindex %0[%c0_i64] : !ascendc.local_tensor<8x16xf32>, i64, !ascendc.local_tensor<3x16xf32>
// CHECK-NEXT:  %2 = ascendc.local_tensor.subindex %0[%c48_i64] : !ascendc.local_tensor<8x16xf32>, i64, !ascendc.local_tensor<5x16xf32>
// CHECK-NEXT:  %3 = builtin.unrealized_conversion_cast %0 : !ascendc.local_tensor<8x16xf32> to !asctile.tile<8x16xf32, UB>
// CHECK-NEXT:  "tensor_user"(%1) : (!ascendc.local_tensor<3x16xf32>) -> ()
// CHECK-NEXT:  %4 = builtin.unrealized_conversion_cast %1 : !ascendc.local_tensor<3x16xf32> to !asctile.tile<3x16xf32, UB>
// CHECK-NEXT:  "tile_user"(%4) : (!asctile.tile<3x16xf32, UB>) -> ()
// CHECK-NEXT:  "tensor_user"(%2) : (!ascendc.local_tensor<5x16xf32>) -> ()
// CHECK-NEXT:  %5 = builtin.unrealized_conversion_cast %2 : !ascendc.local_tensor<5x16xf32> to !asctile.tile<5x16xf32, UB>
// CHECK-NEXT:  "tile_user"(%5) : (!asctile.tile<5x16xf32, UB>) -> ()
// CHECK-NEXT:  return %3 : !asctile.tile<8x16xf32, UB>
// CHECK-NEXT:}
func.func @fully_displace_concat() -> !asctile.tile<8x16xf32, UB> {
  %0 = ascendc.local_tensor_auto veccalc() : <3x16xf32>
  "tensor_user"(%0) : (!ascendc.local_tensor<3x16xf32>) -> ()
  %1 = builtin.unrealized_conversion_cast %0 : !ascendc.local_tensor<3x16xf32> to !asctile.tile<3x16xf32, UB>
  "tile_user"(%1) : (!asctile.tile<3x16xf32, UB>) -> ()
  %2 = ascendc.local_tensor_auto veccalc() : <5x16xf32>
  "tensor_user"(%2) : (!ascendc.local_tensor<5x16xf32>) -> ()
  %3 = builtin.unrealized_conversion_cast %2 : !ascendc.local_tensor<5x16xf32> to !asctile.tile<5x16xf32, UB>
  "tile_user"(%3) : (!asctile.tile<5x16xf32, UB>) -> ()
  %4 = asctile.concat %1, %3 : !asctile.tile<3x16xf32, UB>, !asctile.tile<5x16xf32, UB> -> !asctile.tile<8x16xf32, UB>
  return %4 : !asctile.tile<8x16xf32, UB>
}

// CHECK-LABEL: func.func @convert_concat_fallback(%arg0: !asctile.tile<10xf32, UB>, %arg1: !asctile.tile<6xf32, UB>) -> !asctile.tile<16xf32, UB> {
// CHECK:       %0 = builtin.unrealized_conversion_cast %arg1 : !asctile.tile<6xf32, UB> to !ascendc.local_tensor<6xf32>
// CHECK-NEXT:  %1 = builtin.unrealized_conversion_cast %arg0 : !asctile.tile<10xf32, UB> to !ascendc.local_tensor<10xf32>
// CHECK-NEXT:  %2 = ascendc.local_tensor_auto veccalc() : <16xf32>
// CHECK-NEXT:  %3 = builtin.unrealized_conversion_cast %2 : !ascendc.local_tensor<16xf32> to !asctile.tile<16xf32, UB>
// CHECK-NEXT:  %4 = ascendc.local_tensor.subindex %2[%c0_i64] : !ascendc.local_tensor<16xf32>, i64, !ascendc.local_tensor<10xf32>
// CHECK-NEXT:  ascendc.data_copy_l2 %4, %1, %c10_i64 : !ascendc.local_tensor<10xf32>, !ascendc.local_tensor<10xf32>, i64
// CHECK-NEXT:  %5 = ascendc.local_tensor.subindex %2[%c10_i64] : !ascendc.local_tensor<16xf32>, i64, !ascendc.local_tensor<6xf32>
// CHECK-NEXT:  ascendc.data_copy_l2 %5, %0, %c6_i64 : !ascendc.local_tensor<6xf32>, !ascendc.local_tensor<6xf32>, i64
// CHECK-NEXT:  return %3 : !asctile.tile<16xf32, UB>
func.func @convert_concat_fallback(%arg0: !asctile.tile<10xf32, UB>, %arg1: !asctile.tile<6xf32, UB>) -> !asctile.tile<16xf32, UB> {
  %0 = asctile.concat %arg0, %arg1 : !asctile.tile<10xf32, UB>, !asctile.tile<6xf32, UB> -> !asctile.tile<16xf32, UB>
  return %0 : !asctile.tile<16xf32, UB>
}
