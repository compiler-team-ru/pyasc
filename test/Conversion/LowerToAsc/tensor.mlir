// RUN: ascir-opt -asclower-tensor -canonicalize %s | FileCheck %s

// CHECK-LABEL: func.func @lower_splat(%arg0: f32) -> tensor<32xf32, #asctile.local<UB>> {
// CHECK:       %0 = ascendc.local_tensor_auto veccalc() : <32xf32>
// CHECK-NEXT:  %1 = builtin.unrealized_conversion_cast %0 : !ascendc.local_tensor<32xf32> to tensor<32xf32, #asctile.local<UB>>
// CHECK-NEXT:  ascendc.duplicate_l2 %0, %arg0, %c32_i64 : !ascendc.local_tensor<32xf32>, f32, i64
// CHECK-NEXT:  return %1 : tensor<32xf32, #asctile.local<UB>>
// CHECK-NEXT:}
func.func @lower_splat(%arg0: f32) -> tensor<32xf32, #asctile.local<UB>> {
  %0 = tensor.splat %arg0 : tensor<32xf32, #asctile.local<UB>>
  return %0 : tensor<32xf32, #asctile.local<UB>>
}
