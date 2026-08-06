// RUN: ascir-opt -canonicalize %s | FileCheck %s

// CHECK-LABEL: func.func @if_aic_erase_empty_group(%arg0: !ascendc.local_tensor<16x16xf16>) -> !ascendc.local_tensor<16x16xf16> {
// CHECK-NEXT: return %arg0 : !ascendc.local_tensor<16x16xf16>
// CHECK-NEXT:}
func.func @if_aic_erase_empty_group(%arg0: !ascendc.local_tensor<16x16xf16>) -> !ascendc.local_tensor<16x16xf16> {
  %0 = ascendc.if_aic(%arg0 : !ascendc.local_tensor<16x16xf16>) -> !ascendc.local_tensor<16x16xf16> {
    ascendc.yield %arg0 : !ascendc.local_tensor<16x16xf16>
  }
  return %0 : !ascendc.local_tensor<16x16xf16>
}

// CHECK-LABEL: func.func @if_aiv_erase_empty_group(%arg0: !ascendc.local_tensor<16x16xf16>) -> !ascendc.local_tensor<16x16xf16> {
// CHECK-NEXT: return %arg0 : !ascendc.local_tensor<16x16xf16>
// CHECK-NEXT:}
func.func @if_aiv_erase_empty_group(%arg0: !ascendc.local_tensor<16x16xf16>) -> !ascendc.local_tensor<16x16xf16> {
  %0 = ascendc.if_aiv(%arg0 : !ascendc.local_tensor<16x16xf16>) -> !ascendc.local_tensor<16x16xf16> {
    ascendc.yield %arg0 : !ascendc.local_tensor<16x16xf16>
  }
  return %0 : !ascendc.local_tensor<16x16xf16>
}

// CHECK-LABEL: func.func @if_aic_erase_empty_group_no_operands() {
// CHECK-NEXT: return
// CHECK-NEXT:}
func.func @if_aic_erase_empty_group_no_operands() {
  ascendc.if_aic {
    ascendc.yield
  }
  return
}

// CHECK-LABEL: func.func @if_aiv_erase_empty_group_no_operands() {
// CHECK-NEXT: return
// CHECK-NEXT:}
func.func @if_aiv_erase_empty_group_no_operands() {
  ascendc.if_aiv {
    ascendc.yield
  }
  return
}

// CHECK-LABEL: func.func @if_aic_erase_unused_operands(%arg0: !ascendc.local_tensor<16x16xf16>, %arg1: !ascendc.local_tensor<16x16xf16>) -> !ascendc.local_tensor<16x16xf16> {
// CHECK-NEXT: return %arg0 : !ascendc.local_tensor<16x16xf16>
// CHECK-NEXT:}
func.func @if_aic_erase_unused_operands(%arg0: !ascendc.local_tensor<16x16xf16>, %arg1: !ascendc.local_tensor<16x16xf16>) -> !ascendc.local_tensor<16x16xf16> {
  %0 = ascendc.if_aic(%arg0, %arg1 : !ascendc.local_tensor<16x16xf16>, !ascendc.local_tensor<16x16xf16>) -> !ascendc.local_tensor<16x16xf16> {
    ascendc.yield %arg0 : !ascendc.local_tensor<16x16xf16>
  }
  return %0 : !ascendc.local_tensor<16x16xf16>
}

// CHECK-LABEL: func.func @if_aiv_erase_unused_operands(%arg0: !ascendc.local_tensor<16x16xf16>, %arg1: !ascendc.local_tensor<16x16xf16>) -> !ascendc.local_tensor<16x16xf16> {
// CHECK-NEXT: return %arg0 : !ascendc.local_tensor<16x16xf16>
// CHECK-NEXT:}
func.func @if_aiv_erase_unused_operands(%arg0: !ascendc.local_tensor<16x16xf16>, %arg1: !ascendc.local_tensor<16x16xf16>) -> !ascendc.local_tensor<16x16xf16> {
  %0 = ascendc.if_aiv(%arg0, %arg1 : !ascendc.local_tensor<16x16xf16>, !ascendc.local_tensor<16x16xf16>) -> !ascendc.local_tensor<16x16xf16> {
    ascendc.yield %arg0 : !ascendc.local_tensor<16x16xf16>
  }
  return %0 : !ascendc.local_tensor<16x16xf16>
}

// CHECK-LABEL: func.func @if_aic_erase_unused_result(%arg0: !ascendc.local_tensor<16x16xf16>) {
// CHECK-NEXT: return
// CHECK-NEXT:}
func.func @if_aic_erase_unused_result(%arg0: !ascendc.local_tensor<16x16xf16>) {
  %0 = ascendc.if_aic(%arg0 : !ascendc.local_tensor<16x16xf16>) -> !ascendc.local_tensor<16x16xf16> {
    ascendc.yield %arg0 : !ascendc.local_tensor<16x16xf16>
  }
  return
}

// CHECK-LABEL: func.func @if_aiv_erase_unused_result(%arg0: !ascendc.local_tensor<16x16xf16>) {
// CHECK-NEXT: return
// CHECK-NEXT:}
func.func @if_aiv_erase_unused_result(%arg0: !ascendc.local_tensor<16x16xf16>) {
  %0 = ascendc.if_aiv(%arg0 : !ascendc.local_tensor<16x16xf16>) -> !ascendc.local_tensor<16x16xf16> {
    ascendc.yield %arg0 : !ascendc.local_tensor<16x16xf16>
  }
  return
}

// CHECK-LABEL: func.func @if_aic_used_and_forwarded_arg(%arg0: !ascendc.local_tensor<16x16xf16>, %arg1: !ascendc.local_tensor<16x16xf16>) -> (!ascendc.local_tensor<16x16xf16>, !ascendc.local_tensor<16x16xf16>) {
// CHECK-NEXT: return %arg0, %arg1 : !ascendc.local_tensor<16x16xf16>, !ascendc.local_tensor<16x16xf16>
// CHECK-NEXT:}
func.func @if_aic_used_and_forwarded_arg(%arg0: !ascendc.local_tensor<16x16xf16>, %arg1: !ascendc.local_tensor<16x16xf16>) -> (!ascendc.local_tensor<16x16xf16>, !ascendc.local_tensor<16x16xf16>) {
  %0:2 = ascendc.if_aic(%arg0 : !ascendc.local_tensor<16x16xf16>) -> !ascendc.local_tensor<16x16xf16>, !ascendc.local_tensor<16x16xf16> {
    ascendc.yield %arg0, %arg1 : !ascendc.local_tensor<16x16xf16>, !ascendc.local_tensor<16x16xf16>
  }
  return %0#0, %0#1 : !ascendc.local_tensor<16x16xf16>, !ascendc.local_tensor<16x16xf16>
}

// CHECK-LABEL: func.func @if_aiv_used_and_forwarded_arg(%arg0: !ascendc.local_tensor<16x16xf16>, %arg1: !ascendc.local_tensor<16x16xf16>) -> (!ascendc.local_tensor<16x16xf16>, !ascendc.local_tensor<16x16xf16>) {
// CHECK-NEXT: return %arg0, %arg1 : !ascendc.local_tensor<16x16xf16>, !ascendc.local_tensor<16x16xf16>
// CHECK-NEXT:}
func.func @if_aiv_used_and_forwarded_arg(%arg0: !ascendc.local_tensor<16x16xf16>, %arg1: !ascendc.local_tensor<16x16xf16>) -> (!ascendc.local_tensor<16x16xf16>, !ascendc.local_tensor<16x16xf16>) {
  %0:2 = ascendc.if_aiv(%arg0 : !ascendc.local_tensor<16x16xf16>) -> !ascendc.local_tensor<16x16xf16>, !ascendc.local_tensor<16x16xf16> {
    ascendc.yield %arg0, %arg1 : !ascendc.local_tensor<16x16xf16>, !ascendc.local_tensor<16x16xf16>
  }
  return %0#0, %0#1 : !ascendc.local_tensor<16x16xf16>, !ascendc.local_tensor<16x16xf16>
}

// CHECK-LABEL: func.func @if_aic_inline_nested_group() -> (!ascendc.local_tensor<16x16xf32>, !ascendc.local_tensor<16x16xf32>) {
// CHECK-NEXT:  %0:2 = ascendc.if_aic -> !ascendc.local_tensor<16x16xf32>, !ascendc.local_tensor<16x16xf32> {
// CHECK-NEXT:    %1 = ascendc.local_tensor_v3 co1, 0, 1024 : !ascendc.local_tensor<16x16xf32>
// CHECK-NEXT:    %2 = ascendc.local_tensor_v3 co1, 2048, 1024 : !ascendc.local_tensor<16x16xf32>
// CHECK-NEXT:    ascendc.yield %1, %2 : !ascendc.local_tensor<16x16xf32>, !ascendc.local_tensor<16x16xf32>
// CHECK-NEXT:  }
// CHECK-NEXT:  return %0#0, %0#1 : !ascendc.local_tensor<16x16xf32>, !ascendc.local_tensor<16x16xf32>
// CHECK-NEXT:}
func.func @if_aic_inline_nested_group() -> (!ascendc.local_tensor<16x16xf32>, !ascendc.local_tensor<16x16xf32>) {
  %0:2 = ascendc.if_aic -> !ascendc.local_tensor<16x16xf32>, !ascendc.local_tensor<16x16xf32> {
    %1 = ascendc.local_tensor_v3 co1, 0, 1024 : !ascendc.local_tensor<16x16xf32>
    %2 = ascendc.if_aic -> !ascendc.local_tensor<16x16xf32> {
      %3 = ascendc.local_tensor_v3 co1, 2048, 1024 : !ascendc.local_tensor<16x16xf32>
      ascendc.yield %3 : !ascendc.local_tensor<16x16xf32>
    }
    ascendc.yield %1, %2 : !ascendc.local_tensor<16x16xf32>, !ascendc.local_tensor<16x16xf32>
  }
  return %0#0, %0#1 : !ascendc.local_tensor<16x16xf32>, !ascendc.local_tensor<16x16xf32>
}

// CHECK-LABEL: func.func @if_aiv_inline_nested_group() -> (!ascendc.local_tensor<16x16xf32>, !ascendc.local_tensor<16x16xf32>) {
// CHECK-NEXT:  %0:2 = ascendc.if_aiv -> !ascendc.local_tensor<16x16xf32>, !ascendc.local_tensor<16x16xf32> {
// CHECK-NEXT:    %1 = ascendc.local_tensor_v3 co1, 0, 1024 : !ascendc.local_tensor<16x16xf32>
// CHECK-NEXT:    %2 = ascendc.local_tensor_v3 co1, 2048, 1024 : !ascendc.local_tensor<16x16xf32>
// CHECK-NEXT:    ascendc.yield %1, %2 : !ascendc.local_tensor<16x16xf32>, !ascendc.local_tensor<16x16xf32>
// CHECK-NEXT:  }
// CHECK-NEXT:  return %0#0, %0#1 : !ascendc.local_tensor<16x16xf32>, !ascendc.local_tensor<16x16xf32>
// CHECK-NEXT:}
func.func @if_aiv_inline_nested_group() -> (!ascendc.local_tensor<16x16xf32>, !ascendc.local_tensor<16x16xf32>) {
  %0:2 = ascendc.if_aiv -> !ascendc.local_tensor<16x16xf32>, !ascendc.local_tensor<16x16xf32> {
    %1 = ascendc.local_tensor_v3 co1, 0, 1024 : !ascendc.local_tensor<16x16xf32>
    %2 = ascendc.if_aiv -> !ascendc.local_tensor<16x16xf32> {
      %3 = ascendc.local_tensor_v3 co1, 2048, 1024 : !ascendc.local_tensor<16x16xf32>
      ascendc.yield %3 : !ascendc.local_tensor<16x16xf32>
    }
    ascendc.yield %1, %2 : !ascendc.local_tensor<16x16xf32>, !ascendc.local_tensor<16x16xf32>
  }
  return %0#0, %0#1 : !ascendc.local_tensor<16x16xf32>, !ascendc.local_tensor<16x16xf32>
}
