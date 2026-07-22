// RUN: ascir-opt -ascendc-erase-sync %s | FileCheck %s

// CHECK-LABEL: func.func @test_erase_set_wait_flag() {
// CHECK-NEXT: %0 = ascendc.pipe
// CHECK-NEXT: %1 = ascendc.pipe.fetch_event_id %0, v_s : i8
// CHECK-NEXT: return
// CHECK-NEXT:}
func.func @test_erase_set_wait_flag() {
  %0 = ascendc.pipe
  %1 = ascendc.pipe.fetch_event_id %0, v_s : i8
  ascendc.set_flag v_s, %1 : i8
  ascendc.wait_flag v_s, %1 : i8
  return
}

// CHECK-LABEL: func.func @test_erase_pipe_barrier() {
// CHECK-NEXT: return
// CHECK-NEXT:}
func.func @test_erase_pipe_barrier() {
  ascendc.pipe_barrier pipe_v
  return
}

// CHECK-LABEL: func.func @test_erase_enque(%arg0: !ascendc.global_tensor<*xf16>, %arg1: i32, %arg2: i64) {
// CHECK-NEXT: %0 = ascendc.pipe
// CHECK-NEXT: %1 = ascendc.queue : <vecin, 1>
// CHECK-NEXT: ascendc.pipe.init_queue %0, %1, %arg1, %arg2 : !ascendc.queue<vecin, 1>, i32, i64
// CHECK-NEXT: %2 = ascendc.que_bind.alloc_tensor %1 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<32xf16>
// CHECK-NEXT: return
// CHECK-NEXT:}
func.func @test_erase_enque(%arg0: !ascendc.global_tensor<*xf16>, %arg1: i32, %arg2: i64) {
  %0 = ascendc.pipe
  %1 = ascendc.queue : <vecin, 1>
  ascendc.pipe.init_queue %0, %1, %arg1, %arg2 : !ascendc.queue<vecin, 1>, i32, i64
  %2 = ascendc.que_bind.alloc_tensor %1 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<32xf16>
  ascendc.que_bind.enque_tensor %1, %2 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<32xf16>
  return
}

// CHECK-LABEL: func.func @test_deque_replaced(%arg0: i32, %arg1: i64) {
// CHECK-NEXT: %0 = ascendc.pipe
// CHECK-NEXT: %1 = ascendc.queue : <vecin, 1>
// CHECK-NEXT: ascendc.pipe.init_queue %0, %1, %arg0, %arg1 : !ascendc.queue<vecin, 1>, i32, i64
// CHECK-NEXT: %2 = ascendc.que_bind.alloc_tensor %1 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<32xf16>
// CHECK-NEXT: ascendc.add_l2 %2, %2, %2, %arg0 : !ascendc.local_tensor<32xf16>, !ascendc.local_tensor<32xf16>, !ascendc.local_tensor<32xf16>, i32
// CHECK-NEXT: return
// CHECK-NEXT:}
func.func @test_deque_replaced(%arg0: i32, %arg1: i64) {
  %0 = ascendc.pipe
  %1 = ascendc.queue : <vecin, 1>
  ascendc.pipe.init_queue %0, %1, %arg0, %arg1 : !ascendc.queue<vecin, 1>, i32, i64
  %2 = ascendc.que_bind.alloc_tensor %1 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<32xf16>
  %3 = ascendc.que_bind.deque_tensor %1 : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<32xf16>
  ascendc.add_l2 %3, %3, %3, %arg0 : !ascendc.local_tensor<32xf16>, !ascendc.local_tensor<32xf16>, !ascendc.local_tensor<32xf16>, i32
  return
}

// CHECK-LABEL: func.func private @test_declaration(i32) -> i32
func.func private @test_declaration(%arg0: i32) -> i32
