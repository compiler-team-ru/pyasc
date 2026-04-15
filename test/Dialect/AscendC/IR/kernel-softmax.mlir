// RUN: ascir-opt %s | ascir-opt
// RUN: ascir-opt %s --mlir-print-op-generic | ascir-opt

module attributes {asc.kernel_type = "vector", asc.memory_consumed = {UB = 24768 : i64}, asc.soc_version = "Ascend950PR_9599"} {
  emitc.include "kernel_operator.h"
  func.func @softmax_kernel(%arg0: memref<*xf32, 22> {emitasc.kernel_arg = #emitasc<kernel_arg explicit>}, %arg1: memref<*xf32, 22> {emitasc.kernel_arg = #emitasc<kernel_arg explicit>}, %arg2: i32 {emitasc.kernel_arg = #emitasc<kernel_arg explicit>}, %arg3: i32 {emitasc.kernel_arg = #emitasc<kernel_arg explicit>}) attributes {ascendc.aicore, ascendc.global} {
    %c0_i64 = arith.constant 0 : i64
    %c1024_i64 = arith.constant 1024 : i64
    %c8_i32 = arith.constant 8 : i32
    %c4_i32 = arith.constant 4 : i32
    %c1024_i32 = arith.constant 1024 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst = arith.constant 0.000000e+00 : f32
    %c0_i32 = arith.constant 0 : i32
    %0 = ascendc.pipe
    %1 = ascendc.global_tensor : !ascendc.global_tensor<?x?xf32>
    ascendc.global_tensor.set_global_buffer %1, %arg0 : !ascendc.global_tensor<?x?xf32>, memref<*xf32, 22>
    %2 = ascendc.global_tensor : !ascendc.global_tensor<?x?xf32>
    ascendc.global_tensor.set_global_buffer %2, %arg1 : !ascendc.global_tensor<?x?xf32>, memref<*xf32, 22>
    %3 = ascendc.get_block_idx : i32
    %4 = ascendc.get_block_num : i32
    %5 = ascendc.local_tensor_v3 veccalc, 0, 1024 : !ascendc.local_tensor<1x1024xf32>
    %6 = ascendc.local_tensor_v3 veccalc, 4096, 1024 : !ascendc.local_tensor<1x1024xf32>
    %7 = ascendc.local_tensor_v3 veccalc, 8192, 16 : !ascendc.local_tensor<16xf32>
    %8 = ascendc.local_tensor_v3 veccalc, 8256, 8 : !ascendc.local_tensor<1xf32>
    %9 = ascendc.local_tensor_v3 veccalc, 8288, 1024 : !ascendc.local_tensor<1x1024xf32>
    %10 = ascendc.local_tensor_v3 veccalc, 12384, 1024 : !ascendc.local_tensor<1x1024xf32>
    %11 = ascendc.local_tensor_v3 veccalc, 16480, 1024 : !ascendc.local_tensor<1x1024xf32>
    %12 = ascendc.local_tensor_v3 veccalc, 20576, 16 : !ascendc.local_tensor<16xf32>
    %13 = ascendc.local_tensor_v3 veccalc, 20640, 8 : !ascendc.local_tensor<1xf32>
    %14 = ascendc.local_tensor_v3 veccalc, 20672, 1024 : !ascendc.local_tensor<1x1024xf32>
    %15 = arith.muli %arg2, %arg3 : i32
    %16 = arith.subi %arg3, %c1024_i32 : i32
    %17 = arith.muli %16, %c4_i32 : i32
    scf.for %arg4 = %3 to %arg2 step %4  : i32 {
      %18 = arith.muli %arg4, %arg3 : i32
      %19 = ascendc.global_tensor.subindex %1[%18] : !ascendc.global_tensor<?x?xf32>, i32, !ascendc.global_tensor<?x?xf32>
      %20 = arith.subi %15, %18 : i32
      %21 = arith.minsi %20, %c1024_i32 : i32
      %22 = arith.muli %21, %c4_i32 : i32
      %23 = ascendc.construct !ascendc.data_copy_ext_params(%c1_i32, %22, %17, %c0_i32, %c0_i32) [ui16, ui32, ui32, ui32, ui32] : i32, i32, i32, i32, i32
      %24 = arith.subi %c8_i32, %20 : i32
      %25 = arith.maxsi %24, %c0_i32 : i32
      %26 = ascendc.construct !ascendc.data_copy_pad_ext_params<f32>(%c1_i32, %c0_i32, %25, %cst) [i32, i32, ui8, f32] : i32, i32, i32, f32
      ascendc.get_buf pipe_mte2, 0
      ascendc.data_copy_pad_l0_ext %14, %19, %23, %26 : !ascendc.local_tensor<1x1024xf32>, !ascendc.global_tensor<?x?xf32>, !ascendc.data_copy_ext_params, !ascendc.data_copy_pad_ext_params<f32>
      ascendc.rls_buf pipe_mte2, 0
      ascendc.get_buf pipe_v, 1
      ascendc.get_buf pipe_v, 0
      ascendc.reduce_max_l2 %13, %14, %12, %c1024_i64, %c0_i64 : !ascendc.local_tensor<1xf32>, !ascendc.local_tensor<1x1024xf32>, !ascendc.local_tensor<16xf32>, i64, i64
      ascendc.rls_buf pipe_v, 0
      ascendc.rls_buf pipe_v, 1
      ascendc.get_buf pipe_v, 2
      ascendc.duplicate_l2 %11, %13, %c1024_i64 : !ascendc.local_tensor<1x1024xf32>, !ascendc.local_tensor<1xf32>, i64
      ascendc.rls_buf pipe_v, 2
      ascendc.get_buf pipe_v, 3
      ascendc.get_buf pipe_v, 0
      ascendc.get_buf pipe_v, 2
      ascendc.sub_l2 %10, %14, %11, %c1024_i64 : !ascendc.local_tensor<1x1024xf32>, !ascendc.local_tensor<1x1024xf32>, !ascendc.local_tensor<1x1024xf32>, i64
      ascendc.rls_buf pipe_v, 2
      ascendc.rls_buf pipe_v, 0
      ascendc.rls_buf pipe_v, 3
      ascendc.get_buf pipe_v, 4
      ascendc.get_buf pipe_v, 3
      ascendc.exp_l2 %9, %10, %c1024_i64 : !ascendc.local_tensor<1x1024xf32>, !ascendc.local_tensor<1x1024xf32>, i64
      ascendc.rls_buf pipe_v, 3
      ascendc.rls_buf pipe_v, 4
      ascendc.get_buf pipe_v, 5
      ascendc.get_buf pipe_v, 4
      ascendc.reduce_sum_l2 %8, %9, %7, %c1024_i64 : !ascendc.local_tensor<1xf32>, !ascendc.local_tensor<1x1024xf32>, !ascendc.local_tensor<16xf32>, i64
      ascendc.rls_buf pipe_v, 4
      ascendc.rls_buf pipe_v, 5
      ascendc.get_buf pipe_v, 6
      ascendc.duplicate_l2 %6, %8, %c1024_i64 : !ascendc.local_tensor<1x1024xf32>, !ascendc.local_tensor<1xf32>, i64
      ascendc.rls_buf pipe_v, 6
      ascendc.get_buf pipe_v, 7
      ascendc.get_buf pipe_v, 4
      ascendc.get_buf pipe_v, 6
      ascendc.div_l2 %5, %9, %6, %c1024_i64 : !ascendc.local_tensor<1x1024xf32>, !ascendc.local_tensor<1x1024xf32>, !ascendc.local_tensor<1x1024xf32>, i64
      ascendc.rls_buf pipe_v, 6
      ascendc.rls_buf pipe_v, 4
      ascendc.rls_buf pipe_v, 7
      %27 = ascendc.global_tensor.subindex %2[%18] : !ascendc.global_tensor<?x?xf32>, i32, !ascendc.global_tensor<?x?xf32>
      %28 = ascendc.construct !ascendc.data_copy_ext_params(%c1_i32, %22, %c0_i32, %17, %c0_i32) [ui16, ui32, ui32, ui32, ui32] : i32, i32, i32, i32, i32
      ascendc.get_buf pipe_mte3, 7
      ascendc.data_copy_pad_l2_ext %27, %5, %28 : !ascendc.global_tensor<?x?xf32>, !ascendc.local_tensor<1x1024xf32>, !ascendc.data_copy_ext_params
      ascendc.rls_buf pipe_mte3, 7
    }
    return
  }
}
