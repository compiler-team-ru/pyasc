// RUN: ascir-opt -ascendc-fuse-vf-block %s | FileCheck %s

// CHECK-LABEL: func.func @general_test(%arg0: !ascendc.que_bind<gm, vecin, 1>) {
// CHECK-NEXT:   %c256_i32 = arith.constant 256 : i32
// CHECK-NEXT:   %0 = ascendc.que_bind.alloc_tensor %arg0 : !ascendc.que_bind<gm, vecin, 1>, !ascendc.local_tensor<*xf32>
// CHECK-NEXT:   %1 = ascendc.que_bind.deque_tensor %arg0 : !ascendc.que_bind<gm, vecin, 1>, !ascendc.local_tensor<*xf32>
// CHECK-NEXT:   %2 = ascendc.que_bind.deque_tensor %arg0 : !ascendc.que_bind<gm, vecin, 1>, !ascendc.local_tensor<*xf32>
// CHECK-NEXT:   emitasc.vf_group %0, %2, %1 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32> {
// CHECK-NEXT:     %3 = ascendc.local_tensor.get_phy_addr_v2 %0 : !ascendc.local_tensor<*xf32>, memref<f32, 26>
// CHECK-NEXT:     %4 = ascendc.local_tensor.get_phy_addr_v2 %1 : !ascendc.local_tensor<*xf32>, memref<f32, 26>
// CHECK-NEXT:     %5 = ascendc.local_tensor.get_phy_addr_v2 %2 : !ascendc.local_tensor<*xf32>, memref<f32, 26>
// CHECK-NEXT:     emitasc.vec_scope {
// CHECK-NEXT:       %6 = ascendc.reg_tensor : <f32>
// CHECK-NEXT:       %7 = ascendc.reg_tensor : <f32>
// CHECK-NEXT:       %8 = ascendc.reg_tensor : <f32>
// CHECK-NEXT:       %9 = emitasc.variable %c256_i32 : i32, memref<1xui32>
// CHECK-NEXT:       %10 = ascendc.get_vec_len : index
// CHECK-NEXT:       %c4 = arith.constant 4 : index
// CHECK-NEXT:       %11 = arith.divsi %10, %c4 : index
// CHECK-NEXT:       %12 = arith.index_cast %c256_i32 : i32 to index
// CHECK-NEXT:       %13 = arith.ceildivsi %12, %11 : index
// CHECK-NEXT:       %c0 = arith.constant 0 : index
// CHECK-NEXT:       %c1 = arith.constant 1 : index
// CHECK-NEXT:       scf.for %arg1 = %c0 to %13 step %c1 {
// CHECK-NEXT:         %14 = ascendc.update_mask f32, %9 : memref<1xui32>
// CHECK-NEXT:         %15 = arith.muli %arg1, %11 : index
// CHECK-NEXT:         %16 = emitasc.ptr_offset %5[%15] : memref<f32, 26>, memref<f32, 26>
// CHECK-NEXT:         ascendc.data_copy_vld_micro %8, %16 : !ascendc.reg_tensor<f32>, memref<f32, 26>
// CHECK-NEXT:         %17 = emitasc.ptr_offset %4[%15] : memref<f32, 26>, memref<f32, 26>
// CHECK-NEXT:         ascendc.data_copy_vld_micro %7, %17 : !ascendc.reg_tensor<f32>, memref<f32, 26>
// CHECK-NEXT:         ascendc.add_micro %6, %7, %8, %14 : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
// CHECK-NEXT:         ascendc.mul_micro %6, %6, %8, %14 : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
// CHECK-NEXT:         %18 = emitasc.ptr_offset %3[%15] : memref<f32, 26>, memref<f32, 26>
// CHECK-NEXT:         ascendc.data_copy_vst_micro %18, %6, %14 : memref<f32, 26>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
// CHECK-NEXT:       } {asc.vec_scope_loop}
// CHECK-NEXT:     }
// CHECK-NEXT:   } {operandSegmentSizes = array<i32: 1, 2>}
// CHECK-NEXT:   ascendc.que_bind.enque_tensor %arg0, %0 : !ascendc.que_bind<gm, vecin, 1>, !ascendc.local_tensor<*xf32>
// CHECK-NEXT:   ascendc.que_bind.free_tensor %arg0, %1 : !ascendc.que_bind<gm, vecin, 1>, !ascendc.local_tensor<*xf32>
// CHECK-NEXT:   ascendc.que_bind.free_tensor %arg0, %2 : !ascendc.que_bind<gm, vecin, 1>, !ascendc.local_tensor<*xf32>
// CHECK-NEXT:   return
// CHECK-NEXT: }
func.func @general_test(%que_bind: !ascendc.que_bind<gm, vecin, 1>) {
  %c256_i32 = arith.constant 256 : i32
  %dst = ascendc.que_bind.alloc_tensor %que_bind : !ascendc.que_bind<gm, vecin, 1>, !ascendc.local_tensor<*xf32>
  %src0 = ascendc.que_bind.deque_tensor %que_bind : !ascendc.que_bind<gm, vecin, 1>, !ascendc.local_tensor<*xf32>
  %src1 = ascendc.que_bind.deque_tensor %que_bind : !ascendc.que_bind<gm, vecin, 1>, !ascendc.local_tensor<*xf32>
  ascendc.add_l2 %dst, %src0, %src1, %c256_i32 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, i32
  ascendc.pipe_barrier pipe_v
  ascendc.mul_l2 %dst, %dst, %src1, %c256_i32 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, i32
  ascendc.que_bind.enque_tensor %que_bind, %dst : !ascendc.que_bind<gm, vecin, 1>, !ascendc.local_tensor<*xf32>
  ascendc.que_bind.free_tensor %que_bind, %src0 : !ascendc.que_bind<gm, vecin, 1>, !ascendc.local_tensor<*xf32>
  ascendc.que_bind.free_tensor %que_bind, %src1 : !ascendc.que_bind<gm, vecin, 1>, !ascendc.local_tensor<*xf32>
  return
}

// CHECK-LABEL: func.func @not_convert_if_one_operation(%arg0: !ascendc.local_tensor<*xf32>) {
// CHECK-NEXT:   %c256_i32 = arith.constant 256 : i32
// CHECK-NEXT:   ascendc.add_l2 %arg0, %arg0, %arg0, %c256_i32 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, i32
// CHECK-NEXT:   return
// CHECK-NEXT: }
func.func @not_convert_if_one_operation(%vec: !ascendc.local_tensor<*xf32>) {
  %c256 = arith.constant 256 : i32
  ascendc.add_l2 %vec, %vec, %vec, %c256 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, i32
  return
}

// CHECK-LABEL: func.func @same_group_if_cal_count_and_types_equals(%arg0: !ascendc.local_tensor<*xf32>) {
// CHECK-NEXT:   %c256_i32 = arith.constant 256 : i32
// CHECK-NEXT:   %c256_i32_0 = arith.constant 256 : i32
// CHECK-NEXT:   emitasc.vf_group %arg0, %arg0 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32> {
// CHECK-NEXT:     %0 = ascendc.local_tensor.get_phy_addr_v2 %arg0 : !ascendc.local_tensor<*xf32>, memref<f32, 26>
// CHECK-NEXT:     emitasc.vec_scope {
// CHECK-NEXT:       %1 = ascendc.reg_tensor : <f32>
// CHECK-NEXT:       %2 = emitasc.variable %c256_i32 : i32, memref<1xui32>
// CHECK-NEXT:       %3 = ascendc.get_vec_len : index
// CHECK-NEXT:       %c4 = arith.constant 4 : index
// CHECK-NEXT:       %4 = arith.divsi %3, %c4 : index
// CHECK-NEXT:       %5 = arith.index_cast %c256_i32 : i32 to index
// CHECK-NEXT:       %6 = arith.ceildivsi %5, %4 : index
// CHECK-NEXT:       %c0 = arith.constant 0 : index
// CHECK-NEXT:       %c1 = arith.constant 1 : index
// CHECK-NEXT:       scf.for %arg1 = %c0 to %6 step %c1 {
// CHECK-NEXT:         %7 = ascendc.update_mask f32, %2 : memref<1xui32>
// CHECK-NEXT:         %8 = arith.muli %arg1, %4 : index
// CHECK-NEXT:         %9 = emitasc.ptr_offset %0[%8] : memref<f32, 26>, memref<f32, 26>
// CHECK-NEXT:         ascendc.data_copy_vld_micro %1, %9 : !ascendc.reg_tensor<f32>, memref<f32, 26>
// CHECK-NEXT:         ascendc.add_micro %1, %1, %1, %7 : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
// CHECK-NEXT:         ascendc.mul_micro %1, %1, %1, %7 : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
// CHECK-NEXT:         %10 = emitasc.ptr_offset %0[%8] : memref<f32, 26>, memref<f32, 26>
// CHECK-NEXT:         ascendc.data_copy_vst_micro %10, %1, %7 : memref<f32, 26>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
// CHECK-NEXT:       } {asc.vec_scope_loop}
// CHECK-NEXT:     }
// CHECK-NEXT:   } {operandSegmentSizes = array<i32: 1, 1>}
// CHECK-NEXT:   return
// CHECK-NEXT: }
func.func @same_group_if_cal_count_and_types_equals(%vec: !ascendc.local_tensor<*xf32>) {
  %c256_0 = arith.constant 256 : i32
  %c256_1 = arith.constant 256 : i32
  ascendc.add_l2 %vec, %vec, %vec, %c256_0 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, i32
  ascendc.mul_l2 %vec, %vec, %vec, %c256_1 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, i32
  return
}

// CHECK-LABEL: func.func @different_group_if_cal_count_not_equals(%arg0: !ascendc.local_tensor<*xf32>) {
// CHECK-NEXT:   %c256_i32 = arith.constant 256 : i32
// CHECK-NEXT:   %c128_i32 = arith.constant 128 : i32
// CHECK-NEXT:   emitasc.vf_group %arg0, %arg0 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32> {
// CHECK-NEXT:     %0 = ascendc.local_tensor.get_phy_addr_v2 %arg0 : !ascendc.local_tensor<*xf32>, memref<f32, 26>
// CHECK-NEXT:     emitasc.vec_scope {
// CHECK-NEXT:       %1 = ascendc.reg_tensor : <f32>
// CHECK-NEXT:       %2 = emitasc.variable %c256_i32 : i32, memref<1xui32>
// CHECK-NEXT:       %3 = ascendc.get_vec_len : index
// CHECK-NEXT:       %c4 = arith.constant 4 : index
// CHECK-NEXT:       %4 = arith.divsi %3, %c4 : index
// CHECK-NEXT:       %5 = arith.index_cast %c256_i32 : i32 to index
// CHECK-NEXT:       %6 = arith.ceildivsi %5, %4 : index
// CHECK-NEXT:       %c0 = arith.constant 0 : index
// CHECK-NEXT:       %c1 = arith.constant 1 : index
// CHECK-NEXT:       scf.for %arg1 = %c0 to %6 step %c1 {
// CHECK-NEXT:         %7 = ascendc.update_mask f32, %2 : memref<1xui32>
// CHECK-NEXT:         %8 = arith.muli %arg1, %4 : index
// CHECK-NEXT:         %9 = emitasc.ptr_offset %0[%8] : memref<f32, 26>, memref<f32, 26>
// CHECK-NEXT:         ascendc.data_copy_vld_micro %1, %9 : !ascendc.reg_tensor<f32>, memref<f32, 26>
// CHECK-NEXT:         ascendc.add_micro %1, %1, %1, %7 : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
// CHECK-NEXT:         ascendc.mul_micro %1, %1, %1, %7 : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
// CHECK-NEXT:         %10 = emitasc.ptr_offset %0[%8] : memref<f32, 26>, memref<f32, 26>
// CHECK-NEXT:         ascendc.data_copy_vst_micro %10, %1, %7 : memref<f32, 26>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
// CHECK-NEXT:       } {asc.vec_scope_loop}
// CHECK-NEXT:     }
// CHECK-NEXT:   } {operandSegmentSizes = array<i32: 1, 1>}
// CHECK-NEXT:   emitasc.vf_group %arg0, %arg0 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32> {
// CHECK-NEXT:     %0 = ascendc.local_tensor.get_phy_addr_v2 %arg0 : !ascendc.local_tensor<*xf32>, memref<f32, 26>
// CHECK-NEXT:     emitasc.vec_scope {
// CHECK-NEXT:       %1 = ascendc.reg_tensor : <f32>
// CHECK-NEXT:       %2 = emitasc.variable %c128_i32 : i32, memref<1xui32>
// CHECK-NEXT:       %3 = ascendc.get_vec_len : index
// CHECK-NEXT:       %c4 = arith.constant 4 : index
// CHECK-NEXT:       %4 = arith.divsi %3, %c4 : index
// CHECK-NEXT:       %5 = arith.index_cast %c128_i32 : i32 to index
// CHECK-NEXT:       %6 = arith.ceildivsi %5, %4 : index
// CHECK-NEXT:       %c0 = arith.constant 0 : index
// CHECK-NEXT:       %c1 = arith.constant 1 : index
// CHECK-NEXT:       scf.for %arg1 = %c0 to %6 step %c1 {
// CHECK-NEXT:         %7 = ascendc.update_mask f32, %2 : memref<1xui32>
// CHECK-NEXT:         %8 = arith.muli %arg1, %4 : index
// CHECK-NEXT:         %9 = emitasc.ptr_offset %0[%8] : memref<f32, 26>, memref<f32, 26>
// CHECK-NEXT:         ascendc.data_copy_vld_micro %1, %9 : !ascendc.reg_tensor<f32>, memref<f32, 26>
// CHECK-NEXT:         ascendc.add_micro %1, %1, %1, %7 : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
// CHECK-NEXT:         ascendc.mul_micro %1, %1, %1, %7 : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
// CHECK-NEXT:         %10 = emitasc.ptr_offset %0[%8] : memref<f32, 26>, memref<f32, 26>
// CHECK-NEXT:         ascendc.data_copy_vst_micro %10, %1, %7 : memref<f32, 26>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
// CHECK-NEXT:       } {asc.vec_scope_loop}
// CHECK-NEXT:     }
// CHECK-NEXT:   } {operandSegmentSizes = array<i32: 1, 1>}
// CHECK-NEXT:   return
// CHECK-NEXT: }
func.func @different_group_if_cal_count_not_equals(%vec: !ascendc.local_tensor<*xf32>) {
  %c256 = arith.constant 256 : i32
  %c128 = arith.constant 128 : i32
  ascendc.add_l2 %vec, %vec, %vec, %c256 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, i32
  ascendc.mul_l2 %vec, %vec, %vec, %c256 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, i32
  ascendc.add_l2 %vec, %vec, %vec, %c128 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, i32
  ascendc.mul_l2 %vec, %vec, %vec, %c128 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, i32
  return
}

// CHECK-LABEL: func.func @different_group_if_types_not_equals(%arg0: !ascendc.local_tensor<*xf32>, %arg1: !ascendc.local_tensor<*xf16>) {
// CHECK-NEXT:   %c256_i32 = arith.constant 256 : i32
// CHECK-NEXT:   ascendc.add_l2 %arg0, %arg0, %arg0, %c256_i32 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, i32
// CHECK-NEXT:   ascendc.mul_l2 %arg1, %arg1, %arg1, %c256_i32 : !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, i32
// CHECK-NEXT:   return
// CHECK-NEXT: }
func.func @different_group_if_types_not_equals(%vecf32: !ascendc.local_tensor<*xf32>, %vecf16: !ascendc.local_tensor<*xf16>) {
  %c256 = arith.constant 256 : i32
  ascendc.add_l2 %vecf32, %vecf32, %vecf32, %c256 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, i32
  ascendc.mul_l2 %vecf16, %vecf16, %vecf16, %c256 : !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, i32
  return
}

// CHECK-LABEL: func.func @different_group_if_between_op_exist_other_op(%arg0: !ascendc.local_tensor<*xf32>) {
// CHECK-NEXT:   %c256_i32 = arith.constant 256 : i32
// CHECK-NEXT:   emitasc.vf_group %arg0, %arg0 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32> {
// CHECK-NEXT:     %0 = ascendc.local_tensor.get_phy_addr_v2 %arg0 : !ascendc.local_tensor<*xf32>, memref<f32, 26>
// CHECK-NEXT:     emitasc.vec_scope {
// CHECK-NEXT:       %1 = ascendc.reg_tensor : <f32>
// CHECK-NEXT:       %2 = emitasc.variable %c256_i32 : i32, memref<1xui32>
// CHECK-NEXT:       %3 = ascendc.get_vec_len : index
// CHECK-NEXT:       %c4 = arith.constant 4 : index
// CHECK-NEXT:       %4 = arith.divsi %3, %c4 : index
// CHECK-NEXT:       %5 = arith.index_cast %c256_i32 : i32 to index
// CHECK-NEXT:       %6 = arith.ceildivsi %5, %4 : index
// CHECK-NEXT:       %c0 = arith.constant 0 : index
// CHECK-NEXT:       %c1 = arith.constant 1 : index
// CHECK-NEXT:       scf.for %arg1 = %c0 to %6 step %c1 {
// CHECK-NEXT:         %7 = ascendc.update_mask f32, %2 : memref<1xui32>
// CHECK-NEXT:         %8 = arith.muli %arg1, %4 : index
// CHECK-NEXT:         %9 = emitasc.ptr_offset %0[%8] : memref<f32, 26>, memref<f32, 26>
// CHECK-NEXT:         ascendc.data_copy_vld_micro %1, %9 : !ascendc.reg_tensor<f32>, memref<f32, 26>
// CHECK-NEXT:         ascendc.add_micro %1, %1, %1, %7 : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
// CHECK-NEXT:         ascendc.mul_micro %1, %1, %1, %7 : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
// CHECK-NEXT:         %10 = emitasc.ptr_offset %0[%8] : memref<f32, 26>, memref<f32, 26>
// CHECK-NEXT:         ascendc.data_copy_vst_micro %10, %1, %7 : memref<f32, 26>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
// CHECK-NEXT:       } {asc.vec_scope_loop}
// CHECK-NEXT:     }
// CHECK-NEXT:   } {operandSegmentSizes = array<i32: 1, 1>}
// CHECK-NEXT:   call @blank() : () -> ()
// CHECK-NEXT:   emitasc.vf_group %arg0, %arg0 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32> {
// CHECK-NEXT:     %0 = ascendc.local_tensor.get_phy_addr_v2 %arg0 : !ascendc.local_tensor<*xf32>, memref<f32, 26>
// CHECK-NEXT:     emitasc.vec_scope {
// CHECK-NEXT:       %1 = ascendc.reg_tensor : <f32>
// CHECK-NEXT:       %2 = emitasc.variable %c256_i32 : i32, memref<1xui32>
// CHECK-NEXT:       %3 = ascendc.get_vec_len : index
// CHECK-NEXT:       %c4 = arith.constant 4 : index
// CHECK-NEXT:       %4 = arith.divsi %3, %c4 : index
// CHECK-NEXT:       %5 = arith.index_cast %c256_i32 : i32 to index
// CHECK-NEXT:       %6 = arith.ceildivsi %5, %4 : index
// CHECK-NEXT:       %c0 = arith.constant 0 : index
// CHECK-NEXT:       %c1 = arith.constant 1 : index
// CHECK-NEXT:       scf.for %arg1 = %c0 to %6 step %c1 {
// CHECK-NEXT:         %7 = ascendc.update_mask f32, %2 : memref<1xui32>
// CHECK-NEXT:         %8 = arith.muli %arg1, %4 : index
// CHECK-NEXT:         %9 = emitasc.ptr_offset %0[%8] : memref<f32, 26>, memref<f32, 26>
// CHECK-NEXT:         ascendc.data_copy_vld_micro %1, %9 : !ascendc.reg_tensor<f32>, memref<f32, 26>
// CHECK-NEXT:         ascendc.add_micro %1, %1, %1, %7 : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
// CHECK-NEXT:         ascendc.mul_micro %1, %1, %1, %7 : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
// CHECK-NEXT:         %10 = emitasc.ptr_offset %0[%8] : memref<f32, 26>, memref<f32, 26>
// CHECK-NEXT:         ascendc.data_copy_vst_micro %10, %1, %7 : memref<f32, 26>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
// CHECK-NEXT:       } {asc.vec_scope_loop}
// CHECK-NEXT:     }
// CHECK-NEXT:   } {operandSegmentSizes = array<i32: 1, 1>}
// CHECK-NEXT:   return
// CHECK-NEXT: }
func.func @different_group_if_between_op_exist_other_op(%vec: !ascendc.local_tensor<*xf32>) {
  %c256 = arith.constant 256 : i32
  ascendc.add_l2 %vec, %vec, %vec, %c256 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, i32
  ascendc.mul_l2 %vec, %vec, %vec, %c256 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, i32
  call @blank() : () -> ()
  ascendc.add_l2 %vec, %vec, %vec, %c256 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, i32
  ascendc.mul_l2 %vec, %vec, %vec, %c256 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, i32
  return
}

// CHECK-LABEL: func.func private @blank()
func.func private @blank()

// CHECK-LABEL: func.func @pipe_barrier_v_and_all_between_op_belong_group(%arg0: !ascendc.local_tensor<*xf32>) {
// CHECK-NEXT:   %c256_i32 = arith.constant 256 : i32
// CHECK-NEXT:   emitasc.vf_group %arg0, %arg0 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32> {
// CHECK-NEXT:     %0 = ascendc.local_tensor.get_phy_addr_v2 %arg0 : !ascendc.local_tensor<*xf32>, memref<f32, 26>
// CHECK-NEXT:     emitasc.vec_scope {
// CHECK-NEXT:       %1 = ascendc.reg_tensor : <f32>
// CHECK-NEXT:       %2 = emitasc.variable %c256_i32 : i32, memref<1xui32>
// CHECK-NEXT:       %3 = ascendc.get_vec_len : index
// CHECK-NEXT:       %c4 = arith.constant 4 : index
// CHECK-NEXT:       %4 = arith.divsi %3, %c4 : index
// CHECK-NEXT:       %5 = arith.index_cast %c256_i32 : i32 to index
// CHECK-NEXT:       %6 = arith.ceildivsi %5, %4 : index
// CHECK-NEXT:       %c0 = arith.constant 0 : index
// CHECK-NEXT:       %c1 = arith.constant 1 : index
// CHECK-NEXT:       scf.for %arg1 = %c0 to %6 step %c1 {
// CHECK-NEXT:         %7 = ascendc.update_mask f32, %2 : memref<1xui32>
// CHECK-NEXT:         %8 = arith.muli %arg1, %4 : index
// CHECK-NEXT:         %9 = emitasc.ptr_offset %0[%8] : memref<f32, 26>, memref<f32, 26>
// CHECK-NEXT:         ascendc.data_copy_vld_micro %1, %9 : !ascendc.reg_tensor<f32>, memref<f32, 26>
// CHECK-NEXT:         ascendc.add_micro %1, %1, %1, %7 : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
// CHECK-NEXT:         ascendc.mul_micro %1, %1, %1, %7 : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
// CHECK-NEXT:         %10 = emitasc.ptr_offset %0[%8] : memref<f32, 26>, memref<f32, 26>
// CHECK-NEXT:         ascendc.data_copy_vst_micro %10, %1, %7 : memref<f32, 26>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
// CHECK-NEXT:       } {asc.vec_scope_loop}
// CHECK-NEXT:     }
// CHECK-NEXT:   } {operandSegmentSizes = array<i32: 1, 1>}
// CHECK-NEXT:   return
// CHECK-NEXT: }
func.func @pipe_barrier_v_and_all_between_op_belong_group(%vec: !ascendc.local_tensor<*xf32>) {
  %c256 = arith.constant 256 : i32
  ascendc.add_l2 %vec, %vec, %vec, %c256 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, i32
  ascendc.pipe_barrier pipe_v
  ascendc.pipe_barrier pipe_all
  ascendc.mul_l2 %vec, %vec, %vec, %c256 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, i32
  return
}

// CHECK-LABEL: func.func @different_group_if_between_op_exist_other_pipe_barrier(%arg0: !ascendc.local_tensor<*xf32>) {
// CHECK-NEXT:   %c256_i32 = arith.constant 256 : i32
// CHECK-NEXT:   ascendc.add_l2 %arg0, %arg0, %arg0, %c256_i32 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, i32
// CHECK-NEXT:   ascendc.pipe_barrier pipe_mte2
// CHECK-NEXT:   ascendc.mul_l2 %arg0, %arg0, %arg0, %c256_i32 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, i32
// CHECK-NEXT:   return
// CHECK-NEXT: }
func.func @different_group_if_between_op_exist_other_pipe_barrier(%vec: !ascendc.local_tensor<*xf32>) {
  %c256 = arith.constant 256 : i32
  ascendc.add_l2 %vec, %vec, %vec, %c256 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, i32
  ascendc.pipe_barrier pipe_mte2
  ascendc.mul_l2 %vec, %vec, %vec, %c256 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, i32
  return
}

// CHECK-LABEL: func.func @pipe_barrier_at_beginning_dont_belong_group(%arg0: !ascendc.local_tensor<*xf32>) {
// CHECK-NEXT:   %c256_i32 = arith.constant 256 : i32
// CHECK-NEXT:   ascendc.pipe_barrier pipe_v
// CHECK-NEXT:   emitasc.vf_group %arg0, %arg0 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32> {
// CHECK-NEXT:     %0 = ascendc.local_tensor.get_phy_addr_v2 %arg0 : !ascendc.local_tensor<*xf32>, memref<f32, 26>
// CHECK-NEXT:     emitasc.vec_scope {
// CHECK-NEXT:       %1 = ascendc.reg_tensor : <f32>
// CHECK-NEXT:       %2 = emitasc.variable %c256_i32 : i32, memref<1xui32>
// CHECK-NEXT:       %3 = ascendc.get_vec_len : index
// CHECK-NEXT:       %c4 = arith.constant 4 : index
// CHECK-NEXT:       %4 = arith.divsi %3, %c4 : index
// CHECK-NEXT:       %5 = arith.index_cast %c256_i32 : i32 to index
// CHECK-NEXT:       %6 = arith.ceildivsi %5, %4 : index
// CHECK-NEXT:       %c0 = arith.constant 0 : index
// CHECK-NEXT:       %c1 = arith.constant 1 : index
// CHECK-NEXT:       scf.for %arg1 = %c0 to %6 step %c1 {
// CHECK-NEXT:         %7 = ascendc.update_mask f32, %2 : memref<1xui32>
// CHECK-NEXT:         %8 = arith.muli %arg1, %4 : index
// CHECK-NEXT:         %9 = emitasc.ptr_offset %0[%8] : memref<f32, 26>, memref<f32, 26>
// CHECK-NEXT:         ascendc.data_copy_vld_micro %1, %9 : !ascendc.reg_tensor<f32>, memref<f32, 26>
// CHECK-NEXT:         ascendc.add_micro %1, %1, %1, %7 : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
// CHECK-NEXT:         ascendc.mul_micro %1, %1, %1, %7 : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
// CHECK-NEXT:         %10 = emitasc.ptr_offset %0[%8] : memref<f32, 26>, memref<f32, 26>
// CHECK-NEXT:         ascendc.data_copy_vst_micro %10, %1, %7 : memref<f32, 26>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
// CHECK-NEXT:       } {asc.vec_scope_loop}
// CHECK-NEXT:     }
// CHECK-NEXT:   } {operandSegmentSizes = array<i32: 1, 1>}
// CHECK-NEXT:   return
// CHECK-NEXT: }
func.func @pipe_barrier_at_beginning_dont_belong_group(%vec: !ascendc.local_tensor<*xf32>) {
  %c256 = arith.constant 256 : i32
  ascendc.pipe_barrier pipe_v
  ascendc.add_l2 %vec, %vec, %vec, %c256 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, i32
  ascendc.mul_l2 %vec, %vec, %vec, %c256 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, i32
  return
}

// CHECK-LABEL: func.func @pipe_barrier_at_ending_dont_belong_group(%arg0: !ascendc.local_tensor<*xf32>) {
// CHECK-NEXT:   %c256_i32 = arith.constant 256 : i32
// CHECK-NEXT:   emitasc.vf_group %arg0, %arg0 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32> {
// CHECK-NEXT:     %0 = ascendc.local_tensor.get_phy_addr_v2 %arg0 : !ascendc.local_tensor<*xf32>, memref<f32, 26>
// CHECK-NEXT:     emitasc.vec_scope {
// CHECK-NEXT:       %1 = ascendc.reg_tensor : <f32>
// CHECK-NEXT:       %2 = emitasc.variable %c256_i32 : i32, memref<1xui32>
// CHECK-NEXT:       %3 = ascendc.get_vec_len : index
// CHECK-NEXT:       %c4 = arith.constant 4 : index
// CHECK-NEXT:       %4 = arith.divsi %3, %c4 : index
// CHECK-NEXT:       %5 = arith.index_cast %c256_i32 : i32 to index
// CHECK-NEXT:       %6 = arith.ceildivsi %5, %4 : index
// CHECK-NEXT:       %c0 = arith.constant 0 : index
// CHECK-NEXT:       %c1 = arith.constant 1 : index
// CHECK-NEXT:       scf.for %arg1 = %c0 to %6 step %c1 {
// CHECK-NEXT:         %7 = ascendc.update_mask f32, %2 : memref<1xui32>
// CHECK-NEXT:         %8 = arith.muli %arg1, %4 : index
// CHECK-NEXT:         %9 = emitasc.ptr_offset %0[%8] : memref<f32, 26>, memref<f32, 26>
// CHECK-NEXT:         ascendc.data_copy_vld_micro %1, %9 : !ascendc.reg_tensor<f32>, memref<f32, 26>
// CHECK-NEXT:         ascendc.add_micro %1, %1, %1, %7 : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
// CHECK-NEXT:         ascendc.mul_micro %1, %1, %1, %7 : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
// CHECK-NEXT:         %10 = emitasc.ptr_offset %0[%8] : memref<f32, 26>, memref<f32, 26>
// CHECK-NEXT:         ascendc.data_copy_vst_micro %10, %1, %7 : memref<f32, 26>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
// CHECK-NEXT:       } {asc.vec_scope_loop}
// CHECK-NEXT:     }
// CHECK-NEXT:   } {operandSegmentSizes = array<i32: 1, 1>}
// CHECK-NEXT:   ascendc.pipe_barrier pipe_v
// CHECK-NEXT:   return
// CHECK-NEXT: }
func.func @pipe_barrier_at_ending_dont_belong_group(%vec: !ascendc.local_tensor<*xf32>) {
  %c256 = arith.constant 256 : i32
  ascendc.add_l2 %vec, %vec, %vec, %c256 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, i32
  ascendc.mul_l2 %vec, %vec, %vec, %c256 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, i32
  ascendc.pipe_barrier pipe_v
  return
}

// CHECK-LABEL: func.func @if_group_contains_only_pipe_barrier_then_nothing() {
// CHECK-NEXT:   ascendc.pipe_barrier pipe_v
// CHECK-NEXT:   ascendc.pipe_barrier pipe_all
// CHECK-NEXT:   return
// CHECK-NEXT: }
func.func @if_group_contains_only_pipe_barrier_then_nothing() {
  ascendc.pipe_barrier pipe_v
  ascendc.pipe_barrier pipe_all
  return
}

// CHECK-LABEL: func.func @create_data_copy_load_only_for_input_tensors(%arg0: !ascendc.local_tensor<*xf32>, %arg1: !ascendc.local_tensor<*xf32>) {
// CHECK-NEXT:   %c256_i32 = arith.constant 256 : i32
// CHECK-NEXT:   emitasc.vf_group %arg0, %arg1 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32> {
// CHECK-NEXT:     %0 = ascendc.local_tensor.get_phy_addr_v2 %arg0 : !ascendc.local_tensor<*xf32>, memref<f32, 26>
// CHECK-NEXT:     %1 = ascendc.local_tensor.get_phy_addr_v2 %arg1 : !ascendc.local_tensor<*xf32>, memref<f32, 26>
// CHECK-NEXT:     emitasc.vec_scope {
// CHECK-NEXT:       %2 = ascendc.reg_tensor : <f32>
// CHECK-NEXT:       %3 = ascendc.reg_tensor : <f32>
// CHECK-NEXT:       %4 = emitasc.variable %c256_i32 : i32, memref<1xui32>
// CHECK-NEXT:       %5 = ascendc.get_vec_len : index
// CHECK-NEXT:       %c4 = arith.constant 4 : index
// CHECK-NEXT:       %6 = arith.divsi %5, %c4 : index
// CHECK-NEXT:       %7 = arith.index_cast %c256_i32 : i32 to index
// CHECK-NEXT:       %8 = arith.ceildivsi %7, %6 : index
// CHECK-NEXT:       %c0 = arith.constant 0 : index
// CHECK-NEXT:       %c1 = arith.constant 1 : index
// CHECK-NEXT:       scf.for %arg2 = %c0 to %8 step %c1 {
// CHECK-NEXT:         %9 = ascendc.update_mask f32, %4 : memref<1xui32>
// CHECK-NEXT:         %10 = arith.muli %arg2, %6 : index
// CHECK-NEXT:         %11 = emitasc.ptr_offset %1[%10] : memref<f32, 26>, memref<f32, 26>
// CHECK-NEXT:         ascendc.data_copy_vld_micro %3, %11 : !ascendc.reg_tensor<f32>, memref<f32, 26>
// CHECK-NEXT:         ascendc.add_micro %2, %3, %3, %9 : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
// CHECK-NEXT:         ascendc.mul_micro %2, %2, %3, %9 : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
// CHECK-NEXT:         %12 = emitasc.ptr_offset %0[%10] : memref<f32, 26>, memref<f32, 26>
// CHECK-NEXT:         ascendc.data_copy_vst_micro %12, %2, %9 : memref<f32, 26>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
// CHECK-NEXT:       } {asc.vec_scope_loop}
// CHECK-NEXT:     }
// CHECK-NEXT:   } {operandSegmentSizes = array<i32: 1, 1>}
// CHECK-NEXT:   return
// CHECK-NEXT: }
func.func @create_data_copy_load_only_for_input_tensors(%dst : !ascendc.local_tensor<*xf32>, %src : !ascendc.local_tensor<*xf32>) {
  %c256 = arith.constant 256 : i32
  ascendc.add_l2 %dst, %src, %src, %c256 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, i32
  ascendc.mul_l2 %dst, %dst, %src, %c256 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, i32
  return
}

// CHECK-LABEL: func.func @create_data_copy_store_only_for_output_tensors(%arg0: i32, %arg1: !ascendc.local_tensor<*xf32>, %arg2: !ascendc.local_tensor<*xf32>, %arg3: !ascendc.local_tensor<*xf32>) {
// CHECK-NEXT:   emitasc.vf_group %arg1, %arg2, %arg3 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32> {
// CHECK-NEXT:     %0 = ascendc.local_tensor.get_phy_addr_v2 %arg1 : !ascendc.local_tensor<*xf32>, memref<f32, 26>
// CHECK-NEXT:     %1 = ascendc.local_tensor.get_phy_addr_v2 %arg3 : !ascendc.local_tensor<*xf32>, memref<f32, 26>
// CHECK-NEXT:     %2 = ascendc.local_tensor.get_phy_addr_v2 %arg2 : !ascendc.local_tensor<*xf32>, memref<f32, 26>
// CHECK-NEXT:     emitasc.vec_scope {
// CHECK-NEXT:       %3 = ascendc.reg_tensor : <f32>
// CHECK-NEXT:       %4 = ascendc.reg_tensor : <f32>
// CHECK-NEXT:       %5 = ascendc.reg_tensor : <f32>
// CHECK-NEXT:       %6 = emitasc.variable %arg0 : i32, memref<1xui32>
// CHECK-NEXT:       %7 = ascendc.get_vec_len : index
// CHECK-NEXT:       %c4 = arith.constant 4 : index
// CHECK-NEXT:       %8 = arith.divsi %7, %c4 : index
// CHECK-NEXT:       %9 = arith.index_cast %arg0 : i32 to index
// CHECK-NEXT:       %10 = arith.ceildivsi %9, %8 : index
// CHECK-NEXT:       %c0 = arith.constant 0 : index
// CHECK-NEXT:       %c1 = arith.constant 1 : index
// CHECK-NEXT:       scf.for %arg4 = %c0 to %10 step %c1 {
// CHECK-NEXT:         %11 = ascendc.update_mask f32, %6 : memref<1xui32>
// CHECK-NEXT:         %12 = arith.muli %arg4, %8 : index
// CHECK-NEXT:         %13 = emitasc.ptr_offset %1[%12] : memref<f32, 26>, memref<f32, 26>
// CHECK-NEXT:         ascendc.data_copy_vld_micro %4, %13 : !ascendc.reg_tensor<f32>, memref<f32, 26>
// CHECK-NEXT:         ascendc.add_micro %3, %4, %4, %11 : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
// CHECK-NEXT:         ascendc.mul_micro %5, %3, %4, %11 : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
// CHECK-NEXT:         %14 = emitasc.ptr_offset %2[%12] : memref<f32, 26>, memref<f32, 26>
// CHECK-NEXT:         ascendc.data_copy_vst_micro %14, %5, %11 : memref<f32, 26>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
// CHECK-NEXT:       } {asc.vec_scope_loop}
// CHECK-NEXT:     }
// CHECK-NEXT:   } {operandSegmentSizes = array<i32: 2, 1>}
// CHECK-NEXT:   return
// CHECK-NEXT: }
func.func @create_data_copy_store_only_for_output_tensors(%c0_i32 : i32, %dst0 : !ascendc.local_tensor<*xf32>, %dst1 : !ascendc.local_tensor<*xf32>, %src : !ascendc.local_tensor<*xf32>) {
  ascendc.add_l2 %dst0, %src, %src, %c0_i32 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, i32
  ascendc.mul_l2 %dst1, %dst0, %src, %c0_i32 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, i32
  return
}

// CHECK-LABEL: func.func @create_nested_vec_scope(%arg0: i32, %arg1: !ascendc.local_tensor<*xf32>, %arg2: !ascendc.local_tensor<*xf32>, %arg3: !ascendc.local_tensor<*xf32>) {
// CHECK-NEXT:   %c1 = arith.constant 1 : index
// CHECK-NEXT:   scf.for %arg4 = %c1 to %c1 step %c1 {
// CHECK-NEXT:     emitasc.vf_group %arg1, %arg2, %arg3 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32> {
// CHECK-NEXT:       %0 = ascendc.local_tensor.get_phy_addr_v2 %arg1 : !ascendc.local_tensor<*xf32>, memref<f32, 26>
// CHECK-NEXT:       %1 = ascendc.local_tensor.get_phy_addr_v2 %arg3 : !ascendc.local_tensor<*xf32>, memref<f32, 26>
// CHECK-NEXT:       %2 = ascendc.local_tensor.get_phy_addr_v2 %arg2 : !ascendc.local_tensor<*xf32>, memref<f32, 26>
// CHECK-NEXT:       emitasc.vec_scope {
// CHECK-NEXT:         %3 = ascendc.reg_tensor : <f32>
// CHECK-NEXT:         %4 = ascendc.reg_tensor : <f32>
// CHECK-NEXT:         %5 = ascendc.reg_tensor : <f32>
// CHECK-NEXT:         %6 = emitasc.variable %arg0 : i32, memref<1xui32>
// CHECK-NEXT:         %7 = ascendc.get_vec_len : index
// CHECK-NEXT:         %c4 = arith.constant 4 : index
// CHECK-NEXT:         %8 = arith.divsi %7, %c4 : index
// CHECK-NEXT:         %9 = arith.index_cast %arg0 : i32 to index
// CHECK-NEXT:         %10 = arith.ceildivsi %9, %8 : index
// CHECK-NEXT:         %c0 = arith.constant 0 : index
// CHECK-NEXT:         %c1_0 = arith.constant 1 : index
// CHECK-NEXT:         scf.for %arg5 = %c0 to %10 step %c1_0 {
// CHECK-NEXT:           %11 = ascendc.update_mask f32, %6 : memref<1xui32>
// CHECK-NEXT:           %12 = arith.muli %arg5, %8 : index
// CHECK-NEXT:           %13 = emitasc.ptr_offset %1[%12] : memref<f32, 26>, memref<f32, 26>
// CHECK-NEXT:           ascendc.data_copy_vld_micro %4, %13 : !ascendc.reg_tensor<f32>, memref<f32, 26>
// CHECK-NEXT:           ascendc.add_micro %3, %4, %4, %11 : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
// CHECK-NEXT:           ascendc.mul_micro %5, %3, %4, %11 : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
// CHECK-NEXT:           %14 = emitasc.ptr_offset %2[%12] : memref<f32, 26>, memref<f32, 26>
// CHECK-NEXT:           ascendc.data_copy_vst_micro %14, %5, %11 : memref<f32, 26>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
// CHECK-NEXT:         } {asc.vec_scope_loop}
// CHECK-NEXT:       }
// CHECK-NEXT:     } {operandSegmentSizes = array<i32: 2, 1>}
// CHECK-NEXT:     %true = arith.constant true
// CHECK-NEXT:     scf.if %true {
// CHECK-NEXT:       emitasc.vf_group %arg1, %arg2, %arg3 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32> {
// CHECK-NEXT:         %0 = ascendc.local_tensor.get_phy_addr_v2 %arg1 : !ascendc.local_tensor<*xf32>, memref<f32, 26>
// CHECK-NEXT:         %1 = ascendc.local_tensor.get_phy_addr_v2 %arg3 : !ascendc.local_tensor<*xf32>, memref<f32, 26>
// CHECK-NEXT:         %2 = ascendc.local_tensor.get_phy_addr_v2 %arg2 : !ascendc.local_tensor<*xf32>, memref<f32, 26>
// CHECK-NEXT:         emitasc.vec_scope {
// CHECK-NEXT:           %3 = ascendc.reg_tensor : <f32>
// CHECK-NEXT:           %4 = ascendc.reg_tensor : <f32>
// CHECK-NEXT:           %5 = ascendc.reg_tensor : <f32>
// CHECK-NEXT:           %6 = emitasc.variable %arg0 : i32, memref<1xui32>
// CHECK-NEXT:           %7 = ascendc.get_vec_len : index
// CHECK-NEXT:           %c4 = arith.constant 4 : index
// CHECK-NEXT:           %8 = arith.divsi %7, %c4 : index
// CHECK-NEXT:           %9 = arith.index_cast %arg0 : i32 to index
// CHECK-NEXT:           %10 = arith.ceildivsi %9, %8 : index
// CHECK-NEXT:           %c0 = arith.constant 0 : index
// CHECK-NEXT:           %c1_0 = arith.constant 1 : index
// CHECK-NEXT:           scf.for %arg5 = %c0 to %10 step %c1_0 {
// CHECK-NEXT:             %11 = ascendc.update_mask f32, %6 : memref<1xui32>
// CHECK-NEXT:             %12 = arith.muli %arg5, %8 : index
// CHECK-NEXT:             %13 = emitasc.ptr_offset %1[%12] : memref<f32, 26>, memref<f32, 26>
// CHECK-NEXT:             ascendc.data_copy_vld_micro %4, %13 : !ascendc.reg_tensor<f32>, memref<f32, 26>
// CHECK-NEXT:             ascendc.add_micro %3, %4, %4, %11 : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
// CHECK-NEXT:             ascendc.mul_micro %5, %3, %4, %11 : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
// CHECK-NEXT:             %14 = emitasc.ptr_offset %2[%12] : memref<f32, 26>, memref<f32, 26>
// CHECK-NEXT:             ascendc.data_copy_vst_micro %14, %5, %11 : memref<f32, 26>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
// CHECK-NEXT:           } {asc.vec_scope_loop}
// CHECK-NEXT:         }
// CHECK-NEXT:       } {operandSegmentSizes = array<i32: 2, 1>}
// CHECK-NEXT:     } else {
// CHECK-NEXT:       emitasc.vf_group %arg1, %arg2, %arg3 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32> {
// CHECK-NEXT:         %0 = ascendc.local_tensor.get_phy_addr_v2 %arg1 : !ascendc.local_tensor<*xf32>, memref<f32, 26>
// CHECK-NEXT:         %1 = ascendc.local_tensor.get_phy_addr_v2 %arg3 : !ascendc.local_tensor<*xf32>, memref<f32, 26>
// CHECK-NEXT:         %2 = ascendc.local_tensor.get_phy_addr_v2 %arg2 : !ascendc.local_tensor<*xf32>, memref<f32, 26>
// CHECK-NEXT:         emitasc.vec_scope {
// CHECK-NEXT:           %3 = ascendc.reg_tensor : <f32>
// CHECK-NEXT:           %4 = ascendc.reg_tensor : <f32>
// CHECK-NEXT:           %5 = ascendc.reg_tensor : <f32>
// CHECK-NEXT:           %6 = emitasc.variable %arg0 : i32, memref<1xui32>
// CHECK-NEXT:           %7 = ascendc.get_vec_len : index
// CHECK-NEXT:           %c4 = arith.constant 4 : index
// CHECK-NEXT:           %8 = arith.divsi %7, %c4 : index
// CHECK-NEXT:           %9 = arith.index_cast %arg0 : i32 to index
// CHECK-NEXT:           %10 = arith.ceildivsi %9, %8 : index
// CHECK-NEXT:           %c0 = arith.constant 0 : index
// CHECK-NEXT:           %c1_0 = arith.constant 1 : index
// CHECK-NEXT:           scf.for %arg5 = %c0 to %10 step %c1_0 {
// CHECK-NEXT:             %11 = ascendc.update_mask f32, %6 : memref<1xui32>
// CHECK-NEXT:             %12 = arith.muli %arg5, %8 : index
// CHECK-NEXT:             %13 = emitasc.ptr_offset %1[%12] : memref<f32, 26>, memref<f32, 26>
// CHECK-NEXT:             ascendc.data_copy_vld_micro %4, %13 : !ascendc.reg_tensor<f32>, memref<f32, 26>
// CHECK-NEXT:             ascendc.add_micro %3, %4, %4, %11 : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
// CHECK-NEXT:             ascendc.mul_micro %5, %3, %4, %11 : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
// CHECK-NEXT:             %14 = emitasc.ptr_offset %2[%12] : memref<f32, 26>, memref<f32, 26>
// CHECK-NEXT:             ascendc.data_copy_vst_micro %14, %5, %11 : memref<f32, 26>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
// CHECK-NEXT:           } {asc.vec_scope_loop}
// CHECK-NEXT:         }
// CHECK-NEXT:       } {operandSegmentSizes = array<i32: 2, 1>}
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   return
// CHECK-NEXT: }
func.func @create_nested_vec_scope(%c0_i32 : i32, %dst0 : !ascendc.local_tensor<*xf32>, %dst1 : !ascendc.local_tensor<*xf32>, %src : !ascendc.local_tensor<*xf32>) {
  %c1_idx = arith.constant 1 : index
  scf.for %arg0 = %c1_idx to %c1_idx step %c1_idx {
    ascendc.add_l2 %dst0, %src, %src, %c0_i32 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, i32
    ascendc.mul_l2 %dst1, %dst0, %src, %c0_i32 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, i32
    %true = arith.constant true
    scf.if %true {
      ascendc.add_l2 %dst0, %src, %src, %c0_i32 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, i32
      ascendc.mul_l2 %dst1, %dst0, %src, %c0_i32 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, i32
    }
    else {
      ascendc.add_l2 %dst0, %src, %src, %c0_i32 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, i32
      ascendc.mul_l2 %dst1, %dst0, %src, %c0_i32 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, i32
    }
  }
  return
}

// CHECK-LABEL: func.func @translate_unary_l2(%arg0: i32, %arg1: !ascendc.local_tensor<*xf32>) {
// CHECK-NEXT:   emitasc.vf_group %arg1, %arg1 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32> {
// CHECK-NEXT:     %0 = ascendc.local_tensor.get_phy_addr_v2 %arg1 : !ascendc.local_tensor<*xf32>, memref<f32, 26>
// CHECK-NEXT:     emitasc.vec_scope {
// CHECK-NEXT:       %1 = ascendc.reg_tensor : <f32>
// CHECK-NEXT:       %2 = emitasc.variable %arg0 : i32, memref<1xui32>
// CHECK-NEXT:       %3 = ascendc.get_vec_len : index
// CHECK-NEXT:       %c4 = arith.constant 4 : index
// CHECK-NEXT:       %4 = arith.divsi %3, %c4 : index
// CHECK-NEXT:       %5 = arith.index_cast %arg0 : i32 to index
// CHECK-NEXT:       %6 = arith.ceildivsi %5, %4 : index
// CHECK-NEXT:       %c0 = arith.constant 0 : index
// CHECK-NEXT:       %c1 = arith.constant 1 : index
// CHECK-NEXT:       scf.for %arg2 = %c0 to %6 step %c1 {
// CHECK-NEXT:         %7 = ascendc.update_mask f32, %2 : memref<1xui32>
// CHECK-NEXT:         %8 = arith.muli %arg2, %4 : index
// CHECK-NEXT:         %9 = emitasc.ptr_offset %0[%8] : memref<f32, 26>, memref<f32, 26>
// CHECK-NEXT:         ascendc.data_copy_vld_micro %1, %9 : !ascendc.reg_tensor<f32>, memref<f32, 26>
// CHECK-NEXT:         ascendc.abs_micro %1, %1, %7 : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
// CHECK-NEXT:         ascendc.exp_micro %1, %1, %7 : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
// CHECK-NEXT:         ascendc.ln_micro %1, %1, %7 : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
// CHECK-NEXT:         ascendc.neg_micro %1, %1, %7 : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
// CHECK-NEXT:         ascendc.not_micro %1, %1, %7 : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
// CHECK-NEXT:         ascendc.relu_micro %1, %1, %7 : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
// CHECK-NEXT:         %10 = emitasc.ptr_offset %0[%8] : memref<f32, 26>, memref<f32, 26>
// CHECK-NEXT:         ascendc.data_copy_vst_micro %10, %1, %7 : memref<f32, 26>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
// CHECK-NEXT:       } {asc.vec_scope_loop}
// CHECK-NEXT:     }
// CHECK-NEXT:   } {operandSegmentSizes = array<i32: 1, 1>}
// CHECK-NEXT:   return
// CHECK-NEXT: }
func.func @translate_unary_l2(%calCount : i32, %vec : !ascendc.local_tensor<*xf32>) {
  ascendc.abs_l2 %vec, %vec, %calCount : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, i32
  ascendc.exp_l2 %vec, %vec, %calCount : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, i32
  ascendc.ln_l2 %vec, %vec, %calCount : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, i32
  ascendc.neg_l2 %vec, %vec, %calCount : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, i32
  ascendc.not_l2 %vec, %vec, %calCount : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, i32
  ascendc.relu_l2 %vec, %vec, %calCount : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, i32
  return
}

// CHECK-LABEL: func.func @translate_binary_l2(%arg0: i32, %arg1: !ascendc.local_tensor<*xf32>) {
// CHECK-NEXT:   emitasc.vf_group %arg1, %arg1 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32> {
// CHECK-NEXT:     %0 = ascendc.local_tensor.get_phy_addr_v2 %arg1 : !ascendc.local_tensor<*xf32>, memref<f32, 26>
// CHECK-NEXT:     emitasc.vec_scope {
// CHECK-NEXT:       %1 = ascendc.reg_tensor : <f32>
// CHECK-NEXT:       %2 = emitasc.variable %arg0 : i32, memref<1xui32>
// CHECK-NEXT:       %3 = ascendc.get_vec_len : index
// CHECK-NEXT:       %c4 = arith.constant 4 : index
// CHECK-NEXT:       %4 = arith.divsi %3, %c4 : index
// CHECK-NEXT:       %5 = arith.index_cast %arg0 : i32 to index
// CHECK-NEXT:       %6 = arith.ceildivsi %5, %4 : index
// CHECK-NEXT:       %c0 = arith.constant 0 : index
// CHECK-NEXT:       %c1 = arith.constant 1 : index
// CHECK-NEXT:       scf.for %arg2 = %c0 to %6 step %c1 {
// CHECK-NEXT:         %7 = ascendc.update_mask f32, %2 : memref<1xui32>
// CHECK-NEXT:         %8 = arith.muli %arg2, %4 : index
// CHECK-NEXT:         %9 = emitasc.ptr_offset %0[%8] : memref<f32, 26>, memref<f32, 26>
// CHECK-NEXT:         ascendc.data_copy_vld_micro %1, %9 : !ascendc.reg_tensor<f32>, memref<f32, 26>
// CHECK-NEXT:         ascendc.add_micro %1, %1, %1, %7 : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
// CHECK-NEXT:         ascendc.and_micro %1, %1, %1, %7 : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
// CHECK-NEXT:         ascendc.div_micro %1, %1, %1, %7 : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
// CHECK-NEXT:         ascendc.sub_micro %1, %1, %1, %7 : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
// CHECK-NEXT:         ascendc.max_micro %1, %1, %1, %7 : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
// CHECK-NEXT:         ascendc.min_micro %1, %1, %1, %7 : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
// CHECK-NEXT:         ascendc.mul_micro %1, %1, %1, %7 : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
// CHECK-NEXT:         ascendc.mul_add_dst_micro %1, %1, %1, %7 : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
// CHECK-NEXT:         ascendc.or_micro %1, %1, %1, %7 : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
// CHECK-NEXT:         ascendc.prelu_micro %1, %1, %1, %7 : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
// CHECK-NEXT:         %10 = emitasc.ptr_offset %0[%8] : memref<f32, 26>, memref<f32, 26>
// CHECK-NEXT:         ascendc.data_copy_vst_micro %10, %1, %7 : memref<f32, 26>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
// CHECK-NEXT:       } {asc.vec_scope_loop}
// CHECK-NEXT:     }
// CHECK-NEXT:   } {operandSegmentSizes = array<i32: 1, 1>}
// CHECK-NEXT:   return
// CHECK-NEXT: }
func.func @translate_binary_l2(%calCount : i32, %vec : !ascendc.local_tensor<*xf32>) {
  ascendc.add_l2 %vec, %vec, %vec, %calCount : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, i32
  ascendc.and_l2 %vec, %vec, %vec, %calCount : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, i32
  ascendc.div_l2 %vec, %vec, %vec, %calCount : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, i32
  ascendc.sub_l2 %vec, %vec, %vec, %calCount : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, i32
  ascendc.max_l2 %vec, %vec, %vec, %calCount : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, i32
  ascendc.min_l2 %vec, %vec, %vec, %calCount : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, i32
  ascendc.mul_l2 %vec, %vec, %vec, %calCount : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, i32
  ascendc.mul_add_dst_l2 %vec, %vec, %vec, %calCount : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, i32
  ascendc.or_l2 %vec, %vec, %vec, %calCount : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, i32
  ascendc.prelu_l2 %vec, %vec, %vec, %calCount : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, i32
  return
}
