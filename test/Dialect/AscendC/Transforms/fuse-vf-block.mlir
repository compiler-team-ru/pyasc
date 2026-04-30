// RUN: ascir-opt -ascendc-fuse-vf-block %s | FileCheck %s

// CHECK-LABEL: func.func @general_test(%arg0: !ascendc.que_bind<gm, vecin, 1>) {
// CHECK-NEXT:   %c256_i32 = arith.constant 256 : i32
// CHECK-NEXT:   %0 = ascendc.que_bind.alloc_tensor %arg0 : !ascendc.que_bind<gm, vecin, 1>, !ascendc.local_tensor<*xf32>
// CHECK-NEXT:   %1 = ascendc.que_bind.deque_tensor %arg0 : !ascendc.que_bind<gm, vecin, 1>, !ascendc.local_tensor<*xf32>
// CHECK-NEXT:   %2 = ascendc.que_bind.deque_tensor %arg0 : !ascendc.que_bind<gm, vecin, 1>, !ascendc.local_tensor<*xf32>
// CHECK-NEXT:   emitasc.vf_group %0, %2, %1, %c256_i32 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, i32 {
// CHECK-NEXT:     %3 = ascendc.local_tensor.get_phy_addr_v2 %1 : !ascendc.local_tensor<*xf32>, memref<f32, 26>
// CHECK-NEXT:     %4 = ascendc.local_tensor.get_phy_addr_v2 %2 : !ascendc.local_tensor<*xf32>, memref<f32, 26>
// CHECK-NEXT:     %5 = ascendc.local_tensor.get_phy_addr_v2 %0 : !ascendc.local_tensor<*xf32>, memref<f32, 26>
// CHECK-NEXT:     emitasc.vec_scope {
// CHECK-NEXT:       %6 = ascendc.get_vec_len : index
// CHECK-NEXT:       %c4 = arith.constant 4 : index
// CHECK-NEXT:       %7 = arith.divsi %6, %c4 : index
// CHECK-NEXT:       %8 = arith.index_cast %c256_i32 : i32 to index
// CHECK-NEXT:       %9 = arith.ceildivsi %8, %7 : index
// CHECK:            %10 = emitasc.variable %c256_i32 : i32, memref<1xui32>
// CHECK:            scf.for %arg1 = %c0 to %9 step %c1 {
// CHECK-NEXT:         %18 = arith.muli %arg1, %7 : index
// CHECK-NEXT:         %19 = ascendc.update_mask f32, %10 : memref<1xui32>
// CHECK-NEXT:         %20 = emitasc.ptr_offset %3[%18] : memref<f32, 26>, memref<f32, 26>
// CHECK-NEXT:         ascendc.data_copy_vld_micro %12, %20 : !ascendc.reg_tensor<f32>, memref<f32, 26>
// CHECK-NEXT:         %21 = emitasc.ptr_offset %4[%18] : memref<f32, 26>, memref<f32, 26>
// CHECK-NEXT:         ascendc.data_copy_vld_micro %13, %21 : !ascendc.reg_tensor<f32>, memref<f32, 26>
// CHECK-NEXT:         ascendc.add_micro %14, %12, %13, %19 : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
// CHECK-NEXT:         ascendc.mul_micro %17, %14, %13, %19 : !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
// CHECK-NEXT:         %22 = emitasc.ptr_offset %5[%18] : memref<f32, 26>, memref<f32, 26>
// CHECK-NEXT:         ascendc.data_copy_vst_micro %22, %17, %19 : memref<f32, 26>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
// CHECK-NEXT:       } {asc.vec_scope_loop}
// CHECK-NEXT:     }
// CHECK-NEXT:   } {operandSegmentSizes = array<i32: 1, 2, 1>}
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
  ascendc.mul_l2 %dst, %dst, %src1, %c256_i32 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, i32
  ascendc.que_bind.enque_tensor %que_bind, %dst : !ascendc.que_bind<gm, vecin, 1>, !ascendc.local_tensor<*xf32>
  ascendc.que_bind.free_tensor %que_bind, %src0 : !ascendc.que_bind<gm, vecin, 1>, !ascendc.local_tensor<*xf32>
  ascendc.que_bind.free_tensor %que_bind, %src1 : !ascendc.que_bind<gm, vecin, 1>, !ascendc.local_tensor<*xf32>
  return
}

// CHECK-LABEL: func.func @not_convert_if_one_operation
// CHECK-NOT: emitasc.vf_group
func.func @not_convert_if_one_operation(%vec: !ascendc.local_tensor<*xf32>) {
  %c256 = arith.constant 256 : i32
  ascendc.add_l2 %vec, %vec, %vec, %c256 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, i32
  return
}

// CHECK-LABEL: func.func @same_group_if_cal_count_and_types_equals
// CHECK: emitasc.vf_group
// CHECK-NOT: emitasc.vf_group
func.func @same_group_if_cal_count_and_types_equals(%vec: !ascendc.local_tensor<*xf32>) {
  %c256_0 = arith.constant 256 : i32
  %c256_1 = arith.constant 256 : i32
  ascendc.add_l2 %vec, %vec, %vec, %c256_0 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, i32
  ascendc.mul_l2 %vec, %vec, %vec, %c256_1 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, i32
  return
}

// CHECK-LABEL: func.func @different_group_if_cal_count_not_equals
// CHECK: emitasc.vf_group
// CHECK: emitasc.vf_group
// CHECK-NOT: emitasc.vf_group
func.func @different_group_if_cal_count_not_equals(%vec: !ascendc.local_tensor<*xf32>) {
  %c256 = arith.constant 256 : i32
  %c128 = arith.constant 128 : i32
  ascendc.add_l2 %vec, %vec, %vec, %c256 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, i32
  ascendc.mul_l2 %vec, %vec, %vec, %c256 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, i32
  ascendc.add_l2 %vec, %vec, %vec, %c128 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, i32
  ascendc.mul_l2 %vec, %vec, %vec, %c128 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, i32
  return
}

// CHECK-LABEL: func.func @different_group_if_types_not_equals
// CHECK-NOT: emitasc.vf_group
func.func @different_group_if_types_not_equals(%vecf32: !ascendc.local_tensor<*xf32>, %vecf16: !ascendc.local_tensor<*xf16>) {
  %c256 = arith.constant 256 : i32
  ascendc.add_l2 %vecf32, %vecf32, %vecf32, %c256 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, i32
  ascendc.mul_l2 %vecf16, %vecf16, %vecf16, %c256 : !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, i32
  return
}

// CHECK-LABEL: func.func @different_group_if_between_op_exist_other_op
// CHECK: emitasc.vf_group
// CHECK: emitasc.vf_group
// CHECK-NOT: emitasc.vf_group
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

// CHECK-LABEL: func.func @create_nested_vec_scope
func.func @create_nested_vec_scope(%c0_i32 : i32, %dst0 : !ascendc.local_tensor<*xf32>, %dst1 : !ascendc.local_tensor<*xf32>, %src : !ascendc.local_tensor<*xf32>) {
  %c1_idx = arith.constant 1 : index
// CHECK: scf.for
// CHECK: emitasc.vf_group
// CHECK-NOT: emitasc.vf_group
  scf.for %arg0 = %c1_idx to %c1_idx step %c1_idx {
    ascendc.add_l2 %dst0, %src, %src, %c0_i32 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, i32
    ascendc.mul_l2 %dst1, %dst0, %src, %c0_i32 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, i32
    %true = arith.constant true
// CHECK: scf.if
// CHECK: emitasc.vf_group
// CHECK-NOT: emitasc.vf_group
    scf.if %true {
      ascendc.add_l2 %dst0, %src, %src, %c0_i32 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, i32
      ascendc.mul_l2 %dst1, %dst0, %src, %c0_i32 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, i32
// CHECK: } else {
// CHECK: emitasc.vf_group
// CHECK-NOT: emitasc.vf_group
    } else {
      ascendc.add_l2 %dst0, %src, %src, %c0_i32 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, i32
      ascendc.mul_l2 %dst1, %dst0, %src, %c0_i32 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, i32
    }
  }
  return
}

// CHECK-LABEL: func.func @translate_unary_l2
// CHECK: ascendc.abs_micro {{[^:]*}}: !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
// CHECK-NEXT: ascendc.exp_micro {{[^:]*}}: !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
// CHECK-NEXT: ascendc.ln_micro {{[^:]*}}: !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
// CHECK-NEXT: ascendc.neg_micro {{[^:]*}}: !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
// CHECK-NEXT: ascendc.not_micro {{[^:]*}}: !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
// CHECK-NEXT: ascendc.relu_micro {{[^:]*}}: !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
func.func @translate_unary_l2(%calCount : i32, %vec : !ascendc.local_tensor<*xf32>) {
  ascendc.abs_l2 %vec, %vec, %calCount : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, i32
  ascendc.exp_l2 %vec, %vec, %calCount : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, i32
  ascendc.ln_l2 %vec, %vec, %calCount : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, i32
  ascendc.neg_l2 %vec, %vec, %calCount : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, i32
  ascendc.not_l2 %vec, %vec, %calCount : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, i32
  ascendc.relu_l2 %vec, %vec, %calCount : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, i32
  return
}

// CHECK-LABEL: func.func @translate_binary_l2
// CHECK: ascendc.add_micro {{[^:]*}}: !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
// CHECK-NEXT: ascendc.and_micro {{[^:]*}}: !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
// CHECK-NEXT: ascendc.div_micro {{[^:]*}}: !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
// CHECK-NEXT: ascendc.sub_micro {{[^:]*}}: !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
// CHECK-NEXT: ascendc.max_micro {{[^:]*}}: !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
// CHECK-NEXT: ascendc.min_micro {{[^:]*}}: !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
// CHECK-NEXT: ascendc.mul_micro {{[^:]*}}: !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
// CHECK-NEXT: ascendc.mul_add_dst_micro {{[^:]*}}: !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
// CHECK-NEXT: ascendc.or_micro {{[^:]*}}: !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
// CHECK-NEXT: ascendc.prelu_micro {{[^:]*}}: !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
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

// CHECK-LABEL: func.func @tensor_is_not_output_if_it_has_no_users
func.func @tensor_is_not_output_if_it_has_no_users(%arg0: memref<*xf32, 22>) {
  %c0_i64 = arith.constant 0 : i64
  %c1024_i64 = arith.constant 1024 : i64
  %cst = arith.constant 0.000000e+00 : f32
  %c0_i32 = arith.constant 0 : i32
  %0 = ascendc.global_tensor : !ascendc.global_tensor<?x?xf32>
  ascendc.global_tensor.set_global_buffer %0, %arg0 : !ascendc.global_tensor<?x?xf32>, memref<*xf32, 22>
  %f = ascendc.local_tensor_auto veccalc() : <1x1024xf32>
  %z = ascendc.local_tensor_auto veccalc() : <1x1024xf32>
  %y = ascendc.local_tensor_auto veccalc() : <1x1024xf32>
  %x = ascendc.local_tensor_auto veccalc() : <1x1024xf32>
  %ext_params = ascendc.construct !ascendc.data_copy_ext_params(%c0_i32, %c0_i32, %c0_i32, %c0_i32, %c0_i32) [ui16, ui32, ui32, ui32, ui32] : i32, i32, i32, i32, i32

  // CHECK: emitasc.vf_group
  // CHECK: ascendc.data_copy_vst_micro {{[^:]*}}: memref<1x1024xf32, 26>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
  // CHECK: ascendc.data_copy_vst_micro {{[^:]*}}: memref<1x1024xf32, 26>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
  // CHECK-NOT: ascendc.data_copy_vst_micro
  ascendc.sub_l2 %z, %x, %y, %c1024_i64 : !ascendc.local_tensor<1x1024xf32>, !ascendc.local_tensor<1x1024xf32>, !ascendc.local_tensor<1x1024xf32>, i64
  ascendc.exp_l2 %f, %z, %c1024_i64 : !ascendc.local_tensor<1x1024xf32>, !ascendc.local_tensor<1x1024xf32>, i64
  ascendc.data_copy_pad_l2_ext %0, %f, %ext_params : !ascendc.global_tensor<?x?xf32>, !ascendc.local_tensor<1x1024xf32>, !ascendc.data_copy_ext_params

  // CHECK: emitasc.vf_group
  // CHECK-NOT: ascendc.data_copy_vst_micro
  ascendc.sub_l2 %z, %x, %y, %c1024_i64 : !ascendc.local_tensor<1x1024xf32>, !ascendc.local_tensor<1x1024xf32>, !ascendc.local_tensor<1x1024xf32>, i64
  ascendc.exp_l2 %f, %z, %c1024_i64 : !ascendc.local_tensor<1x1024xf32>, !ascendc.local_tensor<1x1024xf32>, i64

  return
}

// CHECK-LABEL: func.func @dont_create_load_for_loaded_reg_tensor
// CHECK: scf.for
// CHECK: ascendc.data_copy_vld_micro {{[^:]*}}: !ascendc.reg_tensor<f32>, memref<1x1024xf32, 26>
// CHECK: ascendc.data_copy_vld_micro {{[^:]*}}: !ascendc.reg_tensor<f32>, memref<1x1024xf32, 26>
// CHECK-NEXT: ascendc.add_micro {{[^:]*}}: !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
// CHECK-NEXT: ascendc.add_micro {{[^:]*}}: !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
// CHECK: ascendc.data_copy_vst_micro {{[^:]*}}: memref<1x1024xf32, 26>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
func.func @dont_create_load_for_loaded_reg_tensor(%gm: !ascendc.global_tensor<?x?xf32>, %ext_params: !ascendc.data_copy_ext_params, %x: !ascendc.local_tensor<1x1024xf32>, %y: !ascendc.local_tensor<1x1024xf32>, %z: !ascendc.local_tensor<1x1024xf32>, %f: !ascendc.local_tensor<1x1024xf32>, %cal: i64) {
  ascendc.add_l2 %z, %x, %y, %cal : !ascendc.local_tensor<1x1024xf32>, !ascendc.local_tensor<1x1024xf32>, !ascendc.local_tensor<1x1024xf32>, i64
  // %x already loaded
  ascendc.add_l2 %f, %z, %x, %cal : !ascendc.local_tensor<1x1024xf32>, !ascendc.local_tensor<1x1024xf32>, !ascendc.local_tensor<1x1024xf32>, i64
  ascendc.data_copy_pad_l2_ext %gm, %f, %ext_params : !ascendc.global_tensor<?x?xf32>, !ascendc.local_tensor<1x1024xf32>, !ascendc.data_copy_ext_params
  return
}

// func.func @dont_rewrite_memory
// CHECK: scf.for
// CHECK: ascendc.data_copy_vld_micro {{[^:]*}}: !ascendc.reg_tensor<f32>, memref<1x1024xf32, 26>
// CHECK-NEXT: ascendc.exp_micro {{[^:]*}}: !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
// CHECK-NEXT: ascendc.exp_micro {{[^:]*}}: !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
// CHECK-NEXT: ascendc.exp_micro {{[^:]*}}: !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
// CHECK: ascendc.data_copy_vst_micro {{[^:]*}}: memref<1x1024xf32, 26>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
func.func @dont_rewrite_memory(%gm: !ascendc.global_tensor<?x?xf32>, %ext_params: !ascendc.data_copy_ext_params, %x: !ascendc.local_tensor<1x1024xf32>, %y: !ascendc.local_tensor<1x1024xf32>, %z: !ascendc.local_tensor<1x1024xf32>, %f: !ascendc.local_tensor<1x1024xf32>, %cal: i64) {
  ascendc.exp_l2 %f, %f, %cal : !ascendc.local_tensor<1x1024xf32>, !ascendc.local_tensor<1x1024xf32>, i64
  ascendc.exp_l2 %f, %f, %cal : !ascendc.local_tensor<1x1024xf32>, !ascendc.local_tensor<1x1024xf32>, i64
  ascendc.exp_l2 %f, %f, %cal : !ascendc.local_tensor<1x1024xf32>, !ascendc.local_tensor<1x1024xf32>, i64
  ascendc.data_copy_pad_l2_ext %gm, %f, %ext_params : !ascendc.global_tensor<?x?xf32>, !ascendc.local_tensor<1x1024xf32>, !ascendc.data_copy_ext_params
  return
}

// CHECK-LABEL: func.func @softmax_kernel
// CHECK: emitasc.vf_group {{[^:]*}}: !ascendc.local_tensor<1x1024xf32>, !ascendc.local_tensor<1x1024xf32>, i64 {
// CHECK:   emitasc.vec_scope {
// CHECK:     ascendc.duplicate {{[^:]*}}: <f32>, f32
// CHECK:     ascendc.duplicate {{[^:]*}}: <f32>, f32
// CHECK:     scf.for
// CHECK:       ascendc.data_copy_vld_micro {{[^:]*}}: !ascendc.reg_tensor<f32>, memref<1x1024xf32, 26>
// CHECK:       ascendc.max_micro {{[^:]*}}: !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
// CHECK:     }
// CHECK:     ascendc.reduce_max_micro {{[^:]*}}: !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
// CHECK:     ascendc.duplicate_micro {{[^:]*}}: !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
// CHECK:     scf.for
// CHECK:       ascendc.data_copy_vld_micro {{[^:]*}}: !ascendc.reg_tensor<f32>, memref<1x1024xf32, 26>
// CHECK:       ascendc.sub_micro {{[^:]*}}: !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
// CHECK:       ascendc.exp_micro {{[^:]*}}: !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
// CHECK:       ascendc.data_copy_vst_micro {{[^:]*}}: memref<1x1024xf32, 26>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
// CHECK:       ascendc.add_micro {{[^:]*}}: !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
// CHECK:     }
// CHECK:     ascendc.reduce_sum_micro {{[^:]*}}: !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
// CHECK:     ascendc.duplicate_micro {{[^:]*}}: !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
// CHECK:     scf.for
// CHECK:       ascendc.data_copy_vld_micro {{[^:]*}}: !ascendc.reg_tensor<f32>, memref<1x1024xf32, 26>
// CHECK:       ascendc.div_micro {{[^:]*}}: !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
// CHECK:       ascendc.data_copy_vst_micro {{[^:]*}}: memref<1x1024xf32, 26>, !ascendc.reg_tensor<f32>, !ascendc.mask_reg
// CHECK:     }
// CHECK:   }
// CHECK: }
// CHECK: ascendc.data_copy_pad_l2_ext {{[^:]*}}: !ascendc.global_tensor<?x?xf32>, !ascendc.local_tensor<1x1024xf32>, !ascendc.data_copy_ext_params
func.func @softmax_kernel(%arg0: memref<*xf32, 22>) {
  %c0_i64 = arith.constant 0 : i64
  %c1024_i64 = arith.constant 1024 : i64
  %cst = arith.constant 0.000000e+00 : f32
  %c0_i32 = arith.constant 0 : i32
  %0 = ascendc.global_tensor : !ascendc.global_tensor<?x?xf32>
  ascendc.global_tensor.set_global_buffer %0, %arg0 : !ascendc.global_tensor<?x?xf32>, memref<*xf32, 22>
  %4 = ascendc.local_tensor_auto veccalc() output : <1x1024xf32>
  %5 = ascendc.local_tensor_auto veccalc() : <1x1024xf32>
  %6 = ascendc.local_tensor_auto veccalc() : <16xf32>
  %7 = ascendc.local_tensor_auto veccalc() : <1xf32>
  %f = ascendc.local_tensor_auto veccalc() : <1x1024xf32>
  %z = ascendc.local_tensor_auto veccalc() : <1x1024xf32>
  %y = ascendc.local_tensor_auto veccalc() : <1x1024xf32>
  %11 = ascendc.local_tensor_auto veccalc() : <16xf32>
  %12 = ascendc.local_tensor_auto veccalc() : <1xf32>
  %x = ascendc.local_tensor_auto veccalc() input : <1x1024xf32>

  %ext_params = ascendc.construct !ascendc.data_copy_ext_params(%c0_i32, %c0_i32, %c0_i32, %c0_i32, %c0_i32) [ui16, ui32, ui32, ui32, ui32] : i32, i32, i32, i32, i32
  ascendc.reduce_max_l2 %12, %x, %11, %c1024_i64, %c0_i64 : !ascendc.local_tensor<1xf32>, !ascendc.local_tensor<1x1024xf32>, !ascendc.local_tensor<16xf32>, i64, i64
  ascendc.duplicate_l2 %y, %12, %c1024_i64 : !ascendc.local_tensor<1x1024xf32>, !ascendc.local_tensor<1xf32>, i64
  ascendc.sub_l2 %z, %x, %y, %c1024_i64 : !ascendc.local_tensor<1x1024xf32>, !ascendc.local_tensor<1x1024xf32>, !ascendc.local_tensor<1x1024xf32>, i64
  ascendc.exp_l2 %f, %z, %c1024_i64 : !ascendc.local_tensor<1x1024xf32>, !ascendc.local_tensor<1x1024xf32>, i64
  ascendc.reduce_sum_l2 %7, %f, %6, %c1024_i64 : !ascendc.local_tensor<1xf32>, !ascendc.local_tensor<1x1024xf32>, !ascendc.local_tensor<16xf32>, i64
  ascendc.duplicate_l2 %5, %7, %c1024_i64 : !ascendc.local_tensor<1x1024xf32>, !ascendc.local_tensor<1xf32>, i64
  ascendc.div_l2 %4, %f, %5, %c1024_i64 : !ascendc.local_tensor<1x1024xf32>, !ascendc.local_tensor<1x1024xf32>, !ascendc.local_tensor<1x1024xf32>, i64
  ascendc.data_copy_pad_l2_ext %0, %4, %ext_params : !ascendc.global_tensor<?x?xf32>, !ascendc.local_tensor<1x1024xf32>, !ascendc.data_copy_ext_params
  return
}
