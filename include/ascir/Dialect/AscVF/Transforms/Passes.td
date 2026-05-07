/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASCVF_PASSES_TD
#define ASCVF_PASSES_TD

include "mlir/Pass/PassBase.td"

def FuseVFBlock : Pass<"ascvf-fuse-vf-block", "func::FuncOp"> {
  let summary = "Fuse consecutive L2 vector operations into VF blocks with micro-operations and register tensors";
  let constructor = "mlir::ascvf::createFuseVFBlockPass()";
  let dependentDialects = ["ascendc::AscendCDialect", "scf::SCFDialect", "emitasc::EmitAscDialect", "ascvf::AscVFDialect"];
  let description = [{
    Transforms sequences of L2 vector operations into optimized VF (Vector Fusion) blocks:
    1. Groups operations with matching `calCount` and element type, separated by non-vector ops or different types
    2. Wraps groups in `emitasc.vf_group` with input/output tensor lists
    3. Inside `vec_scope`, creates `reg_tensor` for each used local tensor
    4. Generates loop over repeat times with: `data_copy_vld_micro` (load inputs), micro-ops (compute), `data_copy_vst_micro` (store outputs)

    Supported ops: add/mul/sub/div/max/min/relu/abs/exp/ln/neg/not/sqrt L2 operations and `pipe_barrier pipe_v/pipe_all`.
    Groups must have at least 2 operations; single ops are not fused.

    Example transformation:
    ```mlir
    // Before: separate L2 operations
    %dst = ascendc.que_bind.alloc_tensor %queue
    %src0 = ascendc.que_bind.deque_tensor %queue
    %src1 = ascendc.que_bind.deque_tensor %queue
    ascendc.add_l2 %dst, %src0, %src1, %c256
    ascendc.pipe_barrier pipe_v
    ascendc.mul_l2 %dst, %dst, %src1, %c256

    // After: fused VF block with micro-ops
    emitasc.vf_group %dst, %src0, %src1 {
      emitasc.vec_scope {
        %reg = ascendc.reg_tensor : <f32>
        %reg0 = ascendc.reg_tensor : <f32>
        %reg1 = ascendc.reg_tensor : <f32>
        scf.for %i = 0 to %repeat_count {
          ascendc.data_copy_vld_micro %reg0, %src0_ptr[%offset]
          ascendc.data_copy_vld_micro %reg1, %src1_ptr[%offset]
          ascendc.add_micro %reg, %reg0, %reg1, %mask
          ascendc.mul_micro %reg, %reg, %reg1, %mask
          ascendc.data_copy_vst_micro %dst_ptr[%offset], %reg, %mask
        } {asc.vec_scope_loop}
      }
    }
    ```
  }];
}

#endif // ASCVF_PASSES_TD
