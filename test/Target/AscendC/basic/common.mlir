// Copyright (c) 2025 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: ascir-translate -mlir-to-ascendc %s | FileCheck %s

// CHECK-LABEL:void emit_workspace(__gm__ int8_t* v1) {
// CHECK-NEXT:  AscendC::SetSysWorkspace(reinterpret_cast<__gm__ uint8_t*>(v1));
// CHECK-NEXT:  __gm__ uint8_t* v2 = GetSysWorkSpacePtr();
// CHECK-NEXT:  return;
// CHECK-NEXT:}
func.func @emit_workspace(%arg0: memref<?xi8, 22>) {
  ascendc.set_sys_workspace %arg0 : memref<?xi8, 22>
  %0 = ascendc.get_sys_workspace_ptr : memref<?xi8, 22>
  return
}

// CHECK-LABEL:void emit_ascend_is_aicv() {
// CHECK-NEXT:  int8_t v1 = g_coreType == AscendC::AIC;
// CHECK-NEXT:  int8_t v2 = g_coreType == AscendC::AIV;
// CHECK-NEXT:  return;
// CHECK-NEXT:}
func.func @emit_ascend_is_aicv() {
  %0 = ascendc.ascend_is_aic : i8
  %1 = ascendc.ascend_is_aiv : i8
  return
}

// CHECK-LABEL: void emit_data_cache_preload(AscendC::GlobalTensor<uint64_t> v1, int32_t v2) {
// CHECK-NEXT:   AscendC::DataCachePreload(v1, v2);
// CHECK-NEXT:   return;
// CHECK-NEXT: }
func.func @emit_data_cache_preload(
  %src_gm: !ascendc.global_tensor<*xui64>,
  %offset: i32
) attributes {ascendc.aicore, ascendc.global} {
  ascendc.data_cache_preload %src_gm, %offset : !ascendc.global_tensor<*xui64>, i32
  return
}

// CHECK-LABEL: void emit_data_sync_barrier(__gm__ uint64_t* v1) {
// CHECK-NEXT:   set_ffts_base_addr(*v1);
// CHECK-NEXT:   AscendC::DataSyncBarrier<AscendC::MemDsbT::ALL>();
// CHECK-NEXT:   return;
// CHECK-NEXT: }
func.func @emit_data_sync_barrier(%arg0: memref<?xui64, 22>) {
  ascendc.set_ffts_base_addr %arg0 : memref<?xui64, 22>
  ascendc.data_sync_barrier {memDsbType = 0 : i32} : () -> ()
  return
}

// CHECK-LABEL:void emit_pipe_barrier() {
// CHECK-NEXT:  AscendC::PipeBarrier<PIPE_S>();
// CHECK-NEXT:  AscendC::PipeBarrier<PIPE_V>();
// CHECK-NEXT:  AscendC::PipeBarrier<PIPE_M>();
// CHECK-NEXT:  AscendC::PipeBarrier<PIPE_MTE1>();
// CHECK-NEXT:  AscendC::PipeBarrier<PIPE_MTE2>();
// CHECK-NEXT:  AscendC::PipeBarrier<PIPE_MTE3>();
// CHECK-NEXT:  AscendC::PipeBarrier<PIPE_ALL>();
// CHECK-NEXT:  AscendC::PipeBarrier<PIPE_MTE4>();
// CHECK-NEXT:  AscendC::PipeBarrier<PIPE_MTE5>();
// CHECK-NEXT:  AscendC::PipeBarrier<PIPE_V2>();
// CHECK-NEXT:  AscendC::PipeBarrier<PIPE_FIX>();
// CHECK-NEXT:  return;
// CHECK-NEXT: }
func.func @emit_pipe_barrier() {
  ascendc.pipe_barrier pipe_s
  ascendc.pipe_barrier pipe_v
  ascendc.pipe_barrier pipe_m
  ascendc.pipe_barrier pipe_mte1
  ascendc.pipe_barrier pipe_mte2
  ascendc.pipe_barrier pipe_mte3
  ascendc.pipe_barrier pipe_all
  ascendc.pipe_barrier pipe_mte4
  ascendc.pipe_barrier pipe_mte5
  ascendc.pipe_barrier pipe_v2
  ascendc.pipe_barrier pipe_fix
  return
}

// CHECK-LABEL:void emit_reset_mask() {
// CHECK-NEXT:   AscendC::ResetMask();
// CHECK-NEXT:   return;
// CHECK-NEXT: }
func.func @emit_reset_mask() {
  ascendc.reset_mask

  return
}

// CHECK-LABEL: void emit_ib_set(AscendC::GlobalTensor<uint8_t> v1, AscendC::LocalTensor<uint8_t> v2, int32_t v3, int32_t v4) {
// CHECK-NEXT:   AscendC::IBSet(v1, v2, v3, v4);
// CHECK-NEXT:   return;
// CHECK-NEXT: }
func.func @emit_ib_set(
  %gm: !ascendc.global_tensor<*xui8>,
  %ub: !ascendc.local_tensor<*xui8>,
  %bid: i32,
  %eid: i32
) {
  ascendc.ib_set %gm, %ub, %bid, %eid {isAIVOnly} :
      !ascendc.global_tensor<*xui8>, !ascendc.local_tensor<*xui8>, i32, i32
  return
}

// CHECK-LABEL: void emit_ib_wait(AscendC::GlobalTensor<uint8_t> v1, AscendC::LocalTensor<uint8_t> v2, int32_t v3, int32_t v4) {
// CHECK-NEXT:   AscendC::IBWait(v1, v2, v3, v4);
// CHECK-NEXT:   return;
// CHECK-NEXT: }
func.func @emit_ib_wait(
  %gm: !ascendc.global_tensor<*xui8>,
  %ub: !ascendc.local_tensor<*xui8>,
  %bid: i32,
  %eid: i32
) {
  ascendc.ib_wait %gm, %ub, %bid, %eid {isAIVOnly} :
      !ascendc.global_tensor<*xui8>, !ascendc.local_tensor<*xui8>, i32, i32
  return
}

// CHECK-LABEL: void emit_sync_all_soft(AscendC::GlobalTensor<uint8_t> v1, AscendC::LocalTensor<uint8_t> v2, int32_t v3) {
// CHECK-NEXT:   AscendC::SyncAll(v1, v2, v3);
// CHECK-NEXT:   return;
// CHECK-NEXT: }
func.func @emit_sync_all_soft(
  %gm: !ascendc.global_tensor<*xui8>,
  %ub: !ascendc.local_tensor<*xui8>,
  %cores: i32
) {
  ascendc.sync_all_soft %gm, %ub, %cores {isAIVOnly} :
      !ascendc.global_tensor<*xui8>, !ascendc.local_tensor<*xui8>, i32
  return
}

// CHECK-LABEL: void emit_sync_all_hard() {
// CHECK-NEXT:   AscendC::SyncAll();
// CHECK-NEXT:   return;
// CHECK-NEXT: }
func.func @emit_sync_all_hard() {
  ascendc.sync_all_hard {isAIVOnly}
  return
}

// CHECK-LABEL: void emit_cross_core_set_flag(int32_t v1, int32_t v2)
// CHECK-NEXT:   AscendC::CrossCoreSetFlag<0, PIPE_V>(v1);
// CHECK-NEXT:   AscendC::CrossCoreSetFlag<1, PIPE_S>(v2);
// CHECK-NEXT:   return;
// CHECK-NEXT: }
func.func @emit_cross_core_set_flag(%flag1: i32, %flag2: i32) {
  ascendc.cross_core_set_flag %flag1, 0, pipe_v : i32
  ascendc.cross_core_set_flag %flag2, 1, pipe_s : i32
  return
}

// CHECK-LABEL: void emit_cross_core_wait_flag(int32_t v1, int32_t v2)
// CHECK-NEXT:   AscendC::CrossCoreWaitFlag<0, PIPE_V>(v1);
// CHECK-NEXT:   AscendC::CrossCoreWaitFlag<2, PIPE_S>(v2);
// CHECK-NEXT:   return;
// CHECK-NEXT: }
func.func @emit_cross_core_wait_flag(%flag1: i32, %flag2: i32) {
  ascendc.cross_core_wait_flag %flag1, 0, pipe_v : i32
  ascendc.cross_core_wait_flag %flag2, 2, pipe_s : i32
  return
}

// CHECK-LABEL:void emit_set_flag(int32_t v1) {
// CHECK-NEXT:  AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(v1);
// CHECK-NEXT:  AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(v1);
// CHECK-NEXT:  AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(v1);
// CHECK-NEXT:  AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(v1);
// CHECK-NEXT:  AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(v1);
// CHECK-NEXT:  AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(v1);
// CHECK-NEXT:  AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(v1);
// CHECK-NEXT:  AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(v1);
// CHECK-NEXT:  AscendC::SetFlag<AscendC::HardEvent::M_V>(v1);
// CHECK-NEXT:  AscendC::SetFlag<AscendC::HardEvent::V_M>(v1);
// CHECK-NEXT:  AscendC::SetFlag<AscendC::HardEvent::V_V>(v1);
// CHECK-NEXT:  AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE1>(v1);
// CHECK-NEXT:  AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE3>(v1);
// CHECK-NEXT:  AscendC::SetFlag<AscendC::HardEvent::MTE1_V>(v1);
// CHECK-NEXT:  AscendC::SetFlag<AscendC::HardEvent::MTE2_M>(v1);
// CHECK-NEXT:  AscendC::SetFlag<AscendC::HardEvent::M_MTE2>(v1);
// CHECK-NEXT:  AscendC::SetFlag<AscendC::HardEvent::V_MTE1>(v1);
// CHECK-NEXT:  AscendC::SetFlag<AscendC::HardEvent::M_FIX>(v1);
// CHECK-NEXT:  AscendC::SetFlag<AscendC::HardEvent::FIX_M>(v1);
// CHECK-NEXT:  AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(v1);
// CHECK-NEXT:  AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(v1);
// CHECK-NEXT:  AscendC::SetFlag<AscendC::HardEvent::S_V>(v1);
// CHECK-NEXT:  AscendC::SetFlag<AscendC::HardEvent::V_S>(v1);
// CHECK-NEXT:  AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(v1);
// CHECK-NEXT:  AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(v1);
// CHECK-NEXT:  AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(v1);
// CHECK-NEXT:  AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(v1);
// CHECK-NEXT:  AscendC::SetFlag<AscendC::HardEvent::MTE2_FIX>(v1);
// CHECK-NEXT:  AscendC::SetFlag<AscendC::HardEvent::FIX_MTE2>(v1);
// CHECK-NEXT:  AscendC::SetFlag<AscendC::HardEvent::FIX_S>(v1);
// CHECK-NEXT:  AscendC::SetFlag<AscendC::HardEvent::M_S>(v1);
// CHECK-NEXT:  AscendC::SetFlag<AscendC::HardEvent::FIX_MTE3>(v1);
// CHECK-NEXT:  AscendC::SetFlag<AscendC::HardEvent::MTE1_FIX>(v1);
// CHECK-NEXT:  AscendC::SetFlag<AscendC::HardEvent::FIX_MTE1>(v1);
// CHECK-NEXT:  AscendC::SetFlag<AscendC::HardEvent::FIX_FIX>(v1);
// CHECK-NEXT:  AscendC::SetFlag<AscendC::HardEvent::MAX>(v1);
// CHECK-NEXT:  return;
// CHECK-NEXT: }
func.func @emit_set_flag(%arg0: i32) {
  ascendc.set_flag mte2_mte1, %arg0 : i32
  ascendc.set_flag mte1_mte2, %arg0 : i32
  ascendc.set_flag mte1_m, %arg0 : i32
  ascendc.set_flag m_mte1, %arg0 : i32
  ascendc.set_flag mte2_v, %arg0 : i32
  ascendc.set_flag v_mte2, %arg0 : i32
  ascendc.set_flag mte3_v, %arg0 : i32
  ascendc.set_flag v_mte3, %arg0 : i32
  ascendc.set_flag m_v, %arg0 : i32
  ascendc.set_flag v_m, %arg0 : i32
  ascendc.set_flag v_v, %arg0 : i32
  ascendc.set_flag mte3_mte1, %arg0 : i32
  ascendc.set_flag mte1_mte3, %arg0 : i32
  ascendc.set_flag mte1_v, %arg0 : i32
  ascendc.set_flag mte2_m, %arg0 : i32
  ascendc.set_flag m_mte2, %arg0 : i32
  ascendc.set_flag v_mte1, %arg0 : i32
  ascendc.set_flag m_fix, %arg0 : i32
  ascendc.set_flag fix_m, %arg0 : i32
  ascendc.set_flag mte3_mte2, %arg0 : i32
  ascendc.set_flag mte2_mte3, %arg0 : i32
  ascendc.set_flag s_v, %arg0 : i32
  ascendc.set_flag v_s, %arg0 : i32
  ascendc.set_flag s_mte2, %arg0 : i32
  ascendc.set_flag mte2_s, %arg0 : i32
  ascendc.set_flag s_mte3, %arg0 : i32
  ascendc.set_flag mte3_s, %arg0 : i32
  ascendc.set_flag mte2_fix, %arg0 : i32
  ascendc.set_flag fix_mte2, %arg0 : i32
  ascendc.set_flag fix_s, %arg0 : i32
  ascendc.set_flag m_s, %arg0 : i32
  ascendc.set_flag fix_mte3, %arg0 : i32
  ascendc.set_flag mte1_fix, %arg0 : i32
  ascendc.set_flag fix_mte1, %arg0 : i32
  ascendc.set_flag fix_fix, %arg0 : i32
  ascendc.set_flag max, %arg0 : i32
  return
}

// CHECK-LABEL:void emit_wait_flag(int32_t v1) {
// CHECK-NEXT:  AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(v1);
// CHECK-NEXT:  AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(v1);
// CHECK-NEXT:  AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(v1);
// CHECK-NEXT:  AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(v1);
// CHECK-NEXT:  AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(v1);
// CHECK-NEXT:  AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(v1);
// CHECK-NEXT:  AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(v1);
// CHECK-NEXT:  AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(v1);
// CHECK-NEXT:  AscendC::WaitFlag<AscendC::HardEvent::M_V>(v1);
// CHECK-NEXT:  AscendC::WaitFlag<AscendC::HardEvent::V_M>(v1);
// CHECK-NEXT:  AscendC::WaitFlag<AscendC::HardEvent::V_V>(v1);
// CHECK-NEXT:  AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE1>(v1);
// CHECK-NEXT:  AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE3>(v1);
// CHECK-NEXT:  AscendC::WaitFlag<AscendC::HardEvent::MTE1_V>(v1);
// CHECK-NEXT:  AscendC::WaitFlag<AscendC::HardEvent::MTE2_M>(v1);
// CHECK-NEXT:  AscendC::WaitFlag<AscendC::HardEvent::M_MTE2>(v1);
// CHECK-NEXT:  AscendC::WaitFlag<AscendC::HardEvent::V_MTE1>(v1);
// CHECK-NEXT:  AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(v1);
// CHECK-NEXT:  AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(v1);
// CHECK-NEXT:  AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(v1);
// CHECK-NEXT:  AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(v1);
// CHECK-NEXT:  AscendC::WaitFlag<AscendC::HardEvent::S_V>(v1);
// CHECK-NEXT:  AscendC::WaitFlag<AscendC::HardEvent::V_S>(v1);
// CHECK-NEXT:  AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(v1);
// CHECK-NEXT:  AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(v1);
// CHECK-NEXT:  AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(v1);
// CHECK-NEXT:  AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(v1);
// CHECK-NEXT:  AscendC::WaitFlag<AscendC::HardEvent::MTE2_FIX>(v1);
// CHECK-NEXT:  AscendC::WaitFlag<AscendC::HardEvent::FIX_MTE2>(v1);
// CHECK-NEXT:  AscendC::WaitFlag<AscendC::HardEvent::FIX_S>(v1);
// CHECK-NEXT:  AscendC::WaitFlag<AscendC::HardEvent::M_S>(v1);
// CHECK-NEXT:  AscendC::WaitFlag<AscendC::HardEvent::FIX_MTE3>(v1);
// CHECK-NEXT:  AscendC::WaitFlag<AscendC::HardEvent::MTE1_FIX>(v1);
// CHECK-NEXT:  AscendC::WaitFlag<AscendC::HardEvent::FIX_MTE1>(v1);
// CHECK-NEXT:  AscendC::WaitFlag<AscendC::HardEvent::FIX_FIX>(v1);
// CHECK-NEXT:  AscendC::WaitFlag<AscendC::HardEvent::MAX>(v1);
// CHECK-NEXT:  return;
// CHECK-NEXT: }
func.func @emit_wait_flag(%arg0: i32) {
  ascendc.wait_flag mte2_mte1, %arg0 : i32
  ascendc.wait_flag mte1_mte2, %arg0 : i32
  ascendc.wait_flag mte1_m, %arg0 : i32
  ascendc.wait_flag m_mte1, %arg0 : i32
  ascendc.wait_flag mte2_v, %arg0 : i32
  ascendc.wait_flag v_mte2, %arg0 : i32
  ascendc.wait_flag mte3_v, %arg0 : i32
  ascendc.wait_flag v_mte3, %arg0 : i32
  ascendc.wait_flag m_v, %arg0 : i32
  ascendc.wait_flag v_m, %arg0 : i32
  ascendc.wait_flag v_v, %arg0 : i32
  ascendc.wait_flag mte3_mte1, %arg0 : i32
  ascendc.wait_flag mte1_mte3, %arg0 : i32
  ascendc.wait_flag mte1_v, %arg0 : i32
  ascendc.wait_flag mte2_m, %arg0 : i32
  ascendc.wait_flag m_mte2, %arg0 : i32
  ascendc.wait_flag v_mte1, %arg0 : i32
  ascendc.wait_flag m_fix, %arg0 : i32
  ascendc.wait_flag fix_m, %arg0 : i32
  ascendc.wait_flag mte3_mte2, %arg0 : i32
  ascendc.wait_flag mte2_mte3, %arg0 : i32
  ascendc.wait_flag s_v, %arg0 : i32
  ascendc.wait_flag v_s, %arg0 : i32
  ascendc.wait_flag s_mte2, %arg0 : i32
  ascendc.wait_flag mte2_s, %arg0 : i32
  ascendc.wait_flag s_mte3, %arg0 : i32
  ascendc.wait_flag mte3_s, %arg0 : i32
  ascendc.wait_flag mte2_fix, %arg0 : i32
  ascendc.wait_flag fix_mte2, %arg0 : i32
  ascendc.wait_flag fix_s, %arg0 : i32
  ascendc.wait_flag m_s, %arg0 : i32
  ascendc.wait_flag fix_mte3, %arg0 : i32
  ascendc.wait_flag mte1_fix, %arg0 : i32
  ascendc.wait_flag fix_mte1, %arg0 : i32
  ascendc.wait_flag fix_fix, %arg0 : i32
  ascendc.wait_flag max, %arg0 : i32
  return
}

// CHECK-LABEL:void emit_set_hccl_context(__gm__ uint8_t* v1, uint32_t v2) {
// CHECK-NEXT:  AscendC::SetHcclContext<v2>(v1);
// CHECK-NEXT:  return;
// CHECK-NEXT:}
func.func @emit_set_hccl_context(%arg0: memref<?xui8, 22>, %arg1: ui32) {
  ascendc.set_hccl_context %arg1, %arg0 : ui32, memref<?xui8, 22>
  return
}

// CHECK-LABEL:void emit_get_hccl_context(uint32_t v1) {
// CHECK-NEXT:  __gm__ uint8_t* v2 = AscendC::GetHcclContext<v1>();
// CHECK-NEXT:  return;
// CHECK-NEXT:}
func.func @emit_get_hccl_context(%arg0: ui32) {
  %0 = ascendc.get_hccl_context %arg0 : ui32, memref<?xui8, 22>
  return
}

// CHECK-LABEL: void emit_set_aipp_functions(AscendC::GlobalTensor<int8_t> v1, AscendC::GlobalTensor<int8_t> v2, int32_t v3, int8_t v4, int8_t v5, int8_t v6, int8_t v7, bool v8, bool v9, bool v10, bool v11, uint8_t v12, uint8_t v13, uint8_t v14, half v15, half v16, half v17, half v18, half v19, half v20, uint32_t v21, int32_t v22, int8_t v23, bool v24, int16_t v25, int16_t v26, int16_t v27, int16_t v28, int16_t v29, int16_t v30, int16_t v31, int16_t v32, int16_t v33, uint8_t v34, uint8_t v35, uint8_t v36, uint8_t v37, uint8_t v38, uint8_t v39) {
// CHECK:         AscendC::AippParams<int8_t> v40;
// CHECK-NEXT:    v40.paddingParams.paddingMode = v3;
// CHECK-NEXT:    v40.paddingParams.paddingValueCh0 = v4;
// CHECK-NEXT:    v40.paddingParams.paddingValueCh1 = v5;
// CHECK-NEXT:    v40.paddingParams.paddingValueCh2 = v6;
// CHECK-NEXT:    v40.paddingParams.paddingValueCh3 = v7;
// CHECK-NEXT:    v40.swapParams.isSwapRB = v8;
// CHECK-NEXT:    v40.swapParams.isSwapUV = v9;
// CHECK-NEXT:    v40.swapParams.isSwapAX = v10;
// CHECK-NEXT:    v40.singleLineParams.isSingleLineCopy = v11;
// CHECK-NEXT:    v40.dtcParams.dtcMeanCh0 = v12;
// CHECK-NEXT:    v40.dtcParams.dtcMeanCh1 = v13;
// CHECK-NEXT:    v40.dtcParams.dtcMeanCh2 = v14;
// CHECK-NEXT:    v40.dtcParams.dtcMinCh0 = v15;
// CHECK-NEXT:    v40.dtcParams.dtcMinCh1 = v16;
// CHECK-NEXT:    v40.dtcParams.dtcMinCh2 = v17;
// CHECK-NEXT:    v40.dtcParams.dtcVarCh0 = v18;
// CHECK-NEXT:    v40.dtcParams.dtcVarCh1 = v19;
// CHECK-NEXT:    v40.dtcParams.dtcVarCh2 = v20;
// CHECK-NEXT:    v40.dtcParams.dtcRoundMode = v21;
// CHECK-NEXT:    v40.cPaddingParams.cPaddingMode = v22;
// CHECK-NEXT:    v40.cPaddingParams.cPaddingValue = v23;
// CHECK-NEXT:    v40.cscParams.isEnableCsc = v24;
// CHECK-NEXT:    v40.cscParams.cscMatrixR0C0 = v25;
// CHECK-NEXT:    v40.cscParams.cscMatrixR0C1 = v26;
// CHECK-NEXT:    v40.cscParams.cscMatrixR0C2 = v27;
// CHECK-NEXT:    v40.cscParams.cscMatrixR1C0 = v28;
// CHECK-NEXT:    v40.cscParams.cscMatrixR1C1 = v29;
// CHECK-NEXT:    v40.cscParams.cscMatrixR1C2 = v30;
// CHECK-NEXT:    v40.cscParams.cscMatrixR2C0 = v31;
// CHECK-NEXT:    v40.cscParams.cscMatrixR2C1 = v32;
// CHECK-NEXT:    v40.cscParams.cscMatrixR2C2 = v33;
// CHECK-NEXT:    v40.cscParams.cscBiasIn0 = v34;
// CHECK-NEXT:    v40.cscParams.cscBiasIn1 = v35;
// CHECK-NEXT:    v40.cscParams.cscBiasIn2 = v36;
// CHECK-NEXT:    v40.cscParams.cscBiasOut0 = v37;
// CHECK-NEXT:    v40.cscParams.cscBiasOut1 = v38;
// CHECK-NEXT:    v40.cscParams.cscBiasOut2 = v39;
// CHECK:         AscendC::SetAippFunctions(v1, AscendC::AippInputFormat::RGB888_U8, v40);
// CHECK-NEXT:    AscendC::SetAippFunctions(v1, v2, AscendC::AippInputFormat::YUV420SP_U8, v40);
// CHECK-NEXT:    return;
// CHECK-NEXT:  }
func.func @emit_set_aipp_functions(
    %arg0: !ascendc.global_tensor<*xi8>,
    %arg1: !ascendc.global_tensor<*xi8>,
    %arg2: i32, %arg3: i8, %arg4: i8, %arg5: i8, %arg6: i8,
    %arg7: i1, %arg8: i1, %arg9: i1,
    %arg10: i1,
    %arg11: ui8, %arg12: ui8, %arg13: ui8,
    %arg14: f16, %arg15: f16, %arg16: f16,
    %arg17: f16, %arg18: f16, %arg19: f16,
    %arg20: ui32,
    %arg21: i32, %arg22: i8,
    %arg23: i1,
    %arg24: i16, %arg25: i16, %arg26: i16,
    %arg27: i16, %arg28: i16, %arg29: i16,
    %arg30: i16, %arg31: i16, %arg32: i16,
    %arg33: ui8, %arg34: ui8, %arg35: ui8,
    %arg36: ui8, %arg37: ui8, %arg38: ui8
) {
  %padding_params = ascendc.construct !ascendc.aipp_padding_params(%arg2, %arg3, %arg4, %arg5, %arg6) [ui32, i8, i8, i8, i8] : i32, i8, i8, i8, i8
  %swap_params = ascendc.construct !ascendc.aipp_swap_params(%arg7, %arg8, %arg9) [i8, i8, i8] : i1, i1, i1
  %single_line_params = ascendc.construct !ascendc.aipp_single_line_params(%arg10) [i8] : i1
  %dtc_params = ascendc.construct !ascendc.aipp_dtc_params(%arg11, %arg12, %arg13, %arg14, %arg15, %arg16, %arg17, %arg18, %arg19, %arg20) [ui8, ui8, ui8, f16, f16, f16, f16, f16, f16, ui32] : ui8, ui8, ui8, f16, f16, f16, f16, f16, f16, ui32
  %cpadding_params = ascendc.construct !ascendc.aipp_cpadding_params(%arg21, %arg22) [ui32, i8] : i32, i8
  %csc_params = ascendc.construct !ascendc.aipp_csc_params(%arg23, %arg24, %arg25, %arg26, %arg27, %arg28, %arg29, %arg30, %arg31, %arg32, %arg33, %arg34, %arg35, %arg36, %arg37, %arg38) [i8, i16, i16, i16, i16, i16, i16, i16, i16, i16, ui8, ui8, ui8, ui8, ui8, ui8] : i1, i16, i16, i16, i16, i16, i16, i16, i16, i16, ui8, ui8, ui8, ui8, ui8, ui8
  %aipp_params = ascendc.construct !ascendc.aipp_params(%padding_params, %swap_params, %single_line_params, %dtc_params, %cpadding_params, %csc_params) [!ascendc.aipp_padding_params, !ascendc.aipp_swap_params, !ascendc.aipp_single_line_params, !ascendc.aipp_dtc_params, !ascendc.aipp_cpadding_params, !ascendc.aipp_csc_params] : !ascendc.aipp_padding_params, !ascendc.aipp_swap_params, !ascendc.aipp_single_line_params, !ascendc.aipp_dtc_params, !ascendc.aipp_cpadding_params, !ascendc.aipp_csc_params
  "ascendc.set_aipp_functions"(%arg0, %aipp_params) {
  format = 4 : i32
} : (!ascendc.global_tensor<*xi8>, !ascendc.aipp_params) -> ()
  "ascendc.set_aipp_functions"(%arg0, %arg1, %aipp_params) {
  format = 0 : i32
} : (!ascendc.global_tensor<*xi8>, !ascendc.global_tensor<*xi8>, !ascendc.aipp_params) -> ()
  return
}

// CHECK-LABEL:void emit_set_hf32_mode(bool v1) {
// CHECK-NEXT:  AscendC::SetHF32Mode(v1);
// CHECK-NEXT:  return;
// CHECK-NEXT:}
func.func @emit_set_hf32_mode(%mode: i1) {
  ascendc.set_hf32_mode %mode : i1
  return
}

// CHECK-LABEL:void emit_set_hf32_trans_mode(bool v1) {
// CHECK-NEXT:  AscendC::SetHF32TransMode(v1);
// CHECK-NEXT:  return;
// CHECK-NEXT:}
func.func @emit_set_hf32_trans_mode(%mode: i1) {
  ascendc.set_hf32_trans_mode %mode : i1
  return
}

// CHECK-LABEL:void emit_set_mm_layout_transform(bool v1) {
// CHECK-NEXT:  AscendC::SetMMLayoutTransform(v1);
// CHECK-NEXT:  return;
// CHECK-NEXT:}
func.func @emit_set_mm_layout_transform(%mode: i1) {
  ascendc.set_mm_layout_transform %mode : i1
  return
}

// CHECK-LABEL:void emit_set_mask_count() {
// CHECK-NEXT:  AscendC::SetMaskCount();
// CHECK-NEXT:  return;
// CHECK-NEXT:}
func.func @emit_set_mask_count() {
  ascendc.set_mask_count
  return
}

// CHECK-LABEL:void emit_set_mask_norm() {
// CHECK-NEXT:  AscendC::SetMaskNorm();
// CHECK-NEXT:  return;
// CHECK-NEXT:}
func.func @emit_set_mask_norm() {
  ascendc.set_mask_norm
  return
}

// CHECK-LABEL:void emit_set_vector_mask(int32_t v1, int64_t v2, int64_t v3) {
// CHECK-NEXT:  AscendC::SetVectorMask<half, AscendC::MaskMode::COUNTER>(v1)
// CHECK-NEXT:  AscendC::SetVectorMask<float, AscendC::MaskMode::NORMAL>(v2, v3)
// CHECK-NEXT:  return;
// CHECK-NEXT:}
func.func @emit_set_vector_mask(%len: i32, %maskHigh: i64, %maskLow: i64) {
  ascendc.set_vector_mask_l0 %len {dtype = f16, mode = 1 : i32} : i32
  ascendc.set_vector_mask_l1 %maskHigh, %maskLow {dtype = f32, mode = 0 : i32} : i64, i64
  return
}


// CHECK-LABEL:void emit_common_test() {
// CHECK-NEXT:  AscendC::InitSocState();
// CHECK-NEXT:  AscendC::SetStoreAtomicConfig<AscendC::AtomicDtype::ATOMIC_F32, AscendC::AtomicOp::ATOMIC_SUM>();
// CHECK-NEXT:  uint16_t v1;
// CHECK-NEXT:  uint16_t v2;
// CHECK-NEXT:  AscendC::GetStoreAtomicConfig(v1, v2);
// CHECK-NEXT:  constexpr uint8_t v3 = 0;
// CHECK-NEXT:  constexpr uint32_t v4 = 0;
// CHECK-NEXT:  constexpr bool c0_i1 = false;
// CHECK-NEXT:  AscendC::CheckLocalMemoryIAParam v5{v3, v4, v4, c0_i1, c0_i1, c0_i1, c0_i1, c0_i1, c0_i1, c0_i1};
// CHECK-NEXT:  AscendC::CheckLocalMemoryIA(v5);
// CHECK-NEXT:  return;
// CHECK-NEXT:}
func.func @emit_common_test() {
  ascendc.init_soc_state
  ascendc.set_store_atomic_config atomic_f32, atomic_sum
  %type, %op = ascendc.get_store_atomic_config  : ui16, ui16
  %val_8 = "emitc.constant"() <{value = 0 : ui8}> : () -> ui8
  %val_32 = "emitc.constant"() <{value = 0 : ui32}> : () -> ui32
  %false = arith.constant false
  %param = ascendc.construct !ascendc.check_local_memory_ia_param(%val_8, %val_32, %val_32, %false, %false, %false, %false, %false, %false, %false) [ui8, ui32, ui32, i1, i1, i1, i1, i1, i1, i1] : ui8, ui32, ui32, i1, i1, i1, i1, i1, i1, i1
  ascendc.check_local_memory_ia %param : !ascendc.check_local_memory_ia_param
  return
}

// CHECK-LABEL:void emit_get_buf() {
// CHECK-NEXT:  get_buf(PIPE_V, 0, 0);
// CHECK-NEXT:  get_buf(PIPE_MTE2, 1, 1);
// CHECK-NEXT:  return;
// CHECK-NEXT:}
func.func @emit_get_buf() {
  ascendc.get_buf pipe_v, 0
  ascendc.get_buf pipe_mte2, 1 {mode}
  return
}

// CHECK-LABEL:void emit_rls_buf() {
// CHECK-NEXT:  rls_buf(PIPE_V, 0, 0);
// CHECK-NEXT:  rls_buf(PIPE_MTE2, 1, 1);
// CHECK-NEXT:  return;
// CHECK-NEXT:}
func.func @emit_rls_buf() {
  ascendc.rls_buf pipe_v, 0
  ascendc.rls_buf pipe_mte2, 1 {mode}
  return
}
