.. Copyright (c) 2025 Huawei Technologies Co., Ltd.
.. This program is free software, you can redistribute it and/or modify it under the terms and conditions of
.. CANN Open Software License Agreement Version 2.0 (the "License").
.. Please refer to the License for details. You may not use this file except in compliance with the License.
.. THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
.. INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
.. See LICENSE in the root of the software repository for the full text of the License.

asc.language.basic
==============================

.. currentmodule:: asc.language.basic


Common operations
-----------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    axpy
    brcb
    cast
    cast_deq
    check_local_memory_ia
    copy
    cross_core_set_flag
    cross_core_wait_flag
    data_cache_clean_and_invalid
    data_cache_preload
    data_copy
    data_copy_pad
    data_sync_barrier
    dump_acc_chk_point
    dump_tensor
    duplicate
    fixpipe
    gather
    gatherb
    get_block_idx
    get_block_num
    get_cmp_mask
    get_data_block_size_in_bytes
    get_hccl_context
    get_icache_preload_status
    get_mrg_sort_result
    get_program_counter
    get_store_atomic_config
    get_sort_len
    get_sort_offset
    get_sub_block_idx
    get_system_cycle
    get_sys_workspace
    get_task_ratio
    ib_set
    ib_wait
    icache_preload
    init_soc_state
    load_data
    load_data_with_transpose
    load_image_to_local
    metrics_prof_start
    metrics_prof_stop
    mmad
    mrg_sort
    mrg_sort4
    notify_next_block
    pipe_barrier
    printf
    print_time_stamp
    proposal_concat
    proposal_extract
    reset_mask
    rp_sort16
    scatter
    select
    set_aipp_functions
    set_atomic_add
    set_atomic_max
    set_atomic_min
    set_atomic_none
    set_atomic_type
    set_cmp_mask
    set_deq_scale
    set_fix_pipe_pre_quant_flag
    set_hf32_mode
    set_hf32_trans_mode
    set_flag
    set_hccl_context
    set_load_data_boundary
    set_load_data_padding_value
    set_load_data_repeat
    set_mask_count
    set_mask_norm
    set_mm_layout_transform
    set_pad_value
    set_store_atomic_config
    set_vector_mask
    sort
    sort32
    sync_all
    transpose
    trans_data_to_5hd
    trap
    wait_flag
    wait_pre_block


TensorDesc operations
-------------------

.. autoclass:: TensorDesc
.. autosummary::
    :toctree: generated
    :nosignatures:

    TensorDesc.set_shape_addr
    TensorDesc.get_dim
    TensorDesc.get_index
    TensorDesc.get_shape
    TensorDesc.get_data_ptr
    TensorDesc.get_data_obj

.. autoclass:: ListTensorDesc
.. autosummary::
    :toctree: generated
    :nosignatures:

    ListTensorDesc.init
    ListTensorDesc.get_desc
    ListTensorDesc.get_data_ptr
    ListTensorDesc.get_size


Scalar operations
-----------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    count_bits_cnt_same_as_sign_bit
    scalar_cast 
    scalar_count_leading_zero
    scalar_get_count_of_value
    scalar_get_sff_value


Vector binary operations
------------------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    add
    add_deq_relu
    add_relu
    add_relu_cast
    bilinear_interpolation
    bitwise_and
    bitwise_or
    compare
    div
    fused_mul_add
    fused_mul_add_relu
    max
    min
    mul
    mul_add_dst
    mul_cast
    sub
    sub_relu
    sub_relu_cast


Vector reduce operations
------------------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    block_reduce_max
    block_reduce_min
    block_reduce_sum
    pair_reduce_sum
    reduce_max
    reduce_min
    reduce_sum
    repeat_reduce_sum
    whole_reduce_max
    whole_reduce_min
    whole_reduce_sum


Vector-scalar operations
------------------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    adds
    compare_scalar
    leaky_relu
    maxs
    mins
    muls
    shift_left
    shift_right


Vector unary operations
-----------------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    abs
    exp
    ln
    bitwise_not
    gather_mask
    reciprocal
    relu
    rsqrt
    sqrt


Matrix operations
-----------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    init_const_value
    load_data_with_sparse
    mmad_with_sparse
