# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from typing import Optional

from ..core.tensor import LocalTensor
from ..core.ir_value import materialize_ir_value as _mat
from ..core.utils import require_jit, global_builder
from .tiling import SoftmaxTiling
from .types import SoftMaxShapeInfo, SoftmaxConfig


@require_jit
def softmax(dst: LocalTensor, sum: LocalTensor, max: LocalTensor, src: LocalTensor, tiling: SoftmaxTiling,
            temp_buffer: Optional[LocalTensor] = None, reuse_source: bool = False, basic_block: bool = False,
            data_format_nz: bool = False) -> None:
    """
    将输入tensor[m0, m1, ...mt, n]（t大于等于0）的非尾轴长度相乘的结果看作m，则输入tensor的shape看作[m, n]。
    为方便理解，通过Python脚本实现的方式，表达其计算公式（以输入为ND格式为例）如下，其中src是源操作数（输入），dst、sum、max为目的操作数（输出）。

    .. code-block:: python

        def softmax(src):
            # 基于last轴进行rowmax（按行取最大值）处理
            max = np.max(src, axis=-1, keepdims=True)
            sub = src - max
            exp = np.exp(sub)
            # 基于last轴进行rowsum（按行求和）处理
            sum = np.sum(exp, axis=-1, keepdims=True)
            dst = exp / sum
            return dst, max, sum

    **对应的Ascend C函数原型**

    - 接口框架申请临时空间
    
      - LocalTensor的数据类型相同 

        .. code-block:: c++

            template <typename T, bool isReuseSource = false, bool isBasicBlock = false, 
            bool isDataFormatNZ = false, const SoftmaxConfig& config = SOFTMAX_DEFAULT_CFG>
            __aicore__ inline void SoftMax(const LocalTensor<T>& dstTensor, const LocalTensor<T>& sumTensor, 
                                          const LocalTensor<T>& maxTensor, const LocalTensor<T>& srcTensor, 
                                          const SoftMaxTiling& tiling, const SoftMaxShapeInfo& softmaxShapeInfo = {})

      - LocalTensor的数据类型不同

        .. code-block:: c++

            template <typename T, bool isReuseSource = false, bool isBasicBlock = false, 
            bool isDataFormatNZ = false, const SoftmaxConfig& config = SOFTMAX_DEFAULT_CFG>
            __aicore__ inline void SoftMax(const LocalTensor<half>& dstTensor, const LocalTensor<float>& sumTensor, 
                                          const LocalTensor<float>& maxTensor, const LocalTensor<half>& srcTensor, 
                                          const SoftMaxTiling& tiling, const SoftMaxShapeInfo& softmaxShapeInfo = {})

      - 不带sumTensor和maxTensor参数

        .. code-block:: c++

            template <typename T, bool isReuseSource = false, bool isBasicBlock = false, 
            const SoftmaxConfig& config = SOFTMAX_DEFAULT_CFG>
            __aicore__ inline void SoftMax(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor, 
                                        const SoftMaxTiling& tiling, const SoftMaxShapeInfo& softmaxShapeInfo = {})

    - 通过sharedTmpBuffer入参传入临时空间

      - LocalTensor的数据类型相同

        .. code-block:: c++

            template <typename T, bool isReuseSource = false, bool isBasicBlock = false, 
            bool isDataFormatNZ = false, const SoftmaxConfig& config = SOFTMAX_DEFAULT_CFG>
            __aicore__ inline void SoftMax(const LocalTensor<T>& dstTensor, const LocalTensor<T>& sumTensor, 
                                          const LocalTensor<T>& maxTensor, const LocalTensor<T>& srcTensor, 
                                          const LocalTensor<uint8_t>& sharedTmpBuffer, const SoftMaxTiling& tiling, 
                                          const SoftMaxShapeInfo& softmaxShapeInfo = {})
    
      - LocalTensor的数据类型不同

        .. code-block:: c++

            template <typename T, bool isReuseSource = false, bool isBasicBlock = false, 
            bool isDataFormatNZ = false, const SoftmaxConfig& config = SOFTMAX_DEFAULT_CFG>
            __aicore__ inline void SoftMax(const LocalTensor<half>& dstTensor, const LocalTensor<float>& sumTensor, 
                                          const LocalTensor<float>& maxTensor, const LocalTensor<half>& srcTensor, 
                                          const LocalTensor<uint8_t>& sharedTmpBuffer, const SoftMaxTiling& tiling, 
                                          const SoftMaxShapeInfo& softmaxShapeInfo = {})

      - 不带sumTensor和maxTensor参数

        .. code-block:: c++

            template <typename T, bool isReuseSource = false, bool isBasicBlock = false, 
            const SoftmaxConfig& config = SOFTMAX_DEFAULT_CFG>
            __aicore__ inline void SoftMax(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor, 
                                          const LocalTensor<uint8_t>& sharedTmpBuffer, const SoftMaxTiling& tiling, 
                                          const SoftMaxShapeInfo& softmaxShapeInfo = {})
            
    **参数说明**

    - dst：目的操作数。
    - sum：目的操作数。
    - max：目的操作数。
    - src：源操作数。
    - tiling：SoftMax计算所需Tiling信息。
    - tmp_buffer：临时空间。
    - reuse_source：该参数预留，传入默认值false即可。
    - basic_block：src和dst的shape信息和Tiling切分策略满足基本块要求的情况下，可以使能该参数用于提升性能，默认不使能。
    - data_format_nz：当前输入输出的数据格式是否为NZ格式，默认数据格式为ND，即默认取值为false。

    **约束说明**

    - src和dst的Tensor空间可以复用。
    - sum和max为输出，并且last轴长度必须固定32Byte，非last轴大小需要和src以及dst保持一致。
    - sum和max的数据类型需要保持一致。
    - 操作数地址对齐要求请参见 `《Ascend C算子开发接口》 <https://hiascend.com/document/redirect/CannCommunityAscendCApi>`_中的“通用说明和约束-通用地址对齐约束”。
    - 不支持tmp_buffer与源操作数和目的操作数地址重叠。
    开发者需要对GM上的原始输入(ori_src_M, ori_src_K)在M或K方向补齐数据到(src_M, src_K)，补齐的数据会参与部分运算，
    在输入输出复用的场景下，API的计算结果会覆盖src中补齐的原始数据，在输入输出不复用的场景下，
    API的计算结果会覆盖dst中对应src补齐位置的数据。

    **调用示例**

    .. code-block:: python

        src_local = in_queue_src.deque(T)
        sum_temp_local = sum_queue.alloc_tensor(T)
        max_temp_local = max_queue.alloc_tensor(T)
        dst_local = out_queue_dst.alloc_tensor(T)

        src_shape = asc.SoftMaxShapeInfo(height, width, height, width);
        asc.adv.softmax(dst_local, sum_temp_local, max_temp_local, srcLocal, tiling, src_shape);

        out_queue_dst.EnQue(dstLocal)
        max_queue.free_tensor(max_temp_local)
        sum_queue.free_tensor(sum_temp_local)
        in_queue_src.free_tensor(src_local)
    """
    temp_buffer = temp_buffer.to_ir() if temp_buffer is not None else None
    global_builder.get_ir_builder().create_asc_SoftMaxOp(reuseSource=reuse_source, basicBlock=basic_block,
                                                         dataFormatNZ=data_format_nz, dst=dst.to_ir(),
                                                         sumTensor=sum.to_ir(), maxTensor=max.to_ir(), src=src.to_ir(),
                                                         sharedTmpBuffer=temp_buffer, tiling=tiling.to_ir(),
                                                         softmaxShapeInfo=None)


@require_jit
def softmax_flash_v2(dst_tensor: LocalTensor, exp_sum_tensor: LocalTensor, max_tensor: LocalTensor,
                     src_tensor: LocalTensor, exp_max_tensor: LocalTensor, in_exp_sum_tensor: LocalTensor,
                     in_max_tensor: LocalTensor, tiling: SoftmaxTiling,
                     softmax_shape_info: Optional[SoftMaxShapeInfo] = None,
                     shared_tmp_buffer: Optional[LocalTensor] = None, out_reduce_max: Optional[LocalTensor] = None,
                     is_update: bool = False, is_reuse_source: bool = False, is_basic_block: bool = False,
                     is_data_format_nz: bool = False, config: Optional[SoftmaxConfig] = None) -> None:
    """
    SoftmaxFlash增强版本，对应FlashAttention-2算法。

    **对应的Ascend C函数原型**

    - 接口框架申请临时空间

      - LocalTensor的数据类型相同，不输出ReduceMax

        .. code-block:: c++

            template <typename T, bool isUpdate = false, bool isReuseSource = false,
                      bool isBasicBlock = false, bool isDataFormatNZ = false,
                      const SoftmaxConfig& config = SOFTMAX_DEFAULT_CFG>
            __aicore__ inline void SoftmaxFlashV2(
                const LocalTensor<T>& dstTensor, const LocalTensor<T>& expSumTensor,
                const LocalTensor<T>& maxTensor, const LocalTensor<T>& srcTensor,
                const LocalTensor<T>& expMaxTensor, const LocalTensor<T>& inExpSumTensor,
                const LocalTensor<T>& inMaxTensor, const SoftMaxTiling& tiling,
                const SoftMaxShapeInfo& softmaxShapeInfo = {})

      - LocalTensor的数据类型相同，且输出ReduceMax

        .. code-block:: c++

            template <typename T, bool isUpdate = false, bool isReuseSource = false,
                      bool isBasicBlock = false, bool isDataFormatNZ = false,
                      const SoftmaxConfig& config = SOFTMAX_DEFAULT_CFG>
            __aicore__ inline void SoftmaxFlashV2(
                const LocalTensor<T>& dstTensor, const LocalTensor<T>& outReduceMax,
                const LocalTensor<T>& outExpSum, const LocalTensor<T>& outMax,
                const LocalTensor<T>& srcTensor, const LocalTensor<T>& outExpMax,
                const LocalTensor<T>& inExpSum, const LocalTensor<T>& inMax,
                const SoftMaxTiling& tiling, const SoftMaxShapeInfo& softmaxShapeInfo = {})

      - LocalTensor的数据类型不同，不输出ReduceMax

        .. code-block:: c++

            template <typename T, bool isUpdate = false, bool isReuseSource = false,
                      bool isBasicBlock = false, bool isDataFormatNZ = false,
                      const SoftmaxConfig& config = SOFTMAX_DEFAULT_CFG>
            __aicore__ inline void SoftmaxFlashV2(
                const LocalTensor<half>& dstTensor, const LocalTensor<float>& expSumTensor,
                const LocalTensor<float>& maxTensor, const LocalTensor<half>& srcTensor,
                const LocalTensor<half>& expMaxTensor, const LocalTensor<float>& inExpSumTensor,
                const LocalTensor<float>& inMaxTensor, const SoftMaxTiling& tiling,
                const SoftMaxShapeInfo& softmaxShapeInfo = {})

    - 通过sharedTmpBuffer入参传入临时空间

      - LocalTensor的数据类型相同，不输出ReduceMax

        .. code-block:: c++

            template <typename T, bool isUpdate = false, bool isReuseSource = false,
                      bool isBasicBlock = false, bool isDataFormatNZ = false,
                      const SoftmaxConfig& config = SOFTMAX_DEFAULT_CFG>
            __aicore__ inline void SoftmaxFlashV2(
                const LocalTensor<T>& dstTensor, const LocalTensor<T>& outExpSum,
                const LocalTensor<T>& outMax, const LocalTensor<T>& srcTensor,
                const LocalTensor<T>& outExpMax, const LocalTensor<T>& inExpSum,
                const LocalTensor<T>& inMax, const LocalTensor<uint8_t>& sharedTmpBuffer,
                const SoftMaxTiling& tiling, const SoftMaxShapeInfo& softmaxShapeInfo = {})

      - LocalTensor的数据类型相同，且输出ReduceMax

        .. code-block:: c++

            template <typename T, bool isUpdate = false, bool isReuseSource = false,
                      bool isBasicBlock = false, bool isDataFormatNZ = false,
                      const SoftmaxConfig& config = SOFTMAX_DEFAULT_CFG>
            __aicore__ inline void SoftmaxFlashV2(
                const LocalTensor<T>& dstTensor, const LocalTensor<T>& outReduceMax,
                const LocalTensor<T>& expSumTensor, const LocalTensor<T>& maxTensor,
                const LocalTensor<T>& srcTensor, const LocalTensor<T>& expMaxTensor,
                const LocalTensor<T>& inExpSumTensor, const LocalTensor<T>& inMaxTensor,
                const LocalTensor<uint8_t>& sharedTmpBuffer, const SoftMaxTiling& tiling,
                const SoftMaxShapeInfo& softmaxShapeInfo = {})

      - LocalTensor的数据类型不同，不输出ReduceMax

        .. code-block:: c++

            template <typename T, bool isUpdate = false, bool isReuseSource = false,
                      bool isBasicBlock = false, bool isDataFormatNZ = false,
                      const SoftmaxConfig& config = SOFTMAX_DEFAULT_CFG>
            __aicore__ inline void SoftmaxFlashV2(
                const LocalTensor<half>& dstTensor, const LocalTensor<float>& expSumTensor,
                const LocalTensor<float>& maxTensor, const LocalTensor<half>& srcTensor,
                const LocalTensor<half>& expMaxTensor, const LocalTensor<float>& inExpSumTensor,
                const LocalTensor<float>& inMaxTensor,
                const LocalTensor<uint8_t>& sharedTmpBuffer, const SoftMaxTiling& tiling,
                const SoftMaxShapeInfo& softmaxShapeInfo = {})

    **参数说明**

    - dst_tensor：目的操作数，shape与源操作数src_tensor一致。
    - exp_sum_tensor：目的操作数，用于保存softmax计算过程中reducesum的结果。
      - 除config配置为非拓展模式（SoftmaxMode.SOFTMAX_OUTPUT_WITHOUT_BRC）的场景外，
        exp_sum_tensor的last轴长度固定为32Byte，即一个datablock长度。该datablock中的所有数据为同一个值，
        比如float16数据类型下，该datablock中的16个数均为相同的reducesum值。
      - 非last轴的长度与dst_tensor保持一致。
    - max_tensor：目的操作数，用于保存softmax计算过程中reducemax的结果。
      - 除config配置为非拓展模式（SoftmaxMode.SOFTMAX_OUTPUT_WITHOUT_BRC）的场景外，
        max_tensor的last轴长度固定为32Byte，即一个datablock长度。该datablock中的所有数据为同一个值，
        比如float16数据类型下，该datablock中的16个数均为相同的reducemax值。
      - 非last轴的长度与dst_tensor保持一致。
    - src_tensor：源操作数，last轴长度需要32Byte对齐。
    - exp_max_tensor：目的操作数，用于保存in_max_tensor与reducemax差值的e的指数幂结果。
      - 除config配置为非拓展模式（SoftmaxMode.SOFTMAX_OUTPUT_WITHOUT_BRC）的场景外，
        exp_max_tensor的last轴长度固定为32Byte，即一个datablock长度。该datablock中的所有数据为同一个值，
        比如float16数据类型下，该datablock中的16个数均为相同的值。
      - 非last轴的长度与dst_tensor保持一致。
    - in_exp_sum_tensor：源操作数，softmax计算所需的sum值。
      - 除config配置为非拓展模式（SoftmaxMode.SOFTMAX_OUTPUT_WITHOUT_BRC）的场景外，
        in_exp_sum_tensor的last轴长度固定为32Byte，即一个datablock长度。该datablock中的所有数据为同一个值，
        比如float16数据类型下，该datablock中的16个数均为相同的值。
      - 非last轴的长度与dst_tensor保持一致。
    - in_max_tensor：源操作数，softmax计算所需的max值。
      - 除config配置为非拓展模式（SoftmaxMode.SOFTMAX_OUTPUT_WITHOUT_BRC）的场景外，
        in_max_tensor的last轴长度固定为32Byte，即一个datablock长度。该datablock中的所有数据为同一个值，
        比如float16数据类型下，该datablock中的16个数均为相同的值。
      - 非last轴的长度与dst_tensor保持一致。
    - tiling：softmax_flash_v2计算所需的SoftmaxTiling信息。
    - softmax_shape_info：src_tensor的shape信息，类型为SoftMaxShapeInfo。
    - shared_tmp_buffer：可选临时空间，数据类型固定为uint8。用于存储接口内部的中间变量，由开发者提供。
    - out_reduce_max：可选目的操作数，用于保存softmax计算过程中第一次reducemax的结果，
      shape与max_tensor一致。指定该参数时：
      - is_update为False时，不输出该结果。
      - 仅支持ND格式，is_data_format_nz为预留参数，应使用默认值False。
      - config.check_tiling为预留配置，应设为False。
      - config.mode仅支持SoftmaxMode.SOFTMAX_OUTPUT_WITHOUT_BRC。
    - is_update：是否基于in_exp_sum_tensor和in_max_tensor更新softmax状态，默认为False。
    - is_reuse_source：是否复用src_tensor的空间，默认为False。
    - is_basic_block：是否使用基本块模式，默认为False。
    - is_data_format_nz：输入输出是否为NZ格式，默认为False。
    - config：结构体模板参数，此参数可选配，SoftmaxConfig类型。

    **返回值说明**

    无

    **约束说明**

    - src_tensor和dst_tensor的Tensor空间可以复用，max_tensor和in_max_tensor的空间可以复用，
      exp_sum_tensor和in_exp_sum_tensor的空间可以复用。
    - 除config配置为非拓展模式（SoftmaxMode.SOFTMAX_OUTPUT_WITHOUT_BRC）的场景外，
      exp_sum_tensor、max_tensor、exp_max_tensor、in_exp_sum_tensor和in_max_tensor的Tensor空间，
      last轴长度必须固定为32Byte。
    - 指定out_reduce_max时：
      - is_reuse_source、is_data_format_nz和config.check_tiling均为预留配置；
      - config.mode只支持SoftmaxMode.SOFTMAX_OUTPUT_WITHOUT_BRC；
        配置为SoftmaxMode.SOFTMAX_NORMAL时，接口不执行计算，也不保存输出；
      - is_update为False时，不输出out_reduce_max；
      - 除out_reduce_max外，其余输出的计算结果与未指定out_reduce_max时相同。
    - 操作数地址对齐要求请参见通用地址对齐约束。
    - 不支持shared_tmp_buffer与源操作数或目的操作数地址重叠。
    - 当softmax_shape_info中的src_m != ori_src_m或src_k != ori_src_k时，需要将GM上的原始输入
      (ori_src_m, ori_src_k)沿M轴或K轴补齐到(src_m, src_k)。补齐数据会参与部分运算；
      复用输入输出时，结果会覆盖src_tensor中的补齐数据，否则会覆盖dst_tensor中与src_tensor补齐位置对应的数据。

    **调用示例**

    .. code-block:: python

        dst_half = asc.LocalTensor(dtype=asc.float16)
        state_half = asc.LocalTensor(dtype=asc.float16)
        src_half = asc.LocalTensor(dtype=asc.float16)
        exp_max_half = asc.LocalTensor(dtype=asc.float16)
        tiling = asc.adv.SoftmaxTiling(src_m=8, src_k=512, src_size=4096)
        shape = asc.adv.SoftMaxShapeInfo(8, 512, 8, 512)
        asc.adv.softmax_flash_v2(
            dst_half, state_half, state_half, src_half, exp_max_half,
            state_half, state_half, tiling, shape)
    """
    reduce_max_ir = out_reduce_max.to_ir() if out_reduce_max is not None else None
    shared_tmp_ir = shared_tmp_buffer.to_ir() if shared_tmp_buffer is not None else None
    shape_info_ir = softmax_shape_info.to_ir() if softmax_shape_info is not None else None
    config_ir = config.to_ir() if config is not None else None
    global_builder.get_ir_builder().create_asc_SoftmaxFlashV2Op(
        isUpdate=is_update,
        reuseSource=is_reuse_source,
        basicBlock=is_basic_block,
        dataFormatNZ=is_data_format_nz,
        dst=dst_tensor.to_ir(),
        outReduceMax=reduce_max_ir,
        expSumTensor=exp_sum_tensor.to_ir(),
        maxTensor=max_tensor.to_ir(),
        src=src_tensor.to_ir(),
        expMaxTensor=exp_max_tensor.to_ir(),
        inExpSumTensor=in_exp_sum_tensor.to_ir(),
        inMaxTensor=in_max_tensor.to_ir(),
        sharedTmpBuffer=shared_tmp_ir,
        tiling=tiling.to_ir(),
        softmaxShapeInfo=shape_info_ir,
        softmaxConfig=config_ir,
    )


@require_jit
def swiglu(dst_tensor: LocalTensor, src_tensor0: LocalTensor, src_tensor1: LocalTensor, scalar_value: float = 1.0,
           shared_tmp_buffer: Optional[LocalTensor] = None, cal_count: Optional[int] = None) -> None:
    """
    SwiGLU是采用Swish作为激活函数的GLU变体。具体计算公式如下：

    .. code-block:: text

        Swish(x) = x / (1 + e^(-βx))
        SwiGLU = src_tensor0 ⊗ Swish(src_tensor1)

    **对应的Ascend C函数原型**

    - 通过sharedTmpBuffer入参传入临时空间

      - 源操作数Tensor全部/部分参与计算

        .. code-block:: c++

            template <typename T, bool isReuseSource = false>
            __aicore__ inline void SwiGLU(LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor0,
                                          const LocalTensor<T>& srcTensor1, const float& scalarValue,
                                          const LocalTensor<uint8_t>& sharedTmpBuffer, const uint32_t calCount)

      - 源操作数Tensor全部参与计算

        .. code-block:: c++

            template <typename T, bool isReuseSource = false>
            __aicore__ inline void SwiGLU(LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor0,
                                          const LocalTensor<T>& srcTensor1, const float& scalarValue,
                                          const LocalTensor<uint8_t>& sharedTmpBuffer)

    - 接口框架申请临时空间

      - 源操作数Tensor全部/部分参与计算

        .. code-block:: c++

            template <typename T, bool isReuseSource = false>
            __aicore__ inline void SwiGLU(LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor0,
                                          const LocalTensor<T>& srcTensor1, const float& scalarValue,
                                          const uint32_t calCount)

      - 源操作数Tensor全部参与计算

        .. code-block:: c++

            template <typename T, bool isReuseSource = false>
            __aicore__ inline void SwiGLU(LocalTensor<T>& dstTensor, LocalTensor<T>& srcTensor0,
                                          LocalTensor<T>& srcTensor1, const float& scalarValue)

    **参数说明**

    - dst_tensor：目的操作数。
    - src_tensor0：源操作数0。数据类型需要与dst_tensor保持一致。
    - src_tensor1：源操作数1。数据类型需要与dst_tensor保持一致。
    - scalar_value：激活函数中的β参数。默认值为1.0。
    - shared_tmp_buffer：临时缓存。用于SwiGLU内部计算时存储中间变量，由开发者提供。
    - cal_count：实际计算数据元素个数。不指定时为全部元素参与计算。

    **返回值说明**

    无

    **约束说明**

    - 操作数地址对齐要求请参见通用地址对齐约束。
    - 不支持源操作数与目的操作数地址重叠。
    - 当前仅支持ND格式的输入，不支持其他格式。
    - 不支持shared_tmp_buffer与源操作数和目的操作数地址重叠。

    **调用示例**

    .. code-block:: python

        src0_local = in_queue0.deque(T)
        src1_local = in_queue1.deque(T)
        dst_local = out_queue.alloc_tensor(T)
        asc.adv.swiglu(dst_local, src0_local, src1_local, scalar_value=1.0)
        out_queue.enque(dst_local)
        in_queue0.free_tensor(src0_local)
        in_queue1.free_tensor(src1_local)
    """
    tmp_buffer_ir = shared_tmp_buffer.to_ir() if shared_tmp_buffer is not None else None
    scalar_val_ir = _mat(scalar_value, dst_tensor.dtype).to_ir()
    cal_count_ir = _mat(cal_count).to_ir() if cal_count is not None else None
    global_builder.get_ir_builder().create_asc_SwiGLUOp(dst=dst_tensor.to_ir(), srcTensor0=src_tensor0.to_ir(),
                                                        srcTensor1=src_tensor1.to_ir(), scalarValue=scalar_val_ir,
                                                        sharedTmpBuffer=tmp_buffer_ir, calCount=cal_count_ir)
