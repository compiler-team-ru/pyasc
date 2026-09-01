# asc.language.adv.softmax_flash_v2

### asc.language.adv.softmax_flash_v2(dst_tensor: [LocalTensor](../core.md#localtensor), exp_sum_tensor: [LocalTensor](../core.md#localtensor), max_tensor: [LocalTensor](../core.md#localtensor), src_tensor: [LocalTensor](../core.md#localtensor), exp_max_tensor: [LocalTensor](../core.md#localtensor), in_exp_sum_tensor: [LocalTensor](../core.md#localtensor), in_max_tensor: [LocalTensor](../core.md#localtensor), tiling: SoftmaxTiling, softmax_shape_info: SoftMaxShapeInfo | None = None, shared_tmp_buffer: [LocalTensor](../core.md#localtensor) | None = None, out_reduce_max: [LocalTensor](../core.md#localtensor) | None = None, is_update: bool = False, is_reuse_source: bool = False, is_basic_block: bool = False, is_data_format_nz: bool = False, config: SoftmaxConfig | None = None) → None

SoftmaxFlash增强版本，对应FlashAttention-2算法。

**对应的Ascend C函数原型**

- 接口框架申请临时空间
  - LocalTensor的数据类型相同，不输出ReduceMax
    ```c++
    template <typename T, bool isUpdate = false, bool isReuseSource = false,
              bool isBasicBlock = false, bool isDataFormatNZ = false,
              const SoftmaxConfig& config = SOFTMAX_DEFAULT_CFG>
    __aicore__ inline void SoftmaxFlashV2(
        const LocalTensor<T>& dstTensor, const LocalTensor<T>& expSumTensor,
        const LocalTensor<T>& maxTensor, const LocalTensor<T>& srcTensor,
        const LocalTensor<T>& expMaxTensor, const LocalTensor<T>& inExpSumTensor,
        const LocalTensor<T>& inMaxTensor, const SoftMaxTiling& tiling,
        const SoftMaxShapeInfo& softmaxShapeInfo = {})
    ```
  - LocalTensor的数据类型相同，且输出ReduceMax
    ```c++
    template <typename T, bool isUpdate = false, bool isReuseSource = false,
              bool isBasicBlock = false, bool isDataFormatNZ = false,
              const SoftmaxConfig& config = SOFTMAX_DEFAULT_CFG>
    __aicore__ inline void SoftmaxFlashV2(
        const LocalTensor<T>& dstTensor, const LocalTensor<T>& outReduceMax,
        const LocalTensor<T>& outExpSum, const LocalTensor<T>& outMax,
        const LocalTensor<T>& srcTensor, const LocalTensor<T>& outExpMax,
        const LocalTensor<T>& inExpSum, const LocalTensor<T>& inMax,
        const SoftMaxTiling& tiling, const SoftMaxShapeInfo& softmaxShapeInfo = {})
    ```
  - LocalTensor的数据类型不同，不输出ReduceMax
    ```c++
    template <typename T, bool isUpdate = false, bool isReuseSource = false,
              bool isBasicBlock = false, bool isDataFormatNZ = false,
              const SoftmaxConfig& config = SOFTMAX_DEFAULT_CFG>
    __aicore__ inline void SoftmaxFlashV2(
        const LocalTensor<half>& dstTensor, const LocalTensor<float>& expSumTensor,
        const LocalTensor<float>& maxTensor, const LocalTensor<half>& srcTensor,
        const LocalTensor<half>& expMaxTensor, const LocalTensor<float>& inExpSumTensor,
        const LocalTensor<float>& inMaxTensor, const SoftMaxTiling& tiling,
        const SoftMaxShapeInfo& softmaxShapeInfo = {})
    ```
- 通过sharedTmpBuffer入参传入临时空间
  - LocalTensor的数据类型相同，不输出ReduceMax
    ```c++
    template <typename T, bool isUpdate = false, bool isReuseSource = false,
              bool isBasicBlock = false, bool isDataFormatNZ = false,
              const SoftmaxConfig& config = SOFTMAX_DEFAULT_CFG>
    __aicore__ inline void SoftmaxFlashV2(
        const LocalTensor<T>& dstTensor, const LocalTensor<T>& outExpSum,
        const LocalTensor<T>& outMax, const LocalTensor<T>& srcTensor,
        const LocalTensor<T>& outExpMax, const LocalTensor<T>& inExpSum,
        const LocalTensor<T>& inMax, const LocalTensor<uint8_t>& sharedTmpBuffer,
        const SoftMaxTiling& tiling, const SoftMaxShapeInfo& softmaxShapeInfo = {})
    ```
  - LocalTensor的数据类型相同，且输出ReduceMax
    ```c++
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
    ```
  - LocalTensor的数据类型不同，不输出ReduceMax
    ```c++
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
    ```

**参数说明**

- dst_tensor：目的操作数，shape与源操作数src_tensor一致。
- exp_sum_tensor：目的操作数，用于保存softmax计算过程中reducesum的结果。
  - 除config配置为非拓展模式（SoftmaxMode.SOFTMAX_OUTPUT_WITHOUT_BRC）的场景外，
  > exp_sum_tensor的last轴长度固定为32Byte，即一个datablock长度。该datablock中的所有数据为同一个值，
  > 比如float16数据类型下，该datablock中的16个数均为相同的reducesum值。
  - 非last轴的长度与dst_tensor保持一致。
- max_tensor：目的操作数，用于保存softmax计算过程中reducemax的结果。
  - 除config配置为非拓展模式（SoftmaxMode.SOFTMAX_OUTPUT_WITHOUT_BRC）的场景外，
  > max_tensor的last轴长度固定为32Byte，即一个datablock长度。该datablock中的所有数据为同一个值，
  > 比如float16数据类型下，该datablock中的16个数均为相同的reducemax值。
  - 非last轴的长度与dst_tensor保持一致。
- src_tensor：源操作数，last轴长度需要32Byte对齐。
- exp_max_tensor：目的操作数，用于保存in_max_tensor与reducemax差值的e的指数幂结果。
  - 除config配置为非拓展模式（SoftmaxMode.SOFTMAX_OUTPUT_WITHOUT_BRC）的场景外，
  > exp_max_tensor的last轴长度固定为32Byte，即一个datablock长度。该datablock中的所有数据为同一个值，
  > 比如float16数据类型下，该datablock中的16个数均为相同的值。
  - 非last轴的长度与dst_tensor保持一致。
- in_exp_sum_tensor：源操作数，softmax计算所需的sum值。
  - 除config配置为非拓展模式（SoftmaxMode.SOFTMAX_OUTPUT_WITHOUT_BRC）的场景外，
  > in_exp_sum_tensor的last轴长度固定为32Byte，即一个datablock长度。该datablock中的所有数据为同一个值，
  > 比如float16数据类型下，该datablock中的16个数均为相同的值。
  - 非last轴的长度与dst_tensor保持一致。
- in_max_tensor：源操作数，softmax计算所需的max值。
  - 除config配置为非拓展模式（SoftmaxMode.SOFTMAX_OUTPUT_WITHOUT_BRC）的场景外，
  > in_max_tensor的last轴长度固定为32Byte，即一个datablock长度。该datablock中的所有数据为同一个值，
  > 比如float16数据类型下，该datablock中的16个数均为相同的值。
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
  > 配置为SoftmaxMode.SOFTMAX_NORMAL时，接口不执行计算，也不保存输出；
  - is_update为False时，不输出out_reduce_max；
  - 除out_reduce_max外，其余输出的计算结果与未指定out_reduce_max时相同。
- 操作数地址对齐要求请参见通用地址对齐约束。
- 不支持shared_tmp_buffer与源操作数或目的操作数地址重叠。
- 当softmax_shape_info中的src_m != ori_src_m或src_k != ori_src_k时，需要将GM上的原始输入
  (ori_src_m, ori_src_k)沿M轴或K轴补齐到(src_m, src_k)。补齐数据会参与部分运算；
  复用输入输出时，结果会覆盖src_tensor中的补齐数据，否则会覆盖dst_tensor中与src_tensor补齐位置对应的数据。

**调用示例**

```python
dst_half = asc.LocalTensor(dtype=asc.float16)
state_half = asc.LocalTensor(dtype=asc.float16)
src_half = asc.LocalTensor(dtype=asc.float16)
exp_max_half = asc.LocalTensor(dtype=asc.float16)
tiling = asc.adv.SoftmaxTiling(src_m=8, src_k=512, src_size=4096)
shape = asc.adv.SoftMaxShapeInfo(8, 512, 8, 512)
asc.adv.softmax_flash_v2(
    dst_half, state_half, state_half, src_half, exp_max_half,
    state_half, state_half, tiling, shape)
```
