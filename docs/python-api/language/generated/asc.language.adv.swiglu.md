# asc.language.adv.swiglu

### asc.language.adv.swiglu(dst_tensor: [LocalTensor](../core.md#localtensor), src_tensor0: [LocalTensor](../core.md#localtensor), src_tensor1: [LocalTensor](../core.md#localtensor), scalar_value: float = 1.0, shared_tmp_buffer: [LocalTensor](../core.md#localtensor) | None = None, cal_count: int | None = None) → None

SwiGLU是采用Swish作为激活函数的GLU变体。具体计算公式如下：

```text
Swish(x) = x / (1 + e^(-βx))
SwiGLU = src_tensor0 ⊗ Swish(src_tensor1)
```

**对应的Ascend C函数原型**

- 通过sharedTmpBuffer入参传入临时空间
  - 源操作数Tensor全部/部分参与计算
    ```c++
    template <typename T, bool isReuseSource = false>
    __aicore__ inline void SwiGLU(LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor0,
                                  const LocalTensor<T>& srcTensor1, const float& scalarValue,
                                  const LocalTensor<uint8_t>& sharedTmpBuffer, const uint32_t calCount)
    ```
  - 源操作数Tensor全部参与计算
    ```c++
    template <typename T, bool isReuseSource = false>
    __aicore__ inline void SwiGLU(LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor0,
                                  const LocalTensor<T>& srcTensor1, const float& scalarValue,
                                  const LocalTensor<uint8_t>& sharedTmpBuffer)
    ```
- 接口框架申请临时空间
  - 源操作数Tensor全部/部分参与计算
    ```c++
    template <typename T, bool isReuseSource = false>
    __aicore__ inline void SwiGLU(LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor0,
                                  const LocalTensor<T>& srcTensor1, const float& scalarValue,
                                  const uint32_t calCount)
    ```
  - 源操作数Tensor全部参与计算
    ```c++
    template <typename T, bool isReuseSource = false>
    __aicore__ inline void SwiGLU(LocalTensor<T>& dstTensor, LocalTensor<T>& srcTensor0,
                                  LocalTensor<T>& srcTensor1, const float& scalarValue)
    ```

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

```python
src0_local = in_queue0.deque(T)
src1_local = in_queue1.deque(T)
dst_local = out_queue.alloc_tensor(T)
asc.adv.swiglu(dst_local, src0_local, src1_local, scalar_value=1.0)
out_queue.enque(dst_local)
in_queue0.free_tensor(src0_local)
in_queue1.free_tensor(src1_local)
```
