# 05 — MatmulLeakyRelu 融合算子

## 概述

本样例介绍了 MatmulLeakyRelu 融合算子的实现：C = LeakyReLU(A × B + Bias, alpha)。通过 Matmul 高阶 API 与 LeakyReLU 基础 API 的组合，在单次 Kernel 调用中完成矩阵乘和激活计算，避免中间结果的 Global Memory 回传，提升性能。MatmulLeakyRelu 的计算公式为：

```
C = A * B + Bias
C = C >= 0 ? C : C * alpha
```

## 运行环境要求

| 类别 | 要求 |
|------|------|
| AI 处理器 | Ascend 910B / 910C |
| CANN 版本 | 社区版 8.5.0.alpha001 及以上 |
| Python | 3.9 ~ 3.12 |
| PyTorch | 2.7.1 |
| torch_npu | 7.3.0 |

注意：

- 样例支持NPU上板运行（需要NPU硬件）和仿真器模式（不需要NPU硬件）两种运行方式。仿真器模式运行方式，请参考[运行环境变量配置](../../docs/quick_start.md#envvar-config)完成配置。
- PyTorch和torch_npu的安装，请参考[样例运行验证](../../docs/quick_start.md#example-verification)。

## 样例规格

| 参数名称 | 输入/输出 | Shape | 数据类型 | 格式 | 是否转置 |
|----------|-----------|-------|----------|------|----------|
| a | 输入 | [1024, 256] | float16 | ND | 否 |
| b | 输入 | [256, 640] | float16 | ND | 否 |
| bias | 输入 | [1, 640] | float32 | ND | — |
| alpha | 输入（标量） | — | float32 | — | — |
| c | 输出 | [1024, 640] | float32 | ND | — |

## 样例实现

### 整体流程

```
Global Memory (a_gm, b_gm, bias_gm)
    │  Matmul.set_tensor_a / set_tensor_b / set_bias
    ▼
Cube Unit (矩阵乘: C_temp = A × B + Bias)
    │  Matmul.get_tensor_c (输出到 VECCALC)
    ▼
Vector Compute (LeakyReLU)
    │  TQue.enque → TQue.deque
    ▼
Global Memory (c_gm)
    │  data_copy (带 DataCopyParams)
    ▼
结果校验
```

### 关键步骤

- **创建 Matmul 对象** — 矩阵 C 的输出位置设置为 `TPosition.VECCALC`，使得矩阵乘结果保留在 Local Memory 中供后续激活函数直接消费。
- **迭代计算与融合** — 使用 `with matmul.iterate() as count` 逐块迭代。`matmul.get_tensor_c(relu_out_local, en_sequential_write=True)` 获取当前块的矩阵乘结果，`asc.leaky_relu` 就地计算激活，结果通过 TQue 传递给 copy_out 阶段。
- **带参数的数据搬出** — 矩阵 C 以 `base_m × base_n` 的块为单位输出，使用 `asc.DataCopyParams` 指定 block_count、block_len、src_stride、dst_stride，配合 `asc.data_copy(repeat_params=params)` 将分块结果写回 Global Memory。
- **结束计算** — `matmul.end()` 结束矩阵乘操作，`asc.pipe_barrier` 同步。

### 核心接口

| 接口 | 用途 |
|------|------|
| `asc.adv.Matmul` | 矩阵乘高阶 API，支持结果输出到 VECCALC 供融合算子消费 |
| `asc.adv.MatmulType` | 定义矩阵乘操作数的存储位置、数据格式和数据类型 |
| `asc.adv.register_matmul` | 初始化 Matmul 对象，绑定 TPipe、workspace 和 Tiling 参数 |
| `matmul.set_tensor_a` / `set_tensor_b` | 设置矩阵乘的左/右操作数 |
| `matmul.set_bias` | 设置矩阵乘的 Bias 偏置 |
| `matmul.iterate` | 每调用一次计算出一块 baseM × baseN 的 C 矩阵，支持与矢量计算融合 |
| `matmul.get_tensor_c` | 获取当前迭代块的矩阵乘结果 Tensor |
| `asc.leaky_relu` | LeakyReLU 激活函数：dst = src >= 0 ? src : src * alpha |
| `asc.data_copy` (repeat_params) | 带 DataCopyParams 的数据搬运，按 block 步长将 VECCALC 结果写回 GM |
| `asc.DataCopyParams` | 数据搬运参数：block_count、block_len、src_stride、dst_stride |
| `matmul.end` | 结束矩阵乘操作 |
| `asc.pipe_barrier` | 阻塞相同流水，具有数据依赖的相同流水之间需要插入此同步 |

### 分块、多核、Tiling 逻辑

- 平台配置
  - 通过 `set_dim(2)` 指定 2 个核参与计算，通过 `set_fix_split(256, 128, -1)` 设置固定的 M、N 方向分块大小。
- Tiling 生成
  - 创建 `MultiCoreMatmulTiling` 对象，设置 A/B/C/Bias 类型信息和 shape `[1024, 640, 256]`，矩阵 C 输出位置设为 `VECCALC`。
  - 通过 `set_traverse(host.MatrixTraverse.FIRSTM)` 设置遍历方式为先 M 后 N。
- 分块计算
  - `base_m × base_n` 为每次迭代的计算粒度，`single_core_m // base_m` 个 M 方向轮次和 N 方向轮次共同决定每个核的总迭代次数。

## 编译执行

环境配置请参考[quick_start.md](../../docs/quick_start.md#envready)。完成环境配置后，执行如下命令可进行功能验证。

```
cd pyasc/examples/05_matmul_leakyrelu
python3 matmul_leakyrelu.py -r [RUN_MODE] -v [SOC_VERSION]
```

其中脚本参数说明如下：

- RUN_MODE：编译执行方式，可选择NPU仿真，NPU上板，对应参数分别为[Model/NPU]。
- SOC_VERSION：昇腾AI处理器型号，如果无法确定具体的[SOC_VERSION]，则在安装昇腾AI处理器的服务器执行npu-smi info命令进行查询，在查询到的"Name"前增加Ascend信息，例如"Name"对应取值为xxxyy，实际配置的[SOC_VERSION]值为Ascendxxxyy。

示例如下，Ascend910B1请替换为实际的AI处理器型号。

```bash
# 仿真器模式
python3 matmul_leakyrelu.py -r Model -v Ascend910B1

# NPU 上板模式
python3 matmul_leakyrelu.py -r NPU -v Ascend910B1
```

执行成功后输出：

```
[INFO] start process sample matmul_leakyrelu.
[INFO] Sample matmul_leakyrelu run success.
```
