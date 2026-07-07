# 04 — 纯 Cube 模式的 Matmul 算子

## 概述

本样例介绍了纯 Cube 模式（只有矩阵计算）下的 Matmul 算子实现：C = A × B + Bias（Bias 可通过 `ENABLE_BIAS` 开关控制，默认为 False）。通过设置 Kernel 核函数 JIT 编译参数 `matmul_cube_only=True` 启用纯 Cube 模式，仅 AIC 负责 Cube 矩阵计算，AIV 空闲不参与计算。

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
- 纯Cube模式下 `USE_CORE_NUM` 需设置为 AI Core 数量（仅 AIC），启动核数为 `tiling.used_core_num`。

## 样例规格

| 参数名称 | 输入/输出 | Shape | 数据类型 | 格式 | 是否转置 |
|----------|-----------|-------|----------|------|----------|
| a | 输入 | [128, 64] | float16 | ND | 否 |
| b | 输入 | [64, 30720] | float16 | ND | 否 |
| bias | 输入 | [1, 30720] | float32 | ND | — |
| c | 输出 | [128, 30720] | float32 | ND | — |

## 样例实现

### 整体流程

```
Global Memory (a_gm, b_gm, bias_gm)
    │  Matmul.set_tensor_a / set_tensor_b / set_bias
    ▼
Cube Unit (仅 AIC: C = A × B + Bias)
    │  Matmul.iterate_all
    ▼
Global Memory (c_gm)
    │  Matmul.end / pipe_barrier
    ▼
完成
```

### 关键步骤

- **AIC 核判断** — 通过 `asc.ascend_is_aic()` 判断当前是否为 AIC 核，仅 AIC 核执行矩阵乘计算。
- **创建 Matmul 对象** — 通过 `asc.adv.Matmul` 创建矩阵乘对象，通过 `asc.adv.MatmulType` 分别指定矩阵 A、B、C 和 Bias 的位置、格式和数据类型。
- **初始化 Matmul** — 调用 `asc.adv.register_matmul(pipe, workspace, matmul, tiling)` 完成初始化。
- **设置矩阵数据和 Bias** — 通过 `matmul.set_tensor_a/set_tensor_b` 设置左右矩阵，当 `tiling.is_bias` 为 True 时调用 `matmul.set_bias(bias_global)` 设置偏置。通过 `matmul.set_tail(tail_m, tail_n, tiling.k_a)` 设置 M、N、K 三方向的尾块大小。
- **执行与结束** — `matmul.iterate_all(c_global)` 完成计算，`matmul.end()` 结束，`asc.pipe_barrier(asc.PipeID.PIPE_ALL)` 同步。

### 核心接口

| 接口 | 用途 |
|------|------|
| `asc.adv.Matmul` | 矩阵乘高阶 API，封装 Cube 计算单元 |
| `asc.adv.MatmulType` | 定义矩阵乘操作数的存储位置、数据格式和数据类型 |
| `asc.adv.register_matmul` | 初始化 Matmul 对象，绑定 TPipe、workspace 和 Tiling 参数 |
| `matmul.set_tensor_a` / `set_tensor_b` | 设置矩阵乘的左/右操作数 |
| `matmul.set_bias` | 设置矩阵乘的 Bias 偏置操作数 |
| `matmul.set_tail` | 设置 M、N、K 方向的尾块大小，处理非对齐场景 |
| `matmul.iterate_all` | 单次调用完成所有分块的矩阵乘计算，结果写入目标 Tensor |
| `matmul.end` | 结束矩阵乘操作 |
| `asc.ascend_is_aic` | 判断当前核是否为 AIC（Cube Core） |
| `asc.pipe_barrier` | 阻塞相同流水，具有数据依赖的相同流水之间需要插入此同步 |
| `asc.lib.host.MultiCoreMatmulTiling` | Host 侧 Tiling API，获取 Matmul 分块参数 |

### 分块、多核、Tiling 逻辑

- 多核切分
  - 纯Cube模式（只有矩阵计算）下，`USE_CORE_NUM` 用于设置启动多少个Cube（AIC）实例执行，比如Ascend 910B1平台有24个Cube核，建议设置为24。
  - 每个核根据 `block_idx` 计算自己在 M、N 方向的索引和偏移量。
- Tiling 生成
  - 创建 `MultiCoreMatmulTiling` 对象。
  - 设置 A、B、C、Bias 的参数类型信息；M、N、Ka、Kb 形状信息等。
  - 调用 `get_tiling` 接口获取 `TCubeTiling` 结构体。
- 分块计算
  - M 方向按 `single_core_m` 切分，N 方向按 `single_core_n` 切分，每个核负责 `single_core_m × single_core_n` 大小的结果子矩阵。

## 编译执行

环境配置请参考[quick_start.md](../../docs/quick_start.md#envready)。完成环境配置后，执行如下命令可进行功能验证。

```
cd pyasc/examples/04_matmul_cube_only
python3 matmul_cube_only.py -r [RUN_MODE] -v [SOC_VERSION]
```

运行前请根据实际平台的 Cube 核数修改 `matmul_cube_only.py` 中的 `USE_CORE_NUM` 参数。

其中脚本参数说明如下：

- RUN_MODE：编译执行方式，可选择NPU仿真，NPU上板，对应参数分别为[Model/NPU]。
- SOC_VERSION：昇腾AI处理器型号，如果无法确定具体的[SOC_VERSION]，则在安装昇腾AI处理器的服务器执行npu-smi info命令进行查询，在查询到的"Name"前增加Ascend信息，例如"Name"对应取值为xxxyy，实际配置的[SOC_VERSION]值为Ascendxxxyy。

示例如下，Ascend910B1请替换为实际的AI处理器型号。

```bash
# 仿真器模式
python3 matmul_cube_only.py -r Model -v Ascend910B1

# NPU 上板模式
python3 matmul_cube_only.py -r NPU -v Ascend910B1
```

执行成功后输出：

```
[INFO] start process sample matmul_cube_only.
[INFO] Sample matmul_cube_only run success.
```
