# 03 — MIX 模式的 Matmul 算子

## 概述

本样例介绍了 MIX 模式（包含矩阵计算和矢量计算）下的 Matmul 算子实现：C = A × B。通过 Ascend C Python 提供的一组 Matmul 高阶 API，方便用户快速实现 Matmul 矩阵乘法的运算操作。MIX 模式下，AIC 负责 Cube 矩阵计算，AIV 负责 Vector 矢量计算。

## 运行环境要求

| 类别 | 要求 |
|------|------|
| AI 处理器 | Ascend 910B / 910C |
| CANN 版本 | 社区版 8.5.0.alpha001 及以上 |

注意：

- 样例支持NPU上板运行（需要NPU硬件）和仿真器模式（不需要NPU硬件）两种运行方式。仿真器模式运行方式，请参考[运行环境变量配置](../../docs/quick_start.md#envvar-config)完成配置。
- PyTorch和torch_npu的安装，请参考[样例运行验证](../../docs/quick_start.md#example-verification)。
- MIX 模式下 `USE_CORE_NUM` 需设置为 AI Core 数量的 2 倍，启动核数为 `USE_CORE_NUM // 2`。

## 样例规格

| 参数名称 | 输入/输出 | Shape | 数据类型 | 格式 | 是否转置 |
|----------|-----------|-------|----------|------|----------|
| a | 输入 | [512, 512] | float16 | ND | 否 |
| b | 输入 | [512, 1024] | float16 | ND | 否 |
| c | 输出 | [512, 1024] | float32 | ND | — |

## 样例实现

### 整体流程

```
Global Memory (a_gm, b_gm)
    │  Matmul.set_tensor_a / set_tensor_b
    ▼
Cube Unit (AIC: 矩阵乘 C = A × B)
    │  Matmul.iterate_all
    ▼
Global Memory (c_gm)
    │  Matmul.end / pipe_barrier
    ▼
结果校验
```

### 关键步骤

- **创建 Matmul 对象** — 通过 `asc.adv.Matmul` 创建矩阵乘对象，通过 `asc.adv.MatmulType` 分别指定矩阵 A、B、C 的位置（`TPosition.GM`）、数据格式（`CubeFormat.ND`）和数据类型。
- **初始化 Matmul** — 调用 `asc.adv.register_matmul(pipe, workspace, matmul, tiling)` 完成 Matmul 对象初始化，绑定 TPipe 和 workspace 空间。
- **设置矩阵数据** — 通过 `matmul.set_tensor_a(a_global)` 和 `matmul.set_tensor_b(b_global)` 设置左右矩阵，通过 `matmul.set_tail(tail_m, tail_n)` 设置 M、N 方向的尾块大小。
- **执行矩阵乘** — 调用 `matmul.iterate_all(c_global)` 完成全量矩阵乘法计算，结果直接写入 Global Memory 的 C 矩阵。
- **结束计算** — 调用 `matmul.end()` 结束矩阵乘操作，随后 `asc.pipe_barrier(asc.PipeID.PIPE_ALL)` 同步所有流水线。

### 核心接口

| 接口 | 用途 |
|------|------|
| `asc.adv.Matmul` | 矩阵乘高阶 API，封装 Cube 计算单元 |
| `asc.adv.MatmulType` | 定义矩阵乘操作数的存储位置、数据格式和数据类型 |
| `asc.adv.register_matmul` | 初始化 Matmul 对象，绑定 TPipe、workspace 和 Tiling 参数 |
| `matmul.set_tensor_a` / `set_tensor_b` | 设置矩阵乘的左/右操作数 |
| `matmul.set_tail` | 设置 M、N 方向的尾块大小 |
| `matmul.iterate_all` | 单次调用完成所有分块的矩阵乘计算，结果写入目标 Tensor |
| `matmul.end` | 结束矩阵乘操作 |
| `asc.pipe_barrier` | 阻塞相同流水，具有数据依赖的相同流水之间需要插入此同步 |
| `asc.lib.host.MultiCoreMatmulTiling` | Host 侧 Tiling API，获取 Matmul 分块参数 |

### 分块、多核、Tiling 逻辑

- 多核切分
  - MIX 模式（包含矩阵计算和矢量计算）下，启动时，按照AIV和AIC组合启动，`USE_CORE_NUM`用于设置启动多少个组合执行。比如Ascend 910B1平台有48个Vector核和24个Cube核，一个组合是2个Vector核和1个Cube核。本样例设置`USE_CORE_NUM = 48`，实际启动组合为 `48 // 2 = 24` 个组合，即48个Vector核和24个Cube核。注意：该场景下设置的的`USE_CORE_NUM`逻辑核的核数不能超过物理核（2个Vector核和1个Cube核组合为1个物理核）的核数。
  - 每个核根据 `block_idx` 计算自己在 M、N 方向的索引和偏移量，在 Global Memory 中定位自己的数据分块。
- Tiling 生成
  - 创建 `MultiCoreMatmulTiling` 对象。
  - 设置 A、B、C、Bias 的参数类型信息；M、N、Ka、Kb 形状信息等。
  - 调用 `get_tiling` 接口获取 `TCubeTiling` 结构体，包含分块大小（base_m/base_n/base_k）、单核计算量（single_core_m/single_core_n）等参数。
- 分块计算
  - Tiling 数据在 Global Memory 上按单核计算量切分，通过 `m_single_blocks = tiling.m.ceildiv(tiling.single_core_m)` 等方式得到分块数。

## 编译执行

环境配置请参考[quick_start.md](../../docs/quick_start.md#envready)。完成环境配置后，执行如下命令可进行功能验证。

```
cd pyasc/examples/03_matmul_mix
python3 matmul_mix.py -r [RUN_MODE] -v [SOC_VERSION]
```

运行前请根据实际平台的 Vector 核数修改 `matmul_mix.py` 中的 `USE_CORE_NUM` 参数。

其中脚本参数说明如下：

- RUN_MODE：编译执行方式，可选择NPU仿真，NPU上板，对应参数分别为[Model/NPU]。
- SOC_VERSION：昇腾AI处理器型号，如果无法确定具体的[SOC_VERSION]，则在安装昇腾AI处理器的服务器执行npu-smi info命令进行查询，在查询到的"Name"前增加Ascend信息，例如"Name"对应取值为xxxyy，实际配置的[SOC_VERSION]值为Ascendxxxyy。

示例如下，Ascend910B1请替换为实际的AI处理器型号。

```bash
# 仿真器模式
python3 matmul_mix.py -r Model -v Ascend910B1

# NPU 上板模式
python3 matmul_mix.py -r NPU -v Ascend910B1
```

执行成功后输出：

```
[INFO] start process sample matmul_mix.
[INFO] Sample matmul_mix run success.
```
