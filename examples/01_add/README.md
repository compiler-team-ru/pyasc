# 01 — 手动插入同步流水的 Add 算子

## 概述

本样例介绍了手动插入同步流水的Add算子实现，两个向量 x 和 y 的逐元素加法：z = x + y。数据搬运和计算之间的流水同步通过手动调用 set_flag / wait_flag 指令实现，适合学习 Ascend C 流水线同步的基础原理。

## 运行环境要求

| 类别 | 要求 |
|------|------|
| AI 处理器 | Ascend 910B / 910C |
| CANN 版本 | 社区版 8.5.0.alpha001 及以上 |

注意：

- 样例支持NPU上板运行（需要NPU硬件）和仿真器模式（不需要NPU硬件）两种运行方式。仿真器模式运行方式，请参考[运行环境变量配置](../../docs/quick_start.md#envvar-config)完成配置。
- PyTorch和torch_npu的安装，请参考[样例运行验证](../../docs/quick_start.md#example-verification)。

## 样例规格

| 参数名称 | 输入/输出 | Shape | 数据类型 | 格式 |
|----------|-----------|-------|----------|------|
| x | 输入 | [8, 2048] | float32 | ND |
| y | 输入 | [8, 2048] | float32 | ND |
| z | 输出 | [8, 2048] | float32 | ND |

## 样例实现

### 整体流程

```
Global Memory (x_gm, y_gm)
    │  data_copy
    ▼
Local Memory (x_local, y_local)
    │  set_flag / wait_flag
    ▼
Vector Compute (add)
    │  set_flag / wait_flag
    ▼
Local Memory (z_local)
    │  data_copy
    ▼
Global Memory (z_gm)
```

### 关键步骤

- **数据搬入** — 使用 `asc.data_copy` 将输入从 Global Memory 搬运到 Local Memory 的 `x_local`、`y_local` 中。
- **同步等待 MTE2_V** — `asc.set_flag` 后接 `asc.wait_flag`，确保数据搬运完成后再开始计算。
- **矢量加法** — 调用 `asc.add` 对两个 LocalTensor 执行逐元素加法，结果写入 `z_local`。
- **同步等待 V_MTE3** — `asc.set_flag` 后接 `asc.wait_flag`，确保计算完成后，再将结果搬运回 Global Memory。
- **数据搬出** — 将 `z_local` 拷贝到输出 `z_gm`，并用 `asc.set_flag` / `asc.wait_flag` 标记该分块的搬运结束。

### 核心接口

| 接口 | 用途 |
|------|------|
| `asc.GlobalTensor` / `asc.LocalTensor` | 定义 Global/Local 内存张量 |
| `asc.data_copy` | 数据搬运（Global Memory↔Local Memory），支持多种搬运场景 |
| `asc.set_flag` / `asc.wait_flag` | 同一核内不同流水线之间的同步指令，必须成对使用 |
| `asc.add` | 按元素求和 |
| `asc.get_block_idx` | 获取当前核的索引，用于多核切分 |

### 分块、多核、流水线逻辑

- 多核切分
  - 使用 `USE_CORE_NUM = 8` 个核并行计算。
  - 总数据 `total_length` 按核数等分为 `block_length = total_length // USE_CORE_NUM`。
  - 每个核通过 `asc.get_block_idx() * block_length` 计算自己在 Global Memory 中的偏移量。
- 分块计算
  - 每个核内部将数据进一步切分为 `TILE_NUM = 8` 个 tile。
  - 采用双缓冲机制（`BUFFER_NUM = 2`），每个 buffer 容纳一个 tile 的数据量。
  - 每个 tile 的数据量为 `tile_length = block_length // TILE_NUM // BUFFER_NUM`。
- 流水线同步
  本样例使用**手动同步**方式，每轮迭代中显式插入 3 对同步事件：
  - `MTE2_V`：搬运入 → 矢量计算
  - `V_MTE3`：矢量计算 → 搬运出
  - `MTE3_MTE2`：搬运出 → 下一轮搬运入

  双缓冲使得搬运和计算可以在不同 buffer 间流水叠加。

## 编译执行

环境配置请参考[quick_start.md](../../docs/quick_start.md#envready)。完成环境配置后，执行如下命令可进行功能验证。

```
cd pyasc/examples/01_add
python3 add.py -r [RUN_MODE] -v [SOC_VERSION]
```

其中脚本参数说明如下：

- RUN_MODE：编译执行方式，可选择NPU仿真，NPU上板，对应参数分别为[Model/NPU]。
- SOC_VERSION：昇腾AI处理器型号，如果无法确定具体的[SOC_VERSION]，则在安装昇腾AI处理器的服务器执行npu-smi info命令进行查询，在查询到的"Name"前增加Ascend信息，例如"Name"对应取值为xxxyy，实际配置的[SOC_VERSION]值为Ascendxxxyy。

示例如下，Ascend910B1请替换为实际的AI处理器型号。

```bash
# 仿真器模式
python3 add.py -r Model -v Ascend910B1

# NPU 上板模式
python3 add.py -r NPU -v Ascend910B1
```

执行成功后输出：

```
[INFO] start process sample add.
[INFO] Sample add run success.
```
