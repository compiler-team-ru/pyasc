# 02 — 框架自动插入流水同步的 Add 算子

## 概述

本样例介绍了通过 Ascend C 框架（TPipe/TQue）自动插入流水同步的 Add 算子实现：两个向量 x 和 y 的逐元素加法 z = x + y。计算过程中的流水同步通过 Ascend C 框架自动实现，将数据搬运和计算拆分为 copy_in、compute、copy_out 三个子函数，通过 TQue 的 enque/deque 操作自动保证流水同步，适合学习 Ascend C 框架编程模式。

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

| 参数名称 | 输入/输出 | Shape | 数据类型 | 格式 |
|----------|-----------|-------|----------|------|
| x | 输入 | [8, 2048] | float32 | ND |
| y | 输入 | [8, 2048] | float32 | ND |
| z | 输出 | [8, 2048] | float32 | ND |

## 样例实现

### 整体流程

```
Global Memory (x_gm, y_gm)
    │  copy_in: data_copy → TQue.enque
    ▼
TQue (in_queue_x, in_queue_y)
    │  compute: TQue.deque → add → TQue.enque
    ▼
TQue (out_queue_z)
    │  copy_out: TQue.deque → data_copy
    ▼
Global Memory (z_gm)
```

### 关键步骤

- **copy_in** — 使用 `asc.data_copy` 将输入从 Global Memory 搬运到 Local Memory，然后通过 `enque` 将 Tensor 推入队列。计算过程中的 Local Memory 通过 `TQue.alloc_tensor` 接口获取。
- **compute** — 从队列中 `deque` 取出 Tensor，调用 `asc.add` 执行逐元素加法，结果通过 `enque` 推入输出队列，最后 `free_tensor` 释放输入 Tensor。
- **copy_out** — 从输出队列 `deque` 取出结果 Tensor，通过 `asc.data_copy` 搬运回 Global Memory，最后 `free_tensor` 释放。

在此过程中，Ascend C 框架会自动插入对应的同步事件，无需调用 set_flag/wait_flag 设置同步。

### 核心接口

| 接口 | 用途 |
|------|------|
| `asc.TPipe` | 统一管理 Device 端内存和同步事件资源，一个 Kernel 函数必须且只能初始化一个 TPipe 对象 |
| `asc.TQue` | 管理流水任务之间的队列通信和同步，支持 alloc_tensor / enque / deque / free_tensor 操作 |
| `asc.data_copy` | 数据搬运（Global Memory↔Local Memory），支持多种搬运场景 |
| `asc.add` | 按元素求和 |
| `asc.get_block_idx` | 获取当前核的索引，用于多核切分 |

### 分块、多核、流水线逻辑

- 多核切分
  - 使用 `USE_CORE_NUM = 8` 个核并行计算。
  - 总数据 `total_length` 按核数等分为 `block_length = (total_length + USE_CORE_NUM - 1) // USE_CORE_NUM`。
  - 每个核通过 `asc.get_block_idx() * block_length` 计算自己在 Global Memory 中的偏移量。
- 分块计算
  - 每个核内部将数据进一步切分为 `TILE_NUM = 8` 个 tile。
  - 采用双缓冲机制（`BUFFER_NUM = 2`），`tile_length = block_length // TILE_NUM // BUFFER_NUM`。
- 流水线同步
  - 本样例使用 Ascend C 框架自动同步方式。TPipe 通过 init_buffer 接口为 TQue/TBuf 分配内存，在 enque/deque 操作过程中自动插入对应的同步事件。

## 编译执行

环境配置请参考[quick_start.md](../../docs/quick_start.md#envready)。完成环境配置后，执行如下命令可进行功能验证。

```
cd pyasc/examples/02_add_framework
python3 add_framework.py -r [RUN_MODE] -v [SOC_VERSION]
```

其中脚本参数说明如下：

- RUN_MODE：编译执行方式，可选择NPU仿真，NPU上板，对应参数分别为[Model/NPU]。
- SOC_VERSION：昇腾AI处理器型号，如果无法确定具体的[SOC_VERSION]，则在安装昇腾AI处理器的服务器执行npu-smi info命令进行查询，在查询到的"Name"前增加Ascend信息，例如"Name"对应取值为xxxyy，实际配置的[SOC_VERSION]值为Ascendxxxyy。

示例如下，Ascend910B1请替换为实际的AI处理器型号。

```bash
# 仿真器模式
python3 add_framework.py -r Model -v Ascend910B1

# NPU 上板模式
python3 add_framework.py -r NPU -v Ascend910B1
```

执行成功后输出：

```
[INFO] start process sample add_framework.
[INFO] Sample add_framework run success.
```
