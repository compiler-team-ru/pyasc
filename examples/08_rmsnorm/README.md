# 08 — RMSNorm 归一化算子

## 概述

本样例实现了 RMSNorm（Root Mean Square Normalization）归一化算子，是 LLaMA 等 Transformer 架构的标准归一化层。

计算公式：

$$\text{RMSNorm}(x) = \frac{x}{\sqrt{\text{mean}(x^2) + \varepsilon}} \cdot \gamma$$

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

## 目录结构

```text
08_rmsnorm/
├── ascendc/              # Ascend C 手写对标实现
│   ├── build.sh          # 编译脚本，生成 build/demo
│   ├── CMakeLists.txt    # CMake 构建配置
│   └── rmsnorm.asc        # Ascend C kernel 与 ACL 原生 demo 入口
├── bench_rmsnorm.py       # msprof 被测程序
├── profile_msprof.py      # msprof 性能测试脚本
├── README.md
└── rmsnorm.py             # PyASC 算子实现
```

## 样例规格

| 参数名称 | 输入/输出 | Shape | 数据类型 | 格式 |
|----------|-----------|-------|----------|------|
| x | 输入 | [B×S×H] | float32 | ND |
| gamma | 输入 | [H] | float32 | ND |
| y | 输出 | [B×S×H] | float32 | ND |

## 样例实现

### 整体流程

```
Global Memory (x_gm, gamma_gm)
    │  gamma: data_copy → TBuf（一次性加载，set_flag/wait_flag 同步）
    │  x: copy_in → data_copy → TQue.enque
    ▼
TQue (in_queue)
    │  compute: deque → asc.adv.rmsnorm(x, gamma, eps, tiling) → enque
    ▼
TQue (out_queue)
    │  copy_out: deque → data_copy
    ▼
Global Memory (y_gm)
```

### 关键步骤

- **gamma 加载** — gamma 权重通过 `TBuf(VECCALC)` 一次性加载，GM→UB 搬运后插入 `asc.set_flag(asc.HardEvent.MTE2_V)` / `asc.wait_flag(asc.HardEvent.MTE2_V)` 显式同步 MTE2 与 Vector 流水线。
- **copy_in** — 将输入数据 x 从 Global Memory 搬运到 VECIN 队列。
- **compute** — 调用 `asc.adv.rmsnorm` 高阶 API，整块通过 `asc.range()` 循环计算，尾块单独处理，通过 `asc.adv.RmsNormTiling` 传递分块参数。
- **copy_out** — 计算结果从 out_queue 搬回 Global Memory。

### 核心接口

| 接口 | 用途 |
|------|------|
| `asc.TPipe` | TPipe 统一管理 Device 端内存等资源，一个 Kernel 函数必须且只能初始化一个 TPipe 对象。 |
| `asc.TQue` | 流水任务之间通过队列完成任务间通信和同步。 |
| `asc.TBuf` | 临时变量占用的内存可以使用 TBuf 数据结构来管理。 |
| `asc.TPipe.init_buffer` | 为 TQue 等队列和 TBuf 分配内存。 |
| `asc.TQue.alloc_tensor` | 从 Que 中分配 Tensor。 |
| `asc.TQue.enque` | 将 Tensor push 到队列。 |
| `asc.TQue.deque` | 将 Tensor 从队列中取出。 |
| `asc.TQue.free_tensor` | 释放 Que 中的指定 Tensor。 |
| `asc.TBuf.get` | 从 TBuf 上获取指定长度的 Tensor。 |
| `asc.data_copy` | 支持 Local Memory 与 Global Memory 之间的数据搬运。 |
| `asc.adv.rmsnorm` | 实现对 shape 大小为 [B, S, H] 的输入数据的 RMSNorm 归一化。 |
| `asc.adv.RmsNormTiling` | 方便用户获取 RMSNorm kernel 计算时所需的 Tiling 参数。 |
| `asc.set_flag` / `asc.wait_flag` | 同一核内不同流水线之间的同步指令。 |
| `asc.get_block_idx` | 获取当前核的 index，用于多核逻辑控制及多核偏移量计算。 |

### 分块、多核、流水线逻辑

- **多核切分**：
  - 通过 `rt.device_info(RT_MODULE_TYPE_VECTOR_CORE, INFO_TYPE_CORE_NUM)` 查询 AIV 核数。
  - `effective_cores = max(1, min(total_rows, max_core_num))` — 总行数少于可用核数时实际核数等于总行数。
  - `rows_per_core = ceil_div(total_rows, effective_cores)` — 每核处理的行数。
  - 各核通过 `asc.get_block_idx() * block_length` 计算在 Global Memory 中的偏移量。
- **分块计算**：
  - `_rows_per_call(hidden_size)` — 核内每次 RmsNorm 调用处理的行数（hidden_size ≤ 512 → 每次处理 8 行、≤ 1024 → 每次处理 4 行、其他→每次处理 2 行）。
  - `full_groups = rows_per_core // max_rows` — 完整的 max_rows 行组数。
  - `rem_rows = rows_per_core % max_rows` — 剩余尾行数。
  - 整块：
    - `chunk_size = max_rows × hidden_size `— 整块元素数。
    - `aligned_rows = ((max_rows + 16 - 1) // 16) × 16` — 按 16 行对齐。
    - `RmsNormTiling` — `b_length=1`（batch 为 1）、`s_length=max_rows`（行数）、`h_length=hidden_size`（隐藏层维度）、`reciprocal_of_h_length=1.0/hidden_size`、`main_bsh_length=chunk_size`、`main_bs_length_align=aligned_rows`、`input_tail_pos=chunk_size`（无 tail）。
  - 尾块：
    - `rem_bsh = rem_rows × hidden_size` — 尾块元素数。
    - `RmsNormTiling` 中 `s_length=rem_rows`、`main_bsh_length=rem_bsh`、`main_bs_length=rem_rows`、`main_bs_length_align=((rem_rows + 16 - 1) // 16) × 16`、`input_tail_pos=rem_bsh`，其余与整块相同。
  - 双缓冲（`BUFFER_NUM = 2`）。
- **流水线同步**：TPipe + TQue 框架管理搬运同步。gamma 通过 TBuf 加载，不经过 TQue，需额外 `set_flag`/`wait_flag` 同步。

## 编译执行

环境配置请参考[quick_start.md](../../docs/quick_start.md#envready)。

```
cd pyasc/examples/08_rmsnorm
python3 rmsnorm.py -r [RUN_MODE] -v [SOC_VERSION]
```

- RUN_MODE：NPU 仿真（Model）或 NPU 上板（NPU）。
- SOC_VERSION：昇腾 AI 处理器型号。

```bash
# NPU 上板模式
python3 rmsnorm.py -r NPU -v Ascend910B4
```

执行成功后输出：

```
[INFO] start process sample rmsnorm.
[INFO] Sample rmsnorm run success.
```

## 性能测试

### 概述

- 性能测试目的：对比 RMSNorm 的 PyASC、Ascend C 和 torch_npu 实现。
- 性能测试环境：
  - NPU硬件：Ascend 910B4
  - CANN软件版本：社区版8.5.0
  - PyAsc软件版本：1.1.1
  - 输入数据类型：float32
  - 性能数据采集工具：msprof
- 统计方式：每个 shape 通过 `msprof op` 采集 `OpBasicInfo.csv` 的 Task Duration。每次运行 warmup=5（预热）+ iters=10（计时）。
- 复现命令：

  ```bash
  # 编译 Ascend C demo
  cd pyasc/examples/08_rmsnorm/ascendc
  bash build.sh

  # 运行性能测试，输出两份 CSV：
  # summary.csv：每个 shape 的PyAsc、Ascend C、torch_npu 耗时 + py/asc + py/torch 比值
  # pipeline_detail.csv：vec/scalar/mte2/mte3 分项耗时
  cd pyasc/examples/08_rmsnorm
  python3 profile_msprof.py --output ./prof_results
  ```

- 表头说明
  - shape：输入张量 shape，格式为 `(batch, seq, hidden)`。
  - elements：输入张量元素总数。
  - pyasc(us)、ascendc(us)、torch_npu(us)：对应实现的 Task Duration，单位为微秒。
  - py/asc：PyAsc 与 Ascend C 耗时比值。
  - py/torch：PyAsc 与 torch_npu 耗时比值，小于 1 表示 PyAsc 更快。

- 性能测试结果

  | shape | elements | pyasc(us) | ascendc(us) | torch_npu(us) | py/asc | py/torch |
  | --- | --- | --- | --- | --- | --- | --- |
  | (2,64,256) | 32K | **4.40** | 5.62 | 7.64 | 0.783 | 0.576 |
  | (2,64,512) | 64K | **4.58** | 6.02 | 7.68 | 0.761 | 0.596 |
  | (2,128,512) | 128K | **5.04** | 7.18 | 9.04 | 0.702 | 0.558 |
  | (2,256,512) | 256K | **6.30** | 7.50 | 9.14 | 0.840 | 0.689 |
  | (2,512,512) | 512K | **8.66** | 10.74 | 10.22 | 0.806 | 0.847 |
  | (2,1024,512) | 1M | **12.66** | 15.66 | 13.40 | 0.808 | 0.945 |
  | (2,1024,1024) | 2M | 19.40 | 21.78 | **20.02** | 0.891 | 0.969 |
  | (2,2048,512) | 2M | 21.40 | 24.76 | **19.64** | 0.864 | 1.090 |
  | (2,2048,1024) | 4M | 35.54 | 38.60 | **30.68** | 0.921 | 1.159 |

- 流水耗时明细

  | shape | backend | Task Duration(us) | Vec(us) | Scalar(us) | MTE2(us) | MTE3(us) |
  | --- | --- | --- | --- | --- | --- | --- |
  | (2,64,256) | pyasc | **4.40** | 0.42 | **3.16** | **0.83** | **0.21** |
  | (2,64,256) | ascendc | 5.62 | **0.32** | 4.50 | 1.64 | 0.71 |
  | (2,64,256) | torch_npu | 7.64 | 1.22 | 5.94 | 2.09 | 0.37 |
  | (2,64,512) | pyasc | **4.58** | 0.54 | **3.09** | **0.85** | 0.34 |
  | (2,64,512) | ascendc | 6.02 | **0.41** | 4.78 | 1.55 | **0.18** |
  | (2,64,512) | torch_npu | 7.68 | 1.38 | 5.46 | 2.36 | 0.32 |
  | (2,128,512) | pyasc | **5.04** | 0.54 | **3.34** | **0.81** | **0.23** |
  | (2,128,512) | ascendc | 7.18 | **0.53** | 5.69 | 3.35 | 0.67 |
  | (2,128,512) | torch_npu | 9.04 | 1.52 | 7.01 | 3.11 | 0.54 |
  | (2,256,512) | pyasc | **6.30** | 1.07 | **3.81** | **2.13** | 0.82 |
  | (2,256,512) | ascendc | 7.50 | **0.98** | 5.48 | 2.47 | **0.69** |
  | (2,256,512) | torch_npu | 9.14 | 1.80 | 6.41 | 3.09 | 0.76 |
  | (2,512,512) | pyasc | **8.66** | 2.10 | **4.45** | **2.97** | 1.39 |
  | (2,512,512) | ascendc | 10.74 | **1.92** | 7.41 | 3.49 | **1.23** |
  | (2,512,512) | torch_npu | 10.22 | 2.67 | 6.99 | 3.74 | 1.77 |
  | (2,1024,512) | pyasc | **12.66** | 3.71 | **6.80** | **3.42** | 2.07 |
  | (2,1024,512) | ascendc | 15.66 | **3.59** | 9.95 | 5.20 | **2.04** |
  | (2,1024,512) | torch_npu | 13.40 | 4.14 | 7.72 | 6.02 | 2.78 |
  | (2,1024,1024) | pyasc | **19.40** | 7.38 | **7.86** | **6.63** | 3.77 |
  | (2,1024,1024) | ascendc | 21.78 | 7.40 | 10.79 | 7.46 | **3.41** |
  | (2,1024,1024) | torch_npu | 20.02 | **7.12** | 8.51 | 8.58 | 5.00 |
  | (2,2048,512) | pyasc | 21.40 | **6.88** | 10.95 | **6.25** | **3.34** |
  | (2,2048,512) | ascendc | 24.76 | 6.89 | 13.85 | 7.84 | 3.72 |
  | (2,2048,512) | torch_npu | **19.64** | 7.03 | **8.30** | 8.95 | 4.64 |
  | (2,2048,1024) | pyasc | 35.54 | 14.72 | 13.16 | **12.31** | **7.50** |
  | (2,2048,1024) | ascendc | 38.60 | 14.71 | 17.03 | 12.85 | 7.53 |
  | (2,2048,1024) | torch_npu | **30.68** | **13.32** | **11.92** | 14.72 | 8.20 |

- 结果分析
  - PyAsc 在所有 shape 上快于 Ascend C（py/asc 0.702x ~ 0.921x），差距全部来自 Scalar 和 MTE，Vec 时间一致。PyAsc 的 JIT 编译将 `block_length`、`hidden_size`、`max_rows`、`eps` 等参数特化为 `ConstExpr`（编译期常量），bisheng 编译器可做更激进的常量折叠和指令排布优化，在编译期完成地址计算和循环控制，消除运行时 Scalar 指令；Ascend C 使用运行时变量，需在 Scalar 引擎上动态计算。MTE 差距同样源于编译期常量让编译器生成更优的 DataCopy 指令排布。
  - 如果将 Ascend C 的运行时参数（totalRows、hiddenSize、totalRows 等）全部替换为字面量，即可追上 PyAsc。
  - 小 shape 时 PyAsc 快于 torch_npu（py/torch 0.558x ~ 0.847x）：PyAsc 和 Ascend C 使用 40 核，torch_npu 仅 32~37 核，单核数据量更大，导致 Vec 和 Scalar 时间明显偏高。2M 时 torch_npu 也达到 40 核，三方 Vec 基本一致，整体耗时持平（py/torch 0.969x ~ 1.090x）。4M 时 torch_npu 反超（py/torch 1.159x），得益于手写指令排布和流水线调度在大数据量下优于 PyAsc。
