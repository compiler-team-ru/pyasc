# 06 — GELU 激活算子

## 概述

本样例实现了 GELU（Gaussian Error Linear Unit）激活函数算子。GELU 是 BERT、GPT 系列 Transformer 模型的常用激活函数。当前实现采用 tanh 近似公式，通过基础向量算子组合完成计算。

计算公式（tanh 近似）：$GELU(x) \approx 0.5 \cdot x \cdot \left(1 + \tanh\left((2/\pi)^{1/2} \cdot \left(x + 0.044715 x^3\right)\right)\right)$

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
| y | 输出 | [8, 2048] | float32 | ND |

输入范围为 [-2, 2]。

## 样例实现

### 整体流程

```
Global Memory (x_gm)
    │  copy_in: data_copy → TQue.enque
    ▼
TQue (in_queue)
    │  compute: deque → mul(x²) → PIPE_V → mul(x³) → PIPE_V → muls(GELU_CUBIC_COEFF) → PIPE_V
    │                 → add(+x) → PIPE_V → muls(GELU_TANH_SCALE) → PIPE_V → tanh → PIPE_V
    │                 → adds(+1) → PIPE_V → muls(0.5) → PIPE_V → mul(x·t) → enque
    ▼
TQue (out_queue)
    │  copy_out: TQue.deque → data_copy
    ▼
Global Memory (y_gm)
```

### 关键步骤

- **copy_in** — 将输入数据从 Global Memory 搬运到 VECIN 队列。
- **compute** — 通过九步组合计算 GELU，相邻依赖的 Vector 计算之间插入 `asc.pipe_barrier(asc.PipeID.PIPE_V)`，中间结果复用同一块 TBuf：
  - `asc.mul(tmp, x, x)` — 计算 x²
  - `asc.mul(tmp, tmp, x)` — 计算 x³
  - `asc.muls(tmp, tmp, 0.044715)` — GELU_CUBIC_COEFF · x³
  - `asc.add(tmp, tmp, x)` — x + GELU_CUBIC_COEFF · x³
  - `asc.muls(tmp, tmp, (2/π)^(1/2))` — GELU_TANH_SCALE · (x + GELU_CUBIC_COEFF · x³)
  - `asc.adv.tanh(tmp, tmp, count=tile_length)` — tanh(...)
  - `asc.adds(tmp, tmp, 1.0)` — 1 + tanh(...)
  - `asc.muls(tmp, tmp, 0.5)` — 0.5 · (1 + tanh(...))
  - `asc.mul(y, x, tmp)` — x · 0.5 · (1 + tanh(...))
- **copy_out** — 将计算结果从 out_queue 搬回 Global Memory。

### 核心接口

| 接口 | 用途 |
|------|------|
| `asc.TPipe` | TPipe用于统一管理Device端内存等资源，一个Kernel函数必须且只能初始化一个TPipe对象。 |
| `asc.TQue` | 流水任务之间通过队列（Queue）完成任务间通信和同步。TQue是用来执行队列相关操作、管理相关资源的数据结构。TQue继承自TQueBind父类。 |
| `asc.TBuf` | 这些临时变量占用的内存可以使用TBuf数据结构来管理，存储位置通过模板参数来设置，可以设置为不同的TPosition逻辑位置。 |
| `asc.TPipe.init_buffer` | 用于为TQue等队列和TBuf分配内存。 |
| `asc.TQue.alloc_tensor` | 从Que中分配Tensor，Tensor所占大小为InitBuffer时设置的每块内存长度。 |
| `asc.TQue.enque` | 将Tensor push到队列。 |
| `asc.TQue.deque` | 将Tensor从队列中取出，用于后续处理。 |
| `asc.TQue.free_tensor` | 释放Que中的指定Tensor。 |
| `asc.TBuf.get` | 从TBuf上获取指定长度的Tensor，或者获取全部长度的Tensor。 |
| `asc.data_copy` | DataCopy系列接口提供全面的数据搬运功能，支持多种数据搬运场景，并可在搬运过程中实现随路格式转换和量化激活等操作。 该接口支持Local Memory与Global Memory之间的数据搬运，以及Local Memory内部的数据搬运。 |
| `asc.mul` | 按元素求积。 |
| `asc.muls` | 矢量内每个元素与标量求积。 |
| `asc.add` | 按元素求和。 |
| `asc.adds` | 矢量内每个元素与标量求和。 |
| `asc.adv.tanh` | 按元素做逻辑回归 Tanh。 |
| `asc.pipe_barrier` | 阻塞相同流水，具有数据依赖的相同流水之间需要插入此同步。 |
| `asc.get_block_idx` | 获取当前核的index，用于代码内部的多核逻辑控制及多核偏移量计算等。 |

### 分块、多核、流水线逻辑

- 多核切分
  - 最多使用 `USE_CORE_NUM = 8` 个核并行计算，实际核数从 `CORE_CANDIDATES = (1, 2, 4, USE_CORE_NUM)` 中选择，避免小数据量使用过多核。
  - 总数据 `total_length` 先按搬运友好的基本块估算所需核数，再计算 `block_length`。
  - 每个核通过 `asc.get_block_idx() * block_length` 计算自己在 Global Memory 中的偏移量。
- 分块计算
  - 通过 `DATABLOCK_BYTES // dtype_size` 得单个 data_block 容纳的元素数 `vec_align_elems`，作为 tile 最小对齐粒度；再结合 `PREFERRED_COPY_BYTES = 512`、`FALLBACK_COPY_BYTES = 256` 得出首选 tile（`max(vec_align_elems, PREFERRED_COPY_BYTES // dtype_size)`）和降级 tile（`max(vec_align_elems, FALLBACK_COPY_BYTES // dtype_size)`）。当总元素数 >= 首选 tile 时取大 tile，否则取小 tile。这样可以充分利用 DMA 带宽，但如果总数据量还填不满一个大 tile，用大 tile 会搬运无效数据，所以降级到小 tile。
  - 由 tile 大小估算所需核数，经 `CORE_CANDIDATES` 上取整得实际核数 `effective_cores`，再计算 `block_length_raw = ceil_div(total_length, effective_cores)`、`total_tiles = max(1, ceil_div(block_length_raw, tile_length))`、`block_length = total_tiles * tile_length`。
  - 采用双缓冲机制（`BUFFER_NUM = 2`），`TQue` 使用 `num=BUFFER_NUM, len=tile_length * dtype_size`，`TBuf` 使用 `num=tile_length * dtype_size`。
- 流水线同步
  - 本样例使用 TPipe/TQue 管理搬运与队列同步。
  - 相邻依赖的 Vector 计算之间使用 `asc.pipe_barrier(asc.PipeID.PIPE_V)` 同步。
  - TBuf(VECCALC) 用于 x² 到 muls(0.5) 前八步中间结果复用。

## 编译执行

环境配置请参考[quick_start.md](../../docs/quick_start.md#envready)。完成环境配置后，执行如下命令可进行功能验证。

```
cd pyasc/examples/06_gelu
python3 gelu.py -r [RUN_MODE] -v [SOC_VERSION]
```

其中脚本参数说明如下：

- RUN_MODE：编译执行方式，可选择NPU仿真，NPU上板，对应参数分别为[Model/NPU]。
- SOC_VERSION：昇腾AI处理器型号，如果无法确定具体的[SOC_VERSION]，则在安装昇腾AI处理器的服务器执行npu-smi info命令进行查询，在查询到的"Name"前增加Ascend信息，例如"Name"对应取值为xxxyy，实际配置的[SOC_VERSION]值为Ascendxxxyy。

示例如下，Ascend910B1请替换为实际的AI处理器型号。

```bash
# 仿真器模式
python3 gelu.py -r Model -v Ascend910B1

# NPU 上板模式
python3 gelu.py -r NPU -v Ascend910B1
```

执行成功后输出：

```
[INFO] start process sample gelu.
[INFO] Sample gelu run success.
```
