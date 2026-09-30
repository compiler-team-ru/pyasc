<!-- Copyright (c) 2026 Huawei Technologies Co., Ltd. -->
<!-- This program is free software, you can redistribute it and/or modify it under the terms and conditions of -->
<!-- CANN Open Software License Agreement Version 2.0 (the "License"). -->
<!-- Please refer to the License for details. You may not use this file except in compliance with the License. -->
<!-- THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, -->
<!-- INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. -->
<!-- See LICENSE in the root of the software repository for the full text of the License. -->

# asc.jit

`@asc.jit` 是将 Python 函数转译为 Ascend C 核函数、并调用毕昇编译器生成执行二进制的核心入口装饰器。对应原生 C++ 中通过 `extern "C" __global__ __aicore__` 声明核函数的编程模型。

该接口从 `asc` 顶层导出，实现位于 `asc.runtime.jit`。

## 接口列表

| 接口 | 说明 |
|------|------|
| `jit`(fn) | 使用默认编译与启动选项装饰核函数。 |
| `jit`(\*\*options) | 返回装饰器，并将关键字参数转发到 Codegen / Compile / Launch 选项。 |

## jit

### *function* asc.jit(fn: Callable) → JITFunction

### *function* asc.jit(\*\*options) → Callable[[Callable], JITFunction]

将 Python 函数编译为昇腾核函数。

**参数说明**

- fn：被装饰的 Python 函数。省略该位置参数时，返回带默认选项的装饰器。
- \*\*options：代码生成、编译与启动选项，字段名取自下列数据类。未知字段名会在装饰时抛出 `RuntimeError`。
  - [`CompileOptions`](compiler.md)：编译与 IR 变换选项，例如 `debug`、`opt_level`、`always_compile`。
  - [`LaunchOptions`](launcher.md)：启动选项默认值。核数与 Stream 通常在调用时通过中括号传入。
  - `CodegenOptions`（`asc.CodegenOptions`）：代码生成选项，例如 `capture_exceptions`、`ir_multithreading`。

**返回值说明**

返回 `JITFunction`。调用 `kernel[core_num](...)` 或 `kernel[core_num, stream](...)` 时完成编译（或命中缓存）并在设备上启动。

**约束说明**

- 仅被 `@asc.jit` 修饰的 Host 侧核函数会走完整编译与启动流程。Device 侧被 `@asc.jit` 修饰的辅助函数只参与代码生成，其编译参数不生效。
- 核函数形参名不能与上述选项字段名冲突。
- 启动时中括号参数按位置映射到 `LaunchOptions(core_num, stream)`。
- 未指定 `core_num` 时使用当前平台全部可用 AI Core。
- 可通过 `PYASC_DUMP_PATH` 导出编译中间文件，通过 `PYASC_HOME` / `PYASC_CACHE_DIR` 配置 JIT 缓存目录。

**调用示例**

```python
import asc
import asc.runtime.config as config
import asc.lib.runtime as rt


@asc.jit(opt_level=2, always_compile=True)
def vadd_kernel(x: asc.GlobalAddress, y: asc.GlobalAddress, z: asc.GlobalAddress, block_length: int):
    offset = asc.get_block_idx() * block_length
    x_gm = asc.GlobalTensor()
    y_gm = asc.GlobalTensor()
    z_gm = asc.GlobalTensor()
    x_gm.set_global_buffer(x + offset)
    y_gm.set_global_buffer(y + offset)
    z_gm.set_global_buffer(z + offset)
    # ... data_copy / add ...


def launch(x, y, z, core_num=8):
    block_length = z.size // core_num
    vadd_kernel[core_num, rt.current_stream()](x, y, z, block_length)


config.set_platform(config.Backend.NPU)
```
