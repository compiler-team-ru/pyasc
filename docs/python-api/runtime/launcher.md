<!-- Copyright (c) 2026 Huawei Technologies Co., Ltd. -->
<!-- This program is free software, you can redistribute it and/or modify it under the terms and conditions of -->
<!-- CANN Open Software License Agreement Version 2.0 (the "License"). -->
<!-- Please refer to the License for details. You may not use this file except in compliance with the License. -->
<!-- THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, -->
<!-- INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. -->
<!-- See LICENSE in the root of the software repository for the full text of the License. -->

# asc.runtime.launcher

`LaunchOptions` 控制核函数在设备上的启动方式。调用 JIT 函数时，中括号内的位置参数按顺序构造该数据类。

## 接口列表

| 接口 | 说明 |
|------|------|
| `LaunchOptions`(core_num, stream) | 核函数启动与设备运行时选项。 |

## LaunchOptions

### *class* asc.runtime.launcher.LaunchOptions(core_num: int \| None = None, stream: Stream \| None = None)

**参数说明**

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| core_num | int \| None | None | 启动使用的 AI Core 数量。为 `None` 时使用当前平台全部可用核。必须为正整数。 |
| stream | Stream \| None | None | 核函数执行流。为 `None` 时使用 `asc.lib.runtime.current_stream()`。 |

**约束说明**

- `core_num` 若指定，必须大于 0，且不超过硬件实际可用核数。
- 中括号参数按位置映射：`kernel[core_num](...)` 或 `kernel[core_num, stream](...)`。
- Host 侧 Tensor / ndarray 参数会由运行时自动完成 Host 与 Device 之间的拷贝。

**调用示例**

```python
import asc
import asc.lib.runtime as rt
from asc.runtime.launcher import LaunchOptions

@asc.jit
def kernel(x, y):
    ...

kernel[16](x, y)
kernel[16, rt.current_stream()](x, y)

options = LaunchOptions(core_num=8)
```
