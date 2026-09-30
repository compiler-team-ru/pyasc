<!-- Copyright (c) 2026 Huawei Technologies Co., Ltd. -->
<!-- This program is free software, you can redistribute it and/or modify it under the terms and conditions of -->
<!-- CANN Open Software License Agreement Version 2.0 (the "License"). -->
<!-- Please refer to the License for details. You may not use this file except in compliance with the License. -->
<!-- THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, -->
<!-- INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. -->
<!-- See LICENSE in the root of the software repository for the full text of the License. -->

# asc.runtime.config

Ascend C 运行时配置接口，实现位于 `python/asc/runtime/config.py`。用户通过该模块配置后端执行模式、SOC 版本、设备 ID，以及核函数 `KernelType`。

## 接口列表

| [`set_platform`](#set_platform)(backend[, soc_version, device_id, check]) | 设置运行时后端、SOC 版本和设备 ID。当 backend 为 Model 时，soc_version 默认为 Ascend910B1；当 backend 为 NPU 时，soc_version 自动从当前平台获取，且会校验与输入的 soc_version 是否一致。 |

## set_platform

### *function* asc.runtime.config.set_platform(backend, soc_version=None, device_id=None, check=True)

**参数说明**

- backend：执行后端，取 [`Backend`](#backend) 枚举或字符串 `"Model"` / `"NPU"`。
- soc_version：目标 SOC。取 [`Platform`](#platform) 枚举或同名字符串。Model 后端未指定时默认为 `Platform.Ascend910B1`；NPU 后端必须与真实硬件一致。
- device_id：设备 ID。未指定时使用 0 号设备。
- check：为 True 时校验运行时库是否可用。Model 后端不可用时，错误信息会提示设置 `LD_LIBRARY_PATH` 指向对应 simulator。

**约束说明**

- NPU 后端下，若传入的 `soc_version` 与 `rt.current_platform()` 不一致，将抛出 `ValueError`。
- `check=True` 且运行时库不可用时抛出 `RuntimeError`。

**调用示例**

```python
import asc.runtime.config as config

config.set_platform(config.Backend.Model, config.Platform.Ascend910B1)
config.set_platform("NPU", device_id=0)
config.set_platform(config.Backend.Model, "Ascend910B3", check=False)
```

## 枚举类型

### Backend

指定后端执行模式。

| 枚举值 | 说明 |
|--------|------|
| Backend.Model | 使用 Model 后端执行，适用于仿真或模型运行场景 |
| Backend.NPU | 使用 NPU 后端执行，适用于真实 NPU 硬件场景 |

### Platform

指定 SOC 版本。

| 枚举值 | 说明 |
|--------|------|
| Platform.Ascend910B1 | Ascend 910B1 |
| Platform.Ascend910B2 | Ascend 910B2 |
| Platform.Ascend910B2C | Ascend 910B2C |
| Platform.Ascend910B3 | Ascend 910B3 |
| Platform.Ascend910B4 | Ascend 910B4 |
| Platform.Ascend910B4_1 | Ascend 910B4-1 |
| Platform.Ascend910_9362 | Ascend 910 9362 |
| Platform.Ascend910_9372 | Ascend 910 9372 |
| Platform.Ascend910_9381 | Ascend 910 9381 |
| Platform.Ascend910_9382 | Ascend 910 9382 |
| Platform.Ascend910_9391 | Ascend 910 9391 |
| Platform.Ascend910_9392 | Ascend 910 9392 |
| Platform.Ascend950PR_950z | Ascend 950PR 950z |
| Platform.Ascend950PR_9579 | Ascend 950PR 9579 |
| Platform.Ascend950PR_957b | Ascend 950PR 957b |
| Platform.Ascend950PR_957c | Ascend 950PR 957c |
| Platform.Ascend950PR_957d | Ascend 950PR 957d |
| Platform.Ascend950PR_9589 | Ascend 950PR 9589 |
| Platform.Ascend950PR_958b | Ascend 950PR 958b |
| Platform.Ascend950PR_9599 | Ascend 950PR 9599 |

### KernelType

指定核函数类型，可在 `@asc.jit(kernel_type=...)` 或 [`CompileOptions.kernel_type`](compiler.md) 中使用。未指定时由框架自动推导。

| 枚举值 | 说明 |
|--------|------|
| KernelType.AIV_ONLY | 仅 Vector 核 |
| KernelType.AIC_ONLY | 仅 Cube 核 |
| KernelType.MIX_AIV_HARD_SYNC | MIX，以 AIV 为主，硬同步 |
| KernelType.MIX_AIC_HARD_SYNC | MIX，以 AIC 为主，硬同步 |
| KernelType.MIX_AIV_1_0 | MIX AIV 1:0 |
| KernelType.MIX_AIC_1_0 | MIX AIC 1:0 |
| KernelType.MIX_AIC_1_1 | MIX AIC 1:1 |
| KernelType.MIX_AIC_1_2 | MIX AIC 1:2 |
