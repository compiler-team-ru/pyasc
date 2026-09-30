<!-- Copyright (c) 2026 Huawei Technologies Co., Ltd. -->
<!-- This program is free software, you can redistribute it and/or modify it under the terms and conditions of -->
<!-- CANN Open Software License Agreement Version 2.0 (the "License"). -->
<!-- Please refer to the License for details. You may not use this file except in compliance with the License. -->
<!-- THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, -->
<!-- INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. -->
<!-- See LICENSE in the root of the software repository for the full text of the License. -->

# asc.runtime

`asc.runtime` 是 PyAsc 的编译与启动运行时。核函数通过顶层入口 [`@asc.jit`](jit.md) 转译为 Ascend C，再由毕昇编译器生成可执行二进制；启动时通过中括号传入核数与 Stream。平台后端由 [`asc.runtime.config.set_platform`](config.md) 配置。

# Programming models

* [asc.jit](jit.md)
* [asc.runtime.compiler.CompileOptions](compiler.md)
* [asc.runtime.launcher.LaunchOptions](launcher.md)
* [asc.runtime.config](config.md)
