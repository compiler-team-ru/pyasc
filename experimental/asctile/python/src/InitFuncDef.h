/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASCTILE_PYTHON_SRC_INIT_FUNC_DEF_H
#define ASCTILE_PYTHON_SRC_INIT_FUNC_DEF_H

#include "ascir/Extension/PyOpBuilder.h"

#include <pybind11/pybind11.h>

namespace mlir {
namespace asctile {

void initIRModule(pybind11::module&& m);
void initPassesModule(pybind11::module&& m);
void initAsctileBuilder(pybind11::class_<ascir::PyOpBuilder>& clss);

} // namespace asctile
} // namespace mlir

#endif // ASCTILE_PYTHON_SRC_INIT_FUNC_DEF_H
