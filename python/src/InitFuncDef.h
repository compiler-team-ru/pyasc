/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef PYTHON_SRC_INIT_FUNC_DEF_H
#define PYTHON_SRC_INIT_FUNC_DEF_H

#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // automatic casts between containers and python types

namespace pybind11 {
namespace asc {
void pyasc_init_ir(pybind11::module&& m);          // from IR.cpp
void pyasc_init_passes(pybind11::module&& m);      // from Passes.cpp
void pyasc_init_translation(pybind11::module&& m); // from Translation.cpp
void pyasc_init_ir_builder(pybind11::module& m);
} // namespace asc
} // namespace pybind11
#endif // PYTHON_SRC_INIT_FUNC_DEF_H
