/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ascir/Extension/PythonExtension.h"

#include <pybind11/pybind11.h>

#include "InitFuncDef.h"

namespace {

void initAsctileModule(pybind11::module&& m)
{
    m.doc() = "Python bindings to the C++ AscTile API";
    mlir::asctile::initIRModule(m.def_submodule("ir"));
    mlir::asctile::initPassesModule(m.def_submodule("passes"));
}

ASC_PYTHON_EXTENSION(asctile, mlir::asctile::initAsctileBuilder, initAsctileModule);

} // namespace
