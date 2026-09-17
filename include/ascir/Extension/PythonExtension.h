/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASCIR_EXTENSION_PYTHONEXTENSION_H
#define ASCIR_EXTENSION_PYTHONEXTENSION_H

#include "ascir/Extension/PyOpBuilder.h"

#include "llvm/Support/Compiler.h"

#include <pybind11/pybind11.h>

namespace mlir {
namespace ascir {

struct PythonExtensionHooks {
    const char* name;
    void (*initBuilder)(pybind11::class_<PyOpBuilder>&);
    void (*initModule)(pybind11::module&&);
};

class PythonExtensionRegistry {
    SmallVector<PythonExtensionHooks, 8> hooks;

    PythonExtensionRegistry() = default;

public:
    static PythonExtensionRegistry& get()
    {
        static PythonExtensionRegistry instance;
        return instance;
    }

    void add(const PythonExtensionHooks& hooks) { this->hooks.push_back(hooks); }

    void initAllBuilders(pybind11::class_<PyOpBuilder>& clss)
    {
        for (const auto& h : hooks) {
            if (h.initBuilder) {
                h.initBuilder(clss);
            }
        }
    }

    void initAllModules(pybind11::module& parent)
    {
        for (const auto& h : hooks) {
            if (h.initModule) {
                h.initModule(parent.def_submodule(h.name));
            }
        }
    }
};

struct PythonExtensionRegistrar {
    PythonExtensionRegistrar(const PythonExtensionHooks& hooks) { PythonExtensionRegistry::get().add(hooks); }
};

inline void initExtensionBuilders(pybind11::class_<PyOpBuilder>& clss)
{
    PythonExtensionRegistry::get().initAllBuilders(clss);
}

inline void initExtensionModules(pybind11::module& m) { PythonExtensionRegistry::get().initAllModules(m); }

} // namespace ascir
} // namespace mlir

#define ASC_PYTHON_EXTENSION(NAME, INIT_BUILDER, INIT_MODULE) \
    LLVM_ATTRIBUTE_USED ::mlir::ascir::PythonExtensionRegistrar pyext_##NAME({#NAME, INIT_BUILDER, INIT_MODULE})

#endif // ASCIR_EXTENSION_PYTHONEXTENSION_H
