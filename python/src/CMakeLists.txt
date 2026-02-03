# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

set(TARGET_NAME libpyasc)

find_package(Python3 3.9 REQUIRED COMPONENTS Development.Module Interpreter)
find_package(pybind11 2.13.1 CONFIG REQUIRED HINTS "${Python3_SITELIB}")

pybind11_add_module(${TARGET_NAME} MODULE NO_EXTRAS
  IR.cpp
  Module.cpp
  OpBuilder.cpp
  Passes.cpp
  Translation.cpp
)

set_target_properties(${TARGET_NAME} PROPERTIES INTERPROCEDURAL_OPTIMIZATION ON)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)

target_link_libraries(${TARGET_NAME} PRIVATE
  MLIRAsc
  MLIRAscTransforms
  MLIRAscUtils
  MLIRAscTile
  MLIRAscTileTransforms
  MLIREmitAsc
  MLIRIR
  MLIRLowerToAsc
  MLIRPass
  MLIRTransforms
  MLIRTargetAsc
  ${dialect_libs}
  ${conversion_libs}
  ${extension_libs}
)

add_dependencies(${TARGET_NAME}
  AscPybindGen
  AscTilePybindGen
  AscTypesPybindGen
)
