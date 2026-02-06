/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASCTILE_PASSES_TD
#define ASCTILE_PASSES_TD

include "mlir/Pass/PassBase.td"

def DensifyUnrollGroups : Pass<"asctile-densify-unroll-groups", "func::FuncOp"> {
  let summary = "Bring ops from the same group together after loop unroll";
  let constructor = "mlir::asctile::createDensifyUnrollGroupsPass()";
}

def PromotePureOps : Pass<"asctile-promote-pure-ops", "func::FuncOp"> {
  let summary = "Promote pure operations within their regions";
  let constructor = "mlir::asctile::createPromotePureOpsPass()";
}

def TagUnrollGroups : Pass<"asctile-tag-unroll-groups", "func::FuncOp"> {
  let summary = "Tag operations of interest to find them after loop unroll";
  let constructor = "mlir::asctile::createTagUnrollGroupsPass()";
}

def TransformMathOps : Pass<"asctile-transform-math-ops", "func::FuncOp"> {
  let summary = "Transform math operations";
  let constructor = "mlir::asctile::createTransformMathOpsPass()";
  let dependentDialects = ["asctile::AscTileDialect"];
}

def UnrollLoop : Pass<"asctile-unroll-loop", "func::FuncOp"> {
  let summary = "Unroll loops by unroll_factor";
  let constructor = "mlir::asctile::createUnrollLoopPass()";
}

#endif // ASCTILE_PASSES_TD
