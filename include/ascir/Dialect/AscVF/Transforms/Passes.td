/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASCVF_PASSES_TD
#define ASCVF_PASSES_TD

include "mlir/Pass/PassBase.td"

def EliminateCommonMask : Pass<"ascvf-eliminate-common-mask", "func::FuncOp"> {
  let summary = "Eliminates duplicate mask operations within VFGroupOp";
  let description = [{
    Removes redundant UpdateMaskOp and CreateMaskOp operations inside VFGroupOp
    regions by replacing uses of duplicate masks with the first occurrence and
    deleting the duplicates.
  }];
  let constructor = "mlir::ascvf::createEliminateCommonMaskPass()";
  let dependentDialects = ["ascendc::AscendCDialect", "ascvf::AscVFDialect"];
}

def EliminateDataTransfer : Pass<"ascvf-eliminate-data-transfer", "func::FuncOp"> {
  let summary = "Eliminates redundant load/store operations in VFGroupOp";
  let description = [{
    Performs three optimizations: eliminates loads after stores by replacing
    the loaded value with the stored value, removes stores that are immediately
    overwritten, and merges repeated loads from the same memory location.
  }];
  let constructor = "mlir::ascvf::createEliminateDataTransferPass()";
  let dependentDialects = ["emitasc::EmitAscDialect", "ascvf::AscVFDialect"];
}

def FindVFGroup : Pass<"ascvf-find-vf-group", "func::FuncOp"> {
  let summary = "Finds and groups L2 operations into VFGroupOp for vector fusion";
  let description = [{
    Identifies groups of fusible L2 operations (binary, unary, reduce, duplicate)
    that share the same calCount and element type, and wraps them into VFGroupOp
    operations for subsequent vector fusion optimization.
  }];
  let constructor = "mlir::ascvf::createFindVFGroupPass()";
  let dependentDialects = ["ascendc::AscendCDialect", "ascvf::AscVFDialect"];
}

def FuseVFFor : Pass<"ascvf-fuse-vf-for", "func::FuncOp"> {
  let summary = "Fuses consecutive VFForOp loops within VecScopeOp";
  let description = [{
    Merges adjacent VFForOp loops into a single loop by copying operations from
    subsequent loops into the first loop, reducing loop overhead.
  }];
  let constructor = "mlir::ascvf::createFuseVFForPass()";
  let dependentDialects = ["ascvf::AscVFDialect"];
}

def LowerToMicro : Pass<"ascvf-lower-to-micro", "func::FuncOp"> {
  let summary = "Lowers L2 operations to micro-level operations with loops";
  let description = [{
    Converts L2 operations (binary, unary, reduce, duplicate) into micro-level
    operations by wrapping them in VecScopeOp, creating VFForOp loops, allocating
    register tensors, and inserting load/store micro operations.
  }];
  let constructor = "mlir::ascvf::createLowerToMicroPass()";
  let dependentDialects = ["ascendc::AscendCDialect", "emitasc::EmitAscDialect", "ascvf::AscVFDialect"];
}

def MaterializeLoadStore : Pass<"ascvf-materialize-load-store", "func::FuncOp"> {
  let summary = "Materializes abstract load/store micro ops into data copy operations";
  let description = [{
    Converts LoadMicroOp and StoreMicroOp operations into concrete data copy
    operations with physical addresses, creating LocalTensorGetPhyAddrV2Op and
    DataCopyLoadOp/DataCopyStoreOp operations.
  }];
  let constructor = "mlir::ascvf::createMaterializeLoadStorePass()";
  let dependentDialects = ["ascendc::AscendCDialect", "emitasc::EmitAscDialect", "ascvf::AscVFDialect"];
}

def ReorderOpsInVecScope : Pass<"ascvf-reorder-ops-in-vec-scope", "func::FuncOp"> {
  let summary = "Hoists and reorders operations within VecScopeOp blocks";
  let description = [{
    Reorders operations inside VecScopeOp blocks by moving constants, register
    tensors, variables, mask operations, and duplicate operations to the
    beginning in a specific order for better code generation.
  }];
  let constructor = "mlir::ascvf::createReorderOpsInVecScopePass()";
  let dependentDialects = ["ascendc::AscendCDialect", "arith::ArithDialect", "emitasc::EmitAscDialect", "ascvf::AscVFDialect"];
}

#endif // ASCVF_PASSES_TD
