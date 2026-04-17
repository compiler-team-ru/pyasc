/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef LOWERTOASC_PASSES_TD
#define LOWERTOASC_PASSES_TD

include "mlir/Pass/PassBase.td"

def DisplaceConcat : Pass<"asclower-displace-concat", "func::FuncOp"> {
  let summary = "Displace asctile.concat by rewriting related allocations";
  let constructor = "mlir::asclower::createDisplaceConcatPass()";
  let dependentDialects = ["arith::ArithDialect", "ascendc::AscendCDialect"];
}

def ExpandMask : Pass<"asclower-expand-mask", "func::FuncOp"> {
  let summary = "Update vector mask of nested operations and move them outside";
  let constructor = "mlir::asclower::createExpandMaskPass()";
  let dependentDialects = ["emitasc::EmitAscDialect"];
}

def ExpandMath : Pass<"asclower-expand-math", "func::FuncOp"> {
  let summary = "Expand math operations that cannot be lowered directly";
  let constructor = "mlir::asclower::createExpandMathPass()";
  let dependentDialects = ["arith::ArithDialect"];
}

def LowerArith : Pass<"asclower-arith", "func::FuncOp"> {
  let summary = "Lower operations from arith dialect to ascendc";
  let constructor = "mlir::asclower::createLowerArithPass()";
  let dependentDialects = ["ascendc::AscendCDialect"];
}

def LowerArithBinary : Pass<"asclower-arith-binary", "func::FuncOp"> {
  let summary = "Lower binary operations from arith dialect to ascendc";
  let constructor = "mlir::asclower::createLowerArithBinaryPass()";
  let dependentDialects = ["arith::ArithDialect", "ascendc::AscendCDialect"];
}

def LowerAscTile : Pass<"asclower-asctile", "func::FuncOp"> {
  let summary = "Lower operations from asctile dialect to ascendc";
  let constructor = "mlir::asclower::createLowerAscTilePass()";
  let dependentDialects = [
    "arith::ArithDialect", "ascendc::AscendCDialect", "emitasc::EmitAscDialect",
    "scf::SCFDialect",
  ];
}

def LowerAscTileI1 : Pass<"asclower-asctile-i1", "func::FuncOp"> {
  let summary = "Lower asctile operations with i1-like operands to ascendc";
  let constructor = "mlir::asclower::createLowerAscTileI1Pass()";
  let dependentDialects = ["arith::ArithDialect", "ascendc::AscendCDialect"];
}

def LowerAtomic : Pass<"asclower-atomic", "func::FuncOp"> {
  let summary = "Lower atomic operations from asctile dialect to ascendc";
  let constructor = "mlir::asclower::createLowerAtomicPass()";
  let dependentDialects = ["arith::ArithDialect", "ascendc::AscendCDialect", "emitasc::EmitAscDialect"];
}

def LowerMath : Pass<"asclower-math", "func::FuncOp"> {
  let summary = "Lower operations from math dialect to ascendc";
  let constructor = "mlir::asclower::createLowerMathPass()";
  let dependentDialects = ["ascendc::AscendCDialect"];
}

def LowerSCF : Pass<"asclower-scf", "func::FuncOp"> {
  let summary = "Convert operand and result types of scf operations";
  let constructor = "mlir::asclower::createLowerSCFPass()";
  let dependentDialects = ["ascendc::AscendCDialect", "scf::SCFDialect"];
}

def RealizeConversionCast
    : Pass<"asclower-realize-conversion-cast", "func::FuncOp"> {
  let summary = "Realize unrealized conversion casts into ascendc if possible";
  let constructor = "mlir::asclower::createRealizeConversionCastPass()";
  let dependentDialects = ["ascendc::AscendCDialect"];
}

def RedressI1Tile : Pass<"asclower-redress-i1-tile", "func::FuncOp"> {
  let summary = "Redress i1-like tile origin operations to i8-like";
  let constructor = "mlir::asclower::createRedressI1TilePass()";
  let dependentDialects = ["arith::ArithDialect", "asctile::AscTileDialect"];
}

#endif // LOWERTOASC_PASSES_TD
