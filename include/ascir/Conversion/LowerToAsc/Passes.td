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

def LowerArithI1 : Pass<"asclower-arith-i1", "func::FuncOp"> {
  let summary = "Lower arith operations with i1-like operands to ascendc";
  let constructor = "mlir::asclower::createLowerArithI1Pass()";
  let dependentDialects = ["ascendc::AscendCDialect", "emitasc::EmitAscDialect"];
}

def LowerAscTile : Pass<"asclower-asctile", "func::FuncOp"> {
  let summary = "Lower operations from asctile dialect to ascendc";
  let constructor = "mlir::asclower::createLowerAscTilePass()";
  let dependentDialects = ["arith::ArithDialect", "ascendc::AscendCDialect", "emitasc::EmitAscDialect"];
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
  let dependentDialects = ["scf::SCFDialect"];
}

def RealizeConversionCast
    : Pass<"asclower-realize-conversion-cast", "func::FuncOp"> {
  let summary = "Realize unrealized conversion casts into ascendc if possible";
  let constructor = "mlir::asclower::createRealizeConversionCastPass()";
  let dependentDialects = ["ascendc::AscendCDialect"];
}

def RedressI1Tile : Pass<"asclower-redress-i1-tile", "func::FuncOp"> {
  let summary = "Redress i1-like tile origin operations to i16-like";
  let constructor = "mlir::asclower::createRedressI1TilePass()";
  let dependentDialects = ["arith::ArithDialect", "asctile::AscTileDialect"];
}

#endif // LOWERTOASC_PASSES_TD
