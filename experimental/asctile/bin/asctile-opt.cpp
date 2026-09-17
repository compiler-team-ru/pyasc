/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "asctile/Conversion/LowerToAsc/Passes.h"
#include "asctile/Dialect/AscTile/IR/AscTile.h"
#include "asctile/Dialect/AscTile/Transforms/Passes.h"
#include "asctile/Dialect/AscVF/IR/AscVF.h"
#include "asctile/Dialect/AscVF/Transforms/Passes.h"
#include "asctile/Dialect/AscendC/Transforms/Passes.h"

#include "ascir/Dialect/Asc/Utils/Utils.h"
#include "ascir/Dialect/Utils/Registration.h"

#include "mlir/Tools/mlir-opt/MlirOptMain.h"

using namespace mlir;

int main(int argc, char** argv)
{
    DialectRegistry registry;
    ascir::registerDialects(registry);
    ascendc::registerInlinerInterfaces(registry);
    registry.insert<asctile::AscTileDialect, ascvf::AscVFDialect>();
    asctile::registerExternalModels(registry);
    ascir::registerExtensions(registry);
    ascir::registerPasses();
    registerAscTilePasses();
    registerAscVFPasses();
    registerAscTileAscendCPasses();
    registerLowerToAscPasses();
    return asMainReturnCode(MlirOptMain(argc, argv, "AscTile modular optimizer driver\n", registry));
}
