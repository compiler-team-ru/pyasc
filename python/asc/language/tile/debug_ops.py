# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from typing import Optional, Iterable, Tuple

from ..._C import ir
from ..core.dtype import DataType
from ..core.utils import global_builder
from .tile import Tile, TileLocation
from .validation import check_type, verify_shape


def inline_vf(code: str, shape: Tuple[int, ...], dtype: DataType, inputs: Optional[Iterable[Tile]] = None) -> Tile:
    """
    Embed Ascend C VF (vector function) code within a kernel.

    This is an escape hatch for advanced users who need to express vector-fusion operations (e.g., Ascend C MicroAPI
    calls) that are not covered by the built-in API. The provided code string is injected verbatim as the body of a
    ``__VEC_SCOPE__`` block in the generated Ascend C source.

    Tiles are referenced by positional placeholders: ``$0`` is always the output tile, and ``$1``, ``$2``, ... refer
    to the input tiles in the order they appear in :code:`inputs`. Zero or more input tiles are allowed. Each
    placeholder will be replaced with a ``LocalTensor`` allocated for a corresponding tile.

    All input tiles must reside in UB memory. The output tile is always allocated in UB.

    Args:
        code: The raw Ascend C code string to embed (treated as a ``__VEC_SCOPE__`` body).
            Use ``$0`` for the output tile and ``$1``, ``$2``, ... for input tiles.
        shape: The shape of the output tile.
        dtype: The data type of the output tile.
        inputs: An optional iterable of zero or more input tiles referenced as ``$1``, ``$2``, ... in the code.

    Returns:
        Tile: A new UB tile (``$0``) containing the result produced by the inline vector function.

    Raises:
        TypeError: If code is not a str, dtype is not a DataType, or any input is not a Tile.
        RuntimeError: If any input tile is not located in UB memory or shape is invalid.

    Examples:
        Embed an inline vector multiply-add (``x * y + z``) using Ascend C MicroAPI: ::

            out = asc2.inline_vf(
                '''
                auto* out_ptr = reinterpret_cast<__ubuf__ float*>($0.GetPhyAddr());
                auto* x_ptr = reinterpret_cast<__ubuf__ float*>($1.GetPhyAddr());
                auto* y_ptr = reinterpret_cast<__ubuf__ float*>($2.GetPhyAddr());
                auto* z_ptr = reinterpret_cast<__ubuf__ float*>($3.GetPhyAddr());
                AscendC::MicroAPI::RegTensor<float> result_reg;
                . . .
                AscendC::MicroAPI::MaskReg mask_reg = AscendC::MicroAPI::UpdateMask<float>(mask);
                AscendC::MicroAPI::DataCopy(x_reg, x_ptr);
                AscendC::MicroAPI::DataCopy(y_reg, y_ptr);
                AscendC::MicroAPI::Mul(xy_reg, x_reg, y_reg, mask_reg);
                . . .
                ''',
                x.shape, x.dtype, [x, y, z])

        In the example above, ``$0`` placeholder refers to a ``LocalTensor`` corresponding to ``out`` tile;
        ``$1``, ``$2``, and ``$3`` refers to ``x``, ``y``, and ``z`` respectively.
    """
    check_type("code", code, str)
    check_type("dtype", dtype, DataType)
    shape = verify_shape(shape)
    ir_tiles = []
    if inputs is not None:
        for index, tile in enumerate(inputs):
            tile_name = f"inputs[{index}]"
            check_type(tile_name, tile, Tile)
            if tile.location != TileLocation.UB:
                raise RuntimeError(f"{tile_name} tile must have UB location, got {tile.location.name}")
            ir_tiles.append(tile.to_ir())
    ir_type = ir.get_asctile_TileType(shape, dtype.to_ir(), TileLocation.UB)
    handle = global_builder.get_ir_builder().create_asctile_InlineVFOp(ir_type, ir_tiles, code)
    return Tile(handle)
