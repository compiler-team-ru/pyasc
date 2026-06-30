# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from ..basic.sys_var import get_block_idx, get_block_num
from ..core.ir_value import PlainValue
from ..core.utils import require_jit


@require_jit
def block_idx() -> PlainValue:
    """
    Returns the current block (NPU core) index.

    In the PyAsc2 programming model, kernels are executed across multiple NPU blocks (cores). This function returns the
    index of the current block, which can be used to determine which portion of the data to process.

    Returns:
        PlainValue: The current block index (0-based)

    Examples:
        Get the current block index to compute the data offset: ::

            idx = asc2.block_idx()
            offset = idx * TILE_SIZE
            tile = asc2.copy_in(x_gm, [offset], [TILE_SIZE])
    """
    return get_block_idx()


@require_jit
def block_num() -> PlainValue:
    """
    Returns the total number of blocks (NPU cores) allocated for the kernel.

    This function returns the total number of NPU blocks that are executing the kernel, which was specified when
    launching the kernel.

    Returns:
        PlainValue: The total number of blocks

    Examples:
        Use block count to compute a stride across blocks: ::

            idx = asc2.block_idx()
            n_blocks = asc2.block_num()
            stride = n_blocks * TILE_SIZE
    """
    return get_block_num()
