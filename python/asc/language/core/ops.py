# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from typing import Any, Optional, Tuple

from .constexpr import Numeric
from .dtype import DataType
from .ir_value import PlainValue, materialize_ir_value as _mat
from .utils import require_jit, global_builder


@require_jit
def inline(code: str, args: Optional[Tuple[Any]] = None, before_function: bool = False) -> None:
    """
    Inject raw Ascend C++ code into the generated kernel.

    The provided code string is emitted verbatim into the output Ascend C++ source at the current insertion point.
    Optional arguments can be passed to be substituted into the emitted code as IR values.

    Args:
        code: The raw C++ code string to inject into the generated kernel source. Use ``$0``, ``$1``, etc.
            as placeholders to reference arguments from the ``args`` list by index.
        args: Optional list of values to pass as arguments to the emitted code. Each value is materialized
            as an IR value and forwarded to the underlying verbatim operation. In the code string, use
            ``$<index>`` (e.g., ``$0``, ``$1``) to reference these arguments by their position in the list.
        before_function: If ``True``, emit the code before the current function body instead of at the current
            insertion point. Default is ``False``.

    Returns:
        None

    Examples:
        Inject constant declarations into the kernel: ::

            asc.inline(\"\"\"
                constexpr int32_t TOTAL_ROWS = 668;
                constexpr int32_t ELEMENTS_PER_ROW = 32;
            \"\"\")

        Pass kernel arguments to inline code using ``$<index>`` placeholders: ::

            @asc2.jit
            def kernel(x_ptr: asc2.GlobalAddress, y_ptr: asc2.GlobalAddress, size: int):
                asc.inline(\"\"\"
                    auto input_ptr = $0;
                    auto output_ptr = $1;
                    int64_t length = $2;
                    
                    AscendC::GlobalTensor<float> x_gm;
                    x_gm.SetGlobalBuffer(input_ptr);
                \"\"\", [x_ptr, y_ptr, size])
    """
    args = None if args is None else [_mat(arg).to_ir() for arg in args]
    insert_point = None
    builder = global_builder.get_ir_builder()
    if before_function:
        current_function = builder.get_current_function()
        if current_function is not None:
            insert_point = builder.save_insertion_point()
            builder.set_insertion_point(current_function)
    builder.create_emitasc_VerbatimOp(code, args)
    if insert_point is not None:
        builder.restore_insertion_point(insert_point)


@require_jit
def number(value: Numeric, dtype: DataType) -> PlainValue:
    return _mat(value, dtype)
