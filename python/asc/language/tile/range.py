# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from typing import Optional, overload

from ..._C import ir
from ..core.range import BaseRange
from ..core.utils import check_type, global_builder


class range(BaseRange):
    """
    A range loop construct for use in JIT functions.

    This class provides a range-based loop similar to Python's built-in :code:`range`, with additional support for loop
    unrolling and parallel execution on NPU. In fact, built-in :code:`range` automatically becomes :code:`asc2.range`
    when used inside JIT function.

    Args:
        start: Start index of the loop (or stop index if only one argument provided).
        stop: Stop index of the loop (exclusive). If None, start is treated as stop and start is 0.
        step: Step size for the loop iteration.
        unroll_factor: Number of iterations to unroll. Default is 1 (no unrolling).
        parallel: Whether to enable software pipelining. Default is False. When True, iterations may overlap to enable
            software pipelining optimizations. Must be False if loop iterations depend on previous iterations.

    Raises:
        ValueError: If :code:`unroll_factor` is less than 1

    Note:
        The :code:`unroll_factor` parameter controls loop unrolling during compilation.
        Higher values can improve performance but increase code size.

    Examples:
        Basic loop from 0 to N: ::

            for i in asc2.range(N):
                ...

        Loop with unrolling: ::

            for i in asc2.range(0, N, step=1, unroll_factor=4):
                ...

        Parallel loop: ::

            for i in asc2.range(N, parallel=True):
                ...
    """

    @overload
    def __init__(self, start: int, stop: Optional[int] = None, step: int = 1, /, *, unroll_factor: int = 1,
                 parallel: bool = False):
        ...

    def __init__(self, *args, unroll_factor: int = 1, parallel: bool = False):
        check_type("unroll_factor", unroll_factor, int)
        check_type("parallel", parallel, bool)
        if unroll_factor < 1:
            raise ValueError(f"Loop unroll factor must be 1 or greater, got {unroll_factor}")
        super().__init__(*args)
        self.unroll_factor = unroll_factor
        self.parallel = parallel

    def handle_op(self, op: ir.ForOp) -> None:
        builder = global_builder.get_ir_builder()
        op.set_attr(ir.attr.unroll_factor, builder.get_index_attr(self.unroll_factor))
        if self.parallel:
            op.set_attr(ir.attr.parallel, builder.get_unit_attr())
