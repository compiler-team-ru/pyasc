# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

__all__ = []

from asc.language.core.constexpr import ConstExpr
from asc.language.core.dtype import (
    DataType,
    bfloat16,
    bool_,
    float16,
    float32,
    float64,
    float_,
    int8,
    int16,
    int32,
    int64,
    int_,
)
from asc.language.core.ir_value import GlobalAddress
from asc.language.core.ops import inline, number
from asc.language.core.range import static_range
from asc.language.core.utils import ceildiv
from asc.lib import profiling, runtime
from asc.runtime.config import Backend, KernelType, Platform, set_platform

__all__ += [
    "Backend",
    "ConstExpr",
    "DataType",
    "GlobalAddress",
    "KernelType",
    "Platform",
    "bfloat16",
    "bool_",
    "ceildiv",
    "float16",
    "float32",
    "float64",
    "float_",
    "int8",
    "int16",
    "int32",
    "int64",
    "int_",
    "inline",
    "number",
    "profiling",
    "runtime",
    "set_platform",
    "static_range",
]

from asc.language.tile.global_tensor import GlobalTensor, global_tensor
from asc.language.tile.local_tensor import LocalTensor, RoundMode, TensorLocation
from asc.language.tile.range import range

# Tile operations
from asc.language.tile.atomic_ops import (
    atomic_add,
    atomic_max,
    atomic_min,
)
from asc.language.tile.binary_ops import (
    add,
    div,
    equal,
    greater,
    greater_equal,
    left_shift,
    less,
    less_equal,
    matmul,
    matmul_acc,
    maximum,
    minimum,
    mul,
    not_equal,
    right_shift,
    sub,
)
from asc.language.tile.creation_ops import (
    cast,
    concat,
    full,
    full_like,
    zeros,
    zeros_acc,
    zeros_like,
)
from asc.language.tile.debug_ops import (
    inline_vf, )
from asc.language.tile.memory_ops import (
    copy,
    copy_in,
    copy_out,
)
from asc.language.tile.prog_model_ops import (
    block_idx,
    block_num,
)
from asc.language.tile.shape_ops import (
    broadcast_to,
    expand_dims,
    ravel,
    reshape,
    squeeze,
    transpose,
)
from asc.language.tile.unary_ops import (
    abs,
    ceil,
    cos,
    cosh,
    erf,
    exp,
    exp2,
    floor,
    log,
    log2,
    negative,
    relu,
    rms_norm,
    rsqrt,
    sin,
    sinh,
    softmax,
    sqrt,
    tan,
    tanh,
)
from asc.language.tile.indexing_ops import (
    mask,
    where,
)
from asc.language.tile.reduction_ops import (
    reduce_max,
    reduce_min,
    reduce_sum,
    reduce_prod,
)

from .jit import jit

__all__ += [
    # global_tensor
    "GlobalTensor",
    "global_tensor",
    # local_tensor
    "LocalTensor",
    "RoundMode",
    "TensorLocation",
    # range
    "range",
    # atomic_ops
    "atomic_add",
    "atomic_max",
    "atomic_min",
    # binary_ops
    "add",
    "div",
    "equal",
    "greater",
    "greater_equal",
    "left_shift",
    "less",
    "less_equal",
    "matmul",
    "matmul_acc",
    "maximum",
    "minimum",
    "mul",
    "not_equal",
    "right_shift",
    "sub",
    # creation_ops
    "cast",
    "concat",
    "full",
    "full_like",
    "zeros",
    "zeros_acc",
    "zeros_like",
    # debug_ops
    "inline_vf",
    # memory_ops
    "copy",
    "copy_in",
    "copy_out",
    # prog_model_ops
    "block_idx",
    "block_num",
    # shape_ops
    "broadcast_to",
    "expand_dims",
    "ravel",
    "reshape",
    "squeeze",
    "transpose",
    # unary_ops
    "abs",
    "ceil",
    "cos",
    "cosh",
    "erf",
    "exp",
    "exp2",
    "floor",
    "log",
    "log2",
    "negative",
    "relu",
    "rms_norm",
    "rsqrt",
    "sin",
    "sinh",
    "softmax",
    "sqrt",
    "tan",
    "tanh",
    # indexing_ops
    "mask",
    "where",
    # reduction_ops
    "reduce_sum",
    "reduce_max",
    "reduce_min",
    "reduce_prod",
    # .jit
    "jit",
]
