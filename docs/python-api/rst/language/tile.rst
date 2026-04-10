.. Copyright (c) 2025 Huawei Technologies Co., Ltd.
.. This program is free software, you can redistribute it and/or modify it under the terms and conditions of
.. CANN Open Software License Agreement Version 2.0 (the "License").
.. Please refer to the License for details. You may not use this file except in compliance with the License.
.. THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
.. INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
.. See LICENSE in the root of the software repository for the full text of the License.

Tile API
========

.. currentmodule:: asc.language.tile

Programming model
-----------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    ~tile.Tile
    ~tensor.Tensor
    ~tensor.tensor
    ~prog_model_ops.block_idx
    ~prog_model_ops.block_num
    ~prog_model_ops.num_tiles


Creation operations
-------------------

.. currentmodule:: asc.language.tile.creation_ops

.. autosummary::
    :toctree: generated
    :nosignatures:

    full
    full_like
    zeros
    zeros_acc
    zeros_like


Memory operations
-----------------

.. currentmodule:: asc.language.tile.memory_ops

.. autosummary::
    :toctree: generated
    :nosignatures:

    copy
    load
    store


Binary operations
-----------------

.. currentmodule:: asc.language.tile.binary_ops

.. autosummary::
    :toctree: generated
    :nosignatures:

    add
    div
    equal
    greater
    greater_equal
    left_shift
    less
    less_equal
    matmul
    matmul_acc
    maximum
    minimum
    mul
    not_equal
    right_shift
    sub


Unary operations
----------------

.. currentmodule:: asc.language.tile.unary_ops

.. autosummary::
    :toctree: generated
    :nosignatures:

    abs
    ceil
    cos
    cosh
    erf
    exp
    exp2
    floor
    log
    log2
    negative
    relu
    rsqrt
    sin
    sinh
    sqrt
    softmax
    tan
    tanh


Indexing operations
-------------------

.. currentmodule:: asc.language.tile.indexing_ops

.. autosummary::
    :toctree: generated
    :nosignatures:

    mask
    where


Reduction operations
--------------------

.. currentmodule:: asc.language.tile.reduction_ops

.. autosummary::
    :toctree: generated
    :nosignatures:

    reduce_min
    reduce_max
    reduce_sum
    reduce_prod


Shape manipulation operations
-----------------------------

.. currentmodule:: asc.language.tile.shape_ops

.. autosummary::
    :toctree: generated
    :nosignatures:

    broadcast_to
    expand_dims
    reshape
    squeeze


Atomic operations
-----------------

.. currentmodule:: asc.language.tile.atomic_ops

.. autosummary::
    :toctree: generated
    :nosignatures:

    atomic_add
    atomic_max
    atomic_min
