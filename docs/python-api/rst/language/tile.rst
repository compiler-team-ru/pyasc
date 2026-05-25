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

.. autosummary::
    :toctree: generated
    :nosignatures:

    ~tile.Tile
    ~tensor.Tensor

.. currentmodule:: asc2


Programming model
-----------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    tensor
    block_idx
    block_num
    num_tiles


Iterators
---------

.. autosummary::
    :toctree: generated
    :nosignatures:

    range
    static_range


Creation operations
-------------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    concat
    full
    full_like
    zeros
    zeros_acc
    zeros_like


Memory operations
-----------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    copy
    load
    store


Arithmetic operations
---------------------

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
    maximum
    minimum
    mul
    negative
    not_equal
    right_shift
    sub


Math operations
---------------

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
    relu
    rms_norm
    rsqrt
    sin
    sinh
    softmax
    sqrt
    tan
    tanh


Matrix multiplication operations
--------------------------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    matmul
    matmul_acc


Indexing operations
-------------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    mask
    where


Reduction operations
--------------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    reduce_min
    reduce_max
    reduce_sum
    reduce_prod


Shape manipulation operations
-----------------------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    broadcast_to
    expand_dims
    ravel
    reshape
    squeeze
    transpose


Atomic operations
-----------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    atomic_add
    atomic_max
    atomic_min
