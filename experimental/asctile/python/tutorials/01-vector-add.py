# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""
Vector Addition
===============

Your first AscTile kernel: an element-wise vector addition ``out = x + y`` that runs on the Ascend NPU.

.. currentmodule:: asc.experimental

In this tutorial you will learn about:

* The ``@asctile.jit`` decorator and how a Python function becomes an NPU kernel.

* The SPMD programming model: many AI cores run the same kernel, each identified by :py:func:`asctile.block_idx` out of
  :py:func:`asctile.block_num()`.

* Moving data between global memory (GM) and local (on-chip) memory.

* Tiling a long tensor into per-core chunks and per-tile blocks, and overlapping loads with compute (multi-buffering).

* How to launch a kernel from the host and verify its result against PyTorch.
"""

# %%
# Compute Kernel
# --------------
#
# An AscTile kernel is an ordinary function marked with :py:obj:`asctile.jit`. This decorator captures the function AST
# and lowers it to Ascend IR (MLIR) at runtime, then compiles it to an Ascend C binary with the Bisheng compiler. You
# never write Ascend C yourself -- you write Python code (similar to NumPy or Triton), and the toolchain does the rest.
#
# Kernel arguments follow a simple type contract:
#
# * Tensor arguments are annotated ``asctile.GlobalAddress``. At launch time you pass a CPU ``torch.Tensor`` (or numpy
#   array); the runtime hands the kernel a pointer into the device buffer.
#
# * Scalar arguments annotated with a plain Python type (e.g. ``size: int``) become *runtime* values -- they can change
#   from call to call without recompiling.
#
# * Scalar arguments annotated ``asctile.ConstExpr[int]`` are *compile-time constants*: their value is baked into the
#   generated IR, so each distinct value produces a separately compiled (and cached) kernel. Tile sizes must be
#   ``ConstExpr`` because the on-chip buffer they describe must have a static, compile-time-known shape.

from asc.experimental import asctile


@asctile.jit
def vector_add(x_ptr: asctile.GlobalAddress,  # pointer to input tensor x
               y_ptr: asctile.GlobalAddress,  # pointer to input tensor y
               out_ptr: asctile.GlobalAddress,  # pointer to output tensor out
               size: int,  # number of elements (runtime value, may change between launches)
               tile_size: asctile.ConstExpr[int],  # elements per tile (compile-time constant)
               ):

    # A ``GlobalTensor`` is a descriptor for an array living in global memory (GM). It carries a pointer and a (possibly
    # dynamic) shape -- here ``[size]`` -- but owns no storage of its own; the storage is the buffer you passed at
    # launch time.
    x_gm = asctile.global_tensor(x_ptr, [size])
    y_gm = asctile.global_tensor(y_ptr, [size])
    out_gm = asctile.global_tensor(out_ptr, [size])

    # AscTile is SPMD: every launched core runs this same function. We partition the work so that each core owns a
    # *contiguous slice* of the tensor. First split ``size`` evenly across all cores, then divide each core's share into
    # ``tiles_per_core`` tiles with ``tile_size`` elements each:
    #
    #   core 0                   | core 1                   | ... |  core N-1
    #   [tile0][tile1]...[tileM] | [tile0][tile1]...[tileM] |     | [tile0][tile1]...[tileM]
    tiles_per_core = asctile.ceildiv(asctile.ceildiv(size, asctile.block_num()), tile_size)
    core_length = tile_size * tiles_per_core
    core_offset = asctile.block_idx() * core_length

    # ``asctile.range`` is the tiled loop construct. ``unroll_factor=2`` asks the compiler to software-pipeline the loop
    # (multi-buffering): while i-th iteration computes ``x + y`` in UB, the load for i+1-th iteration is already started
    # from GM. This hides the memory latency behind the compute.
    for i in asctile.range(tiles_per_core, unroll_factor=2):
        offset = core_offset + i * tile_size
        # ``copy_in`` moves a tile from GM into local (on-chip) memory. For vector arithmetic the destination is the
        # Unified Buffer (UB); other local memories (L1, L0A, ...) serve the cube unit (see the matmul tutorial). It
        # returns a new ``LocalTensor``.
        x = asctile.copy_in(x_gm, [offset], [tile_size])
        y = asctile.copy_in(y_gm, [offset], [tile_size])
        # Element-wise add runs in UB. Operator overloads -- like + - * / -- call the underlying ``asctile.add``, etc.
        # Each operation returns a *new* tensor (so the language satisfies SSA semantics internally).
        out = x + y
        # ``copy_out`` moves the result tile from local memory (UB) back to GM at the same offset.
        asctile.copy_out(out, out_gm, [offset])


# %%
# Launch and Verify
# -----------------
#
# To run a kernel we first select a target with :py:func:`asctile.set_platform`.
# ``Backend.Model`` is the cycle-accurate Ascend *simulator* (no hardware needed); ``NPU`` runs on a real device.
# The platform ``Ascend950PR_9599`` is a C310-family chip used throughout AscTile tutorials.
#
# Launch syntax: ``kernel[block_num](args...)``.
# The bracketed value is the number of cores; it sets :py:func:`asctile.block_num` seen by the kernel.
# ``ConstExpr`` parameters (here ``tile_size``) are passed as plain Python literals.

if __name__ == "__main__":
    import torch

    asctile.set_platform(asctile.Backend.Model, asctile.Platform.Ascend950PR_9599)

    torch.manual_seed(0)
    # size is chosen so that 16 cores * 4 tiles * 128 elements == 8192 divide it evenly to keep it aligned on purpose.
    size = 8192
    block_num = 16  # number of AI cores to launch -- becomes asctile.block_num() inside
    tile_size = 128  # elements per loop iteration (a ConstExpr at compile time)
    dtype = torch.float32

    x = torch.randn(size, dtype=dtype)
    y = torch.randn(size, dtype=dtype)
    out = torch.empty_like(x)
    vector_add[block_num](x, y, out, size, tile_size)

    # Verify against the reference computed by PyTorch on the CPU.
    reference = x + y
    torch.testing.assert_close(out, reference, atol=1e-5, rtol=1e-5)
    max_diff = (out - reference).abs().max().item()
    print(f"vector_add: PASSED (size={size}, blocks={block_num}, tile={tile_size}, max diff={max_diff:.2e})")
