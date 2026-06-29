.. Copyright (c) 2025 Huawei Technologies Co., Ltd.
.. This program is free software, you can redistribute it and/or modify it under the terms and conditions of
.. CANN Open Software License Agreement Version 2.0 (the "License").
.. Please refer to the License for details. You may not use this file except in compliance with the License.
.. THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
.. INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
.. See LICENSE in the root of the software repository for the full text of the License.

Cube programming capabilities
=============================

.. contents:: Table of Contents
   :local:

This guide covers cube (matrix multiply-accumulate) operations in PyAsc2, which leverage dedicated matrix multiplication units (cube units) on Ascend NPU hardware.

**Supported Features:**

* Data types: float16, bfloat16, float32
* Memory locations: L0A, L0B, L0C, L1 (required)
* Transpose (fused into load)
* Accumulator (K-axis tiling)
* Quantization: float32 → float16/bfloat16
* ReLU activation (fused)
* HF32 mode (for float32)
* Multi-core execution
* Flexible tile shapes

**Not Supported:**

* INT8/INT4 quantization
* DeqScalar parameter
* Dequantization (DEQF16, VDEQF16)
* Bias addition
* Gemm from L1 (direct L1→L0C matmul)
* Batch matmul
* Sparse matrices
* L0C → UB via FixPipe
* 3D tensors
* Standalone transpose on L0A/L0B
* Cast to int types

Basic Matrix Multiplication
---------------------------

PyAsc2 provides simple and intuitive matrix multiplication operations using the :func:`~asc2.matmul` function or the ``@`` operator:

.. code-block:: python

   import asc2

   @asc2.jit
   def matmul_kernel(a_ptr, b_ptr, c_ptr, m, k, n):
       a_gm = asc2.tensor(a_ptr, [m, k])
       b_gm = asc2.tensor(b_ptr, [k, n])
       c_gm = asc2.tensor(c_ptr, [m, n])
       
       # Load matrices to L0A and L0B
       a = asc2.load(a_gm, [0, 0], [m, k], asc2.TileLocation.L0A)
       b = asc2.load(b_gm, [0, 0], [k, n], asc2.TileLocation.L0B)
       
       # Perform matrix multiplication
       c = a @ b  # or asc2.matmul(a, b)
       
       # Store result
       asc2.store(c, c_gm, [0, 0])

Data Types and Memory Locations
--------------------------------

Supported Data Types
~~~~~~~~~~~~~~~~~~~~

The :func:`~asc2.matmul` operation supports the following data types:

* **float16** (half) - Full support
* **bfloat16** - Full support
* **float32** - Full support (with optional HF32 mode)

**Important:** Input matrices must have the same dtype. INT8 and INT4 are not supported.

Memory Location Requirements
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Cube operations have strict memory location requirements:

=====================  =================  ========================================
Component              Required Location  Description
=====================  =================  ========================================
Matrix A (left)        **L0A**            Dedicated memory for left matrix
Matrix B (right)       **L0B**            Dedicated memory for right matrix
Result/Accumulator     **L0C**            Dedicated memory for C matrix
L1 buffer              **L1**             Buffer between GM and L0A/L0B
=====================  =================  ========================================

**Memory Flow Pattern:**

.. code-block::

                         ↱ GM
   GM ➔ L1 ➔ L0A/L0B ➔ L0C
             (matmul)    ↳ L1

L1 is a required buffer for loading data from GM to L0A/L0B. When you call :func:`~asc2.load` with ``location=TileLocation.L0A`` or ``TileLocation.L0B``, the compiler automatically splits the operation into ``GM → L1 → L0A/L0B``.

Example with explicit L1 usage:

.. code-block:: python

   @asc2.jit
   def matmul_explicit_l1_kernel(a_ptr, b_ptr, c_ptr, m, k, n):
       a_gm = asc2.tensor(a_ptr, [m, k])
       b_gm = asc2.tensor(b_ptr, [k, n])
       c_gm = asc2.tensor(c_ptr, [m, n])
       
       # Explicitly load to L1 first
       a_l1 = asc2.load(a_gm, [0, 0], [m, k], asc2.TileLocation.L1)
       b_l1 = asc2.load(b_gm, [0, 0], [k, n], asc2.TileLocation.L1)
       
       # Copy from L1 to L0A/L0B
       a = asc2.copy(a_l1, [0, 0], [m, k], asc2.TileLocation.L0A)
       b = asc2.copy(b_l1, [0, 0], [k, n], asc2.TileLocation.L0B)
       
       c = a @ b
       asc2.store(c, c_gm, [0, 0])

Accumulator Operations
----------------------

For iterative matrix multiplication (e.g., tiling along K dimension), use the accumulator pattern with :func:`~asc2.zeros_acc` and :func:`~asc2.matmul_acc`:

.. code-block:: python

   @asc2.jit
   def matmul_tiled_kernel(a_ptr, b_ptr, c_ptr, m, k, n, k_tiles):
       a_gm = asc2.tensor(a_ptr, [m, k])
       b_gm = asc2.tensor(b_ptr, [k, n])
       c_gm = asc2.tensor(c_ptr, [m, n])
       
       # Initialize accumulator in L0C
       acc = asc2.zeros_acc([m, n], dtype=asc2.float32)
       
       # Tile along K dimension
       tile_k = k // k_tiles
       for i in asc2.range(k_tiles):
           # Load tiles
           a_tile = asc2.load(a_gm, [0, i * tile_k], [m, tile_k], asc2.TileLocation.L0A)
           b_tile = asc2.load(b_gm, [i * tile_k, 0], [tile_k, n], asc2.TileLocation.L0B)
           
           # Accumulate: acc += a_tile @ b_tile
           asc2.matmul_acc(acc, a_tile, b_tile)
       
       # Store final result
       asc2.store(acc, c_gm, [0, 0])

**Accumulator Requirements:**

* Must be created with :func:`~asc2.zeros_acc` (not regular :func:`~asc2.zeros`)
* Must be in L0C location
* Must be float32 dtype
* Must be 2D tile

Bias Support
~~~~~~~~~~~~

Both :func:`~asc2.matmul` and :func:`~asc2.zeros_acc` support optional bias initialization. Bias tiles must be 1D tiles in :code:`BT` location with shape matching the last dimension of the output.

**Supported bias dtypes:** :code:`float16`, :code:`bfloat16`, or :code:`float32`. Bias with :code:`float16` or :code:`bfloat16` dtype is automatically promoted to :code:`float32` to match the accumulator/result type.

**Using bias with matmul:**

.. code-block:: python

   @asc2.jit
   def matmul_bias_kernel(a_ptr, b_ptr, bias_ptr, c_ptr, m, k, n):
       a_gm = asc2.tensor(a_ptr, [m, k])
       b_gm = asc2.tensor(b_ptr, [k, n])
       bias_gm = asc2.tensor(bias_ptr, [n])
       c_gm = asc2.tensor(c_ptr, [m, n])
       
       a = asc2.load(a_gm, [0, 0], [m, k], asc2.TileLocation.L0A)
       b = asc2.load(b_gm, [0, 0], [k, n], asc2.TileLocation.L0B)
       bias = asc2.load(bias_gm, [0], [n], asc2.TileLocation.BT)
       
       # C = A @ B + bias
       c = asc2.matmul(a, b, bias)
       asc2.store(c, c_gm, [0, 0])

**Using bias with zeros_acc for K-tiled accumulation:**

.. code-block:: python

   @asc2.jit
   def matmul_tiled_bias_kernel(a_ptr, b_ptr, bias_ptr, c_ptr, m, k, n, k_tiles):
       a_gm = asc2.tensor(a_ptr, [m, k])
       b_gm = asc2.tensor(b_ptr, [k, n])
       bias_gm = asc2.tensor(bias_ptr, [n])
       c_gm = asc2.tensor(c_ptr, [m, n])
       
       bias = asc2.load(bias_gm, [0], [n], asc2.TileLocation.BT)
       # Initialize accumulator with bias
       acc = asc2.zeros_acc([m, n], dtype=asc2.float32, bias=bias)
       
       tile_k = k // k_tiles
       for i in asc2.range(k_tiles):
           a_tile = asc2.load(a_gm, [0, i * tile_k], [m, tile_k], asc2.TileLocation.L0A)
           b_tile = asc2.load(b_gm, [i * tile_k, 0], [tile_k, n], asc2.TileLocation.L0B)
           asc2.matmul_acc(acc, a_tile, b_tile)
       
       asc2.store(acc, c_gm, [0, 0])

Supported Features
------------------

Matrix Transpose
~~~~~~~~~~~~~~~~

PyAsc2 supports transposing matrices before multiplication using :func:`~asc2.transpose` or the ``.T`` property of tile. Transpose operations on L0A/L0B tiles are automatically fused into the load operation by the compiler.

.. code-block:: python

   @asc2.jit
   def matmul_transpose_kernel(a_ptr, b_ptr, c_ptr, m, k, n):
       a_gm = asc2.tensor(a_ptr, [k, m])  # Note: shape is [k, m] not [m, k]
       b_gm = asc2.tensor(b_ptr, [n, k])  # Note: shape is [n, k] not [k, n]
       c_gm = asc2.tensor(c_ptr, [m, n])
       
       # Load to L1
       a_l1 = asc2.load(a_gm, [0, 0], [k, m], asc2.TileLocation.L1)
       b_l1 = asc2.load(b_gm, [0, 0], [n, k], asc2.TileLocation.L1)
       
       # Copy to L0A/L0B and transpose
       a = asc2.copy(a_l1, [0, 0], [k, m], asc2.TileLocation.L0A)
       a_transpose = asc2.transpose(a)  # Transpose on L0A
       
       b = asc2.copy(b_l1, [0, 0], [n, k], asc2.TileLocation.L0B)
       b_transpose = b.T  # Transpose using .T property of tile
       
       # C = A.T @ B.T
       c = a_transpose @ b_transpose
       asc2.store(c, c_gm, [0, 0])

**Important:**

* Transpose on L0A/L0B is fused automatically by the compiler into load operation
* Standalone transpose on L0A/L0B after matmul is **NOT supported**
* Use ``.T`` property of tile or :func:`~asc2.transpose` function

Quantization and Type Casting
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

After matmul, the result can be cast to other types using the ``.to()`` method. Quantization is achieved by explicit L0C tile cast to the desired type.

**Supported Quantization Modes (from L0C float32):**

1. float32 → float32 (no conversion)
2. float32 → float16
3. float32 → bfloat16

Example:

.. code-block:: python

   @asc2.jit
   def matmul_quant_kernel(a_ptr, b_ptr, c_ptr, m, k, n, quant_type):
       a_gm = asc2.tensor(a_ptr, [m, k])
       b_gm = asc2.tensor(b_ptr, [k, n])
       c_gm = asc2.tensor(c_ptr, [m, n])
       
       a = asc2.load(a_gm, [0, 0], [m, k], asc2.TileLocation.L0A)
       b = asc2.load(b_gm, [0, 0], [k, n], asc2.TileLocation.L0B)
       
       # Matmul produces float32 in L0C
       c = a @ b
       
       # Cast to quantized type (float16 or bfloat16)
       c_quant = c.to(quant_type)  # Uses F322F16 or F322BF16 mode
       asc2.store(c_quant, c_gm, [0, 0])

**Not Supported Quantization Modes:**

The following modes require integer input types or ``deqScalar`` parameter, which are not supported:

* int32 → float16 (dequantization)
* float32 → int8/uint8 (quantization to int8)
* int32 → int8/uint8 (requantization)

ReLU Activation
~~~~~~~~~~~~~~~

ReLU can be applied after matmul using :func:`~asc2.relu` and is automatically fused into the Fixpipe operation for optimization:

.. code-block:: python

   @asc2.jit
   def matmul_relu_quant_kernel(a_ptr, b_ptr, c_ptr, m, k, n, quant_type):
       a_gm = asc2.tensor(a_ptr, [m, k])
       b_gm = asc2.tensor(b_ptr, [k, n])
       c_gm = asc2.tensor(c_ptr, [m, n])
       
       a = asc2.load(a_gm, [0, 0], [m, k], asc2.TileLocation.L0A)
       b = asc2.load(b_gm, [0, 0], [k, n], asc2.TileLocation.L0B)
       
       # Matmul + ReLU + Quantization (fused into Fixpipe)
       c = a @ b
       c = asc2.relu(c).to(quant_type)
       
       asc2.store(c, c_gm, [0, 0])

**ReLU Fusion:** The compiler automatically fuses ReLU into the Fixpipe operation when ReLU is applied directly to L0C result

HF32 Mode
~~~~~~~~~

For float32 inputs, HF32 (high-performance float32) mode can be enabled for optimized performance:

.. code-block:: python

   @asc2.jit
   def matmul_hf32_kernel(a_ptr, b_ptr, c_ptr, m, k, n):
       a_gm = asc2.tensor(a_ptr, [m, k])
       b_gm = asc2.tensor(b_ptr, [k, n])
       c_gm = asc2.tensor(c_ptr, [m, n])
       
       a = asc2.load(a_gm, [0, 0], [m, k], asc2.TileLocation.L0A)
       b = asc2.load(b_gm, [0, 0], [k, n], asc2.TileLocation.L0B)
       
       # Use HF32 mode for float32 inputs
       c = asc2.matmul(a, b, hf32=True)
       
       asc2.store(c, c_gm, [0, 0])

Advanced Usage
--------------

Chained Matmul Operations
~~~~~~~~~~~~~~~~~~~~~~~~~

Chain multiple matmul operations by moving L0C result to L1 and then to L0A for subsequent operations:

.. code-block:: python

   @asc2.jit
   def chained_matmul_kernel(a_ptr, b_ptr, c_ptr, m, k, n, dtype):
       a_gm = asc2.tensor(a_ptr, [m, k])
       b_gm = asc2.tensor(b_ptr, [k, n])
       c_gm = asc2.tensor(c_ptr, [m, n])
       
       # First matmul: A @ B
       a = asc2.load(a_gm, [0, 0], [m, k], asc2.TileLocation.L0A)
       b = asc2.load(b_gm, [0, 0], [k, n], asc2.TileLocation.L0B)
       c = a @ b
       
       # Cast and move to L1
       c_cast = c.to(dtype)
       c_l1 = asc2.copy(c_cast, [0, 0], [m, n], asc2.TileLocation.L1)
       
       # Move to L0A for second matmul
       c_l0a = asc2.copy(c_l1, [0, 0], [m, n], asc2.TileLocation.L0A)
       
       # Second matmul: C @ B
       result = c_l0a @ b
       
       asc2.store(result, c_gm, [0, 0])

Multi-Core Execution
~~~~~~~~~~~~~~~~~~~~

PyAsc2 supports multi-core parallel execution using :func:`~asc2.block_idx` and :func:`~asc2.block_num`:

.. code-block:: python

   @asc2.jit
   def parallel_matmul_kernel(a_ptr, b_ptr, c_ptr, m, k, n):
       a_gm = asc2.tensor(a_ptr, [m, k])
       b_gm = asc2.tensor(b_ptr, [k, n])
       c_gm = asc2.tensor(c_ptr, [m, n])
       
       # Get block index for parallel execution
       block_id = asc2.block_idx()
       num_blocks = asc2.block_num()
       
       # Compute tile for this block
       tile_m = m // num_blocks
       local_m_start = block_id * tile_m
       
       # Load local tiles
       a_tile = asc2.load(a_gm, [local_m_start, 0], [tile_m, k], asc2.TileLocation.L0A)
       b = asc2.load(b_gm, [0, 0], [k, n], asc2.TileLocation.L0B)
       
       # Matmul
       c_tile = a_tile @ b
       
       # Store result for this block
       asc2.store(c_tile, c_gm, [local_m_start, 0])

   # Launch with multiple cores
   parallel_matmul_kernel[8](a, b, c, m, k, n)  # Use 8 cores

Flexible Tile Shapes
~~~~~~~~~~~~~~~~~~~~

Cube operations in PyAsc2 support flexible tile shapes without strict alignment requirements:

.. code-block:: python

   @asc2.jit
   def matmul_flexible_shapes_kernel(a_ptr, b_ptr, c_ptr):
       # Various shapes are supported
       a_gm = asc2.tensor(a_ptr, [1, 32])    # Small tiles
       b_gm = asc2.tensor(b_ptr, [32, 11])
       c_gm = asc2.tensor(c_ptr, [1, 11])
       
       a = asc2.load(a_gm, [0, 0], [1, 32], asc2.TileLocation.L0A)
       b = asc2.load(b_gm, [0, 0], [32, 11], asc2.TileLocation.L0B)
       c = a @ b
       asc2.store(c, c_gm, [0, 0])

**Supported Shape Examples:**

* ``[1, 32] × [32, 11]`` - Small tiles
* ``[11, 19] × [19, 41]`` - Irregular shapes
* ``[47, 21] × [21, 35]`` - Non-power-of-2
* ``[64, 128] × [128, 256]`` - Standard sizes

**Shape Requirements:**

* Must be 2D tiles
* Shapes must be compatible: ``A[M, K] × B[K, N]``
* K dimension must match

Limitations and Constraints
----------------------------

Memory Location Constraints
~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Matrix A must be in L0A (not L0B, L1, or UB)
* Matrix B must be in L0B (not L0A, L1, or UB)
* Result must be in L0C (not L1 or UB)
* L1 tiles cannot be used directly in matmul - must be copied to L0A or L0B

Transpose Constraints
~~~~~~~~~~~~~~~~~~~~~

* Standalone transpose on L0A/L0B after matmul is NOT supported
* Transpose on L0C is NOT supported
* Multiple chained transposes on cube tiles are NOT supported

Accumulator Constraints
~~~~~~~~~~~~~~~~~~~~~~~

* Cannot use regular tiles as accumulators - must use :func:`~asc2.zeros_acc`
* Accumulator must be float32
* Accumulator must be in L0C
* Accumulator must be 2D

HF32 Mode Constraints
~~~~~~~~~~~~~~~~~~~~~

* HF32 mode only works with float32 inputs

Operations Order Constraints
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* ReLU must be applied to L0C directly before moving to other locations
* Cannot apply unary/binary ops on L0A/L0B tiles before matmul

Unsupported Features
~~~~~~~~~~~~~~~~~~~~

* **Data Types:** INT8/INT4 quantization, unsigned integers, cast to int types
* **Quantization:** DeqScalar parameter, dequantization modes (int32 → float16), quantization to int8, vector quantization (VDEQF16, VQF322B8_PRE, VREQ8)
* **Dimensions:** Batch matmul, 3D tensors
* **Gemm from L1:** Direct L1 → L0C matmul operation is not supported
* **L0C → UB via FixPipe:** Copying data from L0C to UB through FixPipe operation is not yet implemented
* **Sparse matrices:** Sparse matrix multiplication is not supported

Best Practices
--------------

Loop Unrolling
~~~~~~~~~~~~~~

Improve performance with loop unrolling in tiled execution:

.. code-block:: python

   for i in asc2.range(k_tiles, unroll_factor=4):
       asc2.matmul_acc(acc, a_tile, b_tile)

Multi-Core Parallelization
~~~~~~~~~~~~~~~~~~~~~~~~~~

Distribute large matrices across multiple cores:

.. code-block:: python

   # Good: Parallel execution
   parallel_matmul_kernel[8](...)  # Use 8 cores

   # Avoid: Single core for large matrices
   matmul_kernel[1](...)  # Only 1 core

Memory Flow Pattern
~~~~~~~~~~~~~~~~~~~

Follow the standard memory flow pattern:

.. code-block::

   # Standard flow
   GM ➔ L1 ➔ L0A/L0B ➔ L0C ➔ GM

   # With chaining
   GM ➔ L1 ➔ L0A/L0B ➔ L0C ➔ L1 ➔ L0A ➔ L0C ➔ GM

Complete Example
----------------

.. code-block:: python

   import asc2
   import torch

   @asc2.jit
   def complete_matmul_pipeline(a_ptr, b_ptr, c_ptr, m, k, n, k_tiles):
       a_gm = asc2.tensor(a_ptr, [m, k])
       b_gm = asc2.tensor(b_ptr, [k, n])
       c_gm = asc2.tensor(c_ptr, [m, n])
       
       # Initialize accumulator
       acc = asc2.zeros_acc([m, n], dtype=asc2.float32)
       
       # K-axis tiling with loop unrolling
       tile_k = k // k_tiles
       for i in asc2.range(k_tiles, unroll_factor=4):
           # Load tiles (internally GM → L1 → L0A/L0B)
           a_tile = asc2.load(a_gm, [0, i * tile_k], [m, tile_k], asc2.TileLocation.L0A)
           b_tile = asc2.load(b_gm, [i * tile_k, 0], [tile_k, n], asc2.TileLocation.L0B)
           
           # Accumulate
           asc2.matmul_acc(acc, a_tile, b_tile)
       
       # Apply ReLU and cast to float16 (fused)
       result = asc2.relu(acc).to(asc2.float16)
       
       # Store result
       asc2.store(result, c_gm, [0, 0])

   # Launch with multiple cores
   m, k, n, k_tiles = 256, 512, 128, 8
   a = torch.rand(m, k, dtype=torch.float16)
   b = torch.rand(k, n, dtype=torch.float16)
   c = torch.zeros(m, n, dtype=torch.float16)

   complete_matmul_pipeline[4](a, b, c, m, k, n, k_tiles)
