.. Copyright (c) 2026 Huawei Technologies Co., Ltd.
.. This program is free software, you can redistribute it and/or modify it under the terms and conditions of
.. CANN Open Software License Agreement Version 2.0 (the "License").
.. Please refer to the License for details. You may not use this file except in compliance with the License.
.. THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
.. INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
.. See LICENSE in the root of the software repository for the full text of the License.

Debug capabilities
==================

.. contents:: Table of Contents
   :local:

.. currentmodule:: asc.experimental.asctile

AscTile provides four debug operations to inspect values and check assumptions while developing kernels.
They fall into two groups, depending on **when** they run:

===========  ===================================  ==============================================  =======================
Group        Operations                           When they run                                   Needs a running kernel?
===========  ===================================  ==============================================  =======================
Device-side  ``device_print``, ``device_assert``  During kernel execution on the NPU              Yes
Host-side    ``static_print``, ``static_assert``  During JIT compilation, before the kernel runs  No
===========  ===================================  ==============================================  =======================

Use the **device-side** operations for runtime values that only exist on the device (a loaded tile, a
computed scalar), and the **host-side** operations for compile-time values (a ``ConstExpr`` argument, a
tile shape, a dtype) — they are cheaper and work without hardware.


Enabling device-side debugging
------------------------------

Device-side operations are **off by default**. To turn them on, decorate the kernel with ``debug=True``:

.. code-block:: python

   @asctile.jit(debug=True)
   def kernel(x_ptr: asctile.GlobalAddress, size: int):
       ...

When ``debug`` is not set (the default), ``device_print`` and ``device_assert`` do nothing and add no
overhead to your kernel — keep it this way for production runs.

.. warning::

   ``debug=True`` leads to slower compilation and may cause performance regressions. Enable it only while developing or
   validating correctness, and switch it off for final builds.

Device-side output is produced only when the kernel is actually executed — on the NPU or the Model
simulator; nothing is printed if the kernel is compiled but never launched. The host-side operations
(``static_print``, ``static_assert``) are different: they run during JIT compilation, before the kernel
runs, so they take effect even before the kernel is launched and without any hardware.


Device-side operations
----------------------

These run during kernel execution on the device.

.. autofunction:: device_print
   :no-index:

.. autofunction:: device_assert
   :no-index:


Host-side operations
--------------------

These run during JIT compilation, before the kernel runs, and work without any hardware.

.. autofunction:: static_print
   :no-index:

.. autofunction:: static_assert
   :no-index:


Environment variables
---------------------

These environment variables complement the debug operations and control what the toolchain keeps on
disk and whether a compiled kernel is actually launched.

``PYASC_DUMP_PATH=<dir>`` (optional)
    directory where the compiler keeps generated artifacts: intermediate IR (``.mlir``), Ascend C source
    (``.cpp``) and the object file (``.o``)

``CAMODEL_LOG_PATH=<dir>`` (optional)
    directory for the ``Model`` simulator logs; only used on the ``Model`` backend — setting it persists
    the logs instead of a temp directory removed on exit

``PYASC_DRY_RUN=1`` (optional)
    when set, the kernel is compiled but not launched on the device — useful to validate compilation
    without hardware
