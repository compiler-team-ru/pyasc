.. Copyright (c) 2025 Huawei Technologies Co., Ltd.
.. This program is free software, you can redistribute it and/or modify it under the terms and conditions of
.. CANN Open Software License Agreement Version 2.0 (the "License").
.. Please refer to the License for details. You may not use this file except in compliance with the License.
.. THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
.. INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
.. See LICENSE in the root of the software repository for the full text of the License.

Runtime API
===========


JIT decorator
-------------

.. autodecorator:: asc2.jit


JIT options
-----------

Attributes from the following classes can be used as keyword arguments in :py:obj:`asc2.jit` decorator.

.. tip::

    The user can mix options from different sections. For example, to disable :code:`CodegenOptions.capture_exceptions`
    and set :code:`CompileOptions.opt_level` to :code:`2`, the kernel function may be decorated as the following:

    .. code-block:: python

        @asc2.jit(capture_exceptions=False, opt_level=2)
        def kernel(x, y):
            ...

.. autoclass:: asc.CodegenOptions
   :members:
   :member-order: bysource

.. autoclass:: asc.CompileOptions
   :members:
   :member-order: bysource

.. autoclass:: asc.LaunchOptions
   :members:
   :member-order: bysource
