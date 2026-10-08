# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from .libpyasc import ir, passes, translation


def _register_submodules():
    """Re-register pybind11 submodules in ``sys.modules``.

    The extension registers its submodules (``asc._C.libpyasc.ir`` etc.) only
    when its module init runs; a later import of the cached extension does not
    re-run init. Tools that snapshot and restore ``sys.modules`` around import
    probing (coverage.py's ``sys_modules_saved`` source resolution) can evict
    those entries, after which dotted-name imports — e.g. unpickling
    ``asc._C.libpyasc.ir.KernelArgument`` from the kernel file cache — fail
    with "'asc._C.libpyasc' is not a package". This package __init__ re-runs on
    every fresh import, so restore the mapping here.
    """
    import sys
    import types

    from . import libpyasc

    def register(module, dotted_name):
        sys.modules.setdefault(dotted_name, module)
        for attr_name in dir(module):
            child = getattr(module, attr_name, None)
            if isinstance(child, types.ModuleType) and \
                    getattr(child, "__name__", None) == f"{dotted_name}.{attr_name}":
                register(child, f"{dotted_name}.{attr_name}")

    register(libpyasc, f"{__name__}.libpyasc")


_register_submodules()
del _register_submodules

__all__ = [
    "ir",
    "passes",
    "translation",
]
