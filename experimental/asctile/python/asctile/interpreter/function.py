# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import inspect
import types
from typing import Any, Callable, Dict, Set, Tuple

import numpy as np

from asc.codegen.specialization import Specialization
from asc.language.core.constexpr import ConstExpr
from asc.runtime.cache import get_mem_cache_key
from asc.runtime.jit import CompilePrereqs
from asc.runtime.launcher import LaunchOptions

from ..runtime.jit import JITFunction
from . import operations
from .patch import make_module
from .types import GlobalAddress


def wrap_argument(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return GlobalAddress(value)
    if isinstance(value, (int, float, np.generic)):
        return value
    try:
        import torch
        if isinstance(value, torch.Tensor):
            if value.dim() == 0:
                return value
            return GlobalAddress(value.numpy(force=True))
    except ModuleNotFoundError:
        pass
    raise TypeError(f"{value.__class__.__name__} arguments are not supported by the interpreter")


def make_exec_function(fn: Callable) -> types.FunctionType:
    namespace = dict(fn.__globals__)
    namespace["asctile"] = make_module()
    return types.FunctionType(fn.__code__, namespace, fn.__name__, fn.__defaults__, fn.__closure__)


class InterpretedFunction(JITFunction):

    def __init__(self, fn, **options):
        super().__init__(fn, **options)
        self._validated: Set[str] = set()

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return make_exec_function(self.fn)(*args, **kwargs)

    def _compile_and_cache(self, prereqs: CompilePrereqs) -> Tuple[Dict[str, ConstExpr], None]:
        cache_key = get_mem_cache_key(self._gen_cache_factors(prereqs))
        if prereqs.compile_options.always_compile or cache_key not in self._validated:
            self._validate(prereqs)
            self._validated.add(cache_key)
        return prereqs.constexprs, None

    def _validate(self, prereqs: CompilePrereqs) -> None:
        self._run_codegen(Specialization(prereqs.arg_types, prereqs.constexprs), prereqs.codegen_options)

    def _run_launcher(self, constexprs: Dict[str, ConstExpr], options: LaunchOptions, runtime_args: Tuple[Any],
                      mem_cache_key: Any) -> None:
        del mem_cache_key
        block_num = options.core_num
        if block_num is None:
            raise NotImplementedError("'core_num' must be set explicitly in interpreter mode")
        if block_num <= 0:
            raise ValueError("'core_num' must be positive")
        arg_values = map(wrap_argument, runtime_args)
        fn_args = tuple(
            ConstExpr.unwrap(constexprs[name]) if name in constexprs else next(arg_values)
            for name in inspect.signature(self.fn).parameters)
        kernel_fn = make_exec_function(self.fn)
        for block_idx in range(block_num):
            operations.block_context.set(block_idx, block_num)
            kernel_fn(*fn_args)
