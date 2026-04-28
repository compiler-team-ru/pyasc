# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import pickle
import inspect
from dataclasses import fields
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, overload, Union

from pathlib import Path
import numpy as np

from ..codegen.function import Function, P, T
from ..codegen.function_visitor import CodegenOptions, FunctionVisitor
from ..codegen.specialization import BaseArgType, PointerArgType, PlainArgType, Specialization, StructArgType, IRArgType
from ..common.compat import get_annotations, merge_dict
from ..language.core.dtype import DataType, KnownTypes as KT
from ..language.core.range import range as asc_range
from ..language.core.struct import Struct
from ..language.core.utils import global_builder
from ..lib import runtime as rt
from .._C import ir
from .compiler import CompileOptions, Compiler
from .kernel_meta import CompiledKernel, LaunchedKernel
from .launcher import LaunchOptions, Launcher
from .cache import get_cache_manager, get_file_cache_key, get_mem_cache_key


class JITFunction(Function[P, T]):
    codegen: Type[FunctionVisitor] = FunctionVisitor
    compiler: Type[Compiler] = Compiler
    launcher: Type[Launcher] = Launcher

    def __init__(self, fn: Callable[P, T], **options):
        super().__init__(fn)
        clashed_args = self.get_clashed_args(fn)
        if clashed_args:
            raise RuntimeError("The following argument names conflict with JIT configuration options: " +
                               ", ".join(clashed_args))
        unknown_options = set(options).difference(self.get_config_keywords())
        if unknown_options:
            raise RuntimeError("The following option names are unknown: " + ", ".join(unknown_options))
        self.default_options: Dict[str, Any] = options
        self.launch_options = LaunchOptions()
        self.kernel_cache: Dict[str, CompiledKernel] = {}

    def __getitem__(self, user_launch_options: Union[int, tuple]) -> Callable:
        try:
            if isinstance(user_launch_options, int):
                self.launch_options = LaunchOptions(core_num=user_launch_options)
            else:
                self.launch_options = LaunchOptions(*user_launch_options)
        except Exception as e:
            raise TypeError("Parse user launch options failed") from e
        else:
            return self._run

    @staticmethod
    def get_clashed_args(fn: Callable) -> List[str]:
        keywords = set(__class__.get_config_keywords())
        signature = inspect.signature(fn)
        keywords.intersection_update(signature.parameters)
        return list(keywords)

    @staticmethod
    def get_arg_type(value: Any) -> BaseArgType:
        if isinstance(value, bool):
            return PlainArgType(KT.bool_)
        if isinstance(value, int):
            return PlainArgType(KT.int_)
        if isinstance(value, float):
            return PlainArgType(KT.float_)
        if isinstance(value, np.ndarray):
            return PointerArgType(DataType(str(np.dtype(value.dtype))))
        if isinstance(value, np.generic):
            return PlainArgType(DataType(str(value.dtype)))
        if isinstance(value, Struct):
            return StructArgType(type(value))
        try:
            import torch
            if isinstance(value, torch.Tensor):
                dtype_str = str(value.dtype)
                dtype = DataType(dtype_str[6:] if dtype_str.startswith("torch.") else dtype_str)
                return PlainArgType(dtype) if value.dim() == 0 else PointerArgType(dtype)
        except ModuleNotFoundError:
            pass
        # Mock arguments
        if isinstance(value, MockTensor):
            return PointerArgType(value.dtype)
        if isinstance(value, MockValue):
            return PlainArgType(value.dtype)
        raise TypeError(f"Argument type in JIT function is not supported: {value.__class__.__name__}")

    @staticmethod
    def extract_kwargs(dataclass: Type, base_kwargs: Dict[str, Any]) -> Any:
        keywords = map(lambda f: f.name, fields(dataclass))
        kwargs = {}
        for keyword in keywords:
            if keyword in base_kwargs:
                kwargs[keyword] = base_kwargs[keyword]
                del base_kwargs[keyword]
        return dataclass(**kwargs)

    @staticmethod
    def create_context() -> ir.Context:
        context = ir.Context()
        context.disable_multithreading()
        ir.load_dialects(context)
        return context

    @staticmethod
    def get_arg_dtype(arg_type: BaseArgType) -> str:
        if isinstance(arg_type, PointerArgType):
            return str(arg_type.dtype)
        if isinstance(arg_type, PlainArgType):
            return str(arg_type.dtype)
        if isinstance(arg_type, IRArgType):
            return str(arg_type.py_type)
        if isinstance(arg_type, StructArgType):
            return str(arg_type.py_type)
        raise NotImplementedError(f"Not implemented for {arg_type.__class__.__name__}")

    @classmethod
    def get_config_keywords(cls) -> List[str]:
        attr = "_config_keywords"
        cached_keywords = getattr(cls, attr, None)
        if cached_keywords is not None:
            return cached_keywords
        keywords = []
        for dataclass in CodegenOptions, CompileOptions, LaunchOptions:
            keywords.extend(f.name for f in fields(dataclass))
        setattr(cls, attr, keywords)
        return keywords

    def _gen_cache_factors(self, arg_types, constexprs, codegen_options, compile_options) -> str:
        cache_factors = []
        separator = ";"
        key = separator.join([f"{name}={val}" for name, val in vars(codegen_options).items()])
        cache_factors.append(key)
        key = separator.join([f"{name}={val}" for name, val in vars(compile_options).items()])
        cache_factors.append(key)
        key = separator.join([f"{name}={repr(val)}" for name, val in constexprs.items()])
        cache_factors.append(key)
        key = separator.join(
            [f"{name}={val.__class__.__name__}:{self.get_arg_dtype(val)}" for name, val in arg_types.items()])
        cache_factors.append(key)
        key = f"fn_name={self.fn_name}"
        cache_factors.append(key)

        separator = "__"
        cache_factors = separator.join(cache_factors)
        return cache_factors

    def _cache_kernel(self, runtime_args, constexprs, codegen_options,
                      compile_options) -> Tuple[CompiledKernel, Optional[str]]:
        arg_types = {name: self.get_arg_type(value) for name, value in runtime_args.items()}
        cache_factors = self._gen_cache_factors(arg_types, constexprs, codegen_options, compile_options)
        mem_cache_key = get_mem_cache_key(cache_factors)
        kernel = self.kernel_cache.get(mem_cache_key, None)
        enable_cache = not compile_options.always_compile
        if enable_cache and kernel is not None:
            return kernel, mem_cache_key

        file_cache_key = get_file_cache_key(self.cache_key, cache_factors)
        file_cache_manager = get_cache_manager(file_cache_key)
        kernel_file_name = self.fn.__name__ + ".o"
        cached_kernel_file = file_cache_manager.get_file(kernel_file_name)

        if enable_cache and cached_kernel_file is not None:
            dst = Path(cached_kernel_file)
            with open(dst, 'rb') as file:
                kernel = pickle.load(file)
        else:
            mod = self._run_codegen(Specialization(arg_types, constexprs), codegen_options)
            kernel = self._run_compiler(mod, compile_options)
            kernel_bin = pickle.dumps(kernel)

        if enable_cache and cached_kernel_file is None:
            file_cache_manager.put(kernel_bin, kernel_file_name)
            self.kernel_cache[mem_cache_key] = kernel

        return kernel, mem_cache_key if enable_cache else None

    def _run_codegen(self, spec: Specialization, options: CodegenOptions) -> ir.ModuleOp:
        self.context = self.create_context()
        if not options.ir_multithreading:
            self.context.disable_multithreading()
        try:
            global_builder.set_ir_builder(self.context)
            visitor = self.codegen(self.src, spec, self.fn.__globals__, self.location, options, is_kernel=True)
            visitor.visit(self.node)
            return global_builder.get_ir_module()
        finally:
            global_builder.teardown()

    def _run_compiler(self, mod: ir.ModuleOp, options: CompileOptions) -> CompiledKernel:
        compiler = self.compiler(options)
        return compiler.run(mod, self.fn.__name__)

    def _run_launcher(self, kernel: CompiledKernel, options: LaunchOptions, runtime_args: Tuple[Any],
                      mem_cache_key: Optional[str]) -> None:
        enable_cache = mem_cache_key is not None

        def save_launched_kernel(handle: rt.Function) -> None:
            if enable_cache:
                self.kernel_cache[mem_cache_key] = LaunchedKernel.from_compiled(kernel, handle)

        launcher = self.launcher(options)
        launcher.run(kernel, self.fn.__name__, runtime_args, not enable_cache, save_launched_kernel)

    def _run(self, *args: P.args, **kwargs: P.kwargs) -> None:
        kwargs = merge_dict(self.default_options, kwargs)
        codegen_options = self.extract_kwargs(CodegenOptions, kwargs)
        compile_options = self.extract_kwargs(CompileOptions, kwargs)
        call_args = inspect.signature(self.fn).bind(*args, **kwargs).arguments
        annotations = get_annotations(self.fn)
        runtime_args, constexprs = self.split_args(call_args, annotations)
        kernel, mem_cache_key = self._cache_kernel(runtime_args, constexprs, codegen_options, compile_options)
        self._run_launcher(kernel, self.launch_options, tuple(runtime_args.values()), mem_cache_key)


@overload
def jit(fn: Callable[P, T]) -> JITFunction[P, T]:
    """Instantiate JIT function without additional options"""
    ...


@overload
def jit(**options) -> Callable[[Callable[P, T]], JITFunction[P, T]]:
    """Instantiate JIT function and provide options (see CodegenOptions, CompileOptions, LaunchOptions)"""
    """for exmaple, use @jit(debug = True) to pass True to CompileOptions.debug"""
    ...


def jit(fn: Optional[Callable[P, T]] = None, **options):
    options.setdefault("custom_builtins", {"range": asc_range})

    def decorator(fn: Callable[P, T]) -> JITFunction[P, T]:
        return JITFunction(fn, **options)

    if fn is None:
        return decorator
    return decorator(fn)


class MockTensor:

    def __init__(self, dtype: DataType):
        self.dtype = dtype


class MockValue:

    def __init__(self, dtype: DataType):
        self.dtype = dtype
