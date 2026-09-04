# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from typing import Any, Iterable, List, Optional, Tuple, overload

from asc._C import ir
from asc.language.core.dtype import DataType, KnownTypes as KT
from asc.language.core.ir_value import IRValue, PlainValue, materialize_ir_value as _mat
from asc.language.core.ops import inline as asc_inline
from asc.language.core.utils import allow_jit, global_builder, require_jit

from .global_tensor import GlobalTensor
from .local_tensor import LocalTensor
from .tensor_location import TensorLocation
from .utils import cast_tensor_location as cast_loc
from .validation import check_type, verify_shape


def escape_percent(s: str) -> str:
    return s.replace("%", "%%")


def get_format_specifier(dtype: DataType) -> str:
    if dtype.is_float():
        return "%f"
    elif dtype.is_signed():
        return "%d"
    elif dtype.is_unsigned():
        return "%u"
    raise ValueError(f"There is no format specifier to print {dtype} value")


@require_jit
def device_assert(test: Any, message: Optional[str] = None) -> None:
    """
    Check a condition during kernel execution on the device.

    If the condition is false, prints an error message with source location and stops execution.

    Args:
        test: Condition to check (boolean, comparison, or value convertible to bool).
        message: Optional error message to include in the report.

    Note:
        Only active when ``debug=True`` is set in the JIT decorator. The built-in ``assert`` 
        statement dispatches to this function inside JIT kernels.

    Examples:
        Assert that the block index is non-negative: ::

            @asctile.jit(debug=True)
            def kernel():
                asctile.device_assert(asctile.block_idx() >= 0, "block_idx must be non-negative")
    """
    check_type("message", message, Optional[str])
    message = escape_percent(message) if message is not None else ""
    global_builder.get_ir_builder().create_asctile_AssertOp(_mat(test, KT.int1).to_ir(), message)


@require_jit
def device_print(*values: Any, sep: Optional[str] = None, end: Optional[str] = None) -> None:
    """
    Print values during kernel execution on the device.

    Outputs values to the device console. Strings are printed as-is, scalars are formatted by type, 
    and tensors are dumped in full.

    Args:
        *values: Values to print (strings, scalars, or tensors).
        sep: Separator between values (default ``" "``).
        end: Line terminator (default ``"\n"``).

    Note:
        Only active when ``debug=True`` is set in the JIT decorator. The built-in ``print()`` 
        function dispatches to this function inside JIT kernels.

    Examples:
        Print a mix of strings, scalars, and tensors: ::

            @asctile.jit(debug=True)
            def kernel():
                x = asctile.zeros([128], asctile.float32)
                asctile.device_print("block", asctile.block_idx(), "tensor", x)
    """
    check_type("sep", sep, Optional[str])
    check_type("end", end, Optional[str])
    fmt_parts: List[str] = []
    ir_args: List[PlainValue] = []
    sep = escape_percent(sep) if sep is not None else " "
    end = escape_percent(end) if end is not None else "\n"
    builder = global_builder.get_ir_builder()

    def flush():
        nonlocal fmt_parts, ir_args
        if not fmt_parts:
            return
        fmt_string = "".join(fmt_parts) + end
        builder.create_asc_PrintfOp(fmt_string, [arg.to_ir() for arg in ir_args])
        fmt_parts = []
        ir_args = []

    for value in values:
        if isinstance(value, (GlobalTensor, LocalTensor)):
            flush()
            builder.create_asctile_DumpTensorOp(value.to_ir())
            continue
        if fmt_parts and sep:
            fmt_parts.append(sep)
        if isinstance(value, PlainValue):
            fmt_parts.append(get_format_specifier(value.dtype))
            ir_args.append(value)
        else:
            if isinstance(value, bool):
                value = int(value)
            fmt_parts.append(escape_percent(str(value)))
    flush()


@allow_jit
def static_assert(test: Any, message: Optional[str] = None) -> None:
    """
    Check a condition during JIT compilation on the host.

    If the condition is false, raises ``AssertionError`` and aborts compilation.

    Args:
        test: Condition to check (must be a compile-time value, not an IR value).
        message: Optional error message for the exception.

    Note:
        Always runs during compilation, regardless of the ``debug`` option. The built-in ``assert`` 
        statement does NOT dispatch to this function — it dispatches to :func:`device_assert` instead. 
        Use this function explicitly for compile-time checks.

    Examples:
        Validate a tile size constant at compile time: ::

            @asctile.jit
            def kernel(TILE: asc.ConstExpr[int]):
                asctile.static_assert(TILE % 16 == 0, "TILE must be a multiple of 16")
    """
    if isinstance(test, IRValue):
        raise TypeError(f"static_assert expects a compile-time value, got {type(test).__name__}, use device_assert")
    check_type("message", message, Optional[str])
    if not test:
        raise AssertionError(message or "")


@overload
def static_print(*values: Any, sep: Optional[str] = None, end: Optional[str] = None, **kwargs) -> None:
    ...


@allow_jit
def static_print(*args: Any, **kwargs: Any) -> None:
    """
    Print values during JIT compilation on the host.

    Outputs values to the host console. Use this to trace compile-time constants and metadata.

    Args:
        *args: Values to print.
        **kwargs: Forwarded to the built-in :func:`print`.

    Note:
        Always runs during compilation, regardless of the ``debug`` option. The built-in ``print()`` 
        function does NOT dispatch to this function — it dispatches to :func:`device_print` instead. 
        Use this function explicitly for compile-time output.

    Examples:
        Log the configured tile size at compile time: ::

            @asctile.jit
            def kernel(TILE: asc.ConstExpr[int]):
                asctile.static_print("compiling kernel with TILE =", TILE)
    """
    print(*args, **kwargs)


@require_jit
def inline(code: str, args: Optional[tuple] = None, before_function: bool = False) -> None:
    """
    Inject raw C++ (Ascend C) code into the generated kernel.

    The provided code string is emitted verbatim into the output Ascend C source at the current insertion point.
    Optional arguments can be passed to be substituted into the emitted code as IR values.

    Args:
        code: The raw C++ code string to inject into the generated kernel source. Use ``$0``, ``$1``, etc.
            as placeholders to reference arguments from the ``args`` list by index.
        args: Optional list of values to pass as arguments to the emitted code. Each value is materialized
            as an IR value and forwarded to the underlying verbatim operation. In the code string, use
            ``$<index>`` (e.g., ``$0``, ``$1``) to reference these arguments by their position in the list.
        before_function: If ``True``, emit the code before the current function body instead of at the current
            insertion point. Default is ``False``.

    Returns:
        None

    Examples:
        Inject constant declarations into the kernel: ::

            asctile.inline('''
                constexpr int32_t TOTAL_ROWS = 668;
                constexpr int32_t ELEMENTS_PER_ROW = 32;
            ''')

        Pass kernel arguments to inline code using ``$<index>`` placeholders: ::

            @asctile.jit
            def kernel(x_ptr: asctile.GlobalAddress, y_ptr: asctile.GlobalAddress, size: int):
                asctile.inline('''
                    auto input_ptr = $0;
                    auto output_ptr = $1;
                    int64_t length = $2;
                    AscendC::GlobalTensor<float> x_gm;
                    x_gm.SetGlobalBuffer(input_ptr);
                ''', [x_ptr, y_ptr, size])
    """
    return asc_inline(code, args, before_function)


@require_jit
def inline_vf(code: str, shape: Tuple[int, ...], dtype: DataType,
              inputs: Optional[Iterable[LocalTensor]] = None) -> LocalTensor:
    """
    Embed Ascend C VF (vector function) code within a kernel.

    This is an escape hatch for advanced users who need to express vector-fusion operations (e.g., Ascend C Reg calls)
    that are not covered by the built-in API. The provided code string is injected verbatim as the body of a
    ``__VEC_SCOPE__`` block in the generated Ascend C source.

    Tensors are referenced by positional placeholders: ``$0`` is always the output tensor, and ``$1``, ``$2``, ... refer
    to the input tensors in the order they appear in ``inputs``. Zero or more input tensors are allowed. Each
    placeholder will be replaced with a ``LocalTensor`` allocated for a corresponding tensor.

    All input tensors must reside in UB memory. The output tensor is always allocated in UB.

    Args:
        code: The raw Ascend C code string to embed (treated as a ``__VEC_SCOPE__`` body).
            Use ``$0`` for the output tensor and ``$1``, ``$2``, ... for input tensors.
        shape: The shape of the output tensor.
        dtype: The data type of the output tensor.
        inputs: An optional iterable of zero or more input tensors referenced as ``$1``, ``$2``, ... in the code.

    Returns:
        LocalTensor: A new UB tensor (``$0``) containing the result produced by the inline vector function.

    Raises:
        TypeError: If code is not a str, dtype is not a DataType, or any input is not a LocalTensor.
        RuntimeError: If any input tensor is not located in UB memory or shape is invalid.

    Examples:
        Embed an inline vector multiply-add (``x * y + z``) using Ascend C register API: ::

            out = asctile.inline_vf(
                '''
                auto* out_ptr = reinterpret_cast<__ubuf__ float*>($0.GetPhyAddr());
                auto* x_ptr = reinterpret_cast<__ubuf__ float*>($1.GetPhyAddr());
                auto* y_ptr = reinterpret_cast<__ubuf__ float*>($2.GetPhyAddr());
                auto* z_ptr = reinterpret_cast<__ubuf__ float*>($3.GetPhyAddr());
                AscendC::Reg::RegTensor<float> result_reg;
                . . .
                AscendC::Reg::MaskReg mask_reg = AscendC::Reg::UpdateMask<float>(mask);
                AscendC::Reg::DataCopy(x_reg, x_ptr);
                AscendC::Reg::DataCopy(y_reg, y_ptr);
                AscendC::Reg::Mul(xy_reg, x_reg, y_reg, mask_reg);
                . . .
                ''',
                x.shape, x.dtype, [x, y, z])

        In the example above, ``$0`` placeholder refers to a ``LocalTensor`` corresponding to ``out`` tensor;
        ``$1``, ``$2``, and ``$3`` refers to ``x``, ``y``, and ``z`` respectively.
    """
    check_type("code", code, str)
    check_type("dtype", dtype, DataType)
    shape = verify_shape(shape)
    ir_tiles = []
    if inputs is not None:
        for index, tensor in enumerate(inputs):
            check_type(f"inputs[{index}]", tensor, LocalTensor)
            ir_tiles.append(cast_loc(tensor, TensorLocation.UB).to_ir())
    ir_type = ir.get_asctile_LocalTensorType(shape, dtype.to_ir(), TensorLocation.UB)
    handle = global_builder.get_ir_builder().create_asctile_InlineVFOp(ir_type, ir_tiles, code)
    return cast_loc(LocalTensor(handle))
