# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from typing import Callable, Optional, Tuple, TypeVar, Union, overload

from ..core.dtype import DataType, KnownTypes as KT
from ..core.ir_value import PlainValue, RuntimeFloat, RuntimeNumeric, IRHandle, materialize_ir_value as _mat
from ..core.utils import global_builder, require_jit
from .local_tensor import LocalTensor, bind_tensor_method
from .validation import check_dtype, check_runtime_float, check_type

T = TypeVar("T")


def op_unary_impl(input: Union[LocalTensor, RuntimeNumeric], build_float: Callable[..., IRHandle],
                  build_int: Optional[Callable[..., IRHandle]] = None, support_dtypes: Optional[Tuple[DataType]] = None,
                  support_scalar: bool = False) -> Union[LocalTensor, PlainValue]:
    constraint = Union[LocalTensor, RuntimeNumeric] if support_scalar else LocalTensor
    check_type("input", input, constraint)
    if isinstance(input, LocalTensor) and support_dtypes is not None:
        check_dtype("input", input, support_dtypes)
    is_scalar = not isinstance(input, LocalTensor)
    input = _mat(input, KT.float32) if is_scalar else input  # TODO: infer dtype using builders availability
    dtype = input.dtype
    if dtype.is_float() and build_float is not None:
        handle = build_float(input.to_ir())
    elif dtype.is_signed() and build_int is not None:
        handle = build_int(input.to_ir())
    else:
        raise RuntimeError(f"Input tensor dtype is not supported: {dtype}")
    if is_scalar:
        return PlainValue(handle)
    return LocalTensor(handle)


def set_docstring(name: str, support_dtypes: Tuple[DataType], support_scalar: bool = False) -> Callable[[T], T]:
    dtypes_str = ", ".join(f"``{dtype}``" for dtype in support_dtypes)
    tensor_info = "tensor or scalar" if support_scalar else "tensor"
    examples = f"""
        Compute the element-wise {name} of all tensor elements: ::

            input = asc2.copy_in(tensor, [0, 0], [128, 256])
            result = asc2.{{fn_name}}(input)
        """
    if support_scalar:
        examples += f"""
        Compute the {name} of given scalar value: ::

            result = asc2.{{fn_name}}(1.0)
        """

    def decorator(fn: T) -> T:
        doc = f"""
    Computes the element-wise {name} of ``input``.

    The supported data types for the input are: {dtypes_str}.

    Args:
        input: The input value ({tensor_info})

    Returns:
        LocalTensor: The result tensor

    Raises:
        RuntimeError: If the input dtype is not supported for this operation

    Examples:
        {examples}
        """
        fn.__doc__ = doc.format(fn_name=fn.__name__)
        return fn

    return decorator


@bind_tensor_method
@require_jit
@set_docstring("cosine", support_dtypes=(KT.float16, KT.float32))
def cos(input: LocalTensor) -> LocalTensor:
    builder = global_builder.get_ir_builder()
    return op_unary_impl(input, builder.create_math_CosOp, support_dtypes=(KT.float16, KT.float32))


@bind_tensor_method
@require_jit
@set_docstring("sine", support_dtypes=(KT.float16, KT.float32))
def sin(input: LocalTensor) -> LocalTensor:
    builder = global_builder.get_ir_builder()
    return op_unary_impl(input, builder.create_math_SinOp, support_dtypes=(KT.float16, KT.float32))


@bind_tensor_method
@require_jit
@set_docstring("tangent", support_dtypes=(KT.float16, KT.float32))
def tan(input: LocalTensor) -> LocalTensor:
    builder = global_builder.get_ir_builder()
    return op_unary_impl(input, builder.create_math_TanOp, support_dtypes=(KT.float16, KT.float32))


@bind_tensor_method
@require_jit
@set_docstring("hyperbolic sine", support_dtypes=(KT.float16, KT.float32))
def sinh(input: LocalTensor) -> LocalTensor:
    builder = global_builder.get_ir_builder()
    return op_unary_impl(input, builder.create_math_SinhOp, support_dtypes=(KT.float16, KT.float32))


@bind_tensor_method
@require_jit
@set_docstring("hyperbolic cosine", support_dtypes=(KT.float16, KT.float32))
def cosh(input: LocalTensor) -> LocalTensor:
    builder = global_builder.get_ir_builder()
    return op_unary_impl(input, builder.create_math_CoshOp, support_dtypes=(KT.float16, KT.float32))


@bind_tensor_method
@require_jit
@set_docstring("hyperbolic tangent", support_dtypes=(KT.float16, KT.float32))
def tanh(input: LocalTensor) -> LocalTensor:
    builder = global_builder.get_ir_builder()
    return op_unary_impl(input, builder.create_math_TanhOp, support_dtypes=(KT.float16, KT.float32))


@overload
def exp(input: LocalTensor) -> LocalTensor:
    ...


@overload
def exp(input: RuntimeFloat) -> PlainValue:
    ...


@bind_tensor_method
@require_jit
@set_docstring("exponential", support_dtypes=(KT.float16, KT.float32), support_scalar=True)
def exp(input: Union[LocalTensor, RuntimeFloat]) -> Union[LocalTensor, PlainValue]:
    return op_unary_impl(input,
                         global_builder.get_ir_builder().create_math_ExpOp, support_dtypes=(KT.float16, KT.float32),
                         support_scalar=True)


@bind_tensor_method
@require_jit
@set_docstring("natural logarithm", support_dtypes=(KT.float16, KT.float32))
def log(input: LocalTensor) -> LocalTensor:
    return op_unary_impl(input,
                         global_builder.get_ir_builder().create_math_LogOp, support_dtypes=(KT.float16, KT.float32))


@bind_tensor_method
@require_jit
@set_docstring("logarithm (base 2)", support_dtypes=(KT.float16, KT.float32))
def log2(input: LocalTensor) -> LocalTensor:
    return op_unary_impl(input,
                         global_builder.get_ir_builder().create_math_Log2Op, support_dtypes=(KT.float16, KT.float32))


@bind_tensor_method
@require_jit
@set_docstring("floor rounding", support_dtypes=(KT.float16, KT.float32))
def floor(input: LocalTensor) -> LocalTensor:
    return op_unary_impl(input,
                         global_builder.get_ir_builder().create_math_FloorOp, support_dtypes=(KT.float16, KT.float32))


@bind_tensor_method
@require_jit
@set_docstring("ceil rounding", support_dtypes=(KT.float16, KT.float32))
def ceil(input: LocalTensor) -> LocalTensor:
    return op_unary_impl(input,
                         global_builder.get_ir_builder().create_math_CeilOp, support_dtypes=(KT.float16, KT.float32))


@bind_tensor_method
@require_jit
@set_docstring("absolute value", support_dtypes=(KT.int8, KT.int16, KT.int32, KT.int64, KT.float16, KT.float32))
def abs(input: LocalTensor) -> LocalTensor:
    builder = global_builder.get_ir_builder()
    return op_unary_impl(input, builder.create_math_AbsFOp, builder.create_math_AbsIOp,
                         support_dtypes=(KT.int8, KT.int16, KT.int32, KT.int64, KT.float16, KT.float32))


@overload
def erf(input: LocalTensor) -> LocalTensor:
    ...


@overload
def erf(input: RuntimeFloat) -> PlainValue:
    ...


@bind_tensor_method
@require_jit
@set_docstring("error function", support_dtypes=(KT.float16, KT.float32), support_scalar=True)
def erf(input: Union[LocalTensor, RuntimeFloat]) -> Union[LocalTensor, PlainValue]:
    return op_unary_impl(input,
                         global_builder.get_ir_builder().create_math_ErfOp, support_dtypes=(KT.float16, KT.float32),
                         support_scalar=True)


@bind_tensor_method
@require_jit
@set_docstring("exponential (base 2)", support_dtypes=(KT.float16, KT.float32))
def exp2(input: LocalTensor) -> LocalTensor:
    return op_unary_impl(input,
                         global_builder.get_ir_builder().create_math_Exp2Op, support_dtypes=(KT.float16, KT.float32))


@bind_tensor_method
@require_jit
@set_docstring("inverse square root", support_dtypes=(KT.float16, KT.float32))
def rsqrt(input: LocalTensor) -> LocalTensor:
    return op_unary_impl(input,
                         global_builder.get_ir_builder().create_math_RsqrtOp, support_dtypes=(KT.float16, KT.float32))


@overload
def sqrt(input: LocalTensor) -> LocalTensor:
    ...


@overload
def sqrt(input: RuntimeFloat) -> PlainValue:
    ...


@bind_tensor_method
@require_jit
@set_docstring("square root", support_dtypes=(KT.float16, KT.float32), support_scalar=True)
def sqrt(input: Union[LocalTensor, RuntimeFloat]) -> Union[LocalTensor, PlainValue]:
    return op_unary_impl(input,
                         global_builder.get_ir_builder().create_math_SqrtOp, support_dtypes=(KT.float16, KT.float32),
                         support_scalar=True)


@bind_tensor_method
@require_jit
@set_docstring("ReLU value", support_dtypes=(KT.float16, KT.float32))
def relu(input: LocalTensor) -> LocalTensor:
    return op_unary_impl(input,
                         global_builder.get_ir_builder().create_asctile_ReluOp, support_dtypes=(KT.float16, KT.float32))


@bind_tensor_method(name="__neg__", unary_op="-")
@require_jit
@set_docstring("negation", support_dtypes=(KT.int16, KT.int32, KT.int64, KT.float16, KT.bfloat16, KT.float32))
def negative(input: LocalTensor) -> LocalTensor:
    check_dtype("input", input, (KT.int16, KT.int32, KT.int64, KT.float16, KT.bfloat16, KT.float32))
    return input * (-1)


@bind_tensor_method
@require_jit
def softmax(input: LocalTensor) -> LocalTensor:
    """
    Computes the row-wise softmax of ``input``.

    For 2D tensors, softmax is applied independently along the last dimension for each row.
    For 1D tensors, softmax is applied over all elements.

    The supported data types for the input are: ``float16``, ``float32``.

    Args:
        input: The input tensor (1D or 2D)

    Returns:
        LocalTensor: The result tensor with the same shape as input

    Raises:
        RuntimeError: If the input dtype is not supported or input has more than 2 dimensions

    Examples:
        Compute row-wise softmax for a 2D tensor: ::

            input = asc2.copy_in(x_gm, [0, 0], [64, 1024])
            result = asc2.softmax(input)  # softmax applied independently to each of 64 rows

        Compute softmax for a 1D tensor: ::

            input = asc2.copy_in(x_gm, [0], [1024])
            result = asc2.softmax(input)  # softmax applied over all 1024 elements
    """
    check_type("input", input, LocalTensor)
    check_dtype("input", input, (KT.float16, KT.float32))
    if len(input.shape) > 2:
        raise RuntimeError("Tensor dimensionality greater than two is not supported")
    handle = global_builder.get_ir_builder().create_asctile_SoftmaxOp(input.to_ir().get_type(), input.to_ir())
    return LocalTensor(handle)


@require_jit
def rms_norm(input: LocalTensor, gamma: LocalTensor, epsilon: RuntimeFloat) -> LocalTensor:
    """
    Computes Root Mean Square Layer Normalization of ``input``.

    RMSNorm normalizes the input by the root mean square and scales by learnable parameters ``gamma``.
    This is commonly used in transformer architectures as an alternative to LayerNorm.

    The supported data types for the inputs are: ``float16``, ``float32``.

    Args:
        input: The input tensor to normalize (1D or 2D)
        gamma: The scale parameter tensor (1D, same length as last dimension of input)
        epsilon: Small constant added for numerical stability

    Returns:
        LocalTensor: The normalized tensor with same shape as input

    Raises:
        TypeError: If input or gamma is not a LocalTensor
        RuntimeError: If input dtype is not supported, input has more than 2 dimensions,
            or gamma dtype is not supported

    Examples:
        Apply RMSNorm to a 2D tensor: ::

            input = asc2.copy_in(x_gm, [0, 0], [32, 128])
            gamma = asc2.copy_in(gamma_gm, [0], [128])
            output = asc2.rms_norm(input, gamma, 1e-5)

        Apply RMSNorm to a 1D tensor: ::

            input = asc2.copy_in(x_gm, [0], [128])
            gamma = asc2.copy_in(gamma_gm, [0], [128])
            output = asc2.rms_norm(input, gamma, 1e-6)
    """
    check_type("input", input, LocalTensor)
    check_dtype("input", input, (KT.float16, KT.float32))
    check_type("gamma", gamma, LocalTensor)
    check_dtype("gamma", gamma, (KT.float16, KT.float32))
    check_runtime_float("epsilon", epsilon)
    if len(input.shape) > 2:
        raise RuntimeError("Tensor dimensionality greater than two is not supported.")
    handle = global_builder.get_ir_builder().create_asctile_RmsNormOp(input.to_ir().get_type(), input.to_ir(),
                                                                      gamma.to_ir(),
                                                                      _mat(epsilon, input.dtype).to_ir())
    return LocalTensor(handle)
