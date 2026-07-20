# Quick start

```{contents} Table of Contents
:local:
```

## Install required packages

1. [Build PyAsc2 from sources](../installation/build-from-source.rst).
2. [Install CANN packages](../installation/setup-runtime-env.rst) if not installed.

## Verify the installation

Run tutorials in `python/tutorials/asc2` directory, for example:

```bash
python3 python/tutorials/asc2/01-vector-add.py
```

## Operator file structure

### Kernel function

The functions which are executed on Ascend NPU must be marked with `@asc2.jit` decorator. In this case:
- Pointers to input and output tensors should have `asc2.GlobalAddress` type.
- Scalar parameters are passed as Python types (e.g. `int`, `float`).
- For optimization purposes it is recommended to pass scalar parameter as constants (e.g. `asc2.ConstExpr[int]`).

```python
@asc2.jit
def vadd_kernel(x_ptr: asc2.GlobalAddress, y_ptr: asc2.GlobalAddress, out_ptr: asc2.GlobalAddress, size: int,
                tile_size: asc2.ConstExpr[int], tile_per_block: asc2.ConstExpr[int]):
```

Tensor descriptor is created from `asc2.GlobalAddress` to represent entire tensor.

```python
x_gm = asc2.global_tensor(x_ptr, [size])
y_gm = asc2.global_tensor(y_ptr, [size])
out_gm = asc2.global_tensor(out_ptr, [size])
```

Python expressions are used to calculate offset and define the loop iterating over tiles:
- `asc2.block_idx()` function is used to get current AICORE index.
- `asc2.block_num()` function provides number of AICOREs launched.
- `unroll_factor` parameter of `asc2.range` in `for` loop can be used to manage software pipelining. Set it to `2` to enable double buffering.
- `parallel` parameter of `asc2.range` in `for` loop enable overlapping of store operation of `i`-th iteration and load of `i+1`-th iteration. It is user responsibility to ensure that there are no data dependencies between overlapped iterations.

```python
base_offset = asc2.block_idx() * tile_size * tile_per_block
for i in asc2.range(tile_per_block, unroll_factor=2):
    tile_offset = base_offset + i * tile_size
```

`asc2.copy_in` is used to create a local tensor object which is used for further calculations. Data movement from GM to UB/L1/L0A/L0B happens in this operation. If `location` parameter is set, then the corresponding memory realm is used to load data to. Otherwise it is loaded to UB.

```python
x = asc2.copy_in(x_gm, [tile_offset], [tile_size], asc2.TensorLocation.UB)
y = asc2.copy_in(y_gm, [tile_offset], [tile_size])  # location can be omitted
```

One or more operations can be applied for tiles. It is compiler responsibility to allocate required number of memory blocks in UB.

```python
out = x + y
```

`asc2.copy_out` is used to move data from L0C/UB back to GM. Source location is defined by the local tensor itself, so no `location` parameter is present in `asc2.copy_out`.

```python
asc2.copy_out(out, out_gm, [tile_offset])
```

### Host code

Regular Python function may be used to invoke the kernel function:

```python
def vadd_launch(x: np.ndarray, y: np.ndarray) -> np.ndarray:
```

It is usually used to allocate tensors and calculate invocation parameters:

```python
out = np.empty_like(x)
size = out.size
core_num = 16
tile_size = 128
num_tiles = asc2.ceildiv(size, tile_size)
```

For the kernel invocation, number of AICOREs should be provided in brackets:

```python
vadd_kernel[core_num](x, y, out, size, tile_size, asc2.ceildiv(num_tiles, core_num))
return out
```

Example:

```python
backend = asc2.Backend.Model # can be "Model" for simulator or "NPU" for device
soc_version = asc2.Platform.Ascend950PR_9599 # Device version
device_id = 0 # might be necessary to provide if more than one NPU device is present in the system
asc2.set_platform(backend, soc_version, device_id)
rng = np.random.default_rng(seed=2026)
size = 8192
x = rng.random(size, dtype=np.float32) * 10
y = rng.random(size, dtype=np.float32) * 10
out = vadd_launch(x, y)
np.testing.assert_allclose(out, x + y)
```

## Debug capabilities

### Capture build artifacts

`PYASC_DUMP_PATH` environment variable can be defined to make PyAsc2 compiler keep generated files in the directory, for example:

``` bash
PYASC_DUMP_PATH=dumps python3 python/tutorials/asc2/01-vector-add.py
```

As a result, the following directory structure is created in the current directory:

``` bash
dumps/
  ascendc.cpp  # generated Ascend C file
  ascir.mlir   # final IR which is used to emit Ascend C
  binary.o     # object file produced by bisheng compiler
  codegen.mlir # input IR captured from JIT function

```

> Note: it is necessary to set `always_compile=True` JIT option to avoid caching and trigger the compilation each time.

### Tune generated Ascend C code

The Ascend C code generated from PyAsc2 can be injected back in PyAsc2 python code:
- pass content of generated kernel as parameter to {py:func}`asc2.inline` method;
- comment out `TPipe` definition in the code (see an example below).

```python
@asc2.jit(kernel_type=asc2.KernelType.AIV_ONLY)
def vadd_kernel(x_ptr: asc2.GlobalAddress, y_ptr: asc2.GlobalAddress, out_ptr: asc2.GlobalAddress, size: int,
                tile_size: asc2.ConstExpr[int], tile_per_block: asc2.ConstExpr[int]):
    asc.inline('''
constexpr int32_t c32_i32 = 32;
constexpr int64_t c128_i64 = 128;
.       .       .
constexpr int32_t c128_i32 = 128;
// AscendC::TPipe v5;
AscendC::GlobalTensor<float> v6;
v6.SetGlobalBuffer(v1_x_ptr);
.       .       .
  get_buf(PIPE_MTE3, 5, 0);
  AscendC::DataCopyPad(v55, v11, v58);
  rls_buf(PIPE_MTE3, 5, 0);
}
return;
    ''')
```

> Note: co-existence of PyAsc2 API and `asc.inline` in the same kernel is not supported for now.

### Passing arguments to inline code

Kernel arguments can be passed to `asc2.inline` via the `args` parameter. Use `$<index>` placeholders (e.g., `$0`, `$1`, `$2`) in the code string to reference arguments by their position in the list:

```python
@asc2.jit
def kernel(x_ptr: asc2.GlobalAddress, y_ptr: asc2.GlobalAddress, out_ptr: asc2.GlobalAddress, size: int):
    asc.inline('''
        auto input_x = $0;
        auto input_y = $1;
        auto output = $2;
        int64_t length = $3;

        AscendC::GlobalTensor<float> x_gm;
        x_gm.SetGlobalBuffer(input_x);
        AscendC::GlobalTensor<float> y_gm;
        y_gm.SetGlobalBuffer(input_y);
        AscendC::GlobalTensor<float> out_gm;
        out_gm.SetGlobalBuffer(output);
    ''', [x_ptr, y_ptr, out_ptr, size])
```

Each argument is materialized as an IR value and substituted into the generated code at the corresponding placeholder position.
