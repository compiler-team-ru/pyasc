# Software pipelining (double-buffering)

## Overview

Double-buffering is managed by developer with `unroll_factor` parameter in user
loop operator. The `unroll_factor` parameter is a loop optimization option
available in `asc2.range()` that controls loop unrolling. Loop unrolling is a
compiler optimization technique that replaces loop iterations with explicit
sequential code, reducing loop overhead and enabling better instruction-level
parallelism. As result the loop body is increased by fuctor `unroll_factor`.

To support cases when `unroll_factor` is not divisor for number of loop
iterations **tail loop** is created. If compiler gets information about
**tail loop** boudaries it can eliminate it (if zero iterations) or reduce
to code block without loop (if one iteration).

## Parameter Definition

### Syntax
```python
for i in asc2.range(start, stop, step, unroll_factor=2, parallel=False):
    # loop body
    pass
```
Unrolling is done during compilation and the value must be available in compile
time.
- **Default Value**: `1` (no unrolling)
- **Minimum Value**: `1` (unroll factor must be 1 or greater)
- **Type**: `int`
- **Scope**: Only applicable to `asc2.range()` loops
Recommended value is 2. 

## Usage Examples

### Example 1: Default Behavior (No Unrolling)
```python
import asc2

# Default unroll_factor=1 (no unrolling)
@asc2.jit()
def simple_kernel(x: asc.GlobalAddress, y: asc.GlobalAddress, size: int):
    x_gm = asc2.tensor(x, [size])
    y_gm = asc2.tensor(y, [size])
    
    for i in asc2.range(size):
        # Loop remains as-is
        x_val = asc2.load(x_gm, [1], offsets=[i])
        y_val = asc2.load(y_gm, [1], offsets=[i])
        result = x_val + y_val
        ascasc2.store(result, y_gm, offsets=[i])
```

### Example 2: Recomended unrolling
```python
@asc2.jit()
def unrolled_kernel(x: asc.GlobalAddress, y: asc.GlobalAddress, size: int):
    x_gm = asc2.tensor(x, [size])
    y_gm = asc2.tensor(y, [size])
    
    for i in asc2.range(size, unroll_factor=2):
        # Loop body will be unrolled 2 times
        x_val = asc2.load(x_gm, [1], offsets=[i])
        y_val = asc2.load(y_gm, [1], offsets=[i])
        result = x_val + y_val
        asc2.store(result, y_gm, offsets=[i])
```

### Example 3: Unrolling with Parallel Execution
`parallel` parameter enables parallel execution of store operation on `n`
iteration with load on `n+1` iteration. The optimization works both for unrolled
and not-unrolled iterations.
```python
# Combine unrolling with parallel execution
@asc2.jit()
def parallel_unroll(x: asc.GlobalAddress, y: asc.GlobalAddress, size: int):
    x_gm = asc2.tensor(x, [size])
    y_gm = asc2.tensor(y, [size])
    
    for i in asc2.range(size, unroll_factor=2, parallel=True):
        # Loop unrolled and executed in parallel
        x_val = asc2.load(x_gm, [1], offsets=[i])
        y_val = asc2.load(y_gm, [1], offsets=[i])
        result = x_val + y_val
        asc2.store(result, y_gm, offsets=[i])
```

### Limitations & Recommendations

**No nested unroll**: Unroll for nested loops is not supported.
**Recommended factor**: Recommended unroll factor is 2 for most cases due to
HW design. Efficient operator will fully utilize one of the execution pipelines
(e.g. MTE1, MTE2, MTE3, VECTOR or CUBE).
**Increase code size**: Unrolling causes code size grow. The following may
cause lowering performance gain or performance degradation:
 - inital programm load time increases;
 - increases icache miss rate (if program or loop block increases icashe size).

## Implementation Details

### Compilation Pipeline

The `unroll_factor` parameter triggers two main passes in the compilation pipeline:

#### 1. TagUnrollGroups Pass
**Purpose**: Identifies and groups operations that should be unrolled together

**Process**:
1. Scan all `asc2.range()` loops with `unroll_factor > 1`
2. Identify contiguous groups of operations within loop body
3. Tag operations with `unroll_group` attribute
4. Group operations to maintain data dependencies

**Code Location**: `lib/Dialect/AscTile/Transforms/TagUnrollGroups.cpp`

#### 2. UnrollLoop Pass
**Purpose**: Physically unrolls loops based on unroll_factor

**Process**:
1. For each loop with `unroll_factor > 1`
2. Clone loop body `unroll_factor` times
3. Adjust iteration indices for each unrolled instance
4. Remove original loop structure
5. Clean up temporary attributes

**Code Location**: `lib/Dialect/AscTile/Transforms/UnrollLoop.cpp`

### IR Transformation

The unrolling process transforms the IR as follows:

#### Before Unrolling
```mlir
# Original loop structure
scf.for %arg0 = %start to %stop step %step {
  %0 = asctile.load %gm_tensor[%arg0] : tensor_type
  %1 = asctile.add %0, %0 : tensor_type
  asctile.store %1, %gm_tensor[%arg0] : tensor_type
}
```

#### After Unrolling (unroll_factor=2)
```mlir
scf.for %arg0 = %start to %stop step %step * 2 {
  # Unrolled loop (2 iterations expanded)
  %0 = asctile.load %gm_tensor[%start] : tensor_type
  %1 = asctile.add %0, %0 : tensor_type
  asctile.store %1, %gm_tensor[%start] : tensor_type

  %2 = asctile.load %gm_tensor[%start + %step] : tensor_type
  %3 = asctile.add %2, %2 : tensor_type
  asctile.store %3, %gm_tensor[%start + %step] : tensor_type
}
  # Number of iterations to be procesed in the tail loop
  %4 = arith.remsi %stop, 2: i32
# Tail iterations still in loop form
scf.for %arg0 = %stop - %4 to %stop step %step {
  # ... loop body
}
```

## Conclusion

The `unroll_factor` parameter is a powerful optimization tool for improving loop performance in pyasc kernels. When used appropriately, it can significantly reduce loop overhead and improve instruction-level parallelism. However, it requires careful consideration of loop characteristics, operation complexity, and hardware constraints.
