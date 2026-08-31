<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
-->

# Automatic synchronization injection

```{contents} Table of Contents
:local:
```

## Overview

PyAsc supports automatic synchronization insertion, which is controlled by the `insert_sync` parameter. This parameter is a JIT compilation option that manages automatic insertion of synchronization instructions in PyAsc2 kernels. Automatic synchronization is **enabled by default** and is essential for correct PyAsc2 kernel execution on Ascend NPU processors.

## Parameter Definition

### Syntax

```python
@asc.jit(insert_sync=True)
def kernel_func(...):
    # kernel code
    pass
```

### Default Behavior

- **Default Value**: `True`
- **When `True`**: Forces synchronization instruction insertion
- **When `False`**: Disables automatic synchronization (for debugging only)

## Implementation Details

### Platform-Specific Behavior

The `insert_sync` parameter triggers different synchronization passes based on the target platform.

#### C220 architecture (Ascend910B, Ascend910_93)

**Passes Executed**:

1. `add_erase_sync` - Remove existing sync instructions
2. `add_hoist_que_bind` - Hoist queue binding operations
3. `add_insert_que_sync` - Insert standard synchronization instructions
4. `add_unify_pipe` - Unify pipe operations
5. `add_canonicalizer` - Canonicalize IR

**Code Location**: `lib/Dialect/Asc/Transforms/InsertSync.cpp`

#### C310 architecture (Ascend950PR_95)

**Passes Executed**:

1. `add_erase_sync` - Remove existing sync instructions
2. `add_hoist_que_bind` - Hoist queue binding operations
3. `add_insert_bufid_sync` - Insert buffer ID-based synchronization
4. `add_canonicalizer` - Canonicalize IR
5. `add_fuse_bufid_sync` - Fuse buffer ID synchronization operations
6. `add_unify_pipe` - Unify pipe operations
7. `add_canonicalizer` - Canonicalize IR

**Code Location**: `lib/Dialect/Asc/Transforms/InsertBufIdSync.cpp`

### Synchronization Pass Pipeline

When `insert_sync=True`, the following pipeline is executed:

```python
if self.options.insert_sync:
    passes.ascendc.add_erase_sync(pm)         # Remove existing sync
    passes.ascendc.add_hoist_que_bind(pm)     # Hoist queue binds

    if self.arch != CompilationArch.C310:
        # Standard sync insertion
        passes.ascendc.add_insert_que_sync(pm)
    else:
        # Buffer ID-based sync insertion
        passes.ascendc.add_insert_bufid_sync(pm)
        passes.common.add_canonicalizer(pm)
        passes.ascendc.add_fuse_bufid_sync(pm)

    passes.ascendc.add_unify_pipe(pm)         # Unify pipes
    passes.common.add_canonicalizer(pm)       # Canonicalize IR
```

### InsertQueSync Pass Implementation

The `InsertQueSync` pass (for C220 architecture platforms) uses Enque/Deque API.

For every operation producing LocalTensor object check if it is used within the same pipe:

- If so, insert `pipe_barrier` instruction. And no follow up actions are needed.
- Otherwise, find/allocate corresponding TQue object and add enque operation:

  - for each enqueued tensor, find first user and insert dequeue;
  - re-enqueue if needed for dominance;
  - replace tensor uses with dequeued version.

Additional synchronization is done for GetValue/SetValue operations. Each call is wrapped with the Set/Wait flags construction:

```cpp
int8_t v44 = v6.FetchEventID<AscendC::HardEvent::V_S>();
AscendC::SetFlag<AscendC::HardEvent::V_S>(v44);
AscendC::WaitFlag<AscendC::HardEvent::V_S>(v44);
float v45 = v26.GetValue(c0_i64);
int8_t v46 = v6.FetchEventID<AscendC::HardEvent::S_V>();
AscendC::SetFlag<AscendC::HardEvent::S_V>(v46);
AscendC::WaitFlag<AscendC::HardEvent::S_V>(v46);
```

### InsertBufIdSync Pass Implementation

The `InsertBufIdSync` pass (for C310 architecture) uses buffer ID-based synchronization:

**Key Concepts**:

1. **Buffer ID Management**: Assigns unique buffer IDs to operations. Each pipe supports up to 32 buffer IDs (0-31). Every operation is assigned with bufID corresponding to each local tensor in the operation.

2. **Pipe Detection**: Identifies which pipe each operation belongs to based on operation type:

   - `CopyToL0Op` → `PIPE_MTE1`
   - `FixpipeOp` → `PIPE_FIX`
   - `DataCopyOp` with `GlobalToLocal` direction → `PIPE_MTE2`
   - `DataCopyOp` with `LocalToGlobal` direction → `PIPE_MTE3`
   - `LocalTensorGetValueOp`, `LocalTensorSetValueOp` → `PIPE_S`
   - `MmadOp`, `MmadWithBiasOp` → `PIPE_M`
   - Default (vector compute operations) → `PIPE_V`

3. **Sync Injection**: Every operation is wrapped with `get_buf`/`rls_buf` commands where bufID corresponds to each local tensor in the operation. If operation requires several synchronization points, next synchronization pair is incapsulated in previous one.

**Buffer ID Overflow Handling**:

The pass uses circular buffer ID allocation to handle the 32-buffer limit. This means:

- Buffer IDs cycle through values 0-31 (total of 32 unique IDs)
- After ID 31, the next allocation wraps back to 0
- Multiple tensors can share the same buffer ID across different synchronization points
- The synchronization ensures correct ordering through proper `get_buf`/`rls_buf` pairing

**Algorithm**:

1. Walk through all operations in the function
2. For each operation that produces or consumes local tensors:

   - Collect all buffer IDs associated with the tensor's definition chain
   - If no buffer ID exists yet, allocate a new one (cycling 0-31)
   - Insert `get_buf` before the operation
   - Insert `rls_buf` after the operation

3. Set buffer ID attributes on operations for later use by code emission
4. For parallel loops, insert additional MTE3 synchronization flags

### Disabling Automatic Synchronization

For debugging purposes, automatic synchronization can be disabled:

```python
@asc2.jit(insert_sync=False)  # Not recommended for production
def debug_kernel(...):
    # Manual synchronization must be inserted if needed
    pass
```

> **Warning**: Disabling automatic synchronization may result in incorrect program behavior on NPU hardware. Use only for debugging purposes.

## Platform-Specific Considerations

### C220 architecture (Ascend910B, Ascend910_93)

- Uses `InsertQueSync` pass
- Flag-based synchronization
- Pipe barrier operations

### C310 architecture (Ascend950PR_95)

- Uses `InsertBufIdSync` pass
- Buffer ID-based synchronization
- Additional fusion pass for optimization
