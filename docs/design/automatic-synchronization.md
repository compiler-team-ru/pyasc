# Automatic synchronization injection

## Overview

The `insert_sync` parameter is a JIT compilation option that controls automatic insertion of synchronization instructions in pyasc kernels. This feature is **enabled by default** and is essential for correct asc2 kernel execution on Ascend AI processors.

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

The `insert_sync` parameter triggers different synchronization passes based on the target platform:

#### Non-Ascend910_95 Platforms (Ascend910B, Ascend910_93)
**Passes Executed**:
1. `add_erase_sync` - Remove existing sync instructions
2. `add_hoist_que_bind` - Hoist queue binding operations
3. `add_insert_sync` - Insert standard synchronization instructions
4. `add_unify_pipe` - Unify pipe operations
5. `add_canonicalizer` - Canonicalize IR

**Code Location**: `lib/Dialect/Asc/Transforms/InsertSync.cpp`

#### Ascend910_95 Platform
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
def _schedule_optimizing(self, pm: passes.PassManager) -> None:
    passes.common.add_licm(pm)                    # Loop invariant code motion
    passes.common.add_sccp(pm)                    # Sparse conditional constant propagation
    passes.common.add_canonicalizer(pm)              # Canonicalize IR
    
    if self.options.insert_sync:
        passes.ascendc.add_erase_sync(pm)             # Remove existing sync
        passes.ascendc.add_hoist_que_bind(pm)         # Hoist queue binds
        
        if self.platform != CompilePlatform.Ascend910_95:
            # Standard sync insertion
            passes.ascendc.add_insert_sync(pm)
        else:
            # Buffer ID-based sync insertion
            passes.ascendc.add_insert_bufid_sync(pm)
            passes.common.add_canonicalizer(pm)
            passes.ascendc.add_fuse_bufid_sync(pm)
        
        passes.ascendc.add_unify_pipe(pm)               # Unify pipes
        passes.common.add_canonicalizer(pm)              # Canonicalize IR
```

### InsertSync Pass Implementation
The `InsertSync` pass (for non-Ascend910_95 platforms) uses Enque/Deque API.

For every operation producing LocalTensor object check if it is used within
the same pipe:
 - If so  insert `pipe_barrier` instruction. And no follow up actions are
   needed.
 - Otherwise, find/allocate corresponding TQue object and add enque operation:
   * for each enqueued tensor, find first user and insert dequeue;
   * re-enqueue if needed for dominance;
   * replace tensor uses with dequeued version.
Additional synchronization is done for GetValue/SetValue operations.
 - Each call is wrapped with the Set/Wait flags construction. 
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

The `InsertBufIdSync` pass (for Ascend910_95 platform) uses buffer ID-based
synchronization:
1. **Buffer ID Management**: Assigns unique buffer IDs to operations. Each pipe
supports up to 32 buffer IDs. Every operation is assigned with bufID
corresponding to each local tensor in the operation.
2. **Pipe Detection**: Identifies which pipe each operation belongs to
3. **Sync injection**: Every operation is wrapped with get/rls_buf
commands where bufID corresponds to each local tensor in the operation. If
operation requires several synchronization points next syncronization pair is
incapsulated in previous one.

### Platform-Specific Considerations

#### Ascend910B/Ascend910_93
- Uses standard `InsertSync` pass
- Flag-based synchronization
- Pipe barrier operations

#### Ascend910_95
- Uses `InsertBufIdSync` pass
- Buffer ID-based synchronization
- Additional fusion pass for optimization
