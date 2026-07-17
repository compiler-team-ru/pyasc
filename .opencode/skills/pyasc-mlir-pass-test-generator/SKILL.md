---
name: pyasc-mlir-pass-test-generator
description: Automatic generation lit test for MLIR pass
compatibility: opencode
---

## Overview

This skill helps AI agents automatically generate comprehensive `lit` tests for MLIR passes in the PyAsc2 codebase. The agent analyzes the pass implementation and creates test cases covering all transformation scenarios.

## When to Use

Use this skill when:
- A developer provides a link to an MLIR pass implementation file and and asks to write tests.
- You need to generate lit tests for a new or modified MLIR pass
- Test coverage for a pass needs to be expanded

## Documentation References

Use these documentation sources during test generation:
1. **Architecture Overview** (`docs/design/project-overview.md`)
   - Understanding PyAsc2 module structure
   - Frontend vs backend modules
   - Directory structure for tests

2. **Developer Guide** (`docs/development/`)
   - Development workflow for new APIs
   - Coding standards and conventions (`docs/development/codestyle.rst`)
   - Module development guidelines

3. **Existing Test Examples** (`test/Dialect/<DialectName>/Transforms/`)
   - Reference for correct MLIR syntax
   - Operation usage patterns
   - Test structure examples

4. **TableGen Definitions** (`include/ascir/Dialect/<DialectName>/IR/`)
    - Operation signatures and attributes
    - Type definitions
    - Constraint specifications

5. **MLIR dialect definitions** (`build/cmake*/docs/Dialects/*.md`)
    - Auto-generated documentation files containing operation definitions
    - **Note**: Must build project with `PYASC_SETUP_DOCS=1` environment variable to generate these files
    - Example path: `build/cmake.linux-x86_64-cpython-3.10/docs/Dialects/*.md`

6. **Agent Guidelines** (`AGENTS.md`)
   - Build, lint, and test commands
   - Code style guidelines
   - Testing commands


## Input Requirements

The developer must provide:
1. **Path to MLIR pass file**: e.g., `lib/Dialect/<DialectName>/Transforms/UnrollLoop.cpp`
2. **Pass** name**: e.g., `UnrollLoop`
3. **Optional**: Specific test scenarios or edge cases to prioritize

## Agent Constraints

**CRITICAL**: The agent MUST NOT modify any files except test files:
- **NEVER** modify MLIR pass implementation files (.cpp, .h)
- **NEVER** modify TableGen definition files (.td)
- **NEVER** modify CMakeLists.txt or build configuration files
- **NEVER** modify existing test files (only create new ones)
- **ONLY** create new test files (.mlir) in appropriate test directories

The agent's role is to:
1. **Read** existing code for analysis (pass files, .td files, existing tests)
2. **Generate** new test files based on analysis
3. **Validate** generated tests by running them
4. **Report** test coverage and any issues found

## Agent Pipeline

### Phase 0: Setup Environment and Build Project

### Phase 1: Analyze Pass Implementation

**Goal**: Understand the pass transformation logic and requirements.

**Steps**:

1. **Read the pass source file**
   - Identify the main pass class and its `runOnOperation()` method
   - Extract transformation patterns (pattern rewrites, walkers, etc.)
   - Note attributes used (e.g., `unrollFactor`)
   - Identify operation types targeted (e.g., `scf::ForOp`, `asctile::LoadOp`)
   - Check for helper functions and utility classes

2. **Identify transformation categories**
   - **Pattern-based rewrites**: Uses `OpRewritePattern`, `RewritePatternSet`
   - **Walk-based transformations**: Uses `op.walk()` or recursive visitors
   - **Attribute-based logic**: Reads/writes operation attributes
   - **Control flow changes**: Creates/destroys loops, blocks, regions

3. **Extract key information**
   - Pass name and dialect (e.g., `asctile::UnrollLoopPass`)
   - Input operation types (what ops are transformed)
   - Output operation types (what ops are created)
   - Required attributes and their types
   - Pass options/parameters
   - Dependencies on other passes

### Phase 2: Determine Test Coverage Requirements

**Goal**: Identify all test cases needed for comprehensive coverage.

**Steps**:

1. **Basic functionality tests**
   - Happy path: minimal working example
   - Multiple operations in sequence
   - Nested structures (loops within loops, etc.)

2. **Edge cases**
   - Empty operations/blocks
   - Single iteration cases
   - Boundary values (0, 1, max values)
   - Missing optional attributes
   - Invalid attribute values

3. **Attribute variations**
   - Different attribute values (e.g., `unrollFactor = 2, 4, 8`)
   - Missing attributes
   - Multiple attributes interacting

4. **Operation type coverage**
   - All supported operation types
   - Mixed operation types

5. **Control flow scenarios**
   - Single basic block
   - Multiple basic blocks
   - Nested regions
   - Dominance relationships

6. **Error handling**
   - Invalid IR structure
   - Type mismatches
   - Missing operands
   - Cyclic dependencies

### Phase 3: Locate Test Directory Structure

**Goal**: Find where tests should be placed.

**Steps**:

1. **Map pass to test directory**
   - AscTile passes → `test/Dialect/AscTile/Transforms/`
   - AscendC passes → `test/Dialect/AscendC/Transforms/`
   - Lowering passes → `test/Conversion/LowerToAsc/`

2. **Check existing tests**
   - List existing test files for the pass
   - Analyze naming conventions (kebab-case)
   - Identify test patterns used

3. **Determine test filename**
   - Use pass name in kebab-case without dialect name prefix:
   **Example**:
   ```td
   def UnrollLoop : Pass<"asctile-unroll-loop", "func::FuncOp"> {
      let summary = "Unroll loops by unroll_factor";
      let constructor = "mlir::asctile::createUnrollLoopPass()";
   }
   ```
   Filename for test: `unroll-loop.mlir`
   - Add suffix if multiple files: `unroll-loop-basic.mlir`, `unroll-loop-edge-cases.mlir`
   In most cases, all tests can be placed in single files. Splitting tests into multiple files should not be a goal.

### Phase 4: Generate Test Cases

**Goal**: Create lit test files with proper structure.

**Test File Template**:

```mlir
// RUN: ascir-opt %s --pass-name | FileCheck %s

// CHECK-LABEL: @test_<test_case_name>
func.func @test_<test_case_name>(...) {
  // Input IR
  // CHECK: Expected output IR
}
```

**Generation Steps**:

1. **Create input IR**
    - Use proper MLIR syntax for the dialect
    - Include necessary function signatures
    - Add attributes required by the pass
    - Create realistic operation sequences

2. **Create expected output IR**
    - Apply pass transformation mentally
    - Use `CHECK` directives for verification
    - Check operation order and structure
    - Verify attributes are added/removed correctly
    - **DO NOT** add verbose test case description comments before test functions - keep CHECK directives concise (only `// CHECK-LABEL:` and necessary `// CHECK:/CHECK-NEXT:/CHECK-NOT:` lines)

3. **Add RUN directives**
   - Primary test: `--pass-name`
   - With other passes if needed: `--pass1 --pass2`
   - With debug flags if useful: `--debug`

4. **Group related tests**
   - Basic tests first
   - Edge cases next
   - Complex scenarios last
   - Add comments explaining each test

### Phase 5: Test Structure Guidelines

In most cases, all **CHECK** directives should be placed before the function definition. Use `// CHECK-LABEL: func.func <name>` to distinguish checks for different functions.

**CRITICAL Rules for CHECK vs CHECK-NEXT**:

1. **After CHECK-LABEL, use CHECK: not CHECK-NEXT:** when there are intervening lines between the function header and the first operation being checked. Intervening lines include:
   - Constant declarations: `%c = arith.constant`
   - Tensor/buffer operations that the pass doesn't transform: `ascendc.tbuf.get_tensor`
   - Any operation that appears in the output but isn't part of the transformation being verified

2. **Use CHECK-NEXT: only for truly consecutive operations** - operations that immediately follow one another with no intervening lines

3. **Every test function must end with**:
```mlir
// CHECK-NEXT: return
// CHECK-NEXT:}
```

**For Pattern-Based Passes with no intervening lines**:

```mlir
// CHECK-LABEL: func.func @test_pattern_match
// CHECK-NEXT: asctile.adds %arg0, %cst : tensor<16xf32, #asctile.local<UB>>  // OK if no constants between
// CHECK-NEXT: return
// CHECK-NEXT:}
func.func @test_pattern_match(%arg0: tensor<16xf32, #asctile.local<UB>>) -> tensor<16xf32, #asctile.local<UB>> {
  %cst = arith.constant 0.0 : f32
  %tile = asctile.splat %cst : tensor<16xf32, #asctile.local<UB>>
  %result = arith.addf %arg0, %tile : tensor<16xf32, #asctile.local<UB>>
  return %result : tensor<16xf32, #asctile.local<UB>>
}
```

**For Walk-Based Passes**:

```mlir
// CHECK-LABEL: func.func @test_walk_transform
// CHECK: asctile.load  // Use CHECK if constant precedes
// CHECK-NEXT: asctile.store  // CHECK-NEXT follows load
// CHECK-NEXT: return
// CHECK-NEXT:}
func.func @test_walk_transform(%arg0: memref<1024xf32>, %arg1: memref<1024xf32>) {
  %tile = asctile.load %arg0[0] : tensor<128xf32, #asctile.local<UB>>
  asctile.store %tile, %arg1[0] : tensor<128xf32, #asctile.local<UB>>
  return
}
```

**For Attribute-Based Passes**:

```mlir
// CHECK-LABEL: func.func @test_attribute_handling
// CHECK: asctile.load
// CHECK-NOT: unroll_factor
// CHECK-NEXT: return
// CHECK-NEXT:}
func.func @test_attribute_handling(%arg0: memref<1024xf32>) {
  %tile = asctile.load %arg0[0] {unroll_factor = 4 : i64} : tensor<128xf32, #asctile.local<UB>>
  return
}
```

**For Nested Regions (scf.for, scf.if) with constants inside**:

When operations appear inside nested regions, use `// CHECK:` before the region opener. Inside the region, use `// CHECK:` if constants are introduced before operations:

```mlir
// CHECK-LABEL: func.func @test_nested_for
// CHECK: scf.for
// CHECK: ascendc.set_flag mte3_mte2  // Use CHECK, constant may precede
// CHECK-NEXT: ascendc.set_flag mte3_mte1  // CHECK-NEXT follows previous flag
// CHECK-NEXT: ascendc.wait_flag mte3_mte2
// CHECK-NEXT: ascendc.wait_flag mte3_mte1
// CHECK-NEXT: }
// CHECK-NEXT: return
// CHECK-NEXT:}
func.func @test_nested_for(%arg0: !ascendc.local_tensor<*xf32>) {
  %c256 = arith.constant 256 : i32
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c10 = arith.constant 10 : i32
  scf.for %i = %c0 to %c10 step %c1 : i32 {
    ascendc.add_l2 %arg0, %arg0, %arg0, %c256 : ...
    // Pass may insert constant before set_flag
  }
  return
}
```

### Phase 6: Verify Generated Tests

**Goal**: Ensure tests are correct and complete.

**Steps**:

1. **Check syntax**
   - Verify MLIR syntax is correct
   - Ensure all types are valid
   - Check attribute formats

2. **Verify Test Logic**
   - Each test should have a clear purpose
   - Expected output should match pass behavior
   - CHECK directives should be precise

3. **Run the pass to get actual output** (CRITICAL)
   - Before finalizing CHECK directives, run the pass on test input to see actual output
   - Command: `ascir-opt --pass-name /path/to/test.mlir`
   - Use actual output to determine:
     - Whether constants or other operations intervene between transformed operations
     - Exact format of operations (e.g., `pipe_v, 0` vs just `pipe_v`) including operands and necessary attributes (e.g., `%arg1`, `{ascendc.buf_id = [0, 1 : i32]}`)
     - Correct buffer ID numbers assigned by the pass
   - This prevents CHECK-NEXT failures due to intervening lines

4. **Run Tests**
   - Execute: `lit -v /path/to/test.mlir`
   - Fix any failures by comparing with actual pass output
   - Ensure all tests pass

5. **Coverage Check**
   - Verify all transformation paths are tested
   - Check edge cases are covered
   - Ensure error cases are tested (if applicable)


## Pre-Generation Validation

Before generating any MLIR operation, the agent MUST:

1. **Read the TableGen definition** for the operation from .td files
   - Search in `include/ascir/Dialect/DialectName/IR/` directory
   - Find the exact operation class definition
   - Extract: operands (with types), attributes (with types), constraints

2. **Find existing examples** in test files
   - Search in `test/Target/AscendC/` directory
   - Look for similar operations in .mlir files
   - Extract the exact syntax pattern

3. **Validate attribute types** before generating
   - Check if attribute is I32EnumAttr, I64EnumAttr, UnitAttr, etc.
   - Use the correct type (i32 vs i64) based on definition
   - Never guess attribute types

4. **Distinguish operands from attributes**
   - Operands are in `arguments = (ins ...)` list
   - Attributes have `$attrName` with specific types (UnitAttr, I32EnumAttr, etc.)
   - Count expected operands vs attributes separately

## Static Validation Rules

1. **Never guess operation signatures** - always read from .td files first
2. **Always check enum constraints** - enum attributes have specific allowed values
3. **Verify operand count** - count operands separately from attributes
4. **Match existing test patterns** - use exact syntax from working examples
5. **Use correct attribute syntax** - `{attrName = value : type}` not as operand

## Iterative Improvement

1. **Extract common patterns** from errors and create reusable rules
2. **Apply fixes to all similar cases** - don't fix one-by-one
3. **Document learned rules** for future sessions

**Common Error Patterns and Fixes**:

| Error | Cause | Fix |
|-------|-------|-----|
| `CHECK-NEXT: is not on the line after the previous match` | Intervening line (constant, tbuf.get_tensor, etc.) between operations | Change `CHECK-NEXT:` to `CHECK:` before the first affected operation |
| `expected CHECK not found` | Operation format doesn't match actual output | Run pass to see actual format, update CHECK directive |
| Missing buffer number | Pass assigns buffer IDs dynamically | Run pass, extract actual buffer numbers from output |
| Wrong pipe type | Different operations use different pipes | Verify pipe type from actual output (pipe_v, pipe_mte2, etc.) |

**After fixing test failures**:
- Update this SKILL.md with new learned patterns if the patterns has complex cases.
- Apply the same fix pattern to all similar test functions in the file

## Troubleshooting

**Test fails with "expected CHECK not found"**:
- Verify that pass actually produces expected output
- Check if pass depends on other passes to run first
- Use `--debug` flags to see actual output

**MLIR syntax errors**:
- Verify type syntax matches dialect
- Check attribute formats
- Ensure all operands are defined before use

**Pass doesn't transform**:
- Check if required attributes are present
- Verify operation types match pass expectations
- Ensure pass conditions are met (e.g., dominance, purity)

**CHECK directive best practices**:
- **Use `// CHECK-NEXT:` only for truly consecutive operations** - operations that immediately follow one another with no intervening lines
- **Use `// CHECK:` after `CHECK-LABEL:` when there are intervening lines** - such as constant declarations (`arith.constant`), tensor operations (`ascendc.tbuf.get_tensor`), or other operations that the pass doesn't transform
- Avoid verbose test case description comments - keep CHECK directives concise
- Use `// CHECK-LABEL:` to separate checks for different functions
- Use `// CHECK:` when the match may have intervening lines (e.g., inside nested regions like `scf.for` or `scf.if` where constants are introduced)
- Use `// CHECK-NOT:` to verify operations are NOT present in output

**Special patterns for AscendC buffer operations**:
- For `ascendc.get_buf` and `ascendc.rls_buf`, always include the full format with pipe type and buffer number: `pipe_<type>, <number>`
- Example: `ascendc.get_buf pipe_v, 0`, `ascendc.rls_buf pipe_mte2, 1`, `ascendc.get_buf pipe_m, 2`
- Common pipe types: `pipe_v`, `pipe_m`, `pipe_mte2`, `pipe_mte3`, `pipe_mte1`, `pipe_fix`, `pipe_s`
- Buffer numbers are assigned by the pass and should be verified from actual output

**Handling intervening constant declarations**:
- After `CHECK-LABEL:`, if the pass introduces operations but there are constant declarations in between, use `// CHECK:` for the first transformed operation
- Example pattern:
```mlir
// CHECK-LABEL: func.func @test_example
// CHECK: arith.constant
// CHECK: ascendc.tbuf.get_tensor
// CHECK-NEXT: ascendc.get_buf pipe_v, 0  // This follows immediately after tbuf.get_tensor
// CHECK-NEXT: ascendc.add_l2 %arg1, %arg2, %c256 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, i32
// CHECK-NEXT: ascendc.rls_buf pipe_v, 0
// CHECK-NEXT: return
// CHECK-NEXT:}
func.func @test_example(%arg0: !ascendc.tbuf<vecin>) {
  %c256 = arith.constant 256 : i32
  %tensor = ascendc.tbuf.get_tensor %arg0 : !ascendc.tbuf<vecin>, !ascendc.local_tensor<*xf32>
  ascendc.add_l2 %tensor, %tensor, %tensor, %c256 : !ascendc.local_tensor<*xf32>, ...
  return
}
```

**Inside nested regions with constants**:
- When constants are introduced inside `scf.for` or `scf.if` before the operations to check, use `// CHECK:` for those operations
- Example:
```mlir
// CHECK-LABEL: func.func @test_nested_for
// CHECK: scf.for
// CHECK: ascendc.set_flag mte3_mte2  // Use CHECK because constant may precede
// CHECK-NEXT: ascendc.set_flag mte3_mte1
// CHECK-NEXT: ascendc.wait_flag mte3_mte2
// CHECK-NEXT: ascendc.wait_flag mte3_mte1
// CHECK-NEXT: return
// CHECK-NEXT:}
```
