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

### Phase 1: Analyze Pass Implementation

**Goal**: Understand the pass transformation logic and requirements.

**Steps**:

1. **Read the pass source file**
   - Identify the main pass class and its `runOnOperation()` method
   - Extract transformation patterns (pattern rewrites, walkers, etc.)
   - Note attributes used (e.g., `unrollFactor`, `unrollGroup`)
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

In most cases, all **CHECK** should be before the function itself. It is also important to add **// CHECK-LABEL**: func.func name/header to distinct checks for different functions.

**For Pattern-Based Passes**:

```mlir
// Test pattern match and replacement
// CHECK-LABEL @test_pattern_match
// CHECK: asctile.adds %arg0, %cst : !asctile.tile<16xf32, UB>
func.func @test_pattern_match(%arg0: !asctile.tile<16xf32, UB>) -> !asctile.tile<16xf32, UB> {
  %cst = arith.constant 0.0 : f32
  %tile = asctile.splat %cst : !asctile.tile<16xf32, UB>
  %result = arith.addf %arg0, %tile : !asctile.tile<16xf32, UB>
  return %result : !asctile.tile<16xf32, UB>
}
```

**For Walk-Based Passes**:

```mlir
// Test operation walking and transformation
// CHECK-LABEL: @test_walk_transform
// CHECK-NEXT: asctile.load
// CHECK-NEXT: asctile.store
func.func @test_walk_transform(%arg0: memref<1024xf32>, %arg1: memref<1024xf32>) {
  %tile = asctile.load %arg0[0] : !asctile.tile<128xf32>
  asctile.store %tile, %arg1[0] : !asctile.tile<128xf32>
  return
}
```

**For Attribute-Based Passes**:

```mlir
// Test attribute handling
// CHECK-LABEL: test_attribute_handling
// CHECK: asctile.load
// CHECK-NOT: unroll_factor
func.func @test_attribute_handling(%arg0: memref<1024xf32>) {
  %tile = asctile.load %arg0[0] {unroll_factor = 4 : i64} : !asctile.tile<128xf32>
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

3. **Run Tests**
   - Execute: `python3 -m lit -v /path/to/test.mlir`
   - Fix any failures
   - Ensure all tests pass

4. **Coverage Check**
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
