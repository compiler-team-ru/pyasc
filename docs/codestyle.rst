.. Copyright (c) 2025 Huawei Technologies Co., Ltd.
.. This program is free software, you can redistribute it and/or modify it under the terms and conditions of
.. CANN Open Software License Agreement Version 2.0 (the "License").
.. Please refer to the License for details. You may not use this file except in compliance with the License.
.. THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
.. INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
.. See LICENSE in the root of the software repository for the full text of the License.

Coding style conventions
========================

This document outlines the coding style conventions for the project. These guidelines are primarily based on the LLVM coding style with a few exceptions and project-specific tweaks. Below are the main rules and conventions:

General conventions for C++ files
----------------------------------

1. **File Extensions**:

   - Header files must have the :code:`.h` extension.
   - Implementation (library) files must have the :code:`.cpp` extension.

2. **Include Guards**:

   - All header files must include traditional include guards (e.g., :code:`#ifndef HEADER_H`, :code:`#define HEADER_H`, :code:`#endif`).
   - **Note**: The use of :code:`#pragma once` is **not allowed**.

3. **Indentation**: The code should use **2 spaces** for indentation (no tabs).

4. **Naming Conventions**:

   - **PascalCase**: Used for naming types such as classes, structs, enums, and typedefs. Example: :code:`MyClass`, :code:`MyEnum`.
   - **camelCase**: Used for variables (local and global), functions, and class members. Example: :code:`myFunction`, :code:`myVariable`.
   - **UPPER_SNAKE_CASE**: Typically used for macros. Example: :code:`MY_CONSTANT`, :code:`MAX_BUFFER_SIZE`.
   - **kebab-case**: Not commonly used in code, but may be used in filenames for tests and other resources. Example: :code:`my-test-case`.
  
5. **Namespace Declarations**:

   - After closing a namespace, add a comment :code:`// namespace <namespace_name>`. If the namespace is anonymous, omit the name.
   - **Note**: Do not use :code:`using namespace` in header files.

6. **Macro Definitions**:

   - Defining macros in header files is **discouraged**, unless absolutely necessary.
   - To define a global constant, :code:`constexpr` syntax and **camelCase** naming should be used.

7. **Include Ordering**:

   - The order of :code:`#include` statements should follow this pattern:

     1. Local project includes.
     2. (An empty line).
     3. MLIR/LLVM/Clang includes.
     4. (An empty line).
     5. Standard library includes (:code:`<optional>`, :code:`<vector>`, etc.).

   - Within each subsection, statements should be sorted alphabetically by file name.
   - Example:

     .. code-block:: cpp

        #include "ascir/Dialect/EmitAsc/IR/EmitAsc.h"
        #include "ascir/Target/Asc/Utils.h"

        #include "mlir/IR/Builders.h"
        #include "mlir/IR/DialectImplementation.h"
        #include "llvm/ADT/TypeSwitch.h"

        #include <optional>
        #include <unordered_map>


8. **Anonymous Namespace**: If a class or function is defined and declared in a :code:`.cpp` file but not used elsewhere in the project, it should be placed inside an anonymous namespace. It should **not** be marked as :code:`static`.

9. **Template Argument Naming**:

   - Template typename arguments should follow **PascalCase**. Example: :code:`typename AttrT`.
   - Non-type template arguments should follow **camelCase**. Example: :code:`size_t size`.
   - Always use :code:`typename` instead of :code:`class` for template arguments.

Conventions for MLIR dialects
-----------------------------

Definitions of operations, types, attributes, interfaces, and other entities should be sorted alphabetically within the corresponding TableGen file. 

Conventions for MLIR passes
----------------------------

1. **Pass File Organization**: Each MLIR pass should be placed in a **separate :code:`.cpp` file** under the :code:`Transforms` directory, within the directory corresponding to the specific MLIR dialect.

2. **File Naming**: The name of the :code:`.cpp` file should match the name of the pass **without the "Pass" suffix**. For example, the file for the :code:`FoldVariablePass` pass should be named :code:`FoldVariable.cpp`.

3. **Pass Declarations**: In the :code:`Passes.td` file, in :code:`CMakeLists.txt`, and in the constructor functions header :code:`Passes.h`, pass names should be listed in **alphabetical order**.

Conventions for LIT tests
--------------------------

1. **Test Directory Structure**:

   - Tests are placed in the :code:`test` directory. 
   - Filenames for test files should use **kebab-case**. Example: :code:`my-test-case.mlir`.

2. **Test File Format**: The first line of the test file should generally contain one or more :code:`// RUN:` commands that specify how the test should be executed.
   
3. **Test File Organization**:

   - When adding a new MLIR operation, type, or attribute, a test for that feature should be added under the :code:`IR` directory within the appropriate dialect's directory.
   - When adding a new MLIR pass, a set of tests (typically multiple :code:`func.func` operations) should be added under the :code:`Transforms` directory of the corresponding dialect where the pass is introduced. Name of file should correspond to a pass name.
   - For the emission of a new operation, a test should be placed under the :code:`Target` directory.
   - End-to-end tests for tools should be located under the :code:`Tools` directory.

Additional considerations
-------------------------

- **Readability**: Code should be written for **clarity and maintainability**, not just brevity. Use meaningful names for functions, variables, and types. Comment complex or non-obvious code to aid future developers.

- **Refactoring**: Refrain from making large, non-essential refactoring changes in areas that are not directly related to the task at hand. Always aim for minimal disruption in the codebase.

- **Version control**:

  - Since commits are squashed into a single commit during the merge, detailed commit messages are not required. However, if needed, include helpful information about the change.
  - Merge request titles should be clear and describe the intention behind the changes. Follow conventional commit styles where possible.
  - Merge request description is optional if the title is sufficiently explanatory. It should be added in case of complex changes to clearly describe what has been implemented.
  - Make sure the code builds and passes tests locally before committing.

Code formatting and static analysis tools
-----------------------------------------

To help ensure that the code adheres to the coding style conventions automatically, it is strongly recommended using the following tools:

1. **clang-format**:

   - :code:`clang-format` is a powerful tool for automatic code formatting, which can be configured to follow the project's coding style guide.
   - The :code:`.clang-format` configuration files are already set up in the project repository, therefore you can format a file with :code:`clang-format` by running:

     .. code-block:: bash

        clang-format -i <filename>

   - Integrating :code:`clang-format` into your IDE or editor can help you format code automatically on save. For example, there is `Clang-Format extension <https://marketplace.visualstudio.com/items?itemName=xaver.clang-format>`__ for Visual Studio Code.

2. **clang-tidy**:

   - :code:`clang-tidy` is a static analysis tool that helps catch common issues and enforces coding standards and best practices. It works by checking your code against predefined or custom checks.
   - :code:`clang-tidy` can help identify issues related to code quality, unused variables, potential bugs, and performance improvements. It is recommended to check an output log of *clang-tidy* job in open merge request and address issues before merge.

This style guide serves to maintain consistency across the codebase, making it easier to read, maintain, and extend. Adhering to these conventions will improve collaboration, reduce errors, and make it easier for new contributors to get up to speed.
