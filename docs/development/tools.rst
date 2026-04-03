Tools
=====

The following tools are provided to project contributors to enhance their development and debugging capabilities.


MLIR LSP server: :code:`ascir-lsp`
----------------------------------

The tool implements language server protocol which is used by IDEs to effectively provide syntax highlighting, as well
as other language processing features, both for built-in MLIR dialects and AscendIR extensions.

To enable :code:`ascir-lsp` server in Visual Studio Code:

1. Install `MLIR extension <https://marketplace.visualstudio.com/items?itemName=llvm-vs-code-extensions.vscode-mlir>`__.
2. Obtain full path to the built executable (e.g. run :code:`which ascir-lsp`).
3. Paste the path to *Mlir: Server_path* setting (:code:`mlir.server_path`).


MLIR optimizer driver: :code:`ascir-opt`
----------------------------------------

The tool supports all features and command line options that are supported by :code:`mlir-opt` (LLVM built-in application),
and is also able to run AscendIR passes with its dialects and extensions.

For example:

.. code-block:: bash

    ascir-opt -ascendc-insert-bufid-sync -canonicalize -mlir-print-ir-before-all test.mlir
