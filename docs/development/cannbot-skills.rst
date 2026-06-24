AI agent skills for Ascend C
============================

`CANNBot repository <https://gitcode.com/cann/cannbot-skills>`__ provides skills
for AI agents with specialized knowledge about Ascend C API, NPU architecture,
Tiling design, and Matmul/GEMM optimization. These skills give the AI coding assistant
context-aware help with operator development, API usage, and explaining Ascend C specifics.
The following instructions are recommended to use with OpenCode client.


Recommended Skills
------------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Skill
     - Purpose
   * - ``ascendc-api-best-practices``
      - Comprehensive reference for all Ascend C API categories: arithmetic, reduction, sort,
        DataCopy, Transpose, Buffer management, precision conversion, pipeline operations,
        and API restrictions
    * - ``ascendc-blaze-best-practice``
      - Matmul/Cube/GEMM/BMM development via Blaze/tensor_api path. Covers AIC/StreamK/FixpOpti
        modes, dispatch modes, Tiling strategies, and workspace configuration
    * - ``ascendc-code-review``
      - Code review pipeline for Ascend C: file review, PR review, large PR handling,
        design consistency checks. Includes 11 reference rule sets covering API usage,
        performance, security, SIMT constraints, and C++ coding standards
    * - ``ascendc-crash-debug``
      - Debugging methodology for hang/crash/deadlock scenarios: symptom-to-cause mapping,
        Buffer deadlock diagnosis, cross-core sync issues, memory error detection via
        mssanitizer. Includes crash workflows and memcheck references
    * - ``ascendc-docs-search``
      - Local search across API documents and code examples from ``asc-devkit``.
        Falls back to online Huawei Ascend Community search when local resources are insufficient
    * - ``ascendc-performance-best-practices``
      - Performance optimization patterns by operator family: MatMul, RadixSort, Scalar,
        Reduction, Elementwise. Covers pingpong, swap, streamk, fullload, and 8 optimization rules
    * - ``ascendc-perf-optimize``
      - 4-step performance optimization strategy: Tiling modeling → inter-card pipeline →
        inter-core pipeline → intra-core pipeline. Bound diagnosis (compute/memory/communication)
        with Tiling parameter refinement. Self-contained tiling models for all operator categories
    * - ``ascendc-precision-debug``
      - Systematic precision issue diagnosis: pipeline sync (EnQue/DeQue), DataCopy alignment,
        Cast RoundMode, FP16/BF16 cross-validation, DumpTensor 7-step method. Includes error
        analysis scripts and boundary test generation
    * - ``ascendc-runtime-debug``
      - Runtime error debugging: comprehensive error code reference (161xxx/361xxx/561xxx),
        kernel binary debugging (compilation, cache, SEL matching, opParaSize), plog analysis.
        Includes debug workflows and error code tables
    * - ``ascendc-tiling-design``
      - Tiling methodology for all operator categories: Reduction, Sort, Elementwise, Broadcast,
        MatMul, Conv. Covers UB budgeting, Buffer strategies, and chunk sizing
    * - ``npu-arch``
      - NPU architecture knowledge: DAV_2201 vs DAV_3510, buffer sizes (UB/L0C/BT),
        Cube/Vector/MTE units, SIMD vs SIMD-RegBase, NDDMA, CCU


Directory Structure
-------------------

The ``cannbot-skills`` repository should be cloned as a **sibling directory** to ``pyasc`` (not inside the project).
Skills are symlinked into ``.opencode/skills/``, and ``asc-devkit`` is symlinked in the project root.
All paths use the ``$CANNBOT_SKILLS`` environment variable.


Installation
------------

.. code-block:: bash

    # Clone cannbot-skills as sibling to pyasc
    export CANNBOT_SKILLS=$PWD/../cannbot-skills
    git clone https://gitcode.com/cann/cannbot-skills.git $CANNBOT_SKILLS

    # Clone asc-devkit (API docs + examples)
    git clone https://gitcode.com/cann/asc-devkit.git \
        $CANNBOT_SKILLS/plugins-official/ops-direct-invoke/asc-devkit

    # Create symlinks
    ln -sfn $CANNBOT_SKILLS/ops/ascendc-api-best-practices .opencode/skills/ascendc-api-best-practices
    ln -sfn $CANNBOT_SKILLS/ops/ascendc-blaze-best-practice .opencode/skills/ascendc-blaze-best-practice
    ln -sfn $CANNBOT_SKILLS/ops/ascendc-code-review .opencode/skills/ascendc-code-review
    ln -sfn $CANNBOT_SKILLS/ops/ascendc-crash-debug .opencode/skills/ascendc-crash-debug
    ln -sfn $CANNBOT_SKILLS/ops/ascendc-docs-search .opencode/skills/ascendc-docs-search
    ln -sfn $CANNBOT_SKILLS/ops/ascendc-performance-best-practices .opencode/skills/ascendc-performance-best-practices
    ln -sfn $CANNBOT_SKILLS/ops/ascendc-perf-optimize .opencode/skills/ascendc-perf-optimize
    ln -sfn $CANNBOT_SKILLS/ops/ascendc-precision-debug .opencode/skills/ascendc-precision-debug
    ln -sfn $CANNBOT_SKILLS/ops/ascendc-runtime-debug .opencode/skills/ascendc-runtime-debug
    ln -sfn $CANNBOT_SKILLS/ops/ascendc-tiling-design .opencode/skills/ascendc-tiling-design
    ln -sfn $CANNBOT_SKILLS/ops/npu-arch .opencode/skills/npu-arch
    ln -sfn $CANNBOT_SKILLS/plugins-official/ops-direct-invoke/asc-devkit ./asc-devkit

    # Verify skills installation
    for s in .opencode/skills/*/; do [ -f "$s/SKILL.md" ] && echo "OK: $s" || echo "MISSING: $s"; done

Restart OpenCode to load the skills.

Hiding Symlinks from Git
~~~~~~~~~~~~~~~~~~~~~~~~

The symlinks created above (``asc-devkit`` and skills under ``.opencode/skills/``) are external
dependencies and should not be tracked by git. It is recommended to add them to the local ``.git/info/exclude`` file:

.. code-block:: bash

    echo "
    # CANNBot skills symlinks
    asc-devkit
    .opencode/skills/ascendc-*
    .opencode/skills/npu-arch" >> .git/info/exclude

This file is local-only and will not be committed.


Updating
--------

.. code-block:: bash

    cd $CANNBOT_SKILLS && git pull
    cd plugins-official/ops-direct-invoke/asc-devkit && git pull

Restart OpenCode to apply updates.


Alternative: Automated Installation
-----------------------------------

Use ``init.sh`` from a plugin directory to install all skills from that plugin:

.. code-block:: bash

    cd $CANNBOT_SKILLS/plugins-official/<plugin-name>
    bash init.sh project opencode /path/to/pyasc

See the `cannbot-skills README <https://gitcode.com/cann/cannbot-skills>`__ for available plugins.
