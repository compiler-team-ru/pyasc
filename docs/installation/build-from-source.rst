.. Copyright (c) 2026 Huawei Technologies Co., Ltd.
.. This program is free software, you can redistribute it and/or modify it under the terms and conditions of
.. CANN Open Software License Agreement Version 2.0 (the "License").
.. Please refer to the License for details. You may not use this file except in compliance with the License.
.. THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
.. INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
.. See LICENSE in the root of the software repository for the full text of the License.

Build from source
=================

.. contents:: Table of contents
    :depth: 3
    :local:


Prerequisites
-------------

Obtain the source code
~~~~~~~~~~~~~~~~~~~~~~

To get the source code, the repository should be downloaded with git:

.. code-block:: bash

    git clone https://gitcode.com/cann/pyasc.git
    cd pyasc
    git fetch origin +refs/merge-requests/85/head:v2
    git checkout v2


Setup the build environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Python 3.9 or newer** with pip is required for PyAsc.

It is recommended to activate `a virtual environment <https://docs.python.org/3/tutorial/venv.html>`__ (venv) instead of using a system-wide installation:

.. code-block:: bash

    python3 -m pip install virtualenv
    python3 -m virtualenv $HOME/.pyasc_venv --prompt pyasc
    source $HOME/.pyasc_venv/bin/activate


Install build dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~

PyAsc requires either ``gcc`` or ``clang`` compiler and the compatible linker.
Optionally, ``ccache`` can be installed to speed up the development flow.

.. hint::

    On Ubuntu, these packages can be installed from apt:

    .. code-block:: bash

        sudo apt install build-essential ccache clang lld


Build and install
-----------------

**One of the approaches below** can be used to install or develop the Python package.

The following environment variables can be **exported** to configure the installation process:

``LLVM_INSTALL_PREFIX=<path>`` (required)
    provide the directory with pre-built LLVM binaries (preliminarily download and unpack the archive for
    `x64 <https://oaitriton.blob.core.windows.net/public/llvm-builds/llvm-86b69c31-ubuntu-x64.tar.gz>`__ or
    `arm64 <https://oaitriton.blob.core.windows.net/public/llvm-builds/llvm-86b69c31-ubuntu-arm64.tar.gz>`__ platform)

``PYASC_SETUP_BUILD_DIR=<path>`` (optional)
    provide the directory for the temporary build files (default: ``build``)

``PYASC_SETUP_CCACHE=1`` (optional)
    enable ccache to speed up the repetitive build flow

``PYASC_SETUP_CLANG_LLD=1`` (optional)
    use clang and lld instead of the default toolchain

``PYASC_SETUP_CONFIG=<config>`` (optional)
    build configuration for CMake (default: ``Release``)

``PYASC_HOME=<path>`` (optional)
    provide the directory that should be used to store pre-built dependencies (default: user home directory)

Also, ``-v`` option can be added to pip arguments to increase the output verbosity.
This will allow to see e.g. CMake and C++ compiler commands.


Install the package (recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This is **the simplest way** to build and install PyAsc using a single command.
The project repository can be safely deleted after the package is installed into the environment.

The following command should be run to collect build dependencies and install Python package into the environment:

.. code-block:: bash

    python3 -m pip install .


.. note::

    After the installation, it is recommended to :doc:`setup runtime environment <setup-runtime-env>` in order to run PyAsc operators.


Advanced installation options
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Option 1: Install the package (preserve build directory)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This option provides the most setup features for the project developers.
It allows to enable :doc:`development tools <../development/tools>`.

These environment variables can be exported additionally to configure the installation process:

``PYASC_SETUP_DEVTOOLS=1`` (optional)
    build and install development tools

``PYASC_SETUP_DOCS=1`` (optional)
    render markdown documentation for IR entities in ``build/cmake*/docs`` directory

.. code-block:: bash

    python3 -m pip install -r requirements-build.txt
    python3 -m pip install --no-build-isolation .


Option 2: Install the editable package
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This option is recommended for the project developers since it allows to modify internal Python files without the need to perform the PyAsc re-installation.
The ``--no-build-isolation`` argument can be added too.
However, the development tools will not be available on path automatically when using this approach.

.. code-block:: bash

    python3 -m pip install -e .


Option 3: Install the distributable package (wheel)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This option can be used if there is a need to obtain the installation archive (wheel) to install the package into another environment or transfer it to other device.

.. code-block:: bash

    python3 -m pip wheel --wheel-dir=wheels --no-deps .
    python3 -m pip install wheels/pyasc*.whl
