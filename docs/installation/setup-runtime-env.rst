Setup runtime environment
=========================

After the :doc:`installation <build-from-source>`, the following dependencies should be installed before running PyAscend operators:

.. contents::
    :depth: 1
    :local:

Ascend CANN
-----------

1. Download installation package from `the download center <https://www.hiascend.com/cann/download>`__
   (recommended version is :code:`9.0.0-beta.2`) **depending on the host platform**:

   .. code:: bash

      wget https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%209.0.T511/Ascend-cann_9.0.0-beta.2_linux-$(arch).run

2. Install downloaded package with the following command:

   .. code:: bash

      bash Ascend-cann_9.0.0-beta.2_linux-$(arch).run --full

3. Enable CANN environment:

   .. code-block:: bash

       source /usr/local/Ascend/cann/set_env.sh
       export LD_LIBRARY_PATH=$ASCEND_HOME_PATH/tools/simulator/Ascend910B1/lib:$LD_LIBRARY_PATH

To learn more about CANN environment, visit `the homepage <https://www.hiascend.com/cann>`__.
Ascend C API reference is available in `the documentation <https://www.hiascend.com/document/detail/zh/canncommercial/850/API/ascendcopapi/atlasascendc_api_07_0003.html>`__.
