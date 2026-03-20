# 构建
## 环境准备<a name="envready"></a>
### 编译环境准备<a name="buildenv"></a>
pyasc支持通过pip快速安装和基于源码编译安装两种方式。

#### 快速安装
您可以通过pip安装pyasc的最新稳定版本：
- 安装pyasc依赖
   ```bash
   pip install pybind11==2.13.1
   pip install "numpy<2"
   pip install typing_extensions
   ```
- 安装pyasc
   ```bash
   pip install pyasc
   ```
   二进制wheel安装包支持CPython 3.9-3.12。

#### 基于源码安装

- 系统要求
   - GCC >= 9.4.0
   - GLIBC >= 2.31
   - cmake >= 3.20

- 依赖

   1. 包版本依赖  
      Python支持版本为：**py3.9-py3.12**。

      推荐使用python虚拟环境（venv）
         ```shell
         python3 -m pip install virtualenv
         python3 -m virtualenv $HOME/.pyasc_venv --prompt asc
         source $HOME/.pyasc_venv/bin/activate
         ```
   2. 下载pyasc源码
         ```shell
         git clone https://gitcode.com/cann/pyasc.git
         cd pyasc
         ```
   3. 安装python依赖  
         ```shell
         python3 -m pip install -r requirements-build.txt # build-time dependencies
         ```

- 构建安装LLVM<a name="llvm_build"></a>

   1. 下载LLVM源码  
      准备LLVM版本，推荐使用LLVM 19.1.7版本。

         ```shell
         git clone https://github.com/llvm/llvm-project.git
         cd llvm-project
         git tag | grep 19.1.7
         git checkout tags/llvmorg-19.1.7
         ```

   2. 构建和安装  
   支持clang和GCC两种编译方式。

      - clang构建安装LLVM  
         推荐使用clang安装LLVM，clang 构建 LLVM 具有原生适配优势，编译更快、内存占用更低，支持最新特性且错误提示清晰，但对老旧系统和小众平台支持有限，适合主流系统（如 Ubuntu、arm），尤其适合频繁开发调试或追求编译效率的场景。

         - 步骤1：环境上请安装clang、lld，并指定版本（推荐版本clang>=15，lld>=15），如未安装，请按下面指令安装：

            ```shell
            apt update
            apt install zlib1g-dev clang-15 lld-15 ccache
            ```

            如果环境上有多个版本的clang，请设置clang为当前安装的版本clang-15，如果clang只有15版本，或已指定15版本则跳过该步骤:

            ```shell
            update-alternatives --install /usr/bin/clang clang /usr/bin/clang-15 20; \
            update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-15 20; \
            update-alternatives --install /usr/bin/lld lld /usr/bin/lld-15 20
            ```

         - 步骤2：设置环境变量 LLVM_INSTALL_PREFIX 为您的目标安装路径：

            ```shell
            export LLVM_INSTALL_PREFIX=${llvm_install_path}
            ```

         - 步骤3：执行以下命令进行构建和安装LLVM：

            ```shell
            cd $HOME/llvm-project  # your clone of LLVM.
            mkdir build
            cd build
            cmake ../llvm \
               -G Ninja \
               -DCMAKE_BUILD_TYPE=Release \
               -DLLVM_ENABLE_ASSERTIONS=ON \
               -DLLVM_ENABLE_PROJECTS="mlir" \
               -DLLVM_TARGETS_TO_BUILD="X86" \
               -DCMAKE_INSTALL_PREFIX=${LLVM_INSTALL_PREFIX} \
               -DCMAKE_C_COMPILER=clang \
               -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON \
               -DLLVM_INSTALL_UTILS=ON \
               -DLLVM_BUILD_UTILS=ON \
               -DCMAKE_CXX_COMPILER=clang++
            ninja install
            ```

         - 步骤4：验证安装：  
            执行以下命令，若输出对应版本信息，说明安装成功：

            ```shell
            clang --version
            ${llvm_install_path}/bin/llvm-config --version
            ```

      - GCC构建安装LLVM  
         GCC 构建 LLVM 兼容性极强，支持几乎所有 GCC 覆盖的平台，与系统 GCC 工具链无缝集成且适合离线构建，但编译较慢、内存占用高，版本可能落后于源码仓库且错误提示较晦涩。适合需与 GCC 工具链深度集成或受限环境下稳定部署的场景。推荐使用clang，如果只能使用GCC安装，请注意[注1] [注2]。

         - 步骤1：设置环境变量 LLVM_INSTALL_PREFIX 为您的目标安装路径：

            ```shell
            export LLVM_INSTALL_PREFIX=${llvm_install_path}
            ```

         - 步骤2：执行以下命令进行构建和安装：

            ```shell
            cd $HOME/llvm-project  # your clone of LLVM.
            mkdir build
            cd build
            cmake -G Ninja  ../llvm  \
               -DLLVM_CCACHE_BUILD=OFF \
               -DCMAKE_BUILD_TYPE=Release \
               -DLLVM_ENABLE_ASSERTIONS=ON \
               -DLLVM_ENABLE_PROJECTS="mlir"  \
               -DLLVM_TARGETS_TO_BUILD="X86" \
               -DLLVM_INSTALL_UTILS=ON \
               -DLLVM_BUILD_UTILS=ON \
               -DCMAKE_INSTALL_PREFIX=${LLVM_INSTALL_PREFIX}
            ninja install
            ```

         - 注1：若在编译时出现错误`ld.lld: error: undefined symbol`，可在步骤2中加入设置`-DLLVM_ENABLE_LLD=ON`。
         - 注2：若环境上ccache已安装且正常运行，可设置`-DLLVM_CCACHE_BUILD=ON`加速构建, 否则请勿开启。

- 构建pyasc
   1. 进入上文下载的pyasc源码目录
      ```shell
      cd pyasc
      ```
   2. 设置环境变量
      ```shell
      export LLVM_INSTALL_PREFIX=${llvm_install_path}
      ```
   3. 执行以下命令进行构建和安装：
      ```shell
      # 普通模式，将项目安装到Python环境的site-packages目录中，本地修改不影响已安装版本，适用于生产环境
      python3 -m pip install .
      # 开发者模式，仅创建符号链接，本地修改实时生效，无需重新安装，适用于开发阶段
      python3 -m pip install -e .
      ```

### 运行环境准备<a name="runtimeenv"></a>

使用基于源码安装时，建议安装社区版<a href="https://www.hiascend.com/developer/download/community/result?module=cann&cann=8.5.0.alpha001">8.5.0.alpha001</a>及以上版本。

使用快速安装时，不同pyasc发行版可支持的硬件平台及所需的[CANN](https://www.hiascend.com/developer/download/community/result?module=cann)版本如下表：

<table style="width: 75%; margin: 0 auto;">
  <colgroup>
    <col style="width: 25%">
    <col style="width: 22%">
    <col style="width: 22%">
  </colgroup>
  <thead>
      <tr>
          <th>pyasc社区版本</th>
          <th>支持CANN包版本</th>
          <th>支持昇腾产品</th>
      </tr>
  </thead>
  <tbody style="text-align: center">
      <tr>
 	           <td>v1.1.0</td>
 	           <td>社区版<a href="https://www.hiascend.com/developer/download/community/result?module=cann&cann=8.5.0.alpha001">8.5.0.alpha001</a>及以上</td>
 	           <td><a href="https://www.hiascend.com/document/detail/zh/AscendFAQ/ProduTech/productform/hardwaredesc_0001.html">Atlas A2训练/推理产品</a> <br>
 	           <a href="https://www.hiascend.com/document/detail/zh/AscendFAQ/ProduTech/productform/hardwaredesc_0001.html">Atlas A3训练/推理产品</a></td>
 	       </tr>
      <tr>
          <td>v1.0.0</td>
          <td>社区版<a href="https://www.hiascend.com/developer/download/community/result?module=cann&cann=8.5.0.alpha001">8.5.0.alpha001</a>、<a href="https://www.hiascend.com/developer/download/community/result?module=cann&cann=8.5.0.alpha002">8.5.0.alpha002</a></td>
          <td><a href="https://www.hiascend.com/document/detail/zh/AscendFAQ/ProduTech/productform/hardwaredesc_0001.html">Atlas A2训练/推理产品</a> <br>
          <a href="https://www.hiascend.com/document/detail/zh/AscendFAQ/ProduTech/productform/hardwaredesc_0001.html">Atlas A3训练/推理产品</a></td>
      </tr>
  </tbody>
</table>

在运行Ascend C Python实现的算子前，请根据如下步骤完成相关环境准备。
1. **安装python依赖**
   ```shell
   python3 -m pip install -r requirements-runtime.txt
   ```

2. **安装社区版CANN toolkit包**

    根据实际环境，下载对应`Ascend-cann-toolkit_${cann_version}_linux-${arch}.run`包。
    
    ```bash
    # 确保安装包具有可执行权限
    chmod +x Ascend-cann-toolkit_${cann_version}_linux-${arch}.run
    # 安装命令
    ./Ascend-cann-toolkit_${cann_version}_linux-${arch}.run --full --force --install-path=${install_path}
    ```
    - \$\{cann\_version\}：表示CANN包版本号。
    - \$\{arch\}：表示CPU架构，如aarch64、x86_64。
    - \$\{install\_path\}：表示指定安装路径，默认安装在`/usr/local/Ascend`目录。

3. **安装社区版CANN ops包（可选）**

    运行接入torch的算子时必须安装本包，若仅编译算子，可跳过本操作。

    根据AI处理器类型，下载对应`Ascend-cann-${soc_name}-ops_${cann_version}_linux-${arch}.run`包（8.5.0.alpha001与8.5.0.alpha002版本中为`Ascend-cann-kernels-${soc_name}_${cann_version}_linux-${arch}.run`包）。
    
    ```bash
    # 确保安装包具有可执行权限
    chmod +x Ascend-cann-${soc_name}-ops_${cann_version}_linux-${arch}.run
    # 安装命令
    ./Ascend-cann-${soc_name}-ops_${cann_version}_linux-${arch}.run --install --install-path=${install_path}
    ```
    - \$\{soc\_name\}：表示NPU型号名称，例如910b。
    - \$\{install\_path\}：表示指定安装路径，需要与toolkit包安装在相同路径，默认安装在`/usr/local/Ascend`目录。

4. **配置环境变量**<a name=envset></a>

   - 默认路径，root用户安装

     ```bash
     source /usr/local/Ascend/cann/set_env.sh
     ```

   - 默认路径，非root用户安装

     ```bash
     source $HOME/Ascend/cann/set_env.sh
     ```

   - 指定路径安装

     ```bash
     source ${cann_install_path}/cann/set_env.sh
     ```

   注1：当pyasc后端采用仿真器模式（如Ascend910B1 simulator）时，需设置以下环境变量：
   ```bash
   export LD_LIBRARY_PATH=$ASCEND_HOME_PATH/tools/simulator/Ascend910B1/lib:$LD_LIBRARY_PATH
   ```

   注2：若pyasc后端采用仿真器模式运行接入torch的算子，需要提前加载仿真动态库libruntime_camodel.so。（原因：torch_npu默认只支持NPU上板，并且在导入torch_npu时自动加载libruntime.so，仿真器模式运行需要提前加载libruntime_camodel.so）。
   ```bash
   # 若pyasc后端采用仿真器模式，需设置LD_PRELOAD环境变量
   export LD_PRELOAD=libruntime_camodel.so
   # 若pyasc后端采用NPU处理器运行，需取消LD_PRELOAD环境变量
   unset LD_PRELOAD
   ```

   **注意：若环境中已安装多个版本的CANN软件包，设置上述环境变量时，请确保${cann_install_path}/latest目录指向的是配套版本的软件包。**

5. **安装PyTorch框架和torch_npu插件**

   编译运行torch输入输出tensor的算子时必须安装本包。

   根据实际环境，选择对应的版本进行安装，具体可以查看[Ascend Extension for PyTorch文档](https://www.hiascend.com/document/detail/zh/Pytorch/730/configandinstg/instg/docs/zh/installation_guide/installation_via_binary_package.md)。

   以`Python3.9`，`x86_64`，`PyTorch2.7.1`，`torch_npu7.2.0`版本为例。
   - 安装`PyTorch`框架：
      ```bash
      # 下载软件包
      wget https://download.pytorch.org/whl/cpu/torch-2.7.1%2Bcpu-cp39-cp39-manylinux_2_28_x86_64.whl
      # 安装命令
      pip3 install torch-2.7.1+cpu-cp39-cp39-manylinux_2_28_x86_64.whl
      ```
   - 安装`torch_npu`插件：
      ```bash
      # 下载插件包
      wget https://gitcode.com/Ascend/pytorch/releases/download/v7.2.0-pytorch2.7.1/torch_npu-2.7.1-cp39-cp39-manylinux_2_28_x86_64.whl
      # 安装命令
      pip3 install torch_npu-2.7.1-cp39-cp39-manylinux_2_28_x86_64.whl
      ```

      注意事项：
      - torch_npu当前不支持python3.12及以上版本。关于torch_npu的更多详细信息可参考[torch_npu](https://gitcode.com/Ascend/pytorch)。

## 样例运行验证（可选）
开发者使用Ascend C Python编程语言实现自定义算子后，可以进行算子功能验证。本代码仓提供了部分算子实现的样例，具体请参考[tutorials](../python/tutorials/)目录下的样例。
以Add算子为例，执行如下命令可进行功能验证。
```bash
cd pyasc
python3 ./python/tutorials/01_add/add.py
```
注：完整的运行命令如下所示，通过参数[RUN_MODE]配置运行模式、参数[SOC_VERSION]配置运行环境，具体请参考[编译执行](../python/tutorials/01_add/README.md/#编译执行)。若缺省参数[RUN_MODE]默认是仿真器模式，缺省参数[SOC_VERSION]，仿真器模式下默认是`Ascend910B1`环境，NPU上板模式下默认自动检测，请确保已完成[运行环境准备](#运行环境准备)中的[配置环境变量](#envset)步骤。
```bash
python3 ./python/tutorials/01_add/add.py -r [RUN_MODE] -v [SOC_VERSION]
```

## UT测试（可选）
本代码仓支持开发者对开发内容进行UT测试。在执行UT测试前，请确保环境已[构建安装LLVM](#llvm_build)并且已安装Python测试框架pytest。pytest安装命令如下：
```bash
pip install pytest
```

### Python模块UT测试
在项目根目录下，执行如下命令可进行Python模块的UT单元测试验证。其中，`${llvm_install_path}`为上文描述的LLVM安装路径。
```bash
cd test
bash build_llt.sh --run_python_ut --llvm_install_path ${llvm_install_path}
```
在项目根目录下，执行如下命令，可在执行UT测试后使用pytest-cov工具生成代码覆盖率报告。具体为，test目录下自动生成HTML报告，打开该文件即可查看详细覆盖率信息，Python前端模块报告在cov_py文件夹下。
```bash
cd test
pip install pytest-cov
bash build_llt.sh --cov --run_python_ut --llvm_install_path ${llvm_install_path}
```

### ASC-IR定义模块UT测试
在执行ASC-IR定义模块的UT测试前，请确保环境已安装[lit](https://llvm.org/docs/CommandGuide/lit.html)工具。安装命令如下：
```bash
pip install lit
```

在项目根目录下，执行如下命令可进行ASC-IR定义模块的UT单元测试验证。其中，`${llvm_install_path}`为LLVM安装路径，`${lit_install_path}`为lit安装路径。  
注：lit工具位于安装路径`${lit_install_path}`的bin目录下。
```bash
cd test
bash build_llt.sh --check-ascir --llvm_install_path ${llvm_install_path} --lit_install_path ${lit_install_path}
```
在项目根目录下，执行如下命令，可在执行UT测试后使用LCOV工具生成代码覆盖率报告。具体为，test目录下自动生成HTML报告，打开该文件即可查看详细覆盖率信息，ASC-IR定义模块报告在cov_ascir文件夹下。
```bash
cd test
sudo apt install lcov
bash build_llt.sh --cov --check-ascir --llvm_install_path ${llvm_install_path} --lit_install_path ${lit_install_path}
```