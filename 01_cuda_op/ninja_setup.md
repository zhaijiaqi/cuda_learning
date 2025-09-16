# CUDA 学习环境配置指南

## 问题描述

在运行 CUDA 扩展编译时遇到以下错误：
1. `RuntimeError: Ninja is required to load C++ extensions`
2. `#error C++17 or later compatible compiler is required to use PyTorch`
3. `fatal error: string_view: No such file or directory`

## 解决方案

### 1. 安装 Ninja 构建工具

```bash
pip install ninja
```

### 2. 安装支持 C++17 的编译器

系统的 GCC 6.5.0 不支持 C++17，需要安装更新的编译器：

```bash
conda activate cuda_learning
conda install gcc_linux-64 gxx_linux-64 -y
```

这将安装 GCC 11.2.0，支持 C++17 标准。

### 3. 设置环境变量

在编译前需要设置正确的编译器路径：

```bash
export CC=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc
export CXX=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++
export TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0;8.6"
```

### 4. 运行 CUDA 扩展

设置好环境变量后，可以成功运行：

```bash
python run_time.py --compiler jit
```

## 完整的运行命令

```bash
# 激活环境
conda activate cuda_learning

# 设置编译器环境变量
export CC=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc
export CXX=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++
export TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0;8.6"

# 运行 CUDA 扩展
python run_time.py --compiler jit
```

## 预期输出

成功运行后应该看到类似输出：

```
Using /mnt/7T/jiaqiz/.cache/torch_extensions/py310_cu124 as PyTorch extensions root...
Detected CUDA files, patching ldflags
Emitting ninja build file /mnt/7T/jiaqiz/.cache/torch_extensions/py310_cu124/add2/build.ninja...
Building extension module add2...
Allowing ninja to set a default number of workers... (overridable by setting the environment variable MAX_JOBS=N)
[1/3] /usr/local/cuda-12.1/bin/nvcc --generate-dependencies-with-compile ...
[2/3] /mnt/7T/jiaqiz/anaconda3/envs/cuda_learning/bin/x86_64-conda-linux-gnu-g++ ...
[3/3] /mnt/7T/jiaqiz/anaconda3/envs/cuda_learning/bin/x86_64-conda-linux-g++ ...
Loading extension module add2...
Running cuda...
Cuda time:  134.635us
Running torch...
Torch time:  50.449us
Kernel test passed.
```

## 环境信息

- **操作系统**: Linux 4.15.0-213-generic
- **Python**: 3.10
- **PyTorch**: 支持 CUDA 12.4
- **CUDA**: 12.1
- **GCC**: 11.2.0 (通过 conda 安装)
- **Ninja**: 最新版本

## 注意事项

1. 每次重新打开终端都需要重新设置环境变量
2. 如果使用其他编译方式（setup.py 或 CMake），也需要设置相同的环境变量
3. `TORCH_CUDA_ARCH_LIST` 可以根据你的 GPU 架构进行调整

## 故障排除

如果仍然遇到问题：

1. 确认 conda 环境已激活
2. 检查编译器路径是否正确
3. 确认 CUDA 和 PyTorch 版本兼容性
4. 清理之前的编译缓存：`rm -rf ~/.cache/torch_extensions/`
