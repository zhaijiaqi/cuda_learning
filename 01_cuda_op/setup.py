from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# 使用 setuptools 的 setup 函数来配置和构建 CUDA 扩展模块
setup(
    name="add2",  # 模块名称
    include_dirs=["include"],  # 指定额外的头文件搜索路径
    # ext 是 extension（扩展）的缩写，表示要构建的扩展模块列表
    ext_modules=[
        # 使用 CUDAExtension 定义 CUDA 扩展模块
        CUDAExtension(
            name = "add2",  # 扩展模块名称
            sources=["kernel/add2_ops.cpp", "kernel/add2_kernel.cu"],
        )
    ],
    # 指定构建扩展的命令类
    cmdclass={
        "build_ext": BuildExtension  # 使用 PyTorch 提供的 BuildExtension 来处理构建过程
    }
)