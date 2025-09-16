#include <torch/extension.h>
#include "add2.h"

void torch_launch_add2(torch::Tensor &c,
                       const torch::Tensor &a,
                       const torch::Tensor &b,
                       int64_t n) {
    launch_add2((float *)c.data_ptr(),
                (const float *)a.data_ptr(),
                (const float *)b.data_ptr(),
                n);
}

// 使用 PYBIND11_MODULE 宏定义 Python 模块
// TORCH_EXTENSION_NAME 是模块名称，m 是 pybind11 模块对象
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // 将 C++ 函数 torch_launch_add2 暴露给 Python
    // 参数说明：
    // 1. "torch_launch_add2" - Python 中调用的函数名
    // 2. &torch_launch_add2 - 要暴露的 C++ 函数指针
    // 3. "add2 kernel warpper" - 函数的文档字符串
    m.def("torch_launch_add2",
          &torch_launch_add2,
          "add2 kernel warpper");
}

// 使用 TORCH_LIBRARY 宏注册 C++ 操作到 PyTorch 中
// add2 是库名称，m 是 torch::Library 对象
TORCH_LIBRARY(add2, m) {
    // 将 torch_launch_add2 函数注册为 TorchScript 操作
    // 参数说明：
    // 1. "torch_launch_add2" - 操作名称
    // 2. torch_launch_add2 - 要注册的 C++ 函数
    m.def("torch_launch_add2", torch_launch_add2);
}