__global__ void MatAdd(float* c,
                            const float* a,
                            const float* b,
                            int n)
{
    // 计算当前线程处理的矩阵元素坐标
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = i*n + j;  // 按行优先
    if (i < n && j < n)
        c[idx] = a[idx] + b[idx];
}

void launch_add2(float* c,
                 const float* a,
                 const float* b,
                 int n) {
    dim3 block(16, 16);
    dim3 grid(n/block.x, n/block.y);

    MatAdd<<<grid, block>>>(c, a, b, n);
}