# Assignment 01 notes

## Module 0: 环境准备

### 0.1 第一个 CUDA 程序
运行命令：`cd cuda && make run/m0_env/01_hello`

启动配置：4 个 block，每个 block 8 个 thread，共 32 个 GPU thread。

观察：连续运行 5 次时，输出均呈现 block 0 到 block 3、每个 block 内 thread 0 到 7 的顺序。但这不是 CUDA 所保证的执行顺序，程序正确性不能依赖 block 或 thread 的打印先后。

### 0.2 GPU 参数
- GPU：NVIDIA GeForce RTX 5090
- compute capability：12.0
- SM 数量：170
- warp 大小：32 threads
- shared memory / block：49,152 B = 48 KiB
- 最大常驻线程 / SM：1,536
- 全局显存：33,668,988,928 B ≈ 33.67 GB
- 最大线程 / block：1,024

## Module 1: 为什么使用 GPU

### 核心概念
- **延迟（latency）**：完成一项工作所需的等待时间。
- **吞吐量（throughput）**：单位时间内完成工作的总量。TFLOPS 主要描述浮点计算吞吐量，而不是单条指令的延迟。
- **FLOP / FLOPS / TFLOPS**：一次浮点运算 / 每秒浮点运算次数 / 每秒万亿次浮点运算。
- **异构系统**：一个 CUDA 程序由 CPU（host）和 GPU（device）协作执行；两者通常有各自直接连接的内存。
- **kernel**：由 CPU 启动、在 GPU 上由大量 thread 并行执行的函数。
- **grid / block / thread**：一次 kernel 启动的全部 block / 可协作的一组 thread / 最小执行单位。一个 block 固定在一个 SM 上执行，不会被拆到多个 SM。
- **SM**：Streaming Multiprocessor。GPU 中执行 block 的基本硬件单元；本实验 GPU 有 170 个 SM。
- **warp**：32 个 thread 组成的硬件执行组。block 的 thread 数通常选为 32 的倍数，避免最后一个 warp 有闲置 lane。
- **SIMT**：Single Instruction, Multiple Threads。程序员按单个 thread 编程，硬件按 warp 执行。
- **branch divergence**：同一 warp 的 thread 走不同的 `if/else` 分支时，部分 lane 会暂时闲置，仍有性能代价。Volta 后独立线程调度不消除这一代价。
- **global memory（显存）**：所有 block 可访问的大容量 GPU 内存。
- **shared memory**：物理上位于 SM，逻辑上由每个 block 独享的高速片上内存。不同 block 的 shared memory 不会混用。
- **`__syncthreads()` / `__syncwarp()`**：分别等待同一 block 的所有 thread / 同一 warp 的参与 thread 到达同步点；不能用于跨 block 同步。

### 1.1 概念判断
- (a) 错。TFLOPS 衡量总吞吐，不保证单条指令延迟低于 CPU。
- (b) 对。高带宽依赖大量、连续或合并的访问；随机零散访问难以达到标称带宽。
- (c) 对。严格串行依赖限制并行度，GPU 的高吞吐难以发挥。
- (d) 错。1000 TFLOPS 不表示单次运算耗时为 10^-15 秒。

### 1.2 串行算法为何无法利用高 FLOPS
GPU 的高吞吐依赖大量可并行的独立工作。严格在线串行算法中，每一步都依赖前一步结果，无法同时展开；每一步的计算、访存和调度延迟会沿依赖链累加。因此，即使 GPU 的总 FLOPS 很高，算法仍可能无法在几秒内完成。

### 1.3 执行层次
| 层次 | 软件含义 | 对应硬件 | 直接可用的存储 | 同步与通信 |
|---|---|---|---|---|
| thread | kernel 的最小执行单位 | 一个 lane | 自己的寄存器 | 自身天然有序 |
| warp | 同一 block 中连续的 32 个 thread | SM 内的 warp 执行组 | 各 thread 的寄存器；所在 block 的 shared memory | `__syncwarp()`；warp-level primitive |
| block / CTA | 可协作的一组 thread | 一个 SM | block 的 shared memory；各 thread 的寄存器 | `__syncthreads()`；shared memory |
| grid | 一次 kernel 启动的全部 block | 整张 GPU（多个 SM） | global memory | 标准 CUDA 中不能在 kernel 内跨 block 同步；通常通过 kernel 边界通信 |

### 1.4 SIMD、SIMT 与分支分歧
SIMD 是一条指令显式处理多个数据元素，通常共享同一控制流。SIMT 是程序员按单个 thread 编程，硬件将 32 个 thread 组成 warp 执行。warp 内的 thread 可走不同分支，但不同分支通常需要分别执行，导致部分 lane 被屏蔽，因此产生 branch divergence 的性能代价。

### 1.5 并行度与吞吐实验
运行命令：`cd cuda && make run/m1_why_gpu/01_scaling`

| 配置 | 耗时 (ms) | ns / 元素 |
|---|---:|---:|
| CPU 单线程 | 10.397 | 2.48 |
| GPU `<<<1, 1>>>` | 134.514 | 32.07 |
| GPU `<<<1, 256>>>` | 2.318 | 0.55 |
| GPU 铺满 grid | 0.026 | 0.01 |

结果均通过正确性检查（`PASS`）。

结论：GPU 单 thread 比 CPU 单 thread 慢很多，因为 GPU 为高吞吐而非低延迟设计；只有一个 thread 时，大多数资源闲置，且访存延迟无法被隐藏。单 block 到铺满 grid 的耗时从 2.318 ms 降至 0.026 ms，约提升 89 倍：后者有 16,384 个 block，足以让 170 个 SM 持续工作，并通过大量 warp 隐藏访存延迟。

## Module 2: 第一个 CUDA kernel

### 核心概念
- **host / CPU**：运行普通 C++ 代码、启动 kernel 的一方。
- **device / GPU**：实际并行执行 kernel 的一方。
- **指针**：保存一片数据起始地址的变量。例如 `float *d_a` 表示“指向 float 数据的地址”。
- **`malloc(bytes)`**：向 CPU 内存申请一片空间。
- **`cudaMalloc(&d_a, bytes)`**：向 GPU 全局显存申请一片空间，并把 GPU 地址写入 `d_a`。
- **`cudaMallocManaged(&a, bytes)`**：申请 Unified Memory；CPU 和 GPU 都可通过同一个指针 `a` 访问。
- **`cudaMemcpy`**：显式在 CPU 内存与 GPU 显存间搬运数据。
- **kernel launch**：`kernel<<<blocks, threads>>>(...)`。第一个参数是 grid 中 block 数，第二个是每个 block 的 thread 数。
- **异步执行**：kernel launch 后 CPU 通常立刻继续执行；GPU 在另一条时间线上执行 kernel。
- **`cudaDeviceSynchronize()`**：让 CPU 等待此前 GPU 工作全部完成。课程的 `CUDA_CHECK_KERNEL()` 同时检查 kernel 错误并调用它。

### 2.1 一维向量加法
每个 thread 负责一个元素：

```cpp
int idx = blockIdx.x * blockDim.x + threadIdx.x;
if (idx < n) c[idx] = a[idx] + b[idx];
```

对任意 `n`，block 数采用向上取整：

```cpp
(n + threadsPerBlock - 1) / threadsPerBlock
```

这样最后不足一个 block 的元素也有 thread 覆盖，而越界 thread 被 `idx < n` 排除。

### 2.2 CUDA 修饰符
| 修饰符 | 用途 |
|---|---|
| `__global__` | CPU 启动、GPU 执行的 kernel |
| `__device__` | 仅在 GPU 执行、仅能被 GPU 函数调用的辅助函数 |
| `__host__ __device__` | CPU 与 GPU 均可调用的函数 |
| `__constant__` | GPU 上全局共享的只读变量，通常定义在函数外 |
| `__shared__` | GPU 上每个 block 独享、同一 block 内 thread 共享的变量 |

普通局部变量若定义在 kernel 内，通常每个 thread 各有一份，优先放在寄存器中；`cudaMalloc` 分配的是所有 block 可访问的 global memory。

### 2.3 Unified Memory
显式管理版本具有 CPU 数组 `h_a/h_b/h_c` 与 GPU 数组 `d_a/d_b/d_c`，需要两次 H→D 和一次 D→H 的 `cudaMemcpy`。

Unified Memory 版本改为只使用 `a/b/c`：

```text
CPU 填 a、b → GPU kernel 读 a、b 并写 c → CPU 读 c
```

CUDA 运行时按实际访问需求迁移数据页，因此删去所有 `cudaMemcpy`；kernel 后、CPU 读取 `c` 前仍必须同步。

在 RTX 5090 上的三次测量：

| 版本 | 三次耗时 (ms) | 平均 (ms) |
|---|---:|---:|
| Unified Memory | 97.0、99.3、95.2 | 97.2 |
| 显式 `cudaMemcpy` | 98.7、98.2、98.4 | 98.4 |

结论：本实验中两种方案性能基本相当，Unified Memory 平均略快约 1.3%，但波动范围重叠，不能据此断言它稳定更快。Unified Memory 的主要优势是简化代码；性能依赖硬件、驱动与访问模式。

### 2.4 异步与 stream
- kernel launch 默认异步：CPU 不等待 kernel 完成。
- 同一 stream 内任务按提交顺序执行；D→H `cudaMemcpy` 会等待此前 kernel 完成。
- kernel 内部非法访存发生在 GPU 实际执行时，通常在 `cudaDeviceSynchronize()`、D→H `cudaMemcpy` 等同步点才报告。

### 2.5 定位非法 kernel 启动
原程序设置 `threads = 2048`，超过 RTX 5090 的 `maxThreadsPerBlock = 1024`，导致 kernel 没有启动。

`d_c` 已被 `cudaMemset` 为 0；未检查错误时，复制回来就表现为 `MISMATCH`。加入 `CUDA_CHECK_KERNEL()` 后，立即报告 `cudaErrorInvalidValue`。将 thread 数改为 256 后通过测试。

### 2.6 二维矩阵加法
二维 thread 坐标通常约定：

```cpp
row = blockIdx.y * blockDim.y + threadIdx.y;
col = blockIdx.x * blockDim.x + threadIdx.x;
```

矩阵按行优先存为一维数组：

```cpp
idx = row * N + col;
```

`dim3 threads(16, 16)` 表示一个 block 有 16 列 × 16 行，即 256 个 thread。

对于 `M = 1000` 行、`N = 700` 列：

```text
blocks.x = ceil(N / 16) = 44，覆盖列
blocks.y = ceil(M / 16) = 63，覆盖行
```

必须使用二维边界保护：

```cpp
if (row < M && col < N) { ... }
```

### 2.7 Grid-stride loop
固定 launch 为 `<<<64, 256>>>` 时，只有 `64 × 256 = 16384` 个 thread，远少于 `n = 2^24`。Grid-stride loop 让每个 thread 处理多个元素：

```cpp
for (int idx = threadIdx.x + blockIdx.x * blockDim.x;
     idx < n;
     idx += gridDim.x * blockDim.x) {
    c[idx] = a[idx] + b[idx];
}
```

步长是整个 grid 的 thread 总数。其价值是 kernel 不依赖“启动 thread 数必须覆盖 n”，可处理任意规模的数据。代价是本题只有 64 个 block，少于 GPU 的 170 个 SM，且每个 thread 要顺序处理大量元素，无法充分利用并行度。

### 2.8 Block 执行顺序
多次运行中 16 个 block 的打印顺序不同。block 的调度顺序由 GPU 硬件调度器和 SM 资源状态决定，不由 block 编号保证。

正确性不能依赖 block 的执行先后。这使 CUDA 程序可扩展到不同规模的 GPU：独立 block 可被任意顺序、分批调度到任意可用 SM，结果仍应相同。

## 日常工作流：编辑、备份、运行

### 每次完成一小段本地工作后
在本地 `ai-infra-coursework` 根目录执行：

```bash
git status
git add assignment01/notes.md <本次修改的代码文件>
git commit -m "简短说明本次完成内容"
git push origin main
```

先用 `git status` 检查变更；只 `git add` 本次应保存的文件。`commit` 是本地存档，`push` 才会上传到个人 GitHub fork。

### 将本地最新版本同步到集群
集群无法连接 GitHub，因此在 Mac 本地终端执行（首次同步整个目录）：

```bash
scp -r -P 10029 "/Users/<用户名>/Documents/Work/WMHPC x LCPU Infra Seminars/ai-infra-coursework" <手机号后8位>@221.238.13.179:~/
```

目标目录已存在后，不要直接重复整目录复制，以免产生嵌套目录。只同步本次改动的文件，例如：

```bash
scp -P 10029 assignment01/notes.md <手机号后8位>@221.238.13.179:~/ai-infra-coursework/assignment01/
scp -P 10029 assignment01/cuda/路径/文件.cu <手机号后8位>@221.238.13.179:~/ai-infra-coursework/assignment01/cuda/路径/
```

### 在集群运行 CUDA 作业
```bash
ssh -p 10029 <手机号后8位>@221.238.13.179
cd ~/ai-infra-coursework/assignment01/cuda
srun -G 1 --time=00:15:00 --pty bash
make run/题目路径
exit
```

`srun` 申请 GPU；`exit` 退出并释放 GPU。本地是唯一编辑与保存来源；集群只用于运行测试。确认 PASS 后，在本地提交并推送 GitHub。
