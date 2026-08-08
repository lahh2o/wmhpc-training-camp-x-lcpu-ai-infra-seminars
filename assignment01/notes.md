# Assignment 01 notes

## Module 0: 环境准备

### 0.1 第一个 CUDA 程序
运行命令：
`cd cuda && make run/m0_env/01_hello`

启动配置：4 个 block，每个 block 8 个 thread，共 32 个 GPU thread。

观察：连续运行 5 次时，输出均呈现 block 0 到 block 3、每个 block 内 thread 0 到 7 的顺序。
但这不是 CUDA 所保证的执行顺序，程序正确性不能依赖 block 或 thread 的打印先后。

### 0.2 GPU 参数
- GPU：NVIDIA GeForce RTX 5090
- compute capability：12.0
- SM 数量：170
- warp 大小：32 threads
- shared memory / block：49,152 B = 48 KiB
- 最大常驻线程 / SM：1,536
- 全局显存：33,668,988,928 B ≈ 33.67 GB
- 最大线程 / block：1,024
