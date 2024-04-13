# qcu
qcu reconstruct

## 重构逻辑

分模块重构 ： 通信、内存、计算、算法

~~TODO-0：泛化SU(N)，使得Nc不再固定为3~~ (此部分移动至MRHS_qcu实现)

TODO-1: 压缩dslash通信部分 向量长度Ns * Nc ---> Ns / 2 * Nc

TODO-2: clover dslash

TODO-3: 新版本的SHIFT函数（更通用更符合逻辑的版本）

## 运行环境
cuda / dcu

## 说明
1. 部分理论部分，将在未来使用Latex进行补全（或许会在另一个repo进行），文档之后公开
2. 由于某些原因，可能尽量少地使用C++的特性及模板，但是考虑到使用方便，还是使用了部分面向对象的内容和部分模板函数
特化（以在不降低效率的情况下同时提高可读性）。由于这些要求，代码可读性难免会差一些。第一次重构结构可能并不是特别的
好，后期尽量进行再重构，不过都是后面的事情了，也可能在MRHS里面进行😂。

## 程序说明
对外暴露的接口为interface/qcu.h，目前的库依赖于老版本的PYQUDA，后期将上传我修改后能运行的pyquda。
当前接口设计只是因为对老版本的代码比较友好，后期可能对接口进行更改。

## 模块说明
### algebra
此部分存放线性代数Functor及部分实现，稀疏矩阵向量乘只留基类作为未来接口，在实现dslash后，此部分调用dslash部分的函数。
此模块对于向量内积、求Lp范数涉及reduce操作，需要进行通信，依赖于`comm`模块
### comm
此部分为通信部分，设计上，此模块不应该依赖于其他模块，这里设计上暂时使用[`nccl`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage.html)通信，后期考虑补全naive MPI部分。
### mempool
通信部分，如果进程数目大于1，dslash部分，以及其他reduce操作，需要多进程通信，需要buffer，mempool用于管理host memory buffer
以及device memory buffer。
### qcd
TO BE DONE
### qcu_storage
TO BE DONE
### solver
TO BE DONE
### tests
TO BE DONE

## 使用说明
TO BE DONE