# qcu
qcu reconstruct

## 重构逻辑

分模块重构 ： 通信、内存、计算、算法

~~~TODO-0：泛化SU(N)，使得Nc不再固定为3~~~ (此部分移动至MRHS_qcu实现)
TODO-1: 压缩dslash通信部分 向量长度Ns * Nc ---> Ns / 2 * Nc

## 运行环境
cuda / dcu
