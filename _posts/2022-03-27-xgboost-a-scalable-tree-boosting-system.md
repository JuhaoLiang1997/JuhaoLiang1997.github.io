---
layout: post
title: 'XGBoost: A Scalable Tree Boosting System'
date: 2022-03-27 08:46 +0800
category: [papers]
tags: [XGBoost, Model Ensemble]
typora-root-url: "../_images"
math: true
comments: false
toc: true
pin: false
---

# Paper 阅读笔记

[XGBoost: A Scalable Tree Boosting System](https://arxiv.org/abs/1603.02754)

之前写 ensemble 那篇笔记的时候碰上了这些 boosting 的方法，看了几个讲解 XGBoost 的视频后还是不太了解，于是开始阅读这篇 paper，读了差不多七八个小时才读完.... 看完后感觉基本的概念都了解一点，需要阅读或者实际使用一下这个系统才能更好的体会到各个技术点的作用。

---

## Contributions:

> - We design and build a **highly scalable end-to-end tree boosting system**.
> - We propose a theoretically justified **weighted quantile sketch** for efficient proposal calculation.
> - We introduce a novel **sparsity-aware algorithm** for **parallel tree learning**.
> - We propose an effective **cache-aware block structure** for **out-of-core tree learning**.

---

## Tree Boosting

### 正则化的目标函数 Regularized Learning Objective

<img src="/2022-03-27-xgboost-a-scalable-tree-boosting-system/截屏2022-03-27 02.34.36.png" alt="截屏2022-03-27 02.34.36" style="zoom:50%;" />
_预测值是所有树的预测之和_

在给定的数据集 $\mathcal{D}$ 中，有 $n$ 条数据 $m$ 个特征，Tree Boosting 中有 $K$ 棵**回归树 (CART树)**，每个回归树的叶子结点都有数值。对于每个输入 $x$，可以使用 $q$ （树结构）计算出所对应的叶子结点，叶子结点的值 $w$ 即为预测结果 $\hat y$。而在整个 ensemble 模型中，所有树的预测结果相加就是最终结果 $\hat y$，如式 (1) 所示。这里可以理解为每一棵树都是为了弥补前一棵树的不足（残差）而训练：$y_{i} = y - y_{i-1}$，这也意味着在 boosting 没有办法像 bagging 的 ensemble 方法一样**并行的训练所有树**。

<img src="/2022-03-27-xgboost-a-scalable-tree-boosting-system/截屏2022-03-27 02.04.06.png" alt="截屏2022-03-27 02.04.06" style="zoom:50%;" />
_预测值_

得到预测值之后，怎样去训练模型使得预测值和真实值更接近呢？通过 minimize 以下**目标函数**训练模型，目标函数由两部分组成：

- 第一部分 $l$ 是损失函数 **loss function**，用于衡量预测值和实际值的差距，这个是可以自定的，只要可以**一阶二阶可求导**的凸函数就好。
- 第二项 $\Omega$ 是惩罚项/正则项 **penalty**，用于限制模型复杂程度，防止过拟合。其中 $T$ 是叶子结点个数，$\omega$ 是叶子结点值的和， $\gamma$ 和 $\lambda$ 是系数，可以自己调整权重。

<img src="/2022-03-27-xgboost-a-scalable-tree-boosting-system/截屏2022-03-27 02.19.58.png" alt="截屏2022-03-27 02.19.58" style="zoom:50%;" />
_目标函数_

### Gradient Tree Boosting

因为式 (2) 中有函数作为参数，所以很难使用传统的方法去优化这个函数。以下式子是在第 $t$ 棵树中的目标函数，其中 $ \hat {y} _ {i} ^ {(t-1)}$ 是第 $i$ 条数据在第 $t-1$ 棵树中的预测值。这里就是式 (2) 中 $\hat y_{i} = \hat y_{i}^{t-1}+f_{t}(x_{i})$  ，也就是补残差。

<img src="/2022-03-27-xgboost-a-scalable-tree-boosting-system/截屏2022-03-27 02.34.13.png" alt="截屏2022-03-27 02.34.13" style="zoom:50%;" />
_目标函数_

然后使用**[二阶泰勒展开](https://zh.wikipedia.org/wiki/%E6%B3%B0%E5%8B%92%E5%85%AC%E5%BC%8F)**/**[牛顿法](https://zh.wikipedia.org/wiki/%E7%89%9B%E9%A1%BF%E6%B3%95)**去近似表达这个目标函数，方便后续更高效的优化目标函数，得到新的目标函数

<img src="/2022-03-27-xgboost-a-scalable-tree-boosting-system/截屏2022-03-27 02.43.17.png" alt="截屏2022-03-27 02.43.17" style="zoom:50%;" />
_泰勒展开式_

<img src="/2022-03-27-xgboost-a-scalable-tree-boosting-system/截屏2022-03-27 02.44.16.png" alt="截屏2022-03-27 02.44.16" style="zoom:50%;" />
_近似的目标函数_

其中就是把 $\hat y_{i}^{t-1}$ 视为泰勒式中的 $a$，把 $f_{t}(x_{i})$ 视为 $(x-a)$ ，$g_{i}$ 是目标函数对上一棵树的预测值 $\hat y_{i}^{t-1}$ 的一阶导数 ${f}'(a)$，$h_{i}$ 为二阶导数 ${f}''  (a)$。移除**常数项** $l(y _ {i} , \hat y ^ {t-1})$ 之后获得以下第 $t$ 棵树的目标函数式 (3)

<img src="/2022-03-27-xgboost-a-scalable-tree-boosting-system/截屏2022-03-27 02.52.27.png" alt="截屏2022-03-27 02.52.27" style="zoom:50%;" />
_目标函数_

接着，定义 $I_{j} = \{i \mid q\left(\mathbf{x}_{i}\right) = j\}$ 为叶子结点 $j$ 的数据集合，将正则项 $\Omega$ 代入到目标函数中

<img src="/2022-03-27-xgboost-a-scalable-tree-boosting-system/截屏2022-03-27 03.04.19.png" alt="截屏2022-03-27 03.04.19" style="zoom:20%;" />
_正则项_

将**用数据求和**改成**用叶子结点的数据集合求和**，即 $$\sum_{i=1}^{n}g_{i} f_{t}\left(\mathbf{x}_{i}\right)=\sum_{j=1}^{T}\left(\sum_{i \in I_{j}} g_{i}\right) w_{j}$$  ，合并同类项得到式 (4)

<img src="/2022-03-27-xgboost-a-scalable-tree-boosting-system/截屏2022-03-27 02.55.43.png" alt="截屏2022-03-27 02.55.43" style="zoom:50%;" />
_目标函数_

接着，在式 (4) 中把 $w_{j}$ 视为变量，目标函数即为**一元二次方程**，取 $-\frac{b}{2a} $ 时为最优解，所以求得叶子结点 $j$ 的最优值 $w_{j}^{*}$ 

<img src="/2022-03-27-xgboost-a-scalable-tree-boosting-system/截屏2022-03-27 03.07.15.png" alt="截屏2022-03-27 03.07.15" style="zoom:50%;" />
_最优叶子结点值_

将最优解代入式 (4) 中，得到最优目标函数值；这个式子可以用来衡量树结构 $q$ 的优劣，利用这个式子就可以像决策树的信息增益一样进行**特征选择**，进而优化树的结构。

<img src="/2022-03-27-xgboost-a-scalable-tree-boosting-system/截屏2022-03-27 03.13.08.png" alt="截屏2022-03-27 03.13.08" style="zoom:50%;" />
_目标函数_

如何判断分割的效果呢？每次树进行分割时，假定左边数据集为 $I_{L}$ 和右边数据集为 $I_{R}$ ，当前结点的总数据集 $I=I_{L}+I_{R}$ ，使用以下式子去计算切分后**目标函数减小的数值**：左子树分数与右子树分数的和减去不分割情况下的分数以及加入新叶子节点引入的复杂度代价。 $\gamma$ 是抑制节点个数的，是节点分裂的阈值；$\lambda$ 是抑制节点值不要太大。

<img src="/2022-03-27-xgboost-a-scalable-tree-boosting-system/截屏2022-03-27 03.21.40.png" alt="截屏2022-03-27 03.21.40" style="zoom:50%;" />
_分割后目标函数减小值_

### Shrinkage and Column Subsampling

除了添加正则项外，还增加了两个额外的技术去防止过拟合。

- 第一个是 **shrinkage**，shrinkage 会给每一个新增的叶子结点数值乘以 $\eta$ ，相当于优化器的**学习率**，它降低了每棵树的叶子结点对将来的树的影响。降低每棵树对模型的优化程度，利用更多的树慢慢的逼近结果，使得学习更加平缓，可以更好的避免过拟合。
- 第二个是 **column (feature) subsampling**，根据用户反馈，使用 column subsampling 可以比传统的 row subsampling 更加有效的防止过拟合，同时也加快了并行计算的速度。和**随机森林**的应用是一样的，支持列抽样可以降低过拟合，同时减少了计算量。

---

## Split Finding Algorithms

### Basic Exact Greedy Algorithm

以上介绍了 boosting 方法，但是如何根据式 (7) 找到最好的分裂树的方法还是最大的问题。如果是在分裂时将所有的特征都计算一遍 $L_{\text {split }}$ 则称为 *exact greedy algorithm* 。如 Algorithm 1 所示，它遍历了所有可能的特征，根据式 (7) 找出最大的 $score$ ，作为其最优分裂方案，为了提升效率，它必须先对叶子结点按照输入值排序，再去累加 $G_{L}$ 和 $H_{L}$，这样效率太低了。

<img src="/2022-03-27-xgboost-a-scalable-tree-boosting-system/截屏2022-03-27 14.43.06.png" alt="截屏2022-03-27 14.43.06" style="zoom:50%;" />
_Exact Greedy Algorithm_

### Approximate Algorithm

Exact Greedy Algorithm 很强但当数据不能全部存入内存时**很难高效**地执行，而且在**分布式计算**中也会有问题。因此，Approximate Algorithm 应运而生。

<img src="/2022-03-27-xgboost-a-scalable-tree-boosting-system/截屏2022-03-27 14.53.59.png" alt="截屏2022-03-27 14.53.59" style="zoom:50%;" />
_Approximate Algorithm_

第一步根据特征分布选出**候选分裂点** (candidate splitting points)，然后将特征根据候选分裂点进行 split，再累加各个区域的特征的 $g_{j}$ 和 $h_{j}$，根据式 (7) 算出分裂效果，然后从这些候选分裂点中找出最佳方案。而候选分裂点密度越高时就越接近 Exact Greedy Algorithm，准确率也就越高；相反，密度越低，计算的越快，拟合程度低，防止过拟合就越好。

选取候选分裂点有两种方案。**global variant** 在生成树结构之前就提出了所有的 candidate splits，后续不更新。**local variant** 会在每次分裂结束后重新提出新的候选方案。Global 的方法可以减少提出候选方案的次数，然而每次的候选数量会比 local 的多。而 local 则是每次分裂结束后都重新选候选方案，对于更深的树，可以选出更加合适的 candidates。以下实验也说明了，在 global 给出足够多的 candidates 时，准确率也可以和 local 的一样。

<img src="/2022-03-27-xgboost-a-scalable-tree-boosting-system/截屏2022-03-27 15.08.05.png" alt="截屏2022-03-27 15.08.05" style="zoom:50%;" />
_两种方案对比_

### Weighted Quantile Sketch

那么如果挑选候选分裂点呢？通常根据 feature 的值均匀的分裂。首先 $D_{k}=\{(x_{1 k}, h_{1}),(x_{2 k}, h_{2}) \cdots(x_{n k}, h_{n})\}$ 表示第 $k$ 个 feature 的所有数据的值和二阶导数，然后定义 *rank functions* 式 (8)，表示第 $k$ 个特征中值小于 $z$ 的数据的比例，由 $h$ 加权。目的就是找到所有的 $z$ 使得每一个区间的比例约等于 $1/\varepsilon $ ，这个 $\varepsilon $ 是一个分裂密度的参数，值越大，表示区间越小，越接近 exact greedy algorithm。

<img src="/2022-03-27-xgboost-a-scalable-tree-boosting-system/截屏2022-03-27 15.17.24.png" alt="截屏2022-03-27 15.17.24" style="zoom:50%;" />
_候选分裂点选取_

为什么要用 $h$ 加权呢？这里将式 (3) 转化为以下式子，这里就是**一元二次方程**转为完全平方式。转换后的目标函数正是对 $g_{i}/h_{i}$ 的加权方差 (weighted squared loss)，权为 $h_{i}$ 。

<img src="/2022-03-27-xgboost-a-scalable-tree-boosting-system/截屏2022-03-27 15.25.52.png" alt="截屏2022-03-27 15.25.52" style="zoom:50%;" />
_目标函数_

<img src="/2022-03-27-xgboost-a-scalable-tree-boosting-system/截屏2022-03-27 15.25.38.png" alt="截屏2022-03-27 15.25.38" style="zoom:50%;" />
_目标函数_

当每一个数据都有相同的权重时，*quantile sketch* 可以解决这问题。然而目前没有算法解决**加权数据**，因此提出了有理论支撑的、可以解决加权数据的 *a novel distributed eighted quantile sketch algorithm* ，大概思路就是一个数据结构支持 *merge* 和 *prune* 操作，每个操作都保持一定的准确率，具体的描述和细节在论文附录中。

### Sparsity-aware Split Finding

在实际问题中，很多输入是 **sparse** 的，有以下几个原因：数据缺失值、很多零值、特征工程的特性（onehot）。因此提出了为每个树节点增加默认的方向 **default direction**，如下图。

<img src="/2022-03-27-xgboost-a-scalable-tree-boosting-system/截屏2022-03-27 15.41.21.png" alt="截屏2022-03-27 15.41.21" style="zoom:50%;" />
_default direction_

当一个值缺失时，会划分到默认的方向。最优的默认方向是从数据中学习而来，如下图 Algorithm 3，$x_{ik}$ 是 no-missing 的数据，只对这些不缺失的数据进行 accumulate，根据这些数据进行划分，不管 missing 数据。

<img src="/2022-03-27-xgboost-a-scalable-tree-boosting-system/截屏2022-03-27 15.44.50.png" alt="截屏2022-03-27 15.44.50" style="zoom:50%;" />
_sparsity-aware split finding_

实验证明，处理了这些 sparse 数据之后，快了 50 倍，证明了这个处理是有必要的。

<img src="/2022-03-27-xgboost-a-scalable-tree-boosting-system/截屏2022-03-27 15.53.09.png" alt="截屏2022-03-27 15.53.09" style="zoom:50%;" />
_sparse实验_

---

## System Design

### Column Block for Parallel Learning

树结构优化最耗时的部分就是**数据排序**，提出了将数据存在一种 in-memory units, *block*，数据在每一个 *block* 中是按照 *compressed column (CSC)* 格式存储的，每一个 column 按照对应的特征值进行排序，该排序能复用。

在 exact greedy algorithm 中，把整个数据集存到一个 blcok 中，然后二分搜索已经提前完成排序的 entries，一次遍历就可以收集到所有叶分支的 split candidates 的 gradient 信息。下图展示了如何将数据集传到 CSC 格式，以及利用 block 结构找到最优分解:

<img src="/2022-03-27-xgboost-a-scalable-tree-boosting-system/截屏2022-03-27 16.02.57.png" alt="截屏2022-03-27 16.02.57" style="zoom:50%;" />
_block structure for parallel learning_

这个 block 结构也帮助 approximate algorithms。使用多个 blocks，每个 block 对应着数据子集，不同 block 可以分布在不同的机器上。使用这种已经完成排序的结构，*quantile finding step* 线性的遍历排好序的 columns。这个对 local proposal algorithms 很有帮助，因为 candidates 很频繁的生成。

**时间复杂度分析**：$d$ 为树的最大深度，$K$ 为树的个数。

- 对于 **exact greedy algorithm**，原始的 sparse aware algorithm 的时间复杂度是 $$O\left(K d\|\mathbf{x}\|_{0} \log n\right)$$ ，这里用 $$\|\mathbf{x}\|_{0}$$ 表示非缺失值的个数。在 block structure 上 tree boosting 时间复杂度为 $$O\left(K d\|\mathbf{x}\|_{0}+\|\mathbf{x}\|_{0} \log n\right)$$ 这里的 $$\|\mathbf{x}\|_{0} \log n$$ 就是 block structure 只计算一遍排序的时间。当需要多次排序时 block strcuture 可以节省很多计算。
- 对于 **approximate algorithm** 来说，用二分搜索原始的时间复杂度是 $$O\left(K d\|\mathbf{x}\|_{0} \log q\right)$$  ，这里的 log 参数 $q$ 通常是在32到100之间；使用 block structure 之后，可以把时间降到 $$O\left(K d\|\mathbf{x}\|_{0}+\|\mathbf{x}\|_{0} \log B\right)$$ ，这里也同样的是把只排序一遍。

### Cache-aware Access

block structure 对搜索有帮助，但对于 gradient statistics 来说还是需要按行来读取。非连续内存的读写会拖慢 split finding 的速度。对于 exact greedy algorithm 来说，可以通过 cache-aware prefetching algorithm 缓解这个问题；具体而言，给每个线程分配了连续的 buffer 用于存储 gradient statistics，使其可以在连续内存中读取信息，然后用 mini-batch 方法进行 accumulation，在数据较多的情况下减少运行时间。如图 7 显示，大数据集情况下运行时间快了两倍。

<img src="/2022-03-27-xgboost-a-scalable-tree-boosting-system/截屏2022-03-27 16.46.34.png" alt="截屏2022-03-27 16.46.34" style="zoom:50%;" />
_Cache-aware Access_

对于 approximate algorithms，通过选择正确的 block size 去解决，这里的 block size 是指每个 block 最多可以存储数据的个数。选择过小的 block size 会减小每个线程的负担导致不能高效的并行运算。过大的 block size 会导致缓存不够，命中率低。经过对比几次实验后，发现每个 block 存储 $2^{16}$ 个 examples 可以很好的平衡 cache property 和 parallelizattion。

### Blocks for Out-of-core Computation

如何将一个机器的所有资源都充分利用呢？除了处理器和内存外，还可以使用**磁盘空间**去处理不在主内存中的数据。为了实现 **out-of-core computation** ，把数据分为多个 blocks 存到磁盘中，在计算的同时，使用单独的线程去 pre-fetch 这些 block 到内存中，因此 computation 可以在并行读取 disk 中发生。然而，这也不能完全解决问题，disk 的读取会耗费很长的 computation time，要想办法减小开销和增加 disk IO 的吞吐量，主要用了两个技术提升 out-of-core-computation。

- **Block Compression**：block 按列压缩，并由独立的线程在加载到主内存时解压缩。对于 row index，只记录对于 beginning index 的偏移量，用 16 bit integer 存储。这里每一个 block 可以存储 $2^{16}$ 个 examples，压缩率达到 26% 到 29%。
- **Block Sharding**：将数据拆分到多个磁盘中，为每个磁盘分配一个 prefetch 线程，将数据取到内存缓冲区中。然后，线程交替地从每个缓冲区读取数据。当有多个磁盘可用时，这有助于提高磁盘读取的吞吐量。

---

## Related Works

论文中按照各个技术列了一下对应的相关工作... 

- **Gradient boosting**: additive optimization in functional space

- **Regularized model**: prevent overfitting

- **Column sampling**: prevent overfitting

- **Sparsity-aware learning**: handle all kinds of sparsity patterns

- **Parallel tree learning**

- **Cache-aware learning**

- **Out-of-core computation**

- **Weighted quantile sketch**: finding quantiles on weighted data

  <img src="/2022-03-27-xgboost-a-scalable-tree-boosting-system/截屏2022-03-27 17.23.39.png" alt="截屏2022-03-27 17.23.39" style="zoom:50%;" />
  _related works_

---

## End to End Evaluation

这一部分就是用 XGBoost 做实验了

---

## Conclusion

> In this paper, we described the lessons we learnt when building XGBoost, a **scalable tree boosting system** that is widely used by data scientists and provides state-of-the-art results on many problems. We proposed a novel **sparsity aware algorithm** for handling sparse data and a theoretically justified **weighted quantile sketch** for approximate learning. Our experience shows that **cache access patterns**, **data compression** and **sharding** are essential elements for building a scalable end-to-end system for tree boosting. These lessons can be applied to other machine learning systems as well. By combining these insights, XGBoost is able to solve real world scale problems using a minimal amount of resources.
>

---

## 读后笔记

- XGBoost 支持了多类型分类器，传统的GBDT采用CART作为基学习器
- XGBoost 支持自定目标函数，只要一阶二阶可导
- XGBoost 增加了正则项：叶子结点数量、叶子结点值的和。
- shrinkage 降低每棵树的提升效果，相当于学习率
- column subsampling 列抽样，防止过拟合，加快运算
- 对所有 sparsity patterns 可以计算 default direction
- 基于特征的并行运算，在树进行分裂时，并行计算各个特征的增益
- 近似分裂算法：在树进行分裂时，可以不去遍历所有情况，而是采用近似划分的方法计算
- 数据排序实现计算并存储于 block 中，加快效率，方便并行运算
- cache-aware 和 Out-of-core computation，利用 cache 和 disk 进行加速
- 做了很多工程上的优化
