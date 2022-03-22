---
layout: post
title: Graph Convolutional Networks for Text Classification
date: 2022-03-20 22:15 +0800
category: [papers]
tags: [NLP, GCN]
typora-root-url: "../_images"
math: true
comments: false
toc: true
pin: false
---

# Paper 阅读笔记

[Graph Convolutional Netowrks for Text Classification](https://arxiv.org/abs/1809.05679)

构建 双层 GCN 异构图，words 和 documents 做节点，words 之间的 edge 用词共现信息表示，word 和 documents 之间的 edge 用 tf-idf 表示。然后做把文本分类转换为点分类，该方法可以在小比例的标记文档中实现强大的分类性能，可以同时学习单词和文档节点 embedding。[源码](https://github.com/yao8839836/text_gcn) 在这里。

**Keywords**: NLP, Text GCN, GCN, Text Classification, Graph

---

## Contributions：

- 提出了一个用于文本分类新的图神经网络方法，可用于联合学习 **word 和 document embeddings**
- 几个基准数据集显示这个方法 state-of-the-art，没有用到预训练词向量和额外知识。

---

## Related Work：

> Deep learning text classification studies can be categorized into two groups.

- Models based on word embeddings
- Model employed deep neural networks

---

## Method：

### GCN

- graph $ G = (V, E)$ where $V, E$ are sets of nodes and edges

- $ (v, v) \in E $  for any  $v$, every node is assumed to be connected to itself

- $X \in \mathbb{R}^{n \times m}$  是特征矩阵，n 个节点 m 维特征；$X_{v}  \in \mathbb{R}^{m}$ 表示 $v$ 的特征向量

- $A$  是  $G$  的邻接矩阵，度矩阵  $D$  where  $D_{i i}=\sum_{j} A_{i j}$  就是邻接矩阵求每个点的度 (邻接数)；$A$ 的主对角线都设为 1 因为每个点都连接自己

- 单层 GCN 时每个点都只能获取距离为一的邻接点的信息，而多层 GCN 叠加的时候就可以把更广的范围信息整合起来

- 单层 GCN，新的 $k$-dimensional node 的特征矩阵 $L^{(1)} \in \mathbb{R}^{n \times k}$ 可以被计算为

   $L^{(1)}=\rho\left(\tilde{A} X W_{0}\right)$ (1) :+1::+1:

  where $\tilde{A}=D^{-\frac{1}{2}} A D^{-\frac{1}{2}}$ is normalized symmetric adjacency matrix and $W_{0} \in \mathbb{R}^{m \times k}$ is a weight matrix,  $\rho$ 是激活函数

- 多层 GCN：

   $L^{(j+1)}=\rho\left(\tilde{A} L^{(j)} W_{j}\right)$  (2)

   where $j$ denotes the layer number and $L^{(0)} = X$ 

### Text GCN

- 建立异构图可以把**词共现信息**显性的表现在graph中，方便图卷积

- 图的点数量 V 等于 documents 数 + words 数 (corpus size + vocabulary size)

- 输入时简单设置特征矩阵 $X = I$ ，每个word/document都表示为 one-hot vector

- Document node 和 word node 的边权重为 word 在 document 中的 TF-IDF，他们测试过 TF-IDF weight 比只用 term frequency 的要好

- 使用 point-wise mutual information (PMI) 计算两个 word node edge 的权重，PMI 要比**共现次数**效果好

- 综上得到邻接矩阵 $A$

  <img src="/2022-03-20-graph-convolutional-networks-for-text-classification/equation1.png" alt="equation1" style="zoom:50%;" />
  
- PMI 值计算 word node $i, j$
  
  <img src="/2022-03-20-graph-convolutional-networks-for-text-classification/equation2.png" alt="equation2" style="zoom:50%;" />
  
  where #W(i) 是 sliding windows 的包含 word $i$ 的数量
  
  #W(i, j) 是包含 word $i, j$ 的 sliding windows 数量
  
   #W 是 sliding windows 在 corpus 的总数量
  
- 一个正的 PMI 值说明语义上两个词强关联，反之弱关联；因此只把 PMI 为正的边添加到 graph 中（threshold -> sparse）

- 建立好 graph 之后把它放进两层 GCN 中：第一层根据公式 (1) 算出新的特征向量，再扔进第二层公式(1) 又算出新的特征向量，然后扔进 *softmax* 里生成等同标签数的概率，进行分类

  $Z=\operatorname{softmax}\left(\tilde{A} \operatorname{ReLU}\left(\tilde{A} X W_{0}\right) W_{1}\right)$ （7）

  loss function 就用的 cross-entropy，只计算了 labeled document 节点:

  <img src="/2022-03-20-graph-convolutional-networks-for-text-classification/equation3.png" alt="equation3" style="zoom:50%;" />(8)

  where $\mathcal{Y}_{D}$ is the set of labeled document indices

  $F$  is the dimension of the output features

  $Y$  is the label indicator matrix (label 矩阵)

- 公式 (7) 的 $W_{0}, W_{1}$ 可以用 gradient descent 训练，示意图如下

  <img src="/2022-03-20-graph-convolutional-networks-for-text-classification/Schematic of Text GCN.png" alt="Schematic of Text GCN" style="zoom:50%;" />
  _Schematic of Text GCN_

- 两层 GCN 最多允许 距离为 2 的信息传递，即使没有 document 之间的连边，两层 GCn 也允许 documents 之间传递信息。他们实验说两层的要比一层的要好，更多层也没有提升效果。

---

## Experiments：

### Determine:

- Can our model achieve satisfactory results in text classification, even with limited labeled data? 能不能使用少量的 labeled data 训练得到不错的结果，在文本分类中
- Can our model learn predictive word and document embeddings?

### Baselines

- **TF-IDF + LR**：tf-idf 特征 + Logistic Regression classifier
- **CNN**
  - CNN-rand：随机初始化embedding
  - CNN-non-static：无监督学习的 embedding，训练中 fine tuned
- **LSTM**：有/无 预训练词向量
- **Bi-LSTM**：预训练词向量
- **PV-DBOW**：BOW，Logistic Regression Classifier
- **PTE**：词向量  average 做 doc向量
- **fastText**：word/n-grams 向量均值做 doc向量 + linear classifier
- **SWEM**：词向量 做 simple pooling strategies
- **LEAM**：词和标签在同一空间进行分类，用到了 label description
- **Graph-CNN-C**：graph CNN + shev filter
- **Graph-CNN-S**：graph CNN + Spline filter
- **Graph-CNN-F**：graph CNN + Fourier filter

### Datasets

用了五个数据集：20-Newsgroups (20NG), Ohsumed, R52 and R8 of Reuters 21578 and Movie Review (MR).

### Preprocess

- Cleaning & Tokenizing

- Removed stop words NLTK, low frequency words less than 5 times for some datasets. MR 数据集太小了就没移除低频词

  <img src="/2022-03-20-graph-convolutional-networks-for-text-classification/Summary statistics of datasets.png" alt="Summary statistics of datasets" style="zoom:50%;" />
_Summary statistics of datasets_

### Settings

- Text GCN, **embedding size** of first convolution as 200, **window size** as 20, **learning rate** 0.02, **dropout rate** 0.5, **L2 loss** 0. 10 fold Cross-validation, 200 **epochs**, **Adam optimizer**, **early stopping** (10 epoch)
- Baseline model with default parameter setting as original papers
- Baseline pretrain word embeddings: 300-dimensional GloVe

### Performance

<img src="/2022-03-20-graph-convolutional-networks-for-text-classification/ Test Accuracy.png" alt=" Test Accuracy" style="zoom:50%;" />
_Test Accuracy_

- Text GCN performs the best and significantly outperforms all baseline models on four datasets except MR. 说明长文本数据集表现不错

> Word nodes can gather comprehensive document label information and act as bridges or key paths in the graph, so that label information can be propagated to the entire graph.
>

- 这里作者说的很棒，document 的标签信息影响到了词向量的表达

> However, we also observed that Text GCN did not outperform CNN and LSTM-based models on MR. This is because GCN ignores word orders that are very useful in sentiment classification, while CNN and LSTM model consecutive word sequences explicitly.

- 总结了在 MR 数据集上表现不如 CNN 和 LSTM 的原因，没有利用到语序，结果在情感分类表现就没有序列模型那么好了。

> Another reason is that the edges in MR text graph are fewer than other text graphs, which limits the message passing among the nodes. There are only few document-word edges because the documents are very short. The number of word-word edges is also limited due to the small number of sliding windows.

- 再有就是在短文本 corpus 里，document 连接的 word node 就很少，导致标签信息很难传开，这也和 sliding windows 大小有关。

#### Parameter Sensitivity

- 作者测试了 不同 sliding windows 大小对模型的影响，实验表明过大和过小都会有负影响。说明了太小的不能生成足够的词共现信息，太大的导致不关联词语也连接上边，减少了差异性
- 也测试了第一层 GCN 不同维度的 embeddings，太小的维度导致 标签信息无法传开；太大的维度导致训练成本增加，同时性能也没提升。

#### Effects of the Size of Labels Data

- 在 20NG 和 R8 数据集上做了几个模型在不同比例 labeled data 的数据集情况下的 accuracy 分析。结果说明了 GCN 在低标签数据集中也可以有很好的表现 20% training documents

#### Document Embeddings

- t-SNE 分析了生成的 document embeddings，和其他几个模型做了对比。说是 Text GCN 可以学习到更有区分度的 embeddings

  <img src="/2022-03-20-graph-convolutional-networks-for-text-classification/document embeddings.png" alt="document embeddings" style="zoom:50%;" />
_document embeddings_

#### Word Embeddings

- 同样用 t-SNE 对比 Text GCN 所生成的 word embeddings 和其他模型。

### Discussion

> From experimental results, we can see the proposed Text GCN can achieve strong text classification re- sults and learn predictive document and word embeddings. However, a major limitation of this study is that the GCN model is inherently transductive, in which test document nodes (without labels) are included in GCN training. Thus Text GCN could not quickly generate embeddings and make prediction for unseen test documents. Possible solutions to the problem are introducing inductive (Hamilton, Ying, and Leskovec 2017) or fast GCN model (Chen, Ma, and Xiao 2018).

- 作者所提出的 Text GCN 可以达到很好的文本分类结果
- 主要的问题是 GCN 本质上是传导的模型，测试数据是被包括在训练中的，因此 Text GCN 不能很快的生成 embeddings 以及作出预测
- 可能的解决方案是 引入 inductive or fast GCn model

---

## Conclusion nad Future Work

> In this study, we propose a novel text classification method termed Text Graph Convolutional Networks (Text GCN). We build a heterogeneous word document graph for a whole corpus and turn document classification into a node classification problem. Text GCN can capture global word co-occurrence information and utilize limited labeled documents well. A simple two-layer Text GCN demonstrates promising results by outperforming numerous state-of-the- art methods on multiple benchmark datasets.
>
> In addition to **generalizing Text GCN model to inductive settings**, some interesting future directions include improving the classification performance using **attention mechanisms** (Velicˇkovic ́ et al. 2018) and developing **unsupervised text GCN framework** for representation learning on large scale unlabeled text data.

---

## 个人总结：

- 本文的方法只用了词词连边、词文连边，词和词之间都是有连边的，相当于把 document 放进了 word 的 graph 里，也就是用 word embedding 表达 doc embedding，本质上是 Bag-of-words model；多了 document 之间信息的传递，把 word2vec 和 doc2vec 两个步骤合在了一起，使得 word 和 document 的 embedding 相互影响。也正如 contribution 所说，可以不利用预训练的词向量和预先知识，同时计算 word 和 document 的 embedding。

- Text GCN 出现新的 document 时会影响到 word 和 word 之间权重，所以需要重新训练调整边的权重，不能很快的得到新 document 的 embedding 的分类结果。

---

## 闲谈：

- 曾几何时我本科毕业论文题目也是 异构图神经网络判别推特水军 ... 不过我没用到文本信息，只是从社交网络 graph 来判别... 

- 读这篇 paper 的初衷是完成 Text as Data 这门课的 coursework... 有空把剩下几篇也做一下笔记

  - Class-Based n-gram Models of Natural Language 

  - Enriching Word Vectors with Subword Information

  - GloVe-Global Vectors for Word Representation

  - Graph Convolutional Networks for Text Classification

  - Latent Dirichlet Allocation

  - Translating Embeddings for Modeling Multi-relational Data

  - Word representations- A simple and general method for semi-supervised learning

    

