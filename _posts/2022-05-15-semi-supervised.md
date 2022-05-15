---
layout: post
title: Semi-supervised
date: 2022-05-15 20:32 +0800
category: [Technical]
tags: [Semi-supervised Learning]
typora-root-url: "../_images"
math: true
comments: false
toc: true
pin: false
---

# Semi-supervised Learning

简单了解一下 semi-supervised learning。

**参考资料**：[ML Lecture 12: Semi-supervised](https://youtu.be/fX_guE7JNnY), [A Survey on Deep Semi-supervised Learning](https://arxiv.org/pdf/2103.00550.pdf)

---

## Overview

半监督学习（semi-supervised learning）指的是大部分训练数据是未标注的，少部分已标注。其中分为 transductive learning 和 inductive learning，前者表示未标注数据是测试数据，且参与训练过程；后者是指未标注数据不是测试数据，测试数据不参与模型训练。

为什么需要半监督学习呢？因为标注数据工作是很昂贵的，半监督学习可以由部分已知数据推断未标注数据。半监督学习通常会伴随着一些假设 assumptions，而这些 assumptions 的质量决定了半监督学习的有效性。

---

## Semi-supervised Learning for Generative Model

在传统 supervised learning 中添加 unlabeled data 影响最终 loss 的计算，需要 solved iteratively 因为不是 convex problem。

$$P_{\theta}\left(x^{u}\right)=P_{\theta}\left(x^{u} \mid C_{1}\right) P\left(C_{1}\right)+P_{\theta}\left(x_{0}^{u} \mid C_{2}\right) P\left(C_{2}\right)$$

### Steps

- Initialization: $$\theta=\left\{P\left(C_{1}\right), P\left(C_{2}\right), \mu^{1}, \mu^{2}, \Sigma\right\}$$

- Step1: compute the posterior probability of unlabeled data: $$P_{\theta}\left(C_{1} \mid x^{u}\right)$$ 计算 unlabeled data 的属于某一类的概率

- Step2: update model. 更新时，P(C1) 会加上所有 unlabeled data 属于 C1 的和，再更新 $\mu$，重复此步骤。 

  N: total number of examples, 

  N1: number of examples belonging to C1

   $$P\left(C_{1}\right)=\frac{N_{1}+\sum_{x^{u}} P\left(C_{1} \mid x^{u}\right)}{N}$$

  $$\mu^{1}=\frac{1}{N_{1}} \sum_{x^{r} \in C_{1}} x^{r}+\frac{1}{\sum_{x^{u}} P\left(C_{1} \mid x^{u}\right)} \sum_{x^{u}} P\left(C_{1} \mid x^{u}\right) x^{u}$$

- Back to step 1

- The algorithm converges eventually, but hte initialization influences the results.

---

## Low-density Separation Assumption

非黑即白。

### Self-training

给定 labelled data 和 unlabeled data，用 labelled data 训练得到模型，再用训练得到的模型预测 unlabeled data 得到 pseudo-label，再用部分 unlabeled data 参与训练模型，利用更新的模型继续预测剩下的 unlabeled data，循环运作。

和 generative model 的区别是 self-training 使用的 hard label，而 generative model 用的是 soft label (probability)。soft label 对 neural network 其实是没有用的，hard label 对 regression model 是没有用的。

### Entropy-based Regularization

相当于在 loss function 中添加表示 unlabeled 不确定程度的 regularization，使得模型在未标注数据的标签预测中更加自信。用 entropy 计算 预测分布是否集中：

$$E\left(y^{u}\right)=-\sum_{m=1}^{5} y_{m}^{u} \ln \left(y_{m}^{u}\right)$$ as small as possible

<img src="/2022-05-15-semi-supervised/Screenshot 2022-05-15 at 14.19.53.png" alt="Screenshot 2022-05-15 at 14.19.53" style="zoom:50%;" />
_Entropy-based regularization loss function_

### Semi-supervised SVM

对未标注数据所有标签的可能都做一次 SVM，判断哪一种情况下 margin 最大且 error 最小，从而选择作为最终 model。

---

## Smoothness Assumption

近朱者赤，近墨者黑。

假设 x1 和 x2 在同一片集中区域，可以认为 y1 和 y2 是相同的，下图中 x2 更趋向于与 x1 有相同的标签。

<img src="/2022-05-15-semi-supervised/Screenshot 2022-05-15 at 14.27.22.png" alt="Screenshot 2022-05-15 at 14.27.22" style="zoom:50%;" />
_Smoothness Assumption_

### Cluster and then Label

大概思路就是用有标注数据+无标注数据去做 clustering，通过聚类结果再给无标注数据标签。需要有强聚类方法才会 work。

### Graph-based Approach

利用所有数据建立 graph，两两之间计算 similarity，再根据数据特征 (heuristic) 建连边。例如使用 K Nearest Neighbour 或 e- Neighbourhood 建连边。又或者使用加权连边，Gaussian Radial Basis Function。The labelled data influence their neighbors, propagate through the graph. 标签信息会随着连边传播。

定义 graph 的 label smoothness，图的标签是否顺滑 (渐变)：计算每条连边的两端节点标签相同的个数，越多相同表示越顺滑，反之则 low smoothness。

Graph Laplacian: L = D - W, D是每个节点的连边权重和，W是节点之间的连边权重

<img src="/2022-05-15-semi-supervised/Screenshot 2022-05-15 at 14.48.55.png" alt="Screenshot 2022-05-15 at 14.48.55" style="zoom:50%;" />
_Label Smoothiness_

最后在 loss function 加上 label smoothiness 值，使得未标注数据的预测结果是 smooth 的。

---

## Better Representation

去芜存菁，化繁为简。

Find the latent factors behind the observation. The latent factors (usually simpler) are better representations。
