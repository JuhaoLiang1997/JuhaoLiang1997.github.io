---
layout: post
title: Transfer Learning
date: 2022-05-14 20:53 +0800
category: [Technical]
tags: [Transfer Learning]
typora-root-url: "../_images"
math: true
comments: false
toc: true
pin: false
---

# Transfer Learning

简单了解一下迁移学习（Transfer Learning）。

**参考文献**：[ML Lecture 19: Transfer Learning](https://youtu.be/qD6iD4TFsdQ), [Transfer Learning](https://github.com/jindongwang/transferlearning)

---

## Overview

根据 **target data**、**source data**（是否带标签）区分为四种类型：

<img src="/2022-05-14-transfer-learning/Screenshot 2022-05-14 at 15.03.05.png" alt="Screenshot 2022-05-14 at 15.03.05" style="zoom:40%;" />

---

## Model Fine-tuning

Target data 和 Source data 都有标签时，可以用 Model fine-tuning 做迁移学习。基本思路就是用 source data 先训练模型，再用 target data 微调模型。

### One-shot learning

当 target data 数量少的时候，可以称为 One-shot learning: only a few examples in target domain。

### Conservative Training

训练时加入 constraint 防止 target data 过少导致的 overfitting。

### Layer Transfer

Layer transfer：直接复制 source data 训练模型的部分 layer。只用 target data 训练其中几层，可以防止 overfitting。

如何选择复制哪些层？

- 语音上通常复制最后几层，表示更深层次的内容，不同个体发声方式不一样。
- 图像中通常复制前面几层，表示更简单的图像构成，根据训练任务不同提取内容。

---

## Multitask Learning

Target data 和 Source data 都有标签时，也可以用 Multitask learning 做迁移学习。基本思路就是一个模型（部分 component）同时训练多个任务。可以处理例如多语言辨识的任务。

<img src="/2022-05-14-transfer-learning/Screenshot 2022-05-14 at 14.14.43.png" alt="Screenshot 2022-05-14 at 14.14.43" style="zoom:40%;" />
_Multitask Learning_

<img src="/2022-05-14-transfer-learning/Screenshot 2022-05-14 at 14.21.25.png" alt="Screenshot 2022-05-14 at 14.21.25" style="zoom:50%;" />
_Progressive Neural Network_

---

## Domain-adversarial training

Target data 无标签，Source data 有标签时，可以用 Domain-adversarial training 做迁移学习。目标在于消除 Target data 和 Source data 的差异。和 GAN 相似，由三个模型组成，feature extractor model 用于提取不同数据集数据的 latent feature，label predictor 根据 latent feature 区分预测标签，Domain classifier 根据 latent feature 判断数据来自于哪个数据集。

<img src="/2022-05-14-transfer-learning/Screenshot 2022-05-14 at 14.34.24.png" alt="Screenshot 2022-05-14 at 14.34.24" style="zoom:40%;" />
_Domain-adversarial training is a big network, but different parts have different goals_

---

## Zero-shot Learning

Target data 无标签，Source data 有标签时，也可以用 Zero-shot learning 做迁移学习。基本思路是提取两个任务中更底层的共有属性。例如在图像分类中，提取出各个动物的属性，再用模型预测是否拥有各个属性（毛发、四肢、尾巴），根据预测结果（attribute embedding）再推测属于哪个动物。attribute 可以用 word embedding 表示。

<img src="/2022-05-14-transfer-learning/Screenshot 2022-05-14 at 14.42.46.png" alt="Screenshot 2022-05-14 at 14.42.46" style="zoom:50%;" />
_Data Attributes Example_

### Convex Combination of Semantic Embedding

根据 source data 训练模型得到的预测结果，各个结果的概率再结合 word embedding 加权求和得到新的动物的表示（例如：0.5 老虎 + 0.5 狮子=狮虎兽）。

---

## Self-taught Learning & Self-taught Clustering

Target data 有标签，Source data 无标签时，可以用 Self-taught Learning 做迁移学习。有点像 semi-supervised learning，但两个数据集是不完全一样的。基本思路就是用无标签数据提取 better representation。

Target data 无标签，Source data 也无标签时，可以用 Self-taught Clustering 做迁移学习。思路和 self-taught learning 差不多，也是预先提取 representation。
