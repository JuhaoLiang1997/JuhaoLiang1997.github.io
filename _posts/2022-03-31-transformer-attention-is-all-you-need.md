---
layout: post
title: 'Transformer: Attention Is All You Need'
date: 2022-03-31 21:42 +0800
category: [papers]
tags: [Transformer]
typora-root-url: "../_images"
math: true
comments: false
toc: true
pin: false
---

# Paper 阅读笔记

[Attention Is All You Need](https://arxiv.org/abs/1706.03762), [【機器學習2021】Transformer](https://www.youtube.com/watch?v=n9TlOhRjYoc), [transformer implementation](https://github.com/hyunwoongko/transformer),  [代码笔记](https://github.com/JuhaoLiang1997/transformer/blob/master/Transformer%E4%B8%AA%E4%BA%BA%E7%AC%94%E8%AE%B0.ipynb), [Transformer 面经总结](https://enze5088.github.io/content/article-4/#)

transformer 主要还是利用了 self-attention 的机制，打破了传统 rnn 基于序列的线性训练方法，增强其并行运算能力，克服长距离依赖问题；但与此同时，局部信息的获取没有 RNN 和 CNN 的强。

---

## 个人笔记

<img src="/2022-03-31-transformer-attention-is-all-you-need/截屏2022-03-31 14.52.59.png" alt="截屏2022-03-31 14.52.59" style="zoom:50%;" />
_Transformer 结构_

宏观上来讲， transformer 由两部分组成 encoder + decoder。以文本翻译为例（英译中），encoder 负责英文原句子的意思理解，decoder 负责用 encoder 所理解的整体句子意思逐字的生成中文句子。

### Encoder 流程

1. 输入表示向量 `inputs (seq_len, d_model)`: `seq_len` 最大 sequence 长度，`d_model` 维特征向量（词向量）

2. 添加 position encoding: `inputs_pos (seq_len, d_model) = inputs (seq_len, d_model) + positional encoding (seq_len, d_model)`

   - 这里 positional encoding 可以用公式计算出来，或者用可训练的（bert）；

   - 直接相加和 concat 到后面没什么区别，不影响原本的向量表达

     <img src="/2022-03-31-transformer-attention-is-all-you-need/截屏2022-03-13 01.42.28.png" alt="截屏2022-03-13 01.42.28" style="zoom:50%;" />
   _position encoding公式_

3. Self Attention的部分，$W_{q}, W_{k}, W_{v}$ 是三个可训练的权重矩阵，用于将 `inputs_pos` 转换成 Query, Key, Value，`Query Matrix`, `Key Matrix`, `Value Matrix` 每一行代表一个 token 的 Query, Key, Value

   ```
   QueryMatrix(seq_len, d_k) = inputs_pos(seq_len, d_model) * Wq(d_model, d_k)
   KeyMatrix(seq_len, d_k) = inputs_pos(seq_len, d_model) * Wk(d_model, d_k)
   ValueMatrix(seq_len, d_v) = inputs_pos(seq_len, d_model) * Wv(d_model, d_v)
   ```

   <img src="/2022-03-31-transformer-attention-is-all-you-need/截屏2022-03-31 16.11.52.png" alt="截屏2022-03-31 16.11.52" style="zoom:50%;" />
   _Query, Key, Value Matrix计算_

4. 求得 Query 和 Key 的点积，表示当前 token 对 所有 tokens 的 attention。如果特征维度 `d_model` 很大的话，这里点积结果会很大，需要做 scaling 保持其 variance 为 1，使得 softmax 结果差距不会太大，从而解决**梯度消失**的问题。[参考](https://www.zhihu.com/question/339723385/answer/782509914)

   ```
   AttentionMatrix(seq_len, seq_len) = QueryMatrix(seq_len, d_k) * KeyMatrix.T(d_k, seq_len) / (d_k)^0.5
   ```

5. 将 attention 按行做 softmax，将 attention 压缩到 0 到 1 之间

   ```
   AttentionMatrix(seq_len, seq_len) = softmax(AttentionMatrix, axis=0)
   ```

6. 利用计算得到的 attention 点乘 value，再相加，得到 outputs

   ```
   layer_outputs(seq_len, d_v) = AttentionMatrix(seq_len, seq_len) * ValueMatrix(seq_len, d_v)
   ```

7. 步骤 3-6 是计算单个 Self-Attention 输出的过程。实际上 Multi-Head Attention 是将步骤 3 中 `QueryMatrix`, `KeyMatrix`, `ValueMatrix` 按照**特征维度** (d_model) 切割成**多个子集**，每个子集的特征维度是 `(d // n_head)`。将切割后的 QKV Matrix 进行 步骤456 Self-Attention 计算得到各自的 `layer_outputs`，再 `concat` 在一起，构成输出 `(seq_len, d_v)`，最后加个 **linear transformer** 将 `(seq_len, d_v)` 转为 `(seq_len, d_model)` 就好。

   ```
   subQueryMatrixs, subKeyMatrixs, subValueMatrixs = split(QueryMatrix), split(KeyMatrix), split(ValueMatrix)
   subWeightedValues = softmax(scale(subQueryMatrixs * subKeyMatrixs)) * subValueMatrixs
   concated_outputs(seq_len, d_v) = concat([subWeightedValue1, subWeightedValue1, ...])  
   transformed_outputs(seq_len, d_model) = concated_outputs(seq_len, d_v) * linear_transformer(d_v, d_model) 
   ```

8. 步骤 7 得到了 Multi-Head Attention 的结果后，需要做一个残差连接，与 `input_pos` 进行 `Residual & Norm & Dropout` 的操作

   ```
   add_outputs(seq_len, d_model) = transformed_outputs(seq_len, d_model) + inputs_pos(seq_len, d_model)
   multihead_outputs(seq_len, d_model) = Dropout(LayerNorm(add_outputs))
   ```

   - `Batch Norm` 是对每一个维度进行 normalization，`Layer Norm` 是对单个 token 的特征向量进行 normalization，`Layer Norm` 会消除同一特征的差异性，一般 `Batch Norm` 用于图像，`Layer Norm` 用于NLP，而 NLP 的 embedding 一般计算 cos similarity 作为相似度，所以单一特征差异性其实不关键。

     <img src="/2022-03-31-transformer-attention-is-all-you-need/截屏2022-03-31 16.41.05.png" alt="截屏2022-03-31 16.41.05" style="zoom:50%;" />
   _Batch Normalization_

9. 这里到了 `Feed-Forward` 的部分了，利用俩全连接层将 `multihead_outputs` 提炼到高维空间 (论文中维度提高了四倍)，利用 `ReLU` 进行激活，原理和 SVM 差不多，将低维特征映射到高维更容易区别特征差异。上一步加 `Layer Norm` 也是为了在这一步激活函数更好的发挥作用

   ```
   latent_represent(seq_len, latent_d) = Dropout(ReLU(multihead_outputs * w1 + b1))
   ff_output(seq_len, d_model) = latent_represent * w2 + b2
   ```

   <img src="/2022-03-31-transformer-attention-is-all-you-need/截屏2022-03-31 16.47.11.png" alt="截屏2022-03-31 16.47.11" style="zoom:50%;" />
   _Feed-Forward network_

10. 同步骤 8，给 Feed-Forward Network 再加一个 `Residual & Norm & Dropout`，作用一是解决梯度消失的问题，二是解决权重矩阵的退化问题

11. 步骤 3-10 是一个 encoder 部分需要重复的 network，论文中重复了 $N=6$ 次。至此就完成了 encoder 的部分了，也就是 transformer 结构图中左半部分，输出结果是和输入特征维度一致的 `(seq_len, d_model)`。

### Decoder 流程

1. 输入 output 的表示向量 `outputs (seq_len, d_model)`: seq_len 个 tokens，d_model 维特征向量。

   - `seq_len` 是由最大 sequence 长度决定的，短的 sequence 由 `<mask>` 做 padding
   - 训练阶段数据就是 target sequence 偏移一位: `[</s>, token1, token2, ...]` ，这个偏移一位的目的是为了接下来的mask比较好做。各个 predict element 的生成是并行的
   - 预测阶段（生成阶段）是需要多次调用 decoder 的部分的，在上一个 decoder 生成输出之后，如果不为结束符，则加入到 predicted_data 里 `[\</s>, token1, token2, ...]`，继续扔进 decoder 里预测下一个 token。各个 predict element 的生成是串行的。

2. 加入 position encoding 信息，同 encoder 步骤 2。

3. 这一步的 Multi-Head Attention 用的是 masked 的，就是 encoder 步骤 4 中将还没有 predict 的部分的 attention 给 mask 掉，这样就可以在那一步只获取之前的 weighted value，而丝毫不受未预测部分的 value 所影响；而至于在后面那个 Multi-Head Attention 为什么不用 mask？是因为那一步中的 key 和 value 是 encoder 的结果，包含的是整个序列的信息，与未预测的真实信息无关，所以不用 mask。其余的同 encoder 的 步骤 3-8。

4. 接下来的一个 Multi-Head Attention 使用了 encoder 的结果，Query 是用上一个 layer 的结果求得的，Key 和 Value 是用 encoder 的结果求得的，`Residual + LayerNorm + Dropout`，得到 `multihead_outputs(seq_len, d_model)`

   ```
   QueryMatrix(seq_len, d_k) = first_MultiLayer_output(seq_len, d_model) * Wq(d_model, d_k)
   KeyMatrix(seq_len, d_k) = encoder_output(seq_len, d_model) * Wk(d_model, d_k)
   ValueMatrix(seq_len, d_v) = encoder_output(seq_len, d_model) * Wv(d_model, d_v)
   
   AttentionMatrix(seq_len, seq_len) = QueryMatrix(seq_len, d_k) * KeyMatrix.T(d_k, seq_len) / (d_k)^0.5
   
   layer_outputs(seq_len, d_v) = AttentionMatrix(seq_len, seq_len) * ValueMatrix(seq_len, d_v)
   ```

5. FF layer 也是同 encoder 步骤9-10。

6. 步骤 3-5 也是要重复 $N$ 次。

7. 最后再 Linear tranformation 降为 token corpus 大小，softmax 求各个 token 的概率。

---

### 小细节

#### Regularization 

1. **Residual Dropout**: 每一层前面都加了 dropout function，包括 encoder 和 decoder 求 `inputs_pos` 

   <img src="/2022-03-31-transformer-attention-is-all-you-need/截屏2022-03-31 17.29.27.png" alt="截屏2022-03-31 17.29.27" style="zoom:50%;" />
_Residual Dropout_

2. **Label Smoothing**: cross-entropy 中使标签平滑

   <img src="/2022-03-31-transformer-attention-is-all-you-need/截屏2022-03-31 17.28.52.png" alt="截屏2022-03-31 17.28.52" style="zoom:50%;" />
   _Label Smoothing_

#### 权重共享

1. Encoder 和 Decoder 间的 Embedding 层权重共享；

   对于共有的一些 tokens 可以更好的表达；但是词表会很大

2. Decoder 中 Embedding 层和 FC 层权重共享

   FC 层通过得到的向量，根据 embedding 层的权重反向求得每个 token 的概率

   ```python
   fc = nn.Linear(d, v, bias=False) # Decoder FC层定义，无 bias
   weight = Parameter(torch.Tensor(out_features, in_features))   # nn.Linear 的权重部分定义
   ```

#### Mask用到的地方

1. 因为所有的文本长度不一样，所以会有一些文本是经过padding处理的，padding 部分用 mask 遮盖

2. decoder 计算 attention 的时候，未预测部分由 mask 遮盖

#### Multi-Head Attention

1. 多头注意力机制是按照特征维度划分的，可以保证**计算量不增加**的情况下，提升特征捕捉能力

   > Due to the reduced dimension of each head, the total computational cost is **similar** to that of single-head attention with full dimensionality.

2. 可以类比CNN中同时使用**多个滤波器**的作用，直观上讲，多头的注意力**有助于网络捕捉到更丰富的特征/信息。**

#### decoder 训练和测试的输入输出差别

1. 训练阶段中，decoder 的**输入**是真实输出序列**向右偏移一位**: `[<\s>, token1, token2, ...]`，在 decoder 的第一个 Multi-Head Attention 中生成 query-key attention 之后对未预测部分做了 mask，如以下 `AttentionMatrix` 表格，目的是为了预测下一个 token 时只用到**当前及之前的 token 信息**，例如预测 token1 时只用到了 `<\s>` 一个 token。那么为什么 decoder 中第二个 Multi-Head Attention 不用 mask 呢？是因为第二个 mask 的 key, value 都是 encoder 生成的，带着整个 sequence 的信息，用于生成下一个 token 是没问题的；在剩下的操作中 `(linear transformer & normalization)` 都是**按行处理**，因此不会发生数据泄露。 

   | query\key            | <\s>.key      | token1.key    | token2.key    | token3.key    | mask.key (padding) |
   | -------------------- | ------------- | ------------- | ------------- | ------------- | ------------------ |
   | <\s>.query           | **attention** | mask          | mask          | mask          | mask               |
   | token1.query         | **attention** | **attention** | mask          | mask          | mask               |
   | token2.query         | **attention** | **attention** | **attention** | mask          | mask               |
   | token3.query         | **attention** | **attention** | **attention** | **attention** | mask               |
   | mask.query (padding) | mask          | mask          | mask          | mask          | mask               |

2. 训练阶段中，decoder 的**输出**应该是 `(seq_len, k)`，每一行都是**独立预测**出来的，没有依赖前一个 token 的预测，例如预测 token5 是由真实 token1 到真实 token4 所生成的。loss function 就是 `predictMatrix (seq_len, vocab_size)` 和 `truthMatrix (seq_len, vocab_size)` 的 `cross-entropy`，其中 `truthMatrix` 使用的不是 onehot，是 *label smoothing*。

3. 测试阶段中，decoder 的**输入**是 sequential 的，即每个 token 按先后顺序预测，根据已预测的 token 去预测新的 token，例如预测第一个 token 的输入应该是: `[<\s>, mask, ...]` 长度固定，**输出**应该就是: `[token1, ....]`。再预测下一个的时候直到预测到结束符就可以停下来了，这时就得到了完整的预测序列了。 

   

   


---

## 论文笔记

### Introduction: 

> In this work we propose the Transformer, a model architecture **eschewing recurrence** and instead relying entirely on an attention mechanism to draw **global dependencies** between input and output. The Transformer allows for significantly more **parallelization** and can reach a new state of the art in translation quality after being trained for as little as twelve hours on eight P100 GPUs.

---

### Model Architecture:


#### Encoder and Decoder Stacks

**Encoder:** Encoder 部分是由 $N=6$ 个相同的 layer 组成，每一个 layer 有两个部分，第一个部分是 multi-head self-attention，第二个部分是一个简单的全连接层。两个部分都用了残差连接 + layer normalization，即 $Output_{sub-layer} = LayerNorm(x+Sublayer(x))$ ，每一个输出的维度都相同 $d_{model} = 512$ 。

**Decoder:** Decoder 部分也是由 $N=6$ 个相同的 layer 组成，与 encoder 不同的是每个 layer 中有一个额外的 Masked Multi-Head Attention Sub-layer，这里的 Mask 是用来遮盖待预测的序列的，只允许使用之前的序列进行预测；而第二个 Multi-Head Attention 是和 encoder 输出部分 concat 之后一起做 self-attention。

#### Attention

Attention 的功能是把一组 input 映射成对应的 query, key, value，利用 query 和各个输入的 key 求出对各个 input 的权重，再根据权重求出 weighted value 的集成，得到最后 query 对应 input 的 output：

<img src="/2022-03-31-transformer-attention-is-all-you-need/截屏2022-03-31 15.14.52.png" alt="截屏2022-03-31 15.14.52" style="zoom:50%;" />
_left: Scaled Dot-Product Attention, right: Multi-Head Attention_

##### Scaled Dot-Product Attention

上图左边是 Scaled Dot-Product Attention，输入是 k 维的 query, key 和 v 维的 value，计算 query 和所有 key 的点积再除以 $\sqrt{d_{k}} $ ，在用 softmax 取得各个 value 的权重。式子如下：

$$\operatorname{Attention}(Q, K, V)=\operatorname{softmax}\left(\frac{Q K^{T}}{\sqrt{d_{k}}}\right) V$$ 

有两种常用的 attention: **dot-product attention** and **additive attention**，这里就是用的 dot-product attention + scaling factor。为了不让点积越来越大，通过除以维度的开方来降低 softmax 的差距。

##### Multi-Head Attention

通过使用多个 self-attentions 机制来获取子空间更多方面的表达，用 concat 把多个 self-attentions 的输出集合起来再用一个 linear transformer 转换到原来的维度，如下图。在论文中使用了 8 层 attention layer。

<img src="/2022-03-31-transformer-attention-is-all-you-need/截屏2022-03-31 15.46.26.png" alt="截屏2022-03-31 15.46.26" style="zoom:50%;" />
_Multi-Head Attention_

##### Applications of Attention in Model

- 在 encoder-decoder attention 层，query 是来自 decoder 上一层的，value 和 key 是来自 encoder 的输出的，这使得解码器中的每个位置都能关注到输入序列中的所有位置。
- encoder 的 self-attention 中，query, key, value 都是来自于上一层的输出，编码器中的每个位置都可以关注到编码器前一层的所有位置。
- 同样，decoder 中的自我关注层允许解码器中的每个位置关注已经预测的位置，防止观察到待预测值

#### Position-wise Feed-Forward Networks

fully connected feed-forward network 包含了两个 linear transformer 和 ReLU 激活函数在中间

<img src="/2022-03-31-transformer-attention-is-all-you-need/截屏2022-03-31 17.49.52.png" alt="截屏2022-03-31 17.49.52" style="zoom:50%;" />
_FFN_

#### Embeddings and Softmax

与其他序列转换模型类似，使用学习到的 emebddings 来转换输入的 tokens 和输出tokens 转换为维度为 $d_{model}$ 的向量。还使用 linear transformation 和 softmax 函数将 decoder 的输出转换为预测的下一个 token 的概率。在模型中，在两个 embedding 层和 pre-softmax linear transformation 之间共享相同的权重矩阵。在 embedding 层，会将其乘以权重 $\sqrt {d_{model}}$ 。

#### Positional Encoding

为了保持序列顺序信息，必须添加 position encoding 信息到 input embeddings 中，为了可以相加，position encoding 和 embedding 维度相同。在论文中，使用的是公式：

<img src="/2022-03-31-transformer-attention-is-all-you-need/截屏2022-03-31 18.00.25.png" alt="截屏2022-03-31 18.00.25" style="zoom:50%;" />
_Positional Encoding_

---

### Why Self-Attention

1. 每一层的计算复杂度
2. 并行能力
3. 解决序列长距离依赖问题

---

### Conclusion

> In this work, we presented the Transformer, **the first sequence transduction model based entirely on attention**, replacing the recurrent layers most commonly used in encoder-decoder architectures with multi-headed self-attention.
>
> For translation tasks, the Transformer can be trained **significantly faster** than architectures based on recurrent or convolutional layers. On both WMT 2014 English-to-German and WMT 2014 English-to-French translation tasks, we achieve a new state of the art. In the former task our best model outperforms even all previously reported ensembles.
>
> We are excited about the future of attention-based models and plan to apply them to other tasks. We plan to extend the Transformer to problems involving input and output modalities other than text and to investigate local, restricted attention mechanisms to efficiently handle large inputs and outputs such as images, audio and video. Making generation less sequential is another research goals of ours.



