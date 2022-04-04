---
layout: post
title: BERT Model
date: 2022-04-02 03:29 +0800
category: [Technical]
tags: [tools, BERT]
typora-root-url: "../_images"
math: true
comments: false
toc: true
pin: false
---

# BERT Model

基于 [Huggingface Transformer实战教程](https://www.heywhale.com/home/competition/61dd2a3dc238c000186ac330) 的 BERT 模型笔记，写的有点杂，回头再整理补充一下。

## 课程相关

### 课程目标

《Huggingface Transformers实战教程 》是专门针对HuggingFace开源的transformers库开发的实战教程，适合从事**自然语言处理研究的学生、研究人员以及工程师**等相关人员的学习与参考，目标是阐释transformers模型以及Bert等预训练模型背后的原理，通俗生动地解释transformers库的如何使用与定制化开发，帮助受众使用当前NLP顶级模型解决实际问题并取得优秀稳定的实践效果。

---

## BERT 

### BERT 框架

BERT整体框架包含 **pre-train** 和 **fine-tune** 两个阶段。pre-train 阶段模型是在无标注的标签数据上进行训练，fine-tune 阶段，BERT模型首先是被 pre-train 模型参数初始化，然后所有的参数会用下游的有标注的数据进行训练。

BERT是用了 Transformer 的 **encoder 侧的网络**，encoder中的 Self-attention 机制在编码一个token的时候同时利用了其上下文的 token，其中‘同时利用上下文’即为双向的体现，而并非想 Bi-LSTM 那样把句子倒序输入一遍。

在它之前是 *GPT*，GPT使用的是 Transformer 的 **decoder 侧的网络**，GPT是一个单向语言模型的预训练过程，更适用于文本生成，通过前文去预测当前的字。

### Embedding

Embedding由三种 Embedding 求和而成：

1. Token Embeddings 是词向量，第一个单词是 CLS 标志，可以用于之后的分类任务
2. Segment Embeddings 用来区别两种句子，因为预训练不光做 LM 还要做以两个句子为输入的分类任务
3. Position Embeddings 和之前文章中的 Transformer 不一样，不是三角函数而是学习出来的

#### 特殊 tokens

1. `[CLS]`：在做分类任务时其最后一层的repr. 会被视为整个输入序列的repr. 一般会被放在输入序列的最前面

2. `[SEP]`：有两个句子的文本会被串接成一个输入序列，并在两句之间插入这个 token 以做区隔

3. `[UNK]`：没出现在 BERT 字典里头的字会被这个 token 取代 

4. `[PAD]`：zero padding 遮罩，将长度不一的输入序列补齐方便做 batch 运算 

5. `[MASK`]：未知遮罩，仅在预训练阶段会用到，一般在 fine-tuning 或是 feature extraction 时不会用到，这边只是为了展示预训练阶段的遮蔽字任务才使用的。

   ```python
   tokenizer.all_special_ids # 特殊 token 的 id
   tokenizer.all_special_tokens # 特殊 token
   ```

   

### Transformer Encoder

在 Transformer 中，模型的输入会被转换成 512 维的向量，然后分为 8 个head，每个 head 的维度是 64 维，但是 BERT 的维度是 768 维度，然后分成 12 个 head，每个 head 的维度是 64 维，这是一个微小的差别。Transformer 中 position Embedding 是用的三角函数，BERT 中也有一个 Postion Embedding 是随机初始化，然后从数据中学出来的。

BERT 模型分为 24 层和 12 层两种，其差别就是使用 transformer encoder 的层数的差异，BERT-base 使用的是 12 层的 Transformer Encoder 结构，BERT-Large 使用的是 24 层的 Transformer Encoder 结构。

### Tokenizer

- `tokenizer.encode(text)` 只返回 `input_ids`

- `tokenizer.encode_plus(text)` 返回所有的 embedding 信息，包括

  - `input_ids`： token embeddings
  - `token_type_ids`：segment embeddings
  - `attention_mask`：指定 token 做 self-attention


### Model 输出

- `pooler_output` 对应的是 `[CLS]` 的输出 `(batch_size, hidden_size)`
- `last_hidden_state` 对应的是序列中所有 token 最后一层的 hidden 输出 `(batch_size, sequence_length, hidden_size)`
- `hidden_states` (optional) 如果输出，需要指定`config.output_hidden_states=True` ，它是一个元组，它的第一个元素是embedding，其余元素是各层的输出，每个元素的形状是(batch_size, sequence_length, hidden_size)
- `attentions` (optional) 如果输出，需要指定`config.output_attentions=True`，它是一个元组 ，它的元素是每一层的注意力权重，用于计算self-attention heads的加权平均值

## 实战相关

### 随机种子

```python
def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

np.random.seed(CONFIG.SEED)
seed_torch(seed=CONFIG.SEED)
```

### 数据统计

```python
train.info() # 查看缺失值
train[train['content'].isna()] # 查看缺失值所在数据
train['label'].value_counts() # 标签分布
train['content']=train['content'].fillna('空值') # 数据预处理
train['text'].map(len).describe() # 文本长度统计
train['text'].nunique() == train.shape[0] # 判断 unique text
```

### 自定义数据集

```python
class CustomDataset(Dataset):
  def __init__(self, texts, labels, tokenizer, ...):
    self.texts = texts, ... 
  def __len__(self):
    return len(self.texts) # 数据个数
 	def __getitem__(self, item):
    # item 是数据索引，处理第 item 条数据
    encoding = self.tokenizer.encode_plus(...) # 可以在 dataset 里做 tokenization
    return {'texts': text, 'encoding': ...}
```

```python
train, test = train_test_split(dataset, ....) # 切割训练集测试集
dataloader = DataLoader(dataset, batch_size=...) # dataloader 分 batch
```

### 训练

#### Optimizer

- [AdamW](https://www.fast.ai/2018/07/02/adam-weight-decay/) (Adam with decoupled weight decay) : 效果更好 [huggingface](https://huggingface.co/docs/transformers/main_classes/optimizer_schedules)
- Adam

##### Scheduler

`get_linear_schedule_with_warmup` ： adaptive learning rate

##### clip_grad_norm_

`nn.utils.clip_grad_norm(parameters, max_norm, norm_type=2)` 设置一个梯度剪切的阈值，如果在更新梯度的时候，梯度超过这个阈值，则会将其限制在这个范围之内，防止梯度爆炸。

- `parameters (Iterable[Tensor] or Tensor)`: 参数
- `max_norm (float or int)` 梯度的最大范数
- `norm_type(float or int)=2.0` 规定范数的类型

##### Loss

BCEWithLogitsLoss = Sigmoid+BCELoss，当网络最后一层使用nn.Sigmoid时，就用BCELoss，当网络最后一层不使用nn.Sigmoid时，就用BCEWithLogitsLoss。
(BCELoss) BCEWithLogitsLoss 用于**单标签二分类**或者**多标签二分类**，输出和目标的维度是(batch_size,n_class)，batch_size 是样本数量，n_class 是类别数量，对于每一个 batch 的 n_class 个值，对每个值求 sigmoid 到 0-1 之间，所以每个 batch 的 n_class 个值之间是没有关系的，相互独立的，所以之和不一定为1。每个C值代表属于一类标签的概率。如果是单标签二分类，那输出和目标的维度是 (batch_size,1) 即可。

##### 保存模型

```python
if val_acc > best_acc:
  torch.save(model.state_dict(), 'best_model_state.bin')
  best_acc = val_acc
# model.load_state_dict(torch.load('best_model_state.bin'))
```

---

### 任务

- **02-文本分类实战：基于Bert的企业隐患排查分类模型**

  分类任务：使用 bert 模型的 `pooled_output` 接入 linear layer 进行分类，`pooled_output` 是整体句子表示

  ```python
  def forward(self, input_ids, attention_mask):
    _, pooled_output = self.bert(
      input_ids=input_ids,
      attention_mask=attention_mask,
      return_dict = False
    )
    output = self.drop(pooled_output) # dropout
    return self.out(output) # linear layer (hidden_size, n_classes)
  ```

- **03-文本多标签分类实战：基于Bert对推特文本进行多标签分类**

  - 多分类就是 bert 模型接一个 linear layer 进行分类
  - 多分类使用 `BCEWithLogitsLoss` ，`BCEWithlogitsloss=sigmoid+BCELoss` pytorch官方为了数值计算稳定性，将 sigmoid 层和 BCELoss 合并到了一起，当网络最后一层使用nn.Sigmoid时，就用BCELoss，当网络最后一层不使用nn.Sigmoid时，就用BCEWithLogitsLoss。
  - 预测阶段就用 sigmoid + threshold (0.5) 得出预测的多标签
  - 验证集中使用 `f1_score` 搜索范围内的 thresholds，首先用大范围找 `range(1, 10)/10`，找到粗略的范围后再精细的找 `range(10)/100`，找到最佳的 threshold

- **04-句子相似性识别实战：基于 Bert 对句子对进行相似性二分类**

  - 二分类任务，`bert_pooler_output + dropout + Linear (hidden_size, 1)`
  - [自动混合精度训练](https://zhuanlan.zhihu.com/p/165152789)，大概意思就是自动调整精度 `torch.float64/32` 在有的情景下能加速运算和降低内存占用。
  - 单标签二分类用 BCELoss / BCEWithLogitsLoss
  - 可根据数据集调整 threshold

- **05-命名实体识别实战：基于预训练模型进行商品实体识别微调**

  - seq2seq 问题，句子生成 label sequence，每一个 token 分别做分类
  - BERT做NER 一个棘手部分是 BERT 依赖于 **wordpiece tokenization**，而不是 word tokenization。比如：Washington的标签为 "b-gpe",分词之后得到， "Wash", "##ing", "##ton","b-gpe", "b-gpe", "b-gpe"
  - 给各个 subword 添加同样的 label
  - 算准确率的时候还得把 mask 的移除 `torch.masked_select`

- **06-多项选择任务实战：基于Bert实现SWAG常识问题的多项选择**

  - 多项选择题，给出情景，给一句话的开头，多个续接选项中选一个
  - 数据预处理就是把上下文、开头和各个选项拼在一起，训练就是各个选项拼接在一起，做分类问题。
  - transformers 库里有 `AutoModelForMultipleChoice` 的类
  - 这个实战讲解了使用 huggingface_hub 的相关操作

- **07-文本生成实战：基于预训练模型实现文本生成**

  - 模型: `AutoModelForCausalLM`
  - seq2seq 问题，取各个 token 对应的 `last_hidden_state` 做词表 softmax

- **08-文本摘要实战：基于预训练模型实现文本摘要任务**

  - gpt-2: `pipe = pipeline("text-generation", model="gpt2-xl")`
  - t5: `pipe = pipeline("summarization", model="t5-large")`
  - bart: `pipe = pipeline("summarization", model="facebook/bart-large-cnn")`
  - pegasus: `pipe = pipeline("summarization", model="google/pegasus-cnn_dailymail")`

- **09-文本翻译实战：基于Bert实现端到端的机器翻译**

  - 翻译任务 WMT dataset
  - 模型：`AutoModelForSeq2SeqLM`

- **10-问答实战：基于预训练模型实现QA**

  - 基于 context 的问答
  - 模型：`AutoModelForQuestionAnswering`


---

## Huggingface 相关

### [Pipeline](https://huggingface.co/docs/transformers/pipeline_tutorial)

> Transformers 库中最基本的对象是`pipeline()`函数。它将模型与其必要的预处理和后处理步骤连接起来，使我们能够**直接输入任何文本并获得答案**：

**流程**：`Tokenizer` -> `Model` -> `Post-Processing`

**常用 pipeline:**


- `"feature-extraction"`: will return a FeatureExtractionPipeline.
- `"text-classification"`: will return a TextClassificationPipeline.
- `"sentiment-analysis"`: (alias of "text-classification") will return a TextClassificationPipeline.
- `"token-classification"`: will return a TokenClassificationPipeline.
- `"ner"` (alias of "token-classification"): will return a TokenClassificationPipeline.
- `"question-answering"`: will return a QuestionAnsweringPipeline.
- `"fill-mask"`: will return a FillMaskPipeline.
- `"summarization"`: will return a SummarizationPipeline.
- `"translation_xx_to_yy"`: will return a TranslationPipeline.
- `"text2text-generation"`: will return a Text2TextGenerationPipeline.
- `"text-generation"`: will return a TextGenerationPipeline.
- `"zero-shot-classification"`: will return a ZeroShotClassificationPipeline.
- `"conversational"`: will return a ConversationalPipeline.

### 预训练模型架构

- [huggingface Auto Classes](https://huggingface.co/docs/transformers/main/model_doc/auto#auto-classes)













