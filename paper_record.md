

# P:机器翻译



## Neural machine translation by jointly learning to align and translate.  

> Dzmitry Bahdanau 2014
>
> 机器翻译

使用attention机制来保证在decoder 的时候能够引用所有的时间步骤的信息，起到主题明确的作用

被作为很多生成模型的基础。

使用GRU作为生成的模型，在每一个时间步以attention 之后的信息向量作为输入。并产生一个输出



## Sequence to sequence learning with neural networks.

Sutskever  2014



## Learning phrase representations using rnn encoder-decoder for statistical machine translation

Kyunghyun Cho 2014





## Attention Is All You Need

> transformer 

self attention机制来进行文本翻译任务



## BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding
> BERT 

- 双向transformer
- 预测句子和mask的单词（D A E）
- 预训练和fine tuning
- 



# P:文本摘要

##  Topical Coherence for Graph-based Extractive Summarization 

> EMNLP 2015

用主题模型表示句向量，用ILP调优重要性，关联性和不冗余性。

##  Extractive Summarization by Maximizing Semantic Volume 

>EMNLP 2015

构建语义空间，从语义空间中选择能够涵盖所有内容的句子。



## Ranking with Recursive Neural Networks and Its Application to Multi-Document Summarization

> 抽取式摘要
>
> 文本打分

递归式的对文本进行评分，递归式的nn，但是不是一般意义的RNN



## Reinforced Extractive Summarization with Question-Focused Rewards

>强化学习
>
>抽取式摘要
>
>ACL2018

##  Jointly Learning Topics in Sentence Embedding for Document Summarization 

> 句子编码
>
> GMM高斯混合模型主题模型



## Neural Document Summarization by Jointly Learning to Score and Select Sentences

> 抽取式摘要
>
> 文本打分

使用神经网络训练出文本打分模型，对文本是否属于摘要进行打分

基于规则的数据集构建，具有更高ROUGE分数的句子作为摘要

## **A Semantic QA-Based Approach for Text Summarization Evaluation**

> 验证策略  文本摘要
>
> AAAI2018

提出一种新的验证策略，认为生成的文档和原始文档都应该可以回答一些问题，可以根据这些回答来从语义角度评价两个文档的情况。



## Faithful to the Original: Fact-Aware Neural Abstractive Summarization
> Ziqiang Cao
>
> AAAI2018



OpenIE 项目来获取原始文档中的关系信息，这些关系信息是包含了事实的概念的。



## Selective Encoding for Abstractive Sentence Summarization

> ACL 2017
>
> 门控机制  优化策略

使用门控机制确定某些信息是否保留

使用GRU获得原始输入在每个时间步上的双向隐状态向量，用门控机制确定是否为这个状态向量添加mask

- 门控机制用于统计多少个样本中是使用了门控机制的，结果发现使用了门控机制的样本包含了更少的unk标签，以此证实门控机制对于信息控制的作用





## A Reinforced Topic-Aware Convolutional Sequence-to-Sequence Model for Abstractive Text Summarization

> ACL 2018
>
> 强化学习 CNNs2s 主题模型信息

提出了使用主题模型的信息作为生成文档的辅助信息，在生成文档的时候，输入的特征是两个方面的，一方面是来自文本自己的embedding向量，一方面是主题向量。

- CNNs2s的使用
- positional embedding没有说怎么得到的，可能类似三角函数。
- joint attetnion机制，对attention机制进行了变体，align之前求和CNN关于状态和关于主题的相似度
- 强化学习方式 SCST（self critical sequence training）



## A Neural Attention Model for Abstractive Sentence Summarization

> EMNLP 2015
>
> 文本摘要 attention机制

第一次提出用s2s机制加上attetnion机制来实现文本生成式摘要。

## Get To The Point: Summarization with Pointer-Generator Networks

> point网络
>
> 抽取式和生成式结合
>
> 



## Neural Latent Extractive Document Summarization  

> 抽取式摘要

传统的抽取式摘要的方法都是使用基于规则标注策略，然后进行训练打分，但是这种标注策略并不一定式最好的。

本文提出一个隐含变量的方式来对句子进行标注。这种基于规则的标注策略标注出来的单词被称为“gold standard”

抽取模型会对输入句子打上标签，0或1，表示是否是摘要的一部分。

这个0或1输入了下面的模型

在训练打标签模型的时候，使用的损失函数值得借鉴。

损失函数包含两个部分，一个部分是准确率一部分式召回率

 使用 (Bahdanau et al., 2015; Rush et al., 2015)的方法. 得到预先训练好的s2s模型，然后给每一个句子隐含变量$ z \in \{0,1\} $ ，这个值来自抽取模型的输出。

在使用生成模型的时候不进行训练，只是得到由这个句子得到生成句子的综合概率。（也就是说这个s2s的模型是个没有感情的计算损失的机器）

然后用这个损失值训练标签模型

这样真的能计算出来梯度吗。。



# P:减少平凡回复问题



## Mutual information and diverse decoding improve neural machine translation

> 模型 改变损失函数 互信息
>
> Jiwei Li and Dan Jurafsky. 2016. 

互信息，重新定义MLE，但是容易生成无语法句子

互信息是指p(x|y)与p(y|x)之间的线形结合，表示的是在x的基础上y的概率和y的基础上x的概率，其中x与y表示的是原始句子和目标句子，（机器翻译）

原始的互信息概念需要生成所有的目标句子生成才可以计算。

**提出策略：**使用Beam search每次列出k个最相关的句子，然后用另一个模型求出反向的相关性，得到互信息。

**另外：本文使用下文提出的多元化的BeamSearch方法



## A simple, fast diverse decoding algorithm for neural generation
> 优化策略 Beam search

更快的多样性解码单元，修改beam search来让具有更多信息的句子达到更高的层级

**beam search**的策略导致生成的句子具有较大的相似性【Solving the problem of cascading errors: Approximate Bayesian inference for linguistic annotation pipelines】，所以使用reranking的beam search方法，在beam search的每一轮计算的时候都会对对原来的权重增加一个偏重，值为该单词在这次生成中的顺序。这样可以减少相似的句子出现的几率。

举个例子

第一次生成： he 0.7  you 0.2  

he的第二次生成： is  0.7  are 0.3

it的第二次生成：  are 0.8  is 0.1

传统方法：最后只会留下前两个概率最高的，都是以he开头的

调整之后 第一步： he 0.7-1 you 0.2-2

## Adversarial learning for neural dialogue generation



> 对话系统  对抗学习
>
> 2017 emnlp

在解码过程中减少平凡回复的产生，使用对抗学习的方式生成对话。

- 一般的强化学习的方式中的reward是在每一个样例被完整的生成之后决定的，但是实际上可能模型的输出的前几步都是合理的，最后一步出错而已。所以提出reward应当在每一个步骤进行操作。

- 当不为生成模型提供明确的方向的时候，生成模型有可能会不知道什么样的生成是好的，所以找不到优化的目标，这时候需要提供一些正确的案例作为刺激。

  

## Deep reinforcement learning for dialogue generation

使用强化学习的方式，用生成结果和8个平凡回复之间的距离来衡量某个回复的平凡性



## Towards Less Generic Responses in Neural Conversation Models: A Statistical Re-weighting Method


>  优化策略  数据集处理
>
> Yahui Liu1∗, Victoria Bi 2018

解决文本平凡回复问题，在大部分的数据集中，平凡的回答都会占有大多数，导致模型的输出大量的平凡回复，本文通过引入一个重新分配权重的方法来解决这个问题。通过统计的方式为文本分配新的权重



为回复增加一个权重项
$$
Φ(y) = αE(y) + βF(y)\\

E(y) = e^{-af(y)}\\

f(y) = max\{0,Count(D(y,yi)>r)\}\\

F(y) = e^{−c||y|−|yˆ|}
$$
E和F分别关联全集中和这个句子同义句子的数量，以及这个句子和平均句子长度的变化量。当这两个指标都比较高的时候，表示句子比较特殊。





