# Vector_store
To create a vector store using embedding technologies, provided by OpenAI，Word2Vec，Meta AI, BaiduPaddle, packages on huggingface.

## openai

OpenAI Embedding 利用基于 transformers 架构的 GPT-3 预训练模型，将单词和句子映射到一个连续的向量空间，这些向量能够捕捉到连续的文本语义信息。最后产生一个输出的向量序列作为文本的向量表示

`openai/vector_store.ipynb`借由 openai embedding 接口完成嵌入，而后利用 LLM 能力进而完成数据库搜索

嵌入的向量数据库保存在 openai/index 下，其中 .bin 文件保存文件字节流序列化及反序列化等，存储的是 Python 中的对象即向量数据；.pkl 文件保存模型的权重及其他相关信息即向量与原始文本之间的映射关系，包括从原始ID映射到uuid，从uuid映射回原始ID以及相关包含索引的元数据

## word2vec

Word2Vec 是一种用于生成词向量的浅层神经网络模型。其基本思想是在大量文本数据上训练模型，使得语义上相似的单词在向量空间中的位置靠近。其隐藏层没有激活函数，也没有偏置项。这使得模型可以直接将one-hot编码的输入映射到词嵌入上，然后再映射到输出上。这样，一旦模型训练完毕，我们可以直接取出隐藏层的权重，作为我们的词向量。

`word2vec/create_vs.py` 通过 Word2Vec 算法来完成嵌入，生成的向量数据库保存在 vector_store.index 

[下载链接](https://drive.google.com/file/d/1YPcl72LZw9kJgo3puVP2CyixmEz5zzws/view?usp=sharing)

## Faiss and Rocket QA

Faiss 是由 Facebook AI 开发的一款用于高效相似性搜索和密集向量聚类的库。对于构建向量知识库的情境，我们主要用到的是 Faiss 的相似性搜索功能。Rocket QA 是飞桨开源的问答系统。利用 Faiss 和 RocketQA（一个基于 Transformer 模型的开源问答系统）来实现一个搜索引擎。搜索引擎的目的是根据用户提供的问题（查询），找出最相关的答案。

<details>
  <summary>Faiss构建索引速度更快的理论基础</summary>
  使用 Faiss 进行大规模相似性搜索通常会比传统的搜索方法更快。这主要是因为 Faiss 使用了一种称为 "近似最近邻搜索" (Approximate Nearest Neighbor Search, ANN) 的方法，这种方法可以大大减少搜索过程中的计算量。在传统的最近邻搜索 (Nearest Neighbor Search, NNS) 中，我们需要计算查询向量与数据库中每一个向量的距离，这种操作在高维度和大数据量的情况下会非常耗时。而在 Faiss 中，使用了一种叫做 "量化" (Quantization) 的方法，将原本需要大量存储和计算的向量进行了压缩，并且在压缩的过程中尽量保持原有的距离关系。这使得在 Faiss 中，我们可以在压缩后的表示上进行计算L2距离即L2范数，从而大大提升了搜索速度。另外，Faiss 还支持 GPU 加速，这对于大规模的相似性搜索任务来说是非常有用的。需要注意的是，Faiss 使用的 ANN 方法在提升搜索速度的同时，可能会对搜索结果的精度产生一定的影响。但在实际应用中，这种影响往往可以接受。
</details>

<details>
  <summary>Rocket QA介绍</summary>
  双塔模型 (Dual Encoder) 主要用于处理大规模的候选检索阶段。在这个阶段，系统将问题和候选答案分别输入两个相同的神经网络（塔）进行编码，然后比较编码结果的相似性来筛选出最相关的候选答案。
  交叉编码器 (Cross Encoder) 在第一阶段筛选出的候选答案中进行精细的排序。它将问题和候选答案作为一个整体输入到模型中，模型会输出一个分数，表示这个答案的相关性。交叉编码器通常比双塔模型更精确，但是计算复杂度更高，所以通常在筛选过的较小的候选集中使用。
</details>

这里我们使用之前已有的 faiss 向量数据库进行查询，文本清洗及文本分段的代码见`faiss/faiss_pre.py`，从Faiss索引中依据query和Rocket QA取回查询结果的代码见`faiss/faiss_retrieval.py`

<details>
  <summary>faiss 数据集下载链接</summary>
  链接: https://pan.baidu.com/s/1vGbwEQlGWTiy8u4LUNf_gg?pwd=pkyh 提取码: pkyh
</details>

## Huggingface

利用 huggingface 上的中文 embedding 库构建向量数据库，并进行向量相似度搜索寻找最符合的句段。

代码见`huggingface/text2vec.py`, colab 版本为`text2vec.ipynb`，所用数据为`huggingface/test_book`下的8本中文书籍。

## 结论

使用大模型的接口较自己搭建 embedding 模型效果更好，速度更快。其中 Faiss 使用 ANN 算法减少计算 L2 范数时的计算量，在数据量较大时更有用；而 OpenAI embedding 接口及之后的相似度搜索采用的是余弦相似度，在数据量较小时更适用。

## References

1. [Embeddings - OpenAI API](https://platform.openai.com/docs/guides/embeddings/what-are-embeddings)
2. [word2vec | TensorFlow Core](https://www.tensorflow.org/tutorials/text/word2vec#:~:text=word2vec%20is%20not%20a%20singular,downstream%20natural%20language%20processing%20tasks.)
3. [搜索召回 | Facebook：亿级向量相似度检索库Faiss原理+应用](https://zhuanlan.zhihu.com/p/432317877)
4. [Faiss Documentation](https://faiss.ai/)
5. [PaddlePaddle/RocketQA](https://github.com/PaddlePaddle/RocketQA)
