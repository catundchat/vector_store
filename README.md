# Vector_store
向量知识库 | 相似度搜索 | 知识图谱 | 微信小程序：AI爱家

为 AI爱家聊天机器人 创建向量知识库，并选取合适方法进行相似度搜索 

To create a vector store using embedding technologies, provided by OpenAI，Word2Vec，Meta AI, BaiduPaddle, packages on huggingface. Then do vector similarity search.

# 目录
- [构建向量知识库](#构建向量知识库)
  - [OpenAI](#openai)
  - [Word2Vec](#word2vec)
  - [Faiss and Rocket QA](#Faiss-and-Rocket-QA)
  - [HuggingFace](#Huggingface)
  - [结论](#结论)
- [向量知识库选择](#向量知识库选择)
  - [多种向量知识库横向对比](#多种向量知识库横向对比)
  - [选择建议](#选择建议)
- [向量知识库搜索方法](#向量知识库搜索方法)
  - [ANN](#ANN)
  - [余弦相似度](#余弦相似度)
- [搜索结果示例](#搜索结果示例)
  - [评价](#评价)
- [References](#References)

## 构建向量知识库

下列构建向量数据库方法基本包括：方法介绍，代码，向量数据库文件及其下载链接

### openai

OpenAI Embedding 利用基于 transformers 架构的 GPT-3 预训练模型，将单词和句子映射到一个连续的向量空间，这些向量能够捕捉到连续的文本语义信息。最后产生一个输出的向量序列作为文本的向量表示
- `openai/vector_store.ipynb`借由 openai embedding 接口完成嵌入，而后利用 LLM 能力进而完成数据库搜索
- embedding 后向量数据库保存在 `openai/index` 下
- .bin 文件保存文件字节流序列化及反序列化等，存储的是 Python 中的对象即向量数据；
- .pkl 文件保存模型的权重及其他相关信息即向量与原始文本之间的映射关系，包括从原始ID映射到uuid，从uuid映射回原始ID以及相关包含索引的元数据

### word2vec

Word2Vec 是一种用于生成词向量的浅层神经网络模型。其基本思想是在大量文本数据上训练模型，使得语义上相似的单词在向量空间中的位置靠近。其隐藏层没有激活函数，也没有偏置项。这使得模型可以直接将one-hot编码的输入映射到词嵌入上，然后再映射到输出上。这样，一旦模型训练完毕，我们可以直接取出隐藏层的权重，作为我们的词向量
- `word2vec/create_vs.py` 通过 Word2Vec 算法来完成 embedding
- 生成的向量数据库保存在 `vector_store.index `  [下载链接](https://drive.google.com/file/d/1YPcl72LZw9kJgo3puVP2CyixmEz5zzws/view?usp=sharing)

### Faiss and Rocket QA

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

faiss 数据集[下载链接](https://pan.baidu.com/s/1vGbwEQlGWTiy8u4LUNf_gg?pwd=pkyh) 提取码: pkyh

### Huggingface

利用 huggingface 上的中文 embedding 库构建向量数据库，并进行向量相似度搜索寻找最符合的句段。
- 本机运行代码见`huggingface/text2vec.py`, CPU Ram 8GB, 对《5%的改变》这本书进行 embedding 耗时1.5h，结果保存在`knowledge_vectors_1.pkl` 共400MB，[下载链接](https://drive.google.com/file/d/1QaNpN4EKys1sippa6SDebsFseQbaN7xV/view?usp=sharing) ;
- Colab 版本为`text2vec.ipynb`且调用 GPU:Tesla T4 加速，所用数据为`huggingface/test_book`下的8本中文书籍，大概耗费6h生成32.65GB大小的`knowledge_vectors.pkl`，[下载链接](https://drive.google.com/file/d/1rh2UXEnc8vocZeVm8_pe7DphMTHRLvZN/view?usp=sharing)
- 文本清洗时使用的中文停用词见`cn_stop_words中文文本停用词.txt`

### 结论

使用大模型的接口较自己搭建 embedding 模型效果更好，速度更快。其中 Faiss 使用 ANN 算法减少计算 L2 范数时的计算量，在数据量较大时更有用；而 OpenAI embedding 接口及之后的相似度搜索采用的是余弦相似度，在数据量较小时更适用。

## 向量知识库选择

向量数据库的功能主要是存储和检索向量，其评判标准我理解并细化为如下几个维度：
- 部署方式：自托管，云托管。即存储在自己的硬件上还是利用云服务提供商 
- 代码可用性：开源，闭源。开源社区的扩展性一般会更好而且免费
- 可扩展性：垂直扩展和水平扩展。扩展性良好有利于存储更多的数据及更多的查询 query
- 算法支持：最好支持 ANNOY(Approximate Nearest Neighbors Oh Yeah)，HNSW(近邻图技术) 和 Faiss 高效相似度搜索技术
- 成本：利用闭源数据库的许可费用和托管成本，利用开源数据库本地部署的硬件费用

### 多种向量知识库横向对比

<details>
  <summary>7种向量数据库横向对比</summary>
  
  Milvus
- 类型：自托管向量数据库
- 代码：开源
- 价值主张：关注整个搜索引擎的可扩展性，如何高效地对向量数据进行索引和重新索引；如何缩放搜索部分。能够使用多种ANN算法对数据进行索引，以比较它们在您的用例中的性能。
- 算法：支持多种基于ANN的索引算法，如FAISS、ANNOY、HNSW、RNSG。

Pinecone
- 类型：托管向量数据库
- 代码：封闭源代码
- 价值主张：完全托管的向量数据库，支持非结构化搜索引擎。支持在一个查询中搜索对象并按元数据进行过滤。
- 算法：由FAISS提供支持的Exact KNN；ANN由专有算法提供支持。

Vespa
- 类型：托管/自托管向量数据库
- 代码：开源
- 价值主张：Vespa是在大型数据集上进行低延迟计算的引擎，提供了面向深度学习的深度数据结构。
- 算法：使用HNSW算法，以及一套重新排序和密集检索方法。

Weaviate
- 类型：托管/自托管向量数据库
- 代码：开源
- 价值主张：支持类Graphql接口的表达查询语法，允许对丰富的实体数据运行探索性数据科学查询。
- 算法：使用自定义实现的HNSW算法。

Vald
- 类型：自托管向量搜索引擎
- 代码：开源
- 价值主张：用于十亿向量规模，提供云原生架构。
- 算法：基于NGT的最快算法。

GSI
- 类型：Elasticsearch/OpenSearch的向量搜索硬件后端
- 代码：封闭源代码
- 价值主张：十亿规模的搜索引擎后端，将Elasticsearch/OpenSearch功能扩展到相似性搜索。
- 算法：保持神经散列的汉明空间局部性。

Qdrant
- 类型：托管/自托管向量搜索引擎和数据库
- 代码：开源
- 价值主张：具有扩展过滤支持的向量相似度引擎。Qdrant 完全用 Rust 语言开发，实现了动态查询计划和有效负载数据索引。向量负载支持多种数据类型和查询条件，包括字符串匹配、数值范围、地理位置等。
- 算法：在 Rust 中自定义的 HNSW 实现。
</details>

### 选择建议

在这里我推荐两个向量数据库：Milvus 和 Pinecone
- Milvus 支持多种基于近似最近邻（ANN）的索引算法，如 FAISS、ANNOY、HNSW、RNSG, 允许用户根据数据类型和查询要求选择合适的索引算法。由于支持多种算法，我们可以进行实验以找到最适合其用例的算法。并且Milvus是开源代码仓库，我们只需在自己的服务器上跑通，不需要支付额外的托管费用。
- Pinecone 使用 FAISS 支持的 Exact KNN 以及由专有算法支持的 ANN, 其作为一个完全托管的服务，Pinecone 可能在算法选择上没有 Milvus 那么灵活，但它可能进行了优化以提供高效的检索性能。它的优点是快速部署和数据库使用便捷

关于具体使用哪种向量数据库，这要取决于嵌入向量的质量、数据库配置和索引算法。而这两种向量数据库所采取的算法基本一致，都是近似最近邻算法，在算法层面并无二致。若要详细比较两者细微差别需要用相同嵌入向量进行基准测试，以比较检索速度、准确性和资源消耗。

## 向量知识库搜索方法

- 构建文本的向量表示：首先，你需要将每段文本转化为向量。这通常可以通过训练好的嵌入模型来实现，比如Word2Vec、GloVe或BERT等。
- 构建哈希表（可选）：然后，使用LSH或其他哈希方法，将每个文本的向量表示映射到哈希空间，并构建哈希表。若要使用需将 scikit-learn 降级至 0.16.1，推荐使用 Faiss 或 Annoy.
- 搜索最近邻：当你有一个查询向量时，你可以首先将查询向量映射到哈希空间，然后在哈希表中搜索最近的哈希值，从而找到最近邻的文本。

### ANN

- Approximate Nearest Neighbor 这种算法可以在牺牲一定精度的前提下，大大提高搜索速度。哈希（Hashing）是ANN中的一种常见方法。基于哈希的ANN通常使用局部敏感哈希（Locality Sensitive Hashing，简称LSH）或其他哈希方法，将原始的高维空间映射到一个低维的哈希空间。在哈希空间中，相似的项会有相同或者相似的哈希值。这样，我们就可以通过比较哈希值来快速找到近似最近邻。
- LSH Forest：局部敏感哈希森林是普通近似最近邻搜索方法的替代方法, LSH Forest 数据结构是使用排序数组、二分搜索和 32 位固定长度哈希来实现的。使用随机投影作为近似余弦距离的哈希族。
- Annoy: APPROXIMATE NEAREST NEIGHBORS OH YEAH，近似最近邻搜索算法是 LSH Forest 这种算法的替代方法（LSH Forest 算法已 deprecated），采用了二叉树这个数据结构来提升查询的效率，目标是把查询的耗时减少至 O(\ln(n)).
-代码示例见：`search/annoy.py`

### 余弦相似度

Cosine Similarity:

$$ \text{cosine similarity} = \frac{\mathbf{A} \cdot \mathbf{B}}{\| \mathbf{A} \| \| \mathbf{B} \|} = \frac{ \sum_{i=1}^{n} A_i B_i }{ \sqrt{\sum_{i=1}^{n} A_i^2} \sqrt{\sum_{i=1}^{n} B_i^2} } $$

注：如果 LaTeX 公式不能正常显示，请安装 Mathjax plugin for github [安装链接](https://chrome.google.com/webstore/detail/mathjax-plugin-for-github/ioemnmodlmafdkllaclgeombjnmnbima)

具体实现时为了提高效率，经常采用计算向量点积后归一化的方法，这里直接给出代码示例：
- `search/cos_similarity_search.ipynb`适用于文本数量较小比如单篇文章
- `search/cos_search_2.py`适用于文本数量较大比如书籍

## 搜索结果示例

下面的 prompt 及搜索出的 completion 分别对应《5%的改变》一书中的标题和对应标题下的文段

![search.PNG](img/search.JPG)

### 评价

经过多次试验后，现在可以控制模型的输出文本长度。基本不会出现 LLM 的幻觉(hallucination)问题；但是因为基于 Top_K 选取最高相似度的前K个句子，所生成的回答偶现语义不连贯现象。若有更好的文本清洗质量，后续将修改算法，选取最高相似度的前K个段落，可以很大程度上增加语意连贯性。

## References

1. [Embeddings - OpenAI API](https://platform.openai.com/docs/guides/embeddings/what-are-embeddings)
2. [word2vec | TensorFlow Core](https://www.tensorflow.org/tutorials/text/word2vec#:~:text=word2vec%20is%20not%20a%20singular,downstream%20natural%20language%20processing%20tasks.)
3. [搜索召回 | Facebook：亿级向量相似度检索库Faiss原理+应用](https://zhuanlan.zhihu.com/p/432317877)
4. [Faiss Documentation](https://faiss.ai/)
5. [PaddlePaddle/RocketQA](https://github.com/PaddlePaddle/RocketQA)
6. [Using Vector Stroes - LlamaIndex](https://gpt-index.readthedocs.io/en/latest/how_to/integrations/vector_stores.html)
7. [text2vec-large-chinese - Hugging Face](https://huggingface.co/GanymedeNil/text2vec-large-chinese)
8. [ANNOY](https://sds-aau.github.io/M3Port19/portfolio/ann/)
9. [墨天轮中国数据库排行](https://www.modb.pro/dbRank?xl0524)
10. [云原生向量数据库Milvus](https://developer.aliyun.com/article/1065666#:~:text=%E4%BB%80%E4%B9%88%E6%98%AFMilvus-,Milvus%20%E6%98%AF%E4%B8%80%E6%AC%BE%E4%BA%91%E5%8E%9F%E7%94%9F%E5%90%91%E9%87%8F%E6%95%B0%E6%8D%AE%E5%BA%93%EF%BC%8C%E5%AE%83%E5%85%B7%E5%A4%87,%E7%9B%B8%E4%BC%BC%E5%BA%A6%E6%A3%80%E7%B4%A2%E7%9A%84%E9%97%AE%E9%A2%98%E3%80%82)
11. [向量数据库对比](https://www.modb.pro/db/516016)
