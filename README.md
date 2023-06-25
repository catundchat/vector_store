# Vector_store
create a vector store by OpenAI，Word2Vec.

## openai

OpenAI Embedding 利用基于 transformers 架构的 GPT-3 预训练模型，将单词和句子映射到一个连续的向量空间，这些向量能够捕捉到连续的文本语义信息。最后产生一个输出的向量序列作为文本的向量表示

`openai/vector_store.ipynb`借由 openai embedding 接口完成嵌入，而后利用 LLM 能力进而完成数据库搜索

嵌入的向量数据库保存在 openai/index 下，其中 .bin 文件保存文件字节流序列化及反序列化等，存储的是 Python 中的对象即向量数据；.pkl 文件保存模型的权重及其他相关信息即向量与原始文本之间的映射关系，包括从原始ID映射到uuid，从uuid映射回原始ID以及相关包含索引的元数据

## word2vec

Word2Vec 是一种用于生成词向量的浅层神经网络模型。其基本思想是在大量文本数据上训练模型，使得语义上相似的单词在向量空间中的位置靠近。其隐藏层没有激活函数，也没有偏置项。这使得模型可以直接将one-hot编码的输入映射到词嵌入上，然后再映射到输出上。这样，一旦模型训练完毕，我们可以直接取出隐藏层的权重，作为我们的词向量。

`word2vec/create_vs.py` 通过 Word2Vec 算法来完成嵌入，生成的向量数据库保存在 vector_store.index 

[下载链接](https://drive.google.com/file/d/1YPcl72LZw9kJgo3puVP2CyixmEz5zzws/view?usp=sharing)

## Faiss



## References

1. [Embeddings - OpenAI API](https://platform.openai.com/docs/guides/embeddings/what-are-embeddings)
2. [word2vec | TensorFlow Core](https://www.tensorflow.org/tutorials/text/word2vec#:~:text=word2vec%20is%20not%20a%20singular,downstream%20natural%20language%20processing%20tasks.)
3. [搜索召回 | Facebook：亿级向量相似度检索库Faiss原理+应用](https://zhuanlan.zhihu.com/p/432317877)
