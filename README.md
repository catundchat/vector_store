# vector_store
create a vector store by OpenAi，Word2Vec.

## openai

`openai/vector_store.ipynb`借由 openai embedding 接口完成嵌入，而后利用 LLM 能力进而完成数据库搜索

嵌入的向量数据库保存在 openai/index 下，其中 .pkl 文件保存文件字节流序列化及反序列化等，存储的是 Python 中的对象；.bin 文件保存模型的权重及其他相关信息

## word2vec

Word2Vec 是一种用于生成词向量的浅层神经网络模型。其基本思想是在大量文本数据上训练模型，使得语义上相似的单词在向量空间中的位置靠近。其隐藏层没有激活函数，也没有偏置项。这使得模型可以直接将one-hot编码的输入映射到词嵌入上，然后再映射到输出上。这样，一旦模型训练完毕，我们可以直接取出隐藏层的权重，作为我们的词向量。

`word2vec/create_vs.py` 通过 Word2Vec 算法来完成嵌入，生成的向量数据库保存在 vector_store.index 

[下载链接](https://drive.google.com/file/d/1YPcl72LZw9kJgo3puVP2CyixmEz5zzws/view?usp=sharing)
