# vector_store
create a vector store by openai or other method

## openai

`openai/vector_store.ipynb`借由 openai embedding 接口完成嵌入，而后利用 LLM 能力进而完成数据库搜索

嵌入的向量数据库保存在 openai/index 下。

## word2vec

`word2vec/create_vs.py` 通过 Word2Vec 算法来完成嵌入，生成的向量数据库保存在 vector_store.index 下，文件太大这里没有上传
