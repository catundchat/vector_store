import os
import jieba
from gensim.models import Word2Vec
import faiss
import numpy as np
from pdfminer.high_level import extract_text

def extract_text_from_pdf(pdf_path):
    return extract_text(pdf_path)

def preprocess_and_tokenize(text):
    return list(jieba.cut(text))

def split_text(text, chunk_size):
    words = list(jieba.cut(text))
    return [''.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

def train_embedding_model(sentences):
    model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)
    model.save("word2vec.model")
    return model

def create_faiss_vector_store(model, texts):
    # 提取所有文本的词向量
    vectors = [model.wv[text] for text in texts]
    vectors = np.array(vectors).astype('float32')

    # 创建Faiss索引
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)

    # 保存索引到文件
    faiss.write_index(index, "vector_store.index")

    return index

def create_vector_store(pdf_paths):
    # initialization of the index
    chunk_size = 200
    text_splitter = split_text(chunk_size=chunk_size, chunk_overlap=0)
    
    # here we will store the vectors
    split_texts = []
    
    for i, pdf_path in enumerate(pdf_paths):
        print(f"Processing file {i+1}/{len(pdf_paths)}: {pdf_path}")
        text = extract_text_from_pdf(pdf_path)
        splits = text_splitter.split(text)
        split_texts.extend(splits)
        
    # train embedding model
    model = train_embedding_model(split_texts)
    
    # create faiss vector store
    vector_store = create_faiss_vector_store(model, split_texts)
    
    return vector_store


# PDF file location
pdf_directory = r'D:\mydocument\juxue\psychological_data\book\book_6_9'

# extract text from PDF files
sentences = []
for filename in os.listdir(pdf_directory):
    if filename.endswith('.pdf'):
        pdf_path = os.path.join(pdf_directory, filename)
        text = extract_text_from_pdf(pdf_path)
        sentences.extend([preprocess_and_tokenize(line) for line in text.split('\n') if line])

model = train_embedding_model(sentences)

index = create_vector_store(model, sentences)

# save vector store
faiss.write_index(index, "vector_store.index")
