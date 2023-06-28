import os
import torch
import jieba
import pickle
import gc
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import PyPDF2
import re

tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
model = AutoModel.from_pretrained("bert-base-chinese")

# set the path to your PDF file
pdf_file = r"D:\mydocument\juxue\vector_knowledge_base\test_book\5%的改变.pdf"

knowledge_vectors = []

# read pdf files
with open(pdf_file, 'rb') as file:
    reader = PyPDF2.PdfReader(file)
    page_count = len(reader.pages) # count the number of pages in the current book
    for page_num in tqdm(range(page_count), desc="Processing book"): # add progress bar for book
        text = reader.pages[page_num].extract_text()

        # split text into sentences with jieba package
        sentences = re.split('。|！|？|；|……|，|：|“|”|（|）|、|《|》|【|】|——|……|……|……|……|……|……|', text)

        # get sentence embedding
        for sentence in sentences:
            # tokenize sentence then do text cleaning
            words = jieba.cut(sentence)
            clean_sentence = ' '.join(words)
            inputs = tokenizer(clean_sentence, return_tensors="pt")
            with torch.no_grad():
                embeddings = model(**inputs).last_hidden_state
            knowledge_vectors.append(embeddings.mean(dim=1).numpy())

        # save knowledge vectors to pickle file then do serialization
        with open(r'D:\mydocument\juxue\vector_knowledge_base\test_book\knowledge_vectors_1.pkl', 'wb') as file:
            pickle.dump(knowledge_vectors,file)



     

