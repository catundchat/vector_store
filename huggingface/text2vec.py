# use huggingface text2vec-large-chinese to get sentence embedding

import PyPDF2
import os
import jieba
import pickle
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
import re

# load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("GanymedeNil/text2vec-large-chinese")
model = AutoModel.from_pretrained("GanymedeNil/text2vec-large-chinese")

# set original file directory
file_directory = r"D:\mydocuments\juxue\vector_knowledge_base\test_book"

# get all pdf files then read and split them into sentences
knowledge_vectors = []
for filename in os.listdir(file_directory):
    if filename.endswith('.pdf'):

        # read pdf files
        with open(os.path.join(file_directory, filename), 'rb') as file:
            reader = PyPDF2.PdfFileReader(file)
            text = ''
            for page_num in range(reader.numPages):
                text += reader.getPage(page_num).extractText()
            
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
with open(r'D:\mydocument\juxue\vector_knowledge_base\test_book\knowledge_vectors.pkl', 'wb') as file:
    pickle.dump(knowledge_vectors, file)
     

