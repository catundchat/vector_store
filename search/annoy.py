from annoy import AnnoyIndex
import pickle
import numpy as np
import jieba

# define a text to vector function
def text_to_vector(text):
    words = jieba.cut(text)
    vector = np.zeros(f)
    for word in words:
        if word in model:
            vector += model[word]
    return vector
# load vectors file
with open(r'D:\mydocument\juxue\vector_knowledge_base\test_book\knowledge_vectors_1.pkl', 'rb') as f:
    # check data type
    data = pickle.load(f)
    print(type(data))  
    if isinstance(data, list):
        print(len(data))  
    elif isinstance(data, tuple):
        print(len(data))  
    else:
        print(data)

# define vectors dimension
f = len(vectors[0])

# create a Annoy index
t = AnnoyIndex(f, 'angular')  # Length of item vector that will be indexed
for i in range(len(vectors)):
    t.add_item(i, vectors[i])

t.build(10)  # 10 trees

# change query into vector
query_text = "一周只有一天想干活"
query_vector = text_to_vector(query_text)

indices = t.get_nns_by_vector(query_vector, 5)  # find the 5 nearest neighbors

# print query result
for i in indices:
    print(texts[i])

