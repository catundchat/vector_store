import os
import sys
import json
import faiss
import numpy as np


class FaissTool():
    """
    Faiss index tools
    """
    def __init__(self, text_filename, index_filename):
        self.engine = faiss.read_index(index_filename)
        self.id2text = []
        for line in open(text_filename):
            self.id2text.append(line.strip())

    def search(self, q_embs, topk=5):
        res_dist, res_pid = self.engine.search(q_embs, topk)
        result_list = []
        for i in range(topk):
            result_list.append(self.id2text[res_pid[0][i]])
        return result_list
    
faiss_tool_wiki = FaissTool('wikibaike_passage.txt', 'wikibaike_passage.index')


import rocketqa
de_model = 'zh_dureader_de_v2'
ce_model = 'zh_dureader_ce_v2'
de_conf = {
    "model": de_model,
    "use_cuda": True,
    "device_id": 2,
    "batch_size": 4
}
ce_conf = {
    "model": ce_model,
    "use_cuda": True,
    "device_id": 2,
    "batch_size": 4
}
dual_encoder = rocketqa.load_model(**de_conf)
cross_encoder = rocketqa.load_model(**ce_conf)

query = '牧野是当今的什么地方'
topk = 5
# encode query
q_embs = dual_encoder.encode_query(query=[query])
q_embs = np.array(list(q_embs))
search_result = faiss_tool.search(q_embs, topk)