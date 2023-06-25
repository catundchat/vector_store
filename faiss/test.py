# from gensim.corpora.wikicorpus import extract_pages,filter_wiki
# import bz2file
# import re
# import opencc
# from tqdm import tqdm
# import codecs

# wiki = extract_pages(bz2file.open('zhwiki-latest-pages-articles.xml.bz2'))

# def wiki_replace(d):
#     s = d[1]
#     s = re.sub(':*{\|[\s\S]*?\|}', '', s)
#     s = re.sub('<gallery>[\s\S]*?</gallery>', '', s)
#     s = re.sub('(.){{([^{}\n]*?\|[^{}\n]*?)}}', '\\1[[\\2]]', s)
#     s = filter_wiki(s)
#     s = re.sub('\* *\n|\'{2,}', '', s)
#     s = re.sub('\n+', '\n', s)
#     s = re.sub('\n[:;]|\n +', '\n', s)
#     s = re.sub('\n==', '\n\n==', s)
#     s = u'【' + d[0] + u'】\n' + s
#     # return opencc.convert(s).strip()
#     return converter.convert(s).strip()
# i = 0
# # f = codecs.open('wiki.txt', 'w', encoding='utf-8')
# f = codecs.open('wiki_uniqe.txt', 'w', encoding='utf-8')

# w = tqdm(wiki, desc=u'已获取0篇文章')
# converter = opencc.OpenCC('t2s.json')
# for d in w:
#     if not re.findall('^[a-zA-Z]+:', d[0]) and d[0] and not re.findall(u'^#', d[1]):
#         s = wiki_replace(d)
#         f.write(s+'这是一片基于维基百科的文章断句符号')
#         i += 1
#         if i % 100 == 0:
#             w.set_description(u'已获取%s篇文章'%i)
# print(i)
# f.close()


# 已获取1301600篇文章
# with open('./wiki.txt') as f:
with open('./wiki_uniqe.txt') as f:
    raw_data = f.read()
data = raw_data.split('这是一片基于维基百科的文章断句符号')

item = data[-3]
while '\n\n\n' in item:
    item = item.replace('\n\n\n','\n\n')
tmp = item.split('\n\n')
js_tmp = {}
js_tmp['title'] =  re.findall(pat_title,tmp[0])[0].replace(' ','')
try:
    js_tmp['summary'] = tmp[1].replace('\n','')
except:
    js_tmp['summary'] = ''
try:
    js_tmp['sections'] = []
    print(tmp)
    for text in tmp[2:]:
        text_tmp = text.replace('\n','')
        t1 = re.findall(pat,text_tmp)
        print(t1)
        if t1:
            for tt in t1:
                if tt:
                    t1 = tt.replace(' ','').replace('=','')
                    break
            if isinstance(t1,list):
                t1 = ''
            t2 = text_tmp.split('==')[-1].replace('=','').replace(' ','')
            if t2:
                js_tmp['sections'].append({'title':t1,'content':t2})
        print(js_tmp)
except:
    js_tmp['sections'] = []
    
