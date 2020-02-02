#!/usr/bin/env python
# coding: utf-8

# In[88]:


import gensim
from gensim.models.word2vec import Word2Vec
import gensim.downloader as api

from rutermextract import TermExtractor

import nltk
nltk.download("stopwords")
#--------#

from nltk.corpus import stopwords
from pymystem3 import Mystem
from string import punctuation

import wget
import zipfile

from os import listdir
from os.path import isfile, join

import re


# In[89]:


#model_name = 'glove-wiki-gigaword-100'
#model_name = 'word2vec-ruscorpora-300'
model_url = 'http://vectors.nlpl.eu/repository/20/182.zip'

word2vec_postfixes = ['_PROPN', '_NOUN', '_ADJ', '_VERB']
russian_stopwords = stopwords.words("russian")


# In[90]:


mystem = Mystem() 
term_extractor = TermExtractor()


# In[91]:


#model = api.load(model_name)

# загрузка word2vec модели из сети
if not 'model.bin' in listdir('.'):
    m = wget.download(model_url)

model_file = model_url.split('/')[-1]
with zipfile.ZipFile(model_file, 'r') as archive:
    stream = archive.open('model.bin')
    model = gensim.models.KeyedVectors.load_word2vec_format(stream, binary=True)


# In[92]:


questions = []
answers = []

dir_name = './feb'

files = [f for f in listdir(dir_name) if isfile(join(dir_name, f))]
for file_name in files:
    file = open(join(dir_name, file_name), 'r')
    lines = file.readlines()
    for i in range(len(lines)):
        lines[i] = lines[i][:-1]
    
    question = lines[0]
    answer = ' '.join(lines[1:])
                
    questions.append(question)
    answers.append(answer)


# In[23]:


#### хорошие словосочетания - все слова есть в словаре word2vec
# плохие словосочетания - есть хотя бы одно слово не из словаря word2vec
# хорошее слово - слово из словаря word2vec
# плохое слово - слово не из словаря word2vec


# find_metrics( report_text, [idx, question] ): [ (idx, bp_metric, gp_metric, bw_metric, gw_metric, s_metric) ]
# 0. приводим все слова из факов к нормальной форме

#    extract_keys( text ): (bad_phrases, good_phrases, bad_word, good_words)
# 1. получаем из обращения важные слова и словосочетания с нормализированными словами
# 2. словосочетания закидываем в функцию классификации результата, на выходе получаются:
#    хорошие словосочетания, плохие словосочетания, хорошие слова, плохие слова

# 3. проходимся по всей базе вопросов-ответов и пробуем найти совпадение с текстом обращения:
#        по плохим словосочетаниям (метрика bp_metric)
#        по хорошим словосочетаниям (метрика gp_metric)
#        по плохим словам (метрика bw_metric)
#        по хорошим словам (метрика gw_metric)
#    в качестве значения метрики можно использовать количество символов или слов !(потом изучим)!
#    находя в факовом вопросе словосочетание из обращение, вытираем словосочетание
#      и все его слова из соответствующей категории
#    находя в факовом вопросе слово, вытираем его

# 4. для оставшихся плохих посчитаем метрику ненайденных плохих слов и словосочетаний !(длина или кол-во слов)!
#    Метрики: nfb_metric

#    find_synonymous( good_words ): 
# 5. если остались хорошие слова и/или словосочетания, занимаемся поиском метрики синонимичности
#    для этого все хорошие словосочетания дробим на слова и формируем один большой список хороших слов
#    и для каждого слова находим N самых близких слов и кладем в общий пул синонимов 


# 5.1 в вопросах факов найдем количество слов из пула синонимов. это количество будет метрикой синонимичности:
#     syn_metric


# In[24]:


# является ли данный текст фразой или словом
def is_phrase(text):
    return ' ' in text

# сопоставить слову словарное слово из word2vec
# map_word2vec_word( word ): (is_success, word2vec_word or source_word)
def map_word2vec_word(word):
    temp_tokens = [word + x for x in word2vec_postfixes]
    for temp_token in temp_tokens:
        try:
            model.word_vec(temp_token)
        except Exception:
            pass
        else:
            return (True, temp_token)
        
    return (False, word)


# In[25]:


# выделение важных слов и словосочетаний
# find_tokens( text ): array of phrases and words
def find_tokens(text):
    tokens = []
    for term in term_extractor(text):
        term_extractor(text)
        tokens.append(term.normalized)
        
    return tokens

# нормализация текста
# normalize_words( text ): normalized_text
def normalize_words(text):
    tokens = mystem.lemmatize(text.lower())
    tokens = [token for token in tokens if token not in russian_stopwords              and token != " "               and token.strip() not in punctuation]
    return ' '.join(tokens)


# In[49]:


def memoize2(f):
    memo = {}
    def helper(*x):
        if x not in memo:            
            memo[x] = f(*x)
        return memo[x]
    return helper

# найти хорошие/плохие словосочетания и слова
# extract_keys( text ): (bad_phrases, good_phrases, bad_words, good_words) - all sets
@memoize2
def extract_keys(text):
    bad_phrases, good_phrases = [], []
    bad_words, good_words = [], []
    
    text = text.replace('?', '.')
    text = text.replace('!', '.')
    sentences = text.split('.')
    tokens = set()
    for sentence in sentences:
        for token in find_tokens(sentence):
            tokens.add(normalize_words(token))
    
    word_tokens = [token for token in tokens if not is_phrase(token)]
    phrase_tokens = [token for token in tokens if is_phrase(token)]
    for phrase in phrase_tokens:
        for word in phrase.split(' '):
            word_tokens.append(word)
    
    word_tokens = [token for token in tokens if not is_phrase(token)]
    phrase_tokens = [token for token in tokens if is_phrase(token)]
    
    # распределение фраз по категориям
    for phrase in phrase_tokens:
        words = phrase.split(' ')
        for word in words:
            success, _ = map_word2vec_word(word)
            if not success:
                bad_phrases.append(phrase)
                break
        else:
            good_phrases.append(phrase)
    
    # распределение слов по категориям
    for word in word_tokens:
        success, _ = map_word2vec_word(word)
        if not success:
            bad_words.append(word)
        else:
            good_words.append(word)
    
    return (set(bad_phrases), set(good_phrases), set(bad_words), set(good_words))


# In[46]:


# значение метрики - длина фразы (в словах)
def metric_value(token):
    if len(token) == 0:
        return 0
    return len(token.split(' '))   

def memoize(f):
    memo = {}
    def helper(*x):
        if x not in memo:            
            memo[x] = f(*x)
        return memo[x]
    return helper
    
# поиск значений различных метрик
# syn_coeff - минимальное значение похожести слов в word2vec для поиска синонимов
# find_metric( report_text, question, syn_coeff ): ( metrics, hits )
# metrics - массив из метрик. порядок метрик такой же как в шапке метода
# hits - тапл из четырех массивов: найденные совпадения плохих словосочетаний,
#        хороших словосочетаний, плохих слов, хороших слов
@memoize
def find_metric( report_text, question, syn_coeff=0.5 ):
    question = normalize_words(question)
    
    bp_metric = 0.0 # плохие словосочетания
    gp_metric = 0.0 # хорошие словосочетания
    bw_metric = 0.0 # плохие слова
    gw_metric = 0.0 # хорошие слова
    syn_metric = 0.0 # метрика по синонимам
    nfb_metric = 0.0 # ненайденные плохие слова и словосочетания
    nfg_metric = 0.0 # ненайденные хорошие слова и словосочетания
    
    bad_phrases, good_phrases, bad_words, good_words = extract_keys(report_text)
    
    # не найденные слова
    nf_bad_words, nf_good_words = set(), set()
    
    # найденные совпадения
    bad_phrases_hits, good_phrases_hits = set(), set()
    bad_words_hits, good_words_hits = set(), set()
    
    # поиск значения метрики плохих словосочетаний
    for phrase in bad_phrases:
        metric = metric_value(phrase)
        if phrase in question:
            bad_phrases_hits.add(phrase)
            bp_metric += metric
            
            # больше не ищем слова из словосочетания
            for word in phrase.split(' '):
                if word in bad_words:
                    bad_words.remove(word)
                if word in good_words:
                    good_words.remove(word)
        else:
            # если не нашли, тогда помещаем все плохие слова из плохих словосочетаний в пул ненайденных плохих слов 
            words = phrase.split(' ')
            for word in words:
                is_good, _ = map_word2vec_word(word)
                if not is_good:
                    nf_bad_words.add(word)
            
    # поиск значения метрики хороших словосочетаний
    for phrase in good_phrases:
        if phrase in question:
            good_phrases_hits.add(phrase)
            bp_metric += metric_value(phrase)
            
            # больше не ищем слова из словосочетания
            for word in phrase.split(' '):
                if word in good_words:
                    good_words.remove(word)
        else:
            # если не нашли, тогда помещаем все хорошие слова из словосочетания в пул ненайденных хороших слов 
            words = phrase.split(' ')
            for word in words:
                nf_good_words.add(word)
                
    
    # поиск значения метрики плохих слов
    for word in bad_words:
        if word in question:
            bad_words_hits.add(word)
            bw_metric += metric_value(word) # можно сделать просто инкремент
        else:
            nf_bad_words.add(word)
            
    # метрика ненайденных плохих слов
    nfb_metric += sum([metric_value(word) for word in nf_bad_words])
            
    # поиск значения метрики хороших слов
    for word in good_words:
        if word in question:
            good_words_hits.add(word)
            gw_metric += metric_value(word) # можно сделать просто инкремент
        else:
            nf_good_words.add(word)
    
    # поиск метрики синонимов
    for word in nf_good_words:
        _, w2v_word = map_word2vec_word(word)
        
        similiarities = model.similar_by_word(w2v_word)
        for similiarity in similiarities:
            syn_word, prob = similiarity
            if prob < syn_coeff:
                break
            
            metric = metric_value(syn_word)
            if syn_word in question:
                syn_metric += metric
            else:
                nfg_metric += metric
            
    metrics = [bp_metric, gp_metric, bw_metric, gw_metric, syn_metric, nfb_metric, nfg_metric]
    hits = (bad_phrases_hits, good_phrases_hits, bad_words_hits, good_words_hits)
    
    return (metrics, hits)


def print_metric(metrics, hits=None):
    titles = [
        'плохие словосочетания', 'хорошие словосочетания',
        'плохие слова', 'хорошие слова',
        'метрика по синонимам',
        'ненайденные плохие слова/сочетания', 'ненайденные хорошие слова/сочетания'
    ]
    for i in range(len(metrics)):
        print(titles[i], ':', metrics[i])

    if hits is not None:
        bp_hits, gp_hits, bw_hits, gw_hits = hits
        print('---')
        print('плохие словосочетания', ':', bp_hits)
        print('хорошие словосочетания', ':', gp_hits)
        print('плохие слова', ':', bw_hits)
        print('хорошие слова', ':', gw_hits)


# In[55]:


# # вопрос, который задал юзер сейчас
# actual_question = 'Здравствуйте, можно ли изменить цену в оферте, которой уже присвоен номер? Или нужно создавать новую оферту?'
# print('actual question:', actual_question)

# # проходимся по всей базе вопросов
# for i in range(min(10, len(questions))):
# #for question in questions:
#     question = questions[i]
#     metric, hits = find_metric(actual_question, question)
    
#     # смотрим на метрики, если сумма положительных метрик больше единицы, то печатаем вопрос из базы
#     if sum(metric[:4]) > 1:
#         print(question)
#         print_metric(metric, hits)
#         print('======')


# # In[54]:


# # вопрос, который задал юзер сейчас
# actual_question = 'Здравствуйте, можно ли изменить цену в оферте, которой уже присвоен номер? Или нужно создавать новую оферту?'
# question = 'не присваивается номер оферте, что делать?'
# print('actual question:', actual_question)

# print(extract_keys(question))
# print(extract_keys('Здравствуйте, можно ли изменить цену в оферте, которой уже присвоен номер'))
# print(extract_keys('Или нужно создавать новую оферту?'))
# metric, hits = find_metric(actual_question, question)
    
# # смотрим на метрики, если сумма положительных метрик больше единицы, то печатаем вопрос из базы
# if sum(metric[:4]) > 1:
#     print(question)
#     print_metric(metric, hits)
#     print('======')


# In[61]:


# # вопрос, который задал юзер сейчас
# actual_question = 'В нашей заявке 2886444-19 мы ждем номер заявки на СТЕ. Как скоро обычно такие заявки рассматриваются?'
# print('actual question:', actual_question)

# # проходимся по всей базе вопросов
# max_similarity = 0
# found_index = None
# idx = -1
# for i in range(min(30, len(questions))):
# #for question in questions:
#     question = questions[i]
#     print("NOW", question)
#     metric, hits = find_metric(actual_question, question)
#     print()
#     if sum(metric[:5]) > max_similarity:
#         max_similarity = sum(metric[:5])
#         idx = i
#         found_index = metric, hits
       
# if max_similarity > 0:
#     print(max_similarity, questions[idx])
#     print_metric(metric, hits)
        
    
    # смотрим на метрики, если сумма положительных метрик больше единицы, то печатаем вопрос из базы
#     if sum(metric[:4]) > 1:
#         print(question)
#         print_metric(metric, hits)
#         print('======')


# In[96]:


# вопрос, который задал юзер сейчас
def get_simple_answer(request):
    # проходимся по всей базе вопросов
    max_similarity = 0
    found_index = None
    idx = -1
#    for i in range(len(questions)):
    for i in range(min(150, len(questions))):
        question = questions[i]
        print("NOW", question)
        metric, hits = find_metric(request, question)
        print(metric[:5])
        if sum(metric[:5]) > max_similarity:
            max_similarity = sum(metric[:5])
            idx = i
            found_index = metric, hits

    if max_similarity > 0:
        print(max_similarity, questions[idx])
        print("ANSWER", answers[idx])
        print_metric(metric, hits)
        
    text = request.replace('?', '.')
    text = text.replace('!', '.')
    sentences = text.split('.')
    tokens = set()
    print(term_extractor(request))
    for sentence in sentences:
        for token in find_tokens(sentence):
            tokens.add(token)
            
    for token in tokens:
        request = request.replace(token, token.upper())
    return answers[idx], request, tokens, max_similarity
            
            
# actual_question = 'В нашей заявке 2886444-19 мы ждем номер заявки на СТЕ. Как скоро обычно такие заявки рассматриваются?'
# get_simple_answer(str(actual_question))

    
        
    
    # смотрим на метрики, если сумма положительных метрик больше единицы, то печатаем вопрос из базы
#     if sum(metric[:4]) > 1:
#         print(question)
#         print_metric(metric, hits)
#         print('======')


# In[ ]:




