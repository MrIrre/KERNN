# -*- coding: utf-8 -*-
import re
import string
from Prepare import help_module
from Prepare.result import nlp_base

import pymorphy2
from pymorphy2.shapes import is_punctuation

pymorph = pymorphy2.MorphAnalyzer()

TOKEN_REGEX = re.compile(r'([^\w_-]|[+])', re.UNICODE)


def word_tokenize(text, _split=TOKEN_REGEX.split):
    return [t for t in _split(text) if t]


# def extract_words(s):
#     return [re.sub('^[{0}]+|[{0}]+$'.format(string.punctuation + '—«»'), '', w) for w in s.split()]


def lemmatize_word(word: str):
    if not is_punctuation(word) and not word.isspace():
        normal_form = pymorph.parse(word)[0].normal_form
        return word, normal_form

    return word, None

def extract_words(s):
    return re.findall(r'\b\S+\b', s)

def lemmatize_text(text):
    res = []

    for token in word_tokenize(text):
        if not is_punctuation(token) and not token.isspace():
            normal_form = pymorph.parse(token)[0].normal_form
            res.append((token, normal_form))
        else:
            res.append((token, None))

    return res


# str = 'This is a string, with words!'
title = 'Александр Сергеевич Пушкин'
lemmatized_title = ['александр', 'сергеевич', 'пушкин']

str = '''
А.С.Пушкина
Пушкин
Александра Пушкина
А. Пушкином
об Александре
с Александром Сергеевичем Пушкиным
'''

# nlp_base.get_search_strings(title, lemmatized_title)
import ujson
import math

i = 0

with open('texts_to_train/all-wiki-pages_with_tfidf.json', 'r', encoding='utf-8') as f:
    for line in f:
        cur_json = ujson.loads(line)

        if cur_json['lemmatized_title']:
            tf_idf = {lemma: cur_json['title_tf'][lemma] * math.log2(cur_json['title_idf'][lemma]) for lemma in cur_json['title_tf']}
            print(f"TF-IDF - {tf_idf}")
            print("--------------------------------")
        i += 1

        if i % 20000 == 0:
            print(i)

print(i)


# words = lemmatize_text(str)
# lemmas = list(map(lambda lemma_info: lemma_info[1] if lemma_info[1] is not None else lemma_info[0], words))
#
# # print(words)
#
# print(lemmas)
# print(''.join(lemmas))
