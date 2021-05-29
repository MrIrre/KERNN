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

nlp_base.get_search_strings(title, lemmatized_title)
from collections import Counter

FULL_NAME_WITH_COMMA_REGEX = re.compile("^([A-ZА-ЯЁ][\w -]+), ([A-ZА-ЯЁ][\w-]+)(?: ([A-ZА-ЯЁ][\w-]+))?$")

TF_IDF_THRESHOLD = 0.02

def get_search_strings(title: str, lemmatized_title=None, title_tf_idf=None):
    to_search = set()

    if lemmatized_title:
        to_search = set(lemmatized_title)

    if title_tf_idf is not None and title_tf_idf:
        title_tf_idf_counter = Counter(title_tf_idf)
        to_search = {title for title in lemmatized_title if title_tf_idf_counter[title] >= TF_IDF_THRESHOLD}

    full_name_res = FULL_NAME_WITH_COMMA_REGEX.findall(title)

    if full_name_res:
        surname, first_name, patronymic = [x.lower() for x in full_name_res[0]]

        to_search.add(surname)
        to_search.add(first_name)  # TODO: Нужно ли?
        to_search.add(f"{first_name} {surname}")
        to_search.add(f"{surname} {first_name}")
        # to_search.add(f"{first_name[0]}. {surname}")

        if patronymic:
            to_search.add(f"{first_name} {patronymic}")
            to_search.add(f"{surname} {first_name} {patronymic}")
            to_search.add(f"{first_name} {patronymic} {surname}")
            to_search.add(f"{first_name[0]}. {patronymic[0]}. {surname}")
    else:
        to_search.add(title.lower())
        to_search.add(''.join(lemma_info[1] if lemma_info[1] is not None else lemma_info[0] for lemma_info in lemmatize_text(title)))

    return to_search


print(get_search_strings("Пушкин, Александр Сергеевич",
                         ["пушкин", "александр", "сергеевич"],
                         Counter({"пушкин": 0.6, "александр": 0.008, "сергеевич": 0.01}))
      )

print(get_search_strings("Саркома",
                         ["саркома"],
                         Counter({"саркома": 0.8}))
      )

print(get_search_strings("Лёгкие",
                         ["лёгкое"],
                         Counter({"лёгкое": 0.02}))
      )

print(get_search_strings("Объём лёгких",
                         ["объём", "лёгкий"],
                         {"объём": 0.01})
      )

print(get_search_strings("Объём лёгких",
                         ["объём", "лёгкий"])
      )

print(get_search_strings("Объём лёгких"))

print(get_search_strings("(G)I-DLE", ['g)i-dle'], {'g)i-dle': 0.6}))



















# import ujson
# import math
# import matplotlib.pyplot as plt
#
# i = 0
#
# without_lemmatized_title = 0
# lemmatized_title_is_empty = 0
# without_tf_idf = 0
# lemmatized_title_len_less_than_tf_idf_len = 0
#
# # tf_idf = []
#
# with open('texts_to_train/all-wiki-pages_with_tfidf.json', 'r', encoding='utf-8') as f:
#     for line in f:
#         cur_json = ujson.loads(line)
#
#         if 'lemmatized_title' not in cur_json:
#             print('lemmatized_title not in JSON !!!')
#             print(cur_json['id'])
#             print('----------------------------------------')
#             without_lemmatized_title += 1
#
#         elif not cur_json['lemmatized_title']:
#             lemmatized_title_is_empty += 1
#
#         elif 'title_tf' not in cur_json:
#             print('title_tf not in JSON !!!')
#             try:
#                 print(cur_json['id'], cur_json['lemmatized_title'])
#             except:
#                 print(cur_json['id'])
#             print('----------------------------------------')
#             without_tf_idf += 1
#
#         elif len(cur_json['lemmatized_title']) < len(cur_json['title_tf']):
#             print('lemmatized_title LENGTH < title_tf LENGTH !!!')
#             try:
#                 print(cur_json['id'], cur_json['lemmatized_title'], cur_json['title_tf'])
#             except:
#                 print(cur_json['id'])
#             print('----------------------------------------')
#             lemmatized_title_len_less_than_tf_idf_len += 1
#
#         i += 1
#
#         if i % 20000 == 0:
#             print(f"{i} записей пройдено")
#
# print(f"Всего: {i}")
# print(f"Без лемматизированного заголовка : {without_lemmatized_title}")
# print(f"С пустым лемматизированным заголовком : {lemmatized_title_is_empty}")
# print(f"Без TF-IDF : {without_tf_idf}")
# print(f"Длина лемматизированного заголовка меньше длины TF-IDF: {lemmatized_title_len_less_than_tf_idf_len}")
# print(f"С проблемами : {without_lemmatized_title + lemmatized_title_is_empty + without_tf_idf + lemmatized_title_len_less_than_tf_idf_len}")

# tf_idf = help_module.flatten(tf_idf)
# plt.plot(tf_idf)
# plt.show()



# words = lemmatize_text(str)
# lemmas = list(map(lambda lemma_info: lemma_info[1] if lemma_info[1] is not None else lemma_info[0], words))
#
# # print(words)
#
# print(lemmas)
# print(''.join(lemmas))
