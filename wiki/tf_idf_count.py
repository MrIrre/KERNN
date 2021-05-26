# -*- coding: utf-8 -*-
import ujson
import html
import math
import os
import re
import unicodedata

from Prepare.result import nlp_base
import collections


# TODO: считать TF-IDF до или после отсечения документов с проверкой нахождения тайтла в тексте?

# unaccentify = {
#     '́': '',
#     '̀': '',
#    'ѝ': 'и',
#    'о́': 'о'
# }
# unaccentify = {unicodedata.normalize('NFKC', i):j for i, j in unaccentify.items()}
# pattern = re.compile('|'.join(unaccentify))
# def unaccentify_text(text):
#     def replacer(match):
#         return unaccentify[match.group(0)]
#
#     source = unicodedata.normalize('NFKC', text)
#     return pattern.sub(replacer, source)

count = 0
# articles_count = 1458496
text_not_found_count = 0

from_file = 'all-wiki-pages-json_with_lemmas'
counted_words_for_idf_file = 'counted_words_for_idf.json'
res_file = 'all-wiki-pages_with_tfidf.json'


if os.path.exists(res_file):
    print(f'FILE {res_file} IS ALREADY EXISTS!!!')
    exit(1)


counted_words_occurrences_in_all_texts_for_idf = collections.Counter()

with open(counted_words_for_idf_file, 'r', encoding='utf-8') as fp:
    counted_words_occurrences_in_all_texts_for_idf = collections.Counter(ujson.loads(fp.read()))

articles_count = 1458355

# if os.path.exists(counted_words_for_idf_file):
#     print(f'FILE {counted_words_for_idf_file} IS ALREADY EXISTS!!!')
#     exit(1)
#
# with open(file=from_file, mode='r', encoding='utf-8') as fp:
#     for _ in range(articles_count):
#         next(fp)
#
#     for line in fp:
#         cur_article_info = ujson.loads(line)
#         lemmatized_cur_text_words_counter = cur_article_info['lemmatized_words_counter']
#
#         lemmas = lemmatized_cur_text_words_counter.keys()
#
#         for lemma in lemmas:
#             counted_idf[lemma] += 1
#
#         if articles_count % 10000 == 0:
#             print(f"{articles_count} штук прочитано")
#
#         articles_count += 1
#
# with open(file=counted_words_for_idf_file, mode='w', encoding='utf-8') as fp:
#     fp.write(ujson.dumps(counted_idf))

with open(file=from_file, mode='r', encoding='utf-8') as fp:
    with open(file=res_file, mode='w', encoding='utf-8') as resfp:
        for line in fp:
            cur_article_info = ujson.loads(line)

            cur_title = cur_article_info['title']
            cur_text = cur_article_info['text']

            lemmatized_cur_text_words_counter = collections.Counter(cur_article_info['lemmatized_words_counter'])
            lemmatized_cur_title = list(filter(lambda lemma: lemmatized_cur_text_words_counter[lemma] > 0, cur_article_info['lemmatized_title']))
            lemmatized_cur_text_words_count = float(sum(lemmatized_cur_text_words_counter.values()))

            if lemmatized_cur_title:
                tf_cur_title = dict((lemma, lemmatized_cur_text_words_counter[lemma]) for lemma in lemmatized_cur_title)
                for lemma in tf_cur_title:
                    tf_cur_title[lemma] /= lemmatized_cur_text_words_count
                cur_article_info['title_tf'] = tf_cur_title

                idf_cur_title = dict((lemma, counted_words_occurrences_in_all_texts_for_idf[lemma]) for lemma in lemmatized_cur_title)
                for lemma in idf_cur_title:
                    idf_cur_title[lemma] = float(articles_count) / idf_cur_title[lemma]
                cur_article_info['title_idf'] = idf_cur_title

            cur_article_info['words_count'] = lemmatized_cur_text_words_count
            del cur_article_info['lemmatized_words_counter']

            count += 1
            resfp.write(ujson.dumps(cur_article_info))
            resfp.write('\n')

            if count % 20000 == 0:
                print(f"{count} штук записано")

print(f"{count} штук записано")
print(f"{text_not_found_count} штук НЕ записано")
