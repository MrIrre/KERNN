from typing import List, Dict

import wikipediaapi
import pymystem3
import auxiliary
import base
import re

import functools
import operator


def flatten(a):
    return functools.reduce(operator.iconcat, a, [])


mystem = pymystem3.Mystem()
wiki = wikipediaapi.Wikipedia(language='ru')

page = wiki.page('Пушкин')
page_title = page.displaytitle
print(f'Page display title -> {page_title}')

to_search = base.get_search_strings(page_title)
print(to_search)

text = page.section_by_title(page.sections[0].sections[0].title).text
print(text)
lemmatized_text = base.lemmatize(text)
print(lemmatized_text)

print('-----------------')


def get_lemma_info(word: str) -> (str, List[str]):
    splitted_lemma = base.to_lemmas(word)[:-1]
    return word, splitted_lemma, 0, 0


def find_key_words(key_words: List[str], text: str) -> Dict[str, List[(int, int)]]:
    cur_candidates = {word_info[0]: (word_info[1], word_info[2], word_info[3]) for word_info in map(get_lemma_info, key_words)}
    # print(cur_candidates)
    res_indexes = {key_word: [] for key_word in key_words}

    analyzed_text = base.analyze(text)
    # print(analyzed_text)

    index = 0

    for cur_word_info in analyzed_text:
        cur_word_text = cur_word_info['text']
        cur_word_text_length = len(cur_word_text)
        cur_word_lexeme = None

        if 'analysis' in cur_word_info and cur_word_info['analysis'] and 'lex' in cur_word_info['analysis'][0]:
            cur_word_lexeme = cur_word_info['analysis'][0]['lex']

        for candidate in cur_candidates:
            candidate_info = cur_candidates[candidate]
            splitted_lexeme = candidate_info[0]
            index_in_splitted_lexeme = candidate_info[1]
            cur_length = candidate_info[2]

            if (cur_word_lexeme == splitted_lexeme[index_in_splitted_lexeme]
                    or cur_word_text == splitted_lexeme[index_in_splitted_lexeme]):
                if index_in_splitted_lexeme == len(splitted_lexeme) - 1:
                    res_indexes[candidate].append((index - cur_length, index + cur_word_text_length))
                    cur_candidates[candidate] = (splitted_lexeme, 0, 0)
                else:
                    cur_candidates[candidate] = (splitted_lexeme, index_in_splitted_lexeme + 1, cur_length + cur_word_text_length)
            else:
                cur_candidates[candidate] = (splitted_lexeme, 0, 0)

        index += cur_word_text_length






        # index += len(lem_info['text'])

# for search_text in to_search:
#     if search_text not in res_indexes:
#         res_indexes[search_text] = []
#
#     for i in range(len(analyzed_text)):
#         word_info = analyzed_text[i]
#         if 'analysis' in word_info and word_info['analysis'] and word_info['analysis'][0]['lex'] == search_text:
#             res_indexes[search_text].append(i)

    print(res_indexes)

    return res_indexes

print('-----------------')

res_indexes = find_key_words(to_search, text)

all_indexes = flatten(res_indexes.values())
print(all_indexes)

for index in all_indexes:
    begin = index[0]
    end = index[1]

    print(text[begin:end])




# lemmatized = mystem.lemmatize(page.text)
# print(''.join(lemmatized))
