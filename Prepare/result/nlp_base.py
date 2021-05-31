# -*- coding: utf-8 -*-
from collections import Counter
import re
from time import time
from typing import Dict, List, Set, Tuple
import unicodedata

import pymorphy2
from pymorphy2.shapes import is_punctuation

from Prepare.constants import FULL_NAME_WITH_COMMA_REGEX
from Prepare import help_module

# mystem = pymystem3.Mystem()
pymorph = pymorphy2.MorphAnalyzer()

# titles = [
#     'Пушкин, Александр Сергеевич',
#     'Теистический эволюционизм',
#     'Паскаль, Блез',
#     'Арифмометр',
#     'Эффект Матфея',
#     'Си (язык программирования)',
#     'Криштиану Роналду',
#     'Де Бройль, Луи',
#     'Объектно-ориентированный язык программирования'
# ]

TOKEN_REGEX = re.compile(r'([^\w_-]|[+])', re.UNICODE)


def word_tokenize(text, _split=TOKEN_REGEX.split):
    return [t for t in _split(text) if t]


def lemmatize_word(word: str) -> Tuple[str, str]:
    if not is_punctuation(word) and not word.isspace():
        normal_form = pymorph.parse(word)[0].normal_form
        return word, normal_form

    return word, None


def lemmatize_text(text, log_time=False) -> List[Tuple[str, str]]:
    res = []

    if log_time:
        start_lemmatize_time = time()

    for token in word_tokenize(text):
        if not is_punctuation(token) and not token.isspace():
            normal_form = pymorph.parse(token)[0].normal_form
            res.append((token, normal_form))
        else:
            res.append((token, None))

    if log_time:
        help_module.get_time(start_lemmatize_time, time(), "Лемматизация текста из токенов")

    # lemmatized_text = ''.join(map(lambda x: x[1] if x[1] is not None else x[0], res))
    # print(f'Длина текста до лемматизации - {len(text)}')
    # print(f'Длина текста после лемматизации - {len(lemmatized_text)}')
    # print(f'Длина текста до лемматизации (сплит по пробелу) - {len(text.split(" "))}')
    # print(f'Длина текста после лемматизации (сплит по пробелу) - {len(lemmatized_text.split(" "))}')
    # print(f"Текст после лемматизации - {lemmatized_text}")

    return res


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

# for title in titles:
#     get_search_strings(title)


unaccentify = {
    '́': '',
    '̀': '',
   'ѝ': 'и',
   'о́': 'о'
}
unaccentify = {unicodedata.normalize('NFKC', i):j for i, j in unaccentify.items()}
pattern = re.compile('|'.join(unaccentify))
def unaccentify_text(text):
    def replacer(match):
        return unaccentify[match.group(0)]

    source = unicodedata.normalize('NFKC', text)
    return pattern.sub(replacer, source)


def get_lemma_info(lemma: str) -> Tuple[str, List[str], int, int]:
    splitted_lemma = word_tokenize(lemma)  # list(map(lambda lemma_info: lemma_info[1] if lemma_info[1] is not None else lemma_info[0], lemmatize_text(word)))
    return lemma, splitted_lemma, 0, 0


def find_key_words(key_words_with_infos: Dict[str, Tuple[List[str], int, int]], text: str) -> Set[Tuple[int, int]]:
    cur_candidates = key_words_with_infos
    # print(cur_candidates)
    res_indexes = set()  # {key_word: [] for key_word in key_words}

    # start_time = time()
    analyzed_text = lemmatize_text(text)
    # help_module.get_time(start_time, time(), "Лемматизация текста с помощью PyMorphy2")
    # print(analyzed_text)

    index = 0

    for cur_word_info in analyzed_text:
        cur_word_text = cur_word_info[0]
        cur_word_text_length = len(cur_word_text)
        cur_word_lexeme = None

        if cur_word_info[1] is not None:
            cur_word_lexeme = cur_word_info[1]

        for candidate in cur_candidates:
            candidate_info = cur_candidates[candidate]
            splitted_lexeme = candidate_info[0]
            index_in_splitted_lexeme = candidate_info[1]
            cur_length = candidate_info[2]

            if (cur_word_lexeme == splitted_lexeme[index_in_splitted_lexeme]
                    or cur_word_text == splitted_lexeme[index_in_splitted_lexeme]):
                if index_in_splitted_lexeme == len(splitted_lexeme) - 1:
                    res_indexes.add((index - cur_length, index + cur_word_text_length))
                    cur_candidates[candidate] = (splitted_lexeme, 0, 0)
                else:
                    cur_candidates[candidate] = (splitted_lexeme, index_in_splitted_lexeme + 1, cur_length + cur_word_text_length)
            else:
                cur_candidates[candidate] = (splitted_lexeme, 0, 0)

        index += cur_word_text_length

    # print(res_indexes)
    return res_indexes


# TODO: skip { <UNK>, <PAD> }?
def find_keyword_token_positions_in_bpe(
        keywords_positions_in_original_text: Set[Tuple[int, int]],
        bpe_string_tokens_text: List[str],
        start_index_in_real_text: int = 0
) -> List[int]:
    NOT_KEYWORD_CODE = 0
    KEYWORD_CODE = 1
    keywords_token_positions = []

    # if len(bpe_string_tokens_text) == 0:
    #     print(keywords_positions_in_original_text)

    # print(bpe_string_tokens_text[0], bpe_string_tokens_text[1], bpe_string_tokens_text[2], bpe_string_tokens_text[3], bpe_string_tokens_text[4])

    if bpe_string_tokens_text[0] == '<BOS>':
        keywords_token_positions.append(NOT_KEYWORD_CODE)
        bpe_string_tokens_text = bpe_string_tokens_text[1:]

    cur_index_in_real_text = start_index_in_real_text

    for bpe_string_token in bpe_string_tokens_text:
        if bpe_string_token == '<EOS>':
            keywords_token_positions.append(NOT_KEYWORD_CODE)
            break

        cur_indexes = (cur_index_in_real_text, cur_index_in_real_text + len(bpe_string_token))

        for position_in_real_text in keywords_positions_in_original_text:
            if position_in_real_text[0] <= cur_indexes[0] < position_in_real_text[1] \
                    or cur_indexes[0] <= position_in_real_text[0] < cur_indexes[1]:
                keywords_token_positions.append(KEYWORD_CODE)
                break
        else:
            keywords_token_positions.append(NOT_KEYWORD_CODE)

        cur_index_in_real_text = cur_indexes[1]

    return keywords_token_positions


def rfind(text: str, to_find: List[str]) -> int:
    return max(text.rfind(i) for i in to_find)


CHUNK_END_SYMBOLS = [' ', '<n>']


def get_chunks(text, max_chunk_size=200, step=None):
    chunks_start_indexes = []
    chunks = []

    if step is None:
        step = max_chunk_size // 2

    chunk_start_index = 0
    chunks_start_indexes.append(chunk_start_index)

    while len(text) > max_chunk_size:
        line_length = rfind(text[:max_chunk_size], CHUNK_END_SYMBOLS)

        if line_length <= 0:
            line_length = max_chunk_size

        chunks.append(text[:line_length])

        chunk_start_index = rfind(text[:step], CHUNK_END_SYMBOLS) + 1

        step_add_val = cur_step = step

        while chunk_start_index == 0:
            step_add_val //= 2
            cur_step += step_add_val

            if step_add_val != 0:
                chunk_start_index = rfind(text[:cur_step], CHUNK_END_SYMBOLS) + 1
            else:
                chunk_start_index = line_length

        chunks_start_indexes.append(chunks_start_indexes[len(chunks_start_indexes) - 1] + chunk_start_index)
        text = text[chunk_start_index:]

    if not text.isspace():
        chunks.append(text)

    return chunks, chunks_start_indexes


TOO_MANY_SPACES_OR_NEW_LINES_REGEX = re.compile(r'\n+')


def get_nn_data(tokenizer, entity_to_search, text, text_chunk_size=None, with_new_lines=None):
    text = TOO_MANY_SPACES_OR_NEW_LINES_REGEX.sub('<n>', text)
    if text_chunk_size is not None:
        chunks = get_chunks(text, max_chunk_size=text_chunk_size)
        chunks = chunks[0]
    else:
        raise NotImplementedError

    tokenized_chunks = tokenizer.encode(chunks)
    # help_module.get_time(start_time, time(), "Токенизация текста(chunks)")
    symbol_tokenized_chunks = list(map(lambda chunk: [tokenizer.id_to_subword(token) for token in chunk], tokenized_chunks))
    joined_symbol_tokenized_chunks = list(map(lambda chunk: ''.join(chunk), symbol_tokenized_chunks))
    space_replaced_joined_symbol_tokenized_chunks = list(map(lambda chunk: chunk.replace('▁', ' '), joined_symbol_tokenized_chunks))

    key_words_with_infos = {word_info[0]: (word_info[1], word_info[2], word_info[3]) for word_info in map(get_lemma_info, entity_to_search)}
    all_indexes = list(map(lambda chunk: find_key_words(key_words_with_infos, chunk), space_replaced_joined_symbol_tokenized_chunks))
    # all_indexes = list(map(lambda indexes: help_module.flatten(indexes), res_indexes))

    # non_empty_chunks_with_indexes = list(filter(lambda chunk_and_chunk_indexes: chunk_and_chunk_indexes[1], zip(symbol_tokenized_chunks, all_indexes)))
    non_empty_chunks_with_indexes = [
        (tokenized_chunk, symbol_tokenized_chunk, chunk_indexes)
        for tokenized_chunk, symbol_tokenized_chunk, chunk_indexes
        in zip(tokenized_chunks, symbol_tokenized_chunks, all_indexes) if chunk_indexes
    ]

    if not non_empty_chunks_with_indexes:
        return None

    # token_positions_in_tokenized_chunks = list(map(lambda chunk_with_indexes: find_keyword_token_positions_in_bpe(chunk_with_indexes[1], chunk_with_indexes[0]), non_empty_chunks_with_indexes))
    filtered_tokenized_chunks, filtered_token_positions = zip(*[
        (tokenized_chunk, find_keyword_token_positions_in_bpe(chunk_indexes, symbol_tokenized_chunk))
        for tokenized_chunk, symbol_tokenized_chunk, chunk_indexes
        in non_empty_chunks_with_indexes
    ])

    return filtered_tokenized_chunks, filtered_token_positions


