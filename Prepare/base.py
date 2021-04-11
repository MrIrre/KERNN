import re
import pymorphy2
from pymorphy2.shapes import is_punctuation

from typing import List, Dict, Tuple
import unicodedata

from time import time

import help_module

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


FULL_NAME_WITH_COMMA_REGEX = re.compile("^([а-яА-Я -]+), (\w+) ?(\w+)?$")


def get_search_strings(title):
    to_search = []

    word_infos = lemmatize_text(title)
    words = list(filter(lambda x: x[1] if x[1] is not None else x[0], word_infos))

    full_name_res = FULL_NAME_WITH_COMMA_REGEX.findall(title)

    if full_name_res:
        surname, first_name, patronymic = full_name_res[0]

        to_search.append(surname)
        to_search.append(first_name)
        to_search.append(f"{first_name} {surname}")
        to_search.append(f"{surname} {first_name}")

        if patronymic:
            to_search.append(f"{first_name} {patronymic}")
            to_search.append(f"{surname} {first_name} {patronymic}")
            to_search.append(f"{first_name} {patronymic} {surname}")
            to_search.append(f"{first_name[0]}. {patronymic[0]}. {surname}")
    else:
        print(words)
        # TODO: Think about it

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


def get_lemma_info(word: str) -> Tuple[str, List[str], int, int]:
    splitted_lemma = list(map(lambda lemma_info: lemma_info[1] if lemma_info[1] is not None else lemma_info[0], lemmatize_text(word)))
    return word, splitted_lemma, 0, 0


def find_key_words(key_words: List[str], text: str) -> Dict[str, List[Tuple[int, int]]]:
    cur_candidates = {word_info[0]: (word_info[1], word_info[2], word_info[3]) for word_info in map(get_lemma_info, key_words)}
    # print(cur_candidates)
    res_indexes = {key_word: [] for key_word in key_words}

    start_time = time()
    analyzed_text = lemmatize_text(text, log_time=True)
    help_module.get_time(start_time, time(), "Лемматизация текста с помощью PyMorphy2")
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
                    res_indexes[candidate].append((index - cur_length, index + cur_word_text_length))
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
        keywords_positions_in_original_text: List[Tuple[int, int]],
        bpe_string_tokens_text: List[str],
        start_index_in_real_text: int = 0
) -> List[int]:
    NOT_KEYWORD_CODE = 0
    KEYWORD_CODE = 1
    keywords_token_positions = []

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
