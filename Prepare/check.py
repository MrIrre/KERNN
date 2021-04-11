import youtokentome as yttm
import matplotlib.pyplot as plt
import numpy as np

from time import time

import help_module

import wikipediaapi
import auxiliary
import base
import wiki_helper

start_time = time()

page = wiki_helper.download_page('Пушкин')
page_title = page.displaytitle
to_search = base.get_search_strings(page_title)
# print(f"SEARCHING IN TEXT -> {to_search}")
text = wiki_helper.get_page_needed_text(page)


# text = 'Х' + text[1:]
# print(text)

# res_indexes = base.find_key_words(to_search, text)
#
# all_indexes = help_module.flatten(res_indexes.values())
#
# help_module.get_time(start_time, time(), "Поиск позиций ключевой сущности")

#
# for index in all_indexes:
#     begin = index[0]
#     end = index[1]
#
#     print(text[begin:end])

# text = 'Мама мыла раму. А.С.Пушкин'

BPE_MODEL_FILENAME = './models/1_bpe.yttm'
TRAIN_TEXTS_FILENAME = './models/1_bpe_train.txt'
texts = [text]

help_module.save_texts_to_file(texts, TRAIN_TEXTS_FILENAME)
yttm.BPE.train(data=TRAIN_TEXTS_FILENAME, vocab_size=300, model=BPE_MODEL_FILENAME)
tokenizer = yttm.BPE(BPE_MODEL_FILENAME)
token_vocab = tokenizer.vocab()
# print(f"Словарь BPE токенов -> {token_vocab}")
token_vocab_lengths = list(map(len, token_vocab))
# print(f"Длины BPE токенов в словаре -> {token_vocab_lengths}")


tokenized_text = tokenizer.encode(texts)[0]
help_module.get_time(start_time, time(), "Токенизация текста")
symbol_tokenized_text = [tokenizer.id_to_subword(token) for token in tokenized_text]
# print(f"Текст из токенов -> {symbol_tokenized_text}")
joined_symbol_tokenized_text = ''.join(symbol_tokenized_text)
# print(f"Склеенный из токенов текст -> {joined_symbol_tokenized_text}")
space_replaced_joined_symbol_tokenized_text = joined_symbol_tokenized_text.replace('▁', ' ')
# print(f"Склеенный из токенов текст, где заменены \"▁\" на пробелы -> {space_replaced_joined_symbol_tokenized_text}")


# cur_candidates = {word_info[0]: (word_info[1], word_info[2], word_info[3]) for word_info in map(base.get_lemma_info, to_search)}
# # print(cur_candidates)
# res_indexes = {key_word: [] for key_word in to_search}

# import pymystem3
# m = pymystem3.Mystem()
# starrrt = time()
# analyzed_text = m.analyze(text)
# help_module.get_time(starrrt, time(), "Анализ текста из токенов1")



cur_candidates = {word_info[0]: (word_info[1], word_info[2], word_info[3]) for word_info in map(base.get_lemma_info, to_search)}
# print(cur_candidates)
res_indexes = base.find_key_words(to_search, space_replaced_joined_symbol_tokenized_text)
all_indexes = help_module.flatten(res_indexes.values())
# print(res_indexes)

NOT_KEYWORD_CODE = 0
KEYWORD_CODE = 1
keywords_token_positions = []

token_positions_for_bpe_texts = base.find_keyword_token_positions_in_bpe(all_indexes, symbol_tokenized_text)

help_module.get_time(start_time, time(), "Получение позиций сущности в токенах")

# for token_info in zip(symbol_tokenized_text, token_positions_for_bpe_texts):
#     if token_info[1] == 1:
#         print(token_info)

# for index in all_indexes:
#     begin = index[0]
#     end = index[1]
#     print(space_replaced_joined_symbol_tokenized_text[begin:end])





# m = pymystem3.Mystem()

# print(pymorph.parse("«Современник»"))

# splitted_text = text.split(' ')

# word_regexp = re.compile('[a-zа-яё\d-]+', re.IGNORECASE)

# with open('temp', 'w', encoding='utf-8') as f:
#     f.write(text)
#
# help_module.get_time(starrrt, time(), "Анализ текста из токенов")






# for split_piece in space_replaced_joined_symbol_tokenized_text.split(' '):
#     # print(split_piece)
#     if word_regexp.match(split_piece):
#         # print(split_piece)
#         r = pymorph.parse(split_piece)
#         res.append((split_piece, r[0].normal_form))
#
#     else:
#         words_in_split_piece = list(word_regexp.finditer(split_piece))
#         if words_in_split_piece:
#             lemmatized_split_piece = ''
#             index = 0
#             for word_find_info in words_in_split_piece:
#                 found_word = word_find_info[0]
#                 start_idx = word_find_info.start()
#                 end_idx = word_find_info.end()
#                 r = pymorph.parse(found_word)
#                 normal_form = r[0].normal_form
#                 lemmatized_split_piece += split_piece[index:start_idx] + normal_form
#                 index = end_idx
#             lemmatized_split_piece += split_piece[index:]
#             # print(words_in_split_piece)
#             res.append((split_piece, lemmatized_split_piece))
#         else:
#             res.append((split_piece, None))





# index = 0
#
# for cur_word_info in analyzed_text:
#     cur_word_text = cur_word_info['text']
#     cur_word_text_length = len(cur_word_text)
#     cur_word_lexeme = None
#
#     if 'analysis' in cur_word_info and cur_word_info['analysis'] and 'lex' in cur_word_info['analysis'][0]:
#         cur_word_lexeme = cur_word_info['analysis'][0]['lex']
#
#     for candidate in cur_candidates:
#         candidate_info = cur_candidates[candidate]
#         splitted_lexeme = candidate_info[0]
#         index_in_splitted_lexeme = candidate_info[1]
#         cur_length = candidate_info[2]
#
#         if (cur_word_lexeme == splitted_lexeme[index_in_splitted_lexeme]
#                 or cur_word_text == splitted_lexeme[index_in_splitted_lexeme]):
#             if index_in_splitted_lexeme == len(splitted_lexeme) - 1:
#                 res_indexes[candidate].append((index - cur_length, index + cur_word_text_length))
#                 cur_candidates[candidate] = (splitted_lexeme, 0, 0)
#             else:
#                 cur_candidates[candidate] = (splitted_lexeme, index_in_splitted_lexeme + 1, cur_length + cur_word_text_length)
#         else:
#             cur_candidates[candidate] = (splitted_lexeme, 0, 0)
#
#     index += cur_word_text_length
#
# print(res_indexes)


# actual = joined_symbol_tokenized_text[1:].replace('▁', ' ')
# expected = text
# print(actual)
# print(expected.replace('\n', ' '))



# max_chunk_size = 200
# step = max_chunk_size // 2  # TODO: Можем перепрыгнуть какое-то слово, но очень маловероятно
#
# start_time_before_chunks = time()
# train_chunks, chunks_start_indexes = help_module.get_chunks(text, max_chunk_size=200, step=step)
# print(f"Стартовые позиции чанков - {chunks_start_indexes}")
# # print(text)
# print(f"Количество чанков = {len(train_chunks)}")
# help_module.get_time(start_time_before_chunks, time(), "Разбиение на чанки")


# print(train_texts[100])

# BPE_MODEL_FILENAME = './models/1_bpe.yttm'
# TRAIN_TEXTS_FILENAME = './models/1_bpe_train.txt'
#
# help_module.save_texts_to_file(train_chunks, TRAIN_TEXTS_FILENAME)
# yttm.BPE.train(data=TRAIN_TEXTS_FILENAME, vocab_size=300, model=BPE_MODEL_FILENAME)
# tokenizer = yttm.BPE(BPE_MODEL_FILENAME)
# print(' '.join(tokenizer.vocab()))
# print(tokenizer.encode(chunks))

# train_token_ids = tokenizer.encode(train_chunks, bos=True, eos=True)

# plt.hist([len(sent) for sent in train_token_ids], bins=30)
# plt.title('Распределение длин фрагментов в токенах')
# plt.yscale('log')
# plt.show()
#
# token_counts = np.bincount([token_id for text in train_token_ids for token_id in text])
# plt.hist(token_counts, bins=100)
# plt.title('Распределение количества упоминаний токенов')
# # plt.yscale('log')
# plt.show()

# res_indexes = base.find_key_words(to_search, text)
# all_indexes = help_module.flatten(res_indexes.values())

# for index in all_indexes:
#     begin = index[0]
#     end = index[1]
#
#     print(text[begin:end])

# bpe_texts_to_test = train_token_ids
# bpe_string_token_texts_to_test = list(map(lambda bpe_text: list(map(lambda token: tokenizer.id_to_subword(token), bpe_text)), bpe_texts_to_test))




# print(bpe_texts_to_test)
# print(bpe_string_tokens_text_to_test)

# token_positions_for_bpe_texts = map(
#     lambda enum_elem: base.find_keyword_token_positions_in_bpe(
#         all_indexes,
#         enum_elem[0],
#         start_index_in_real_text=enum_elem[1]
#     ),
#     zip(bpe_string_token_texts_to_test, chunks_start_indexes)
# )
#
# for bpe_string_token_text, token_positions in zip(bpe_string_token_texts_to_test, token_positions_for_bpe_texts):
#     for token_info in zip(bpe_string_token_text, token_positions):
#         if token_info[1] == 1:
#             print(token_info)
#
#     print("-------------------------------------")
