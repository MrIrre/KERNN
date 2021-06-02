# -*- coding: utf-8 -*-
# import ujson
#
# filename = 'all_nn_data_15000_with_tf-idf-complete'
# from_file = 'result/nn_data/' + filename
#
# res_train_file = 'result/nn_data/result_data/train/' + filename + '_train_shuffled'
# res_val_file = 'result/nn_data/result_data/validation/' + filename + '_val_shuffled'
# res_test_file = 'result/nn_data/result_data/test/' + filename + '_test_shuffled'
#
# with open(file=from_file, mode='r', encoding='utf-8') as f:
#     # for _ in range(2):
#     #     next(f)
#
#     for line in f:
#         cur_json = ujson.loads(line)
#
#         if len(cur_json[0]) != len(cur_json[1]):
#             print(f'INPUT - {len(cur_json[0])}')
#             print(f'ANSWER - {len(cur_json[1])}')
#             print('----------------------------')


# import constants
# import youtokentome as yttm
#
# vocab_size = 15000
#
# tokenizer = yttm.BPE('result/' + constants.BPE_MODEL_FILENAME + '_' + str(vocab_size))
#
# string = """Hello!
# Second line!
# Third LINE!!!
# AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
# """
#
# print('BEGIN:')
# print(string)
# print('-------------------------------')
# codes = tokenizer.encode([string])
# result = tokenizer.decode(codes[0])
# print('AFTER:')
# print(result[0])


import matplotlib.pyplot as plt

a = [[1, 2, 3], [4, 5, 6]]

for i, m in enumerate(a):
    plt.plot(m, label=str(m))
    plt.title(i)
    plt.xlabel('x' + str(i))
    plt.ylabel('y' + str(i))
    plt.legend()
    plt.show()













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
