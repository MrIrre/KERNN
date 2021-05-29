# -*- coding: utf-8 -*-
from Prepare import constants
from time import time
import math
import ujson
import nlp_base
import os
import youtokentome as yttm


class ContinueOuterLoop(Exception):
    pass


vocab_size = 15000  # 20000 25000 50000

bpe_tokenizer = yttm.BPE(constants.BPE_MODEL_FILENAME + '_' + str(vocab_size))
CHUNK_SIZE = 200

cur_index = start_index = 0
with_answer_in_text = 0
without_answer_in_text = 0

file_with_texts = '../texts_to_train/all-wiki-pages_with_tfidf.json'
result_file = 'nn_data/all_nn_data'+'_'+str(vocab_size)+'_with_tf-idf'

if os.path.exists(result_file):
    print(f'File with name "{result_file}" already exists')
    exit(1)

start_time = time()

try:
    with open(file=file_with_texts, mode='r', encoding='utf-8') as fromfile:
        with open(file=result_file, mode='w', encoding='utf-8') as resfile:
            for _ in range(cur_index):
                next(fromfile)

            for line in fromfile:
                cur_json = ujson.loads(line)

                page_id = cur_json['id']
                page_title = cur_json['title']
                page_text = cur_json['text']
                page_lemmatized_title = cur_json['lemmatized_title']  # Maybe is empty

                entities_to_search = []

                if page_lemmatized_title:
                    # Если есть леммы из заголовка (НЕ факт, что для каждой из них посчитан tf и idf)
                    title_tf_idf = None

                    if 'title_tf' in cur_json:
                        title_tf = cur_json['title_tf']
                        title_idf = cur_json['title_idf']

                        # len(title_tf) == len(title_idf) всегда
                        title_tf_idf = {lemma: title_tf[lemma] * math.log2(title_idf[lemma]) for lemma in title_tf}  # считаем tf_idf

                    entities_to_search = nlp_base.get_search_strings(page_title, page_lemmatized_title, title_tf_idf)
                else:  # Если леммы из заголовка не были взяты
                    entities_to_search = nlp_base.get_search_strings(page_title)

                if not entities_to_search:
                    without_answer_in_text += 1
                    cur_index += 1
                    continue

                try:
                    tokenized_chunks, token_positions = nlp_base.get_nn_data(bpe_tokenizer, entities_to_search, page_text, CHUNK_SIZE)
                except Exception as e:
                    print('GET NN DATA EXCEPTION!!!!!')
                    print(e)
                    print(f'Line {cur_index} failed!!!!!')
                    print(f'С ответом - {with_answer_in_text}')
                    print(f'Без ответа - {without_answer_in_text}')
                    print(f'Page Title - {page_title}')
                    print(f'Page Text - {page_text}')
                    print('-------------------------')
                    continue

                test_input = tokenized_chunks
                test_answer = token_positions

                for chunk in test_answer:
                    if any(chunk):
                        break
                else:
                    without_answer_in_text += 1
                    cur_index += 1
                    continue

                res_dict = {'input': test_input, 'answer': test_answer}
                res_json = ujson.dumps(res_dict)
                resfile.write(res_json)
                resfile.write('\n')

                cur_index += 1
                with_answer_in_text += 1

                if with_answer_in_text % 1000 == 0:
                    print(f'С ответом - {with_answer_in_text}')

                if cur_index % 10000 == 0:
                    print(f'Проанализировано {cur_index - start_index} статей за {time() - start_time} секунд')

except Exception as e:
    print(f'Cycle exception!')
    print(e)
    print(f'Line {cur_index} failed!!!!!')
    print(f'С ответом - {with_answer_in_text}')
    print(f'Без ответа - {without_answer_in_text}')
    print(f'Page Title - {page_title}')
    print(f'Page Text - {page_text}')


print(f'With answer - {with_answer_in_text}')
print(f'Without answer - {without_answer_in_text}')
