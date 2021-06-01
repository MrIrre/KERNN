# -*- coding: utf-8 -*-
from Prepare import constants
from time import time
from itertools import repeat
import math
import ujson
import nlp_base
import os
import youtokentome as yttm


class ContinueOuterLoop(Exception):
    pass


vocab_size = 15000  # 20000 25000 50000

bpe_tokenizer = yttm.BPE(constants.BPE_MODEL_FILENAME + '_' + str(vocab_size))
CHUNK_SIZE = 450

cur_index = start_index = 750000
written_chunks = 0
with_answer_in_text = 0
without_answer_in_text = 0
with_exception = 0
count_to_prepare = 1500000

file_with_texts = '../texts_to_train/all-wiki-pages_with_tfidf.json'

nn_data_folder = 'with_newlines'
result_file = 'nn_data/'+nn_data_folder+'/all_nn_data'+'_'+str(vocab_size)+'_with_tf-idf-complete_3'
result_file_for_ids = result_file + '_ids'

if os.path.exists(result_file):
    print(f'File with name "{result_file}" already exists')
    exit(1)

if os.path.exists(result_file_for_ids):
    print(f'File with name "{result_file_for_ids}" already exists')
    exit(1)

start_time = time()

all_ids = []

try:
    with open(file=file_with_texts, mode='r', encoding='utf-8') as fromfile:
        with open(file=result_file, mode='w', encoding='utf-8') as resfile:
            for _ in range(cur_index):
                next(fromfile)

            for line in fromfile:
                cur_json = ujson.loads(line)

                page_id = int(cur_json['id'])
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
                    nn_data = nlp_base.get_nn_data(bpe_tokenizer, entities_to_search, page_text, CHUNK_SIZE)
                    if nn_data is None:
                        without_answer_in_text += 1
                        cur_index += 1
                        continue
                except Exception as e:
                    print('GET NN DATA EXCEPTION!!!!!')
                    print(e)
                    print(f'Line {cur_index} failed!!!!!')
                    print(f'Id = {page_id}')
                    print(f'С ответом - {with_answer_in_text}')
                    print(f'Без ответа - {without_answer_in_text}')
                    try:
                        print(f'Page Title - {page_title}')
                        print(f'Page Text - {page_text}')
                    except UnicodeEncodeError:
                        pass

                    print(f'Записано {written_chunks} чанков')
                    print(f'Записано {len(all_ids)} айдишников')
                    print('-------------------------')
                    with_exception += 1
                    cur_index += 1
                    continue

                tokenized_chunks, token_positions = nn_data[0], nn_data[1]

                # test_input = tokenized_chunks
                # test_answer = token_positions

                # for chunk in test_answer:
                #     if any(chunk):
                #         break
                # else:
                #     without_answer_in_text += 1
                #     cur_index += 1
                #     continue

                # res_dict = {'input': test_input, 'answer': test_answer}
                # res_json = ujson.dumps(res_dict)

                chunks_with_positions = list(zip(tokenized_chunks, token_positions, repeat(page_id)))  # Добавляем к каждому чанку id, для дальнейшего шаффла

                for chunk in chunks_with_positions:
                    resfile.write("%s\n" % ujson.dumps(chunk))
                    written_chunks += 1

                all_ids.append(page_id)

                cur_index += 1
                with_answer_in_text += 1

                if with_answer_in_text % 2000 == 0:
                    print(f'С ответом - {with_answer_in_text}')

                if cur_index % 10000 == 0:
                    print(f'Проанализировано {cur_index - start_index} статей за {time() - start_time} секунд')
                    print(f'Записано {written_chunks} чанков')

                if cur_index > count_to_prepare:
                    break

        print(f'Записываем все id в отдельный файл - {time() - start_time}')
        with open(file=result_file_for_ids, mode='w', encoding='utf-8') as res_ids_file:
            res_ids_file.write(ujson.dumps(all_ids))
        print(f'Записали все id в отдельный файл - {time() - start_time}')


except Exception as e:
    print(f'!!! Cycle exception !!!')
    print(e)
    print(f'Line {cur_index} failed!!!!!')
    print(f'С ответом - {with_answer_in_text}')
    print(f'Без ответа - {without_answer_in_text}')
    print(f'С ошибкой - {with_exception}')
    print(f'Page Title - {page_title}')
    print(f'Page Text - {page_text}')
    print(f'Записано {written_chunks} чанков')
    print(f'Записано {len(all_ids)} айдишников')
except KeyboardInterrupt as e:
    print('!!! Keyboard Interrupt !!!')
    print(e)
    print(f'Line {cur_index} failed!!!!!')
    print(f'С ответом - {with_answer_in_text}')
    print(f'Без ответа - {without_answer_in_text}')
    print(f'С ошибкой - {with_exception}')
    print(f'Записано {written_chunks} чанков')
    print(f'Записано {len(all_ids)} айдишников')

print(f'Проанализировано {cur_index - start_index} статей за {time() - start_time} секунд')
print(f'With answer - {with_answer_in_text}')
print(f'Without answer - {without_answer_in_text}')
print(f'With exception - {with_exception}')
print(f'Записано {written_chunks} чанков')
print(f'Записано {len(all_ids)} айдишников')
