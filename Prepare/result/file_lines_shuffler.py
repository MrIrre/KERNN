# -*- coding: utf-8 -*-
import os
import random
import ujson


# def get_max_chunk_length(chunks_strings):
#     return len(max(map(lambda input_with_answer: input_with_answer[0], chunks_strings), key=len))

def get_max_chunk_length(indexes, chunks):
    max_chunk_length = 0
    for i in indexes:
        cur_len = len(ujson.loads(chunks[i])[0])

        if max_chunk_length < cur_len:
            max_chunk_length = cur_len

    return max_chunk_length

# def write_data_in_file(chunks_strings, file_path):
#     with open(file_path, "w") as f:
#         f.write(str(len(chunks_strings)))  # Кол-во чанков
#         f.write('\n')
#         max_chunk_len = get_max_chunk_length(map(ujson.loads, chunks_strings))  # Размер максимального чанка
#         f.write(str(max_chunk_len))
#         f.write('\n')
#         f.writelines(chunks_strings)


def write_data_in_file(ids, chunks, file_path):
    chunk_indexes = []

    for i in range(len(chunks)):
        # chunk = [input, answers, page_id]
        if get_chunk_id(chunks[i]) in ids:
            chunk_indexes.append(i)

    chunks_len = len(chunk_indexes)  # Кол-во чанков
    max_chunk_len = get_max_chunk_length(chunk_indexes, chunks)  # Размер максимального чанка

    with open(file_path, "w") as f:
        f.write(str(chunks_len))
        f.write('\n')
        f.write(str(max_chunk_len))
        f.write('\n')

        for i in chunk_indexes:
            f.write(chunks[i])

    print(f'Записано {len(chunk_indexes)} чанков')


def get_chunk_id(cur_chunk_string):
    return int(cur_chunk_string[(cur_chunk_string.rfind(',') + 1):cur_chunk_string.rfind(']')])


filename = 'all_nn_data_15000_with_tf-idf-complete_3'
newlines = 'with_newlines'
from_file = 'nn_data/' + newlines + '/' + filename
from_file_ids = 'nn_data/' + newlines + '/' + filename + '_ids'

res_train_file = 'nn_data/result_data/train/' + filename + '_train_shuffled'
res_val_file = 'nn_data/result_data/validation/' + filename + '_val_shuffled'
res_test_file = 'nn_data/result_data/test/' + filename + '_test_shuffled'

if os.path.exists(res_train_file):
    print(f"File {res_train_file} is already exists")
    exit(1)

if os.path.exists(res_val_file):
    print(f"File {res_val_file} is already exists")
    exit(1)

if os.path.exists(res_test_file):
    print(f"File {res_test_file} is already exists")
    exit(1)

with open(from_file_ids) as f:
    all_ids = ujson.loads(f.read())

chunks = []
i = 0
with open(from_file) as f:
    for line in f:
        chunks.append(line)

        if i % 50000 == 0:
            print(f'Прочитано {i} чанков')
        i += 1

print(f'Прочитано в итоге {i} чанков')

ALL_SPLIT = int(len(all_ids) * 0.8)
TRAIN_SPLIT = int(ALL_SPLIT * 0.8)

print('До шаффла')
random.shuffle(all_ids)
random.shuffle(chunks)
print('После шаффла')

train_data_ids = set(all_ids[:ALL_SPLIT][:TRAIN_SPLIT])
val_data_ids = set(all_ids[:ALL_SPLIT][TRAIN_SPLIT:])
test_data_ids = set(all_ids[ALL_SPLIT:])

# console_input_text = ''
# while True:
#     print(f'Количество чанков для обучения - {count_chunks(train_data_ids, chunks)}')
#     print(f'Количество чанков для валидации - {count_chunks(val_data_ids, chunks)}')
#     print(f'Количество чанков для теста - {count_chunks(test_data_ids, chunks)}')
#
#     console_input_text = input('Подходящее распределение? (y/n)\n').lower()
#
#     if console_input_text == 'y':
#         break
#
#     random.shuffle(all_ids)
#
#     train_data_ids = all_ids[:ALL_SPLIT][:TRAIN_SPLIT]
#     val_data_ids = all_ids[:ALL_SPLIT][TRAIN_SPLIT:]
#     test_data_ids = all_ids[ALL_SPLIT:]


write_data_in_file(train_data_ids, chunks, res_train_file)
write_data_in_file(val_data_ids, chunks, res_val_file)
write_data_in_file(test_data_ids, chunks, res_test_file)
