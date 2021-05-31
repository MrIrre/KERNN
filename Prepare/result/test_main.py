# -*- coding: utf-8 -*-
from time import time

from random import shuffle
import numpy as np
import torch
import torch.nn.functional as F

import youtokentome as yttm

from Prepare import constants
# import ujson
# from wiki_dataset import WikiTextDataset
from wiki_lazy_dataset import WikiTextLazyDataset
from neural_network import MyNN
from utils import train_eval_loop

# start_time = time()

DEVICE = 'cuda'
assert DEVICE in {'cpu', 'cuda'}
yttm.BPE
vocab_size = 15000

tokenizer = yttm.BPE(constants.BPE_MODEL_FILENAME + '_' + str(vocab_size))
# print(tokenizer.vocab())


vocab_size = len(tokenizer.vocab())

filename = 'all_nn_data_15000_with_tf-idf-final_3'
train_file_path = 'nn_data/result_data/train/' + filename + '_train_shuffled'
val_file_path = 'nn_data/result_data/validation/' + filename + '_val_shuffled'
test_file_path = 'nn_data/result_data/test/' + filename + '_test_shuffled'


# all_nn_data = []
# line_index = 0
#
# with open('nn_data/all_nn_data_' + str(vocab_size) + '_with_tf-idf-ready', 'r', encoding='utf-8') as f:
#     for line in f:
#         cur_json = ujson.loads(line)
#         cur_input = cur_json['input']
#         cur_answer = cur_json['answer']
#         all_nn_data.append((cur_input, cur_answer))
#
#         line_index += 1
#         if line_index % 10000 == 0:
#             print(f'Прочитано {line_index} строк')
#
# print('Файл прочитан')

# line_offsets = []
# line_index = 0
# MAX_TEXT_SIZE = 0
# MAX_CHUNK_SIZE = 0
#
# nn_data_filepath = 'nn_data/all_nn_data_' + str(vocab_size) + '_with_tf-idf'
# line_offsets_filepath = 'nn_data/line_offsets/all_nn_data_' + str(vocab_size) + '_with_tf-idf_line_offsets'


# # Сохранить смещения каждой строки в файле nn_data
# with open(nn_data_filepath, 'rb') as f:
#     line_offsets.append(f.tell())
#     for line in f:
#         cur_json = ujson.loads(line)
#
#         if MAX_TEXT_SIZE < len(cur_json['input']):
#             MAX_TEXT_SIZE = len(cur_json['input'])
#
#         cur_chunk_max_len = len(max(cur_json['input'], key=len))
#
#        # if cur_chunk_max_len in {10028, 780, 326, 1285, 1194, 902}:  # TODO: костыль, который починен
#        #     continue
#
#         if MAX_CHUNK_SIZE < cur_chunk_max_len:
#             MAX_CHUNK_SIZE = cur_chunk_max_len
#
#         line_offsets.append(f.tell())
#
#         line_index += 1
#         if line_index % 10000 == 0:
#             print(f'Записаны позиции {line_index} строк')
#
# del line_offsets[-1]
# print(f'Line index = {line_index}')
# assert len(line_offsets) == line_index
#
# with open(file=line_offsets_filepath, mode='w', encoding='utf-8') as f:
#     result = {
#         'line_offsets': line_offsets,
#         'max_chunk_size': MAX_CHUNK_SIZE,
#         'max_text_size': MAX_TEXT_SIZE
#     }
#
#     f.write(ujson.dumps(result))
# # ---------------------------------------------------------------------

# with open(file=line_offsets_filepath, mode='r', encoding='utf-8') as f:
#     result = ujson.loads(f.read())
#     line_offsets = result['line_offsets']
#     MAX_CHUNK_SIZE = result['max_chunk_size']
#     MAX_TEXT_SIZE = result['max_text_size']
#
# print(f'Line offsets length = {len(line_offsets)}')
# print(f'Все позиции строк из файла записаны. Количество строк - {line_index}')

nn = MyNN(vocab_size, embedding_size=64)
print('Количество параметров', sum(np.product(t.shape) for t in nn.parameters()))

train_dataset = WikiTextLazyDataset(filepath=train_file_path)
val_dataset = WikiTextLazyDataset(filepath=val_file_path)


def lr_scheduler(optim):
    return torch.optim.lr_scheduler.ReduceLROnPlateau(optim,
                                                      patience=2,
                                                      factor=0.5,
                                                      verbose=True)


# import gc
#
# gc.collect()
# torch.cuda.empty_cache()


device = torch.device(DEVICE)
nn.load_state_dict(torch.load('./nn_models/nn_model_15000_tf-idf_1622456520.161046.pth'))
# nn.eval()
# nn.to(device)

batch_size = 256
(best_val_loss, best_nn_model) = train_eval_loop(model=nn,
                                                 train_dataset=train_dataset,
                                                 val_dataset=val_dataset,
                                                 criterion=F.binary_cross_entropy,
                                                 lr=0.001,
                                                 epoch_n=10,
                                                 batch_size=batch_size,
                                                 early_stopping_patience=5,
                                                 max_batches_per_epoch_train=train_dataset.line_count // batch_size,
                                                 max_batches_per_epoch_val=val_dataset.line_count // batch_size,
                                                 lr_scheduler_ctor=lr_scheduler,
                                                 device=DEVICE,
                                                 shuffle_train=False)
torch.save(best_nn_model.state_dict(), f'nn_models/nn_model_{str(vocab_size)}_tf-idf_{str(time())}.pth')


del train_dataset
del val_dataset





# from torch.utils.data import DataLoader
# from matplotlib import pyplot as plt
# import utils
# from Prepare import help_module
#
# from sklearn.metrics import classification_report
#
# dataset = WikiTextLazyDataset(filepath=test_file_path)
# loader = DataLoader(dataset, batch_size=256, shuffle=False)
#
# batch_count = 1500  # 3500
#
# y_true = []
# y_pred = []
#
# loss_results = []
# loss_func = F.binary_cross_entropy
#
# for batch_i, (batch_x, batch_y) in enumerate(loader):
#     if batch_i > batch_count:
#         break
#     # print(batch_i)
#     batch_x = utils.copy_data_to_device(batch_x, device)
#     batch_y = utils.copy_data_to_device(batch_y, device)
#
#     pred = nn(batch_x)
#
#     y_pred.append(pred.tolist())
#
#     loss = loss_func(pred, batch_y)
#     loss_results.append(loss.item())
#
#     if batch_i % 100 == 0:
#         print(f'Пройдено {batch_i} батчей')
#
#     # print(pred)
#     # print(res)
#     # print("-")

