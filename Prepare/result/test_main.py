# -*- coding: utf-8 -*-
from time import time

from random import shuffle
import numpy as np
import torch
import torch.nn.functional as F

import youtokentome as yttm

from Prepare import constants
import json
from wiki_dataset import WikiTextDataset
from neural_network import MyNN
from utils import train_eval_loop

# start_time = time()

DEVICE = 'cuda'
assert DEVICE in {'cpu', 'cuda'}

vocab_size = 50000

tokenizer = yttm.BPE(constants.BPE_MODEL_FILENAME + '_' + str(vocab_size))
# print(tokenizer.vocab())


vocab_size = len(tokenizer.vocab())
all_nn_data = []

with open('nn_data/all_nn_data_' + str(vocab_size), 'r', encoding='utf-8') as f:
    for line in f:
        cur_json = json.loads(line)
        cur_input = cur_json['input']
        cur_answer = cur_json['answer']
        all_nn_data.append((cur_input, cur_answer))


MAX_TEXT_SIZE = max(map(lambda text: len(text[0]), all_nn_data))  # размер самого большого текста
MAX_CHUNK_SIZE = max(map(lambda cur_data: max(map(lambda chunk: len(chunk), cur_data[0])),
                         all_nn_data))  # размер самого большого чанка среди всех текстов
nn = MyNN(vocab_size, embedding_size=64)
print('Количество параметров', sum(np.product(t.shape) for t in nn.parameters()))

ALL_SPLIT = int(len(all_nn_data) * 0.8)
TRAIN_SPLIT = int(ALL_SPLIT * 0.8)

shuffle(all_nn_data)

train_data = all_nn_data[:ALL_SPLIT][:TRAIN_SPLIT]
val_data = all_nn_data[:ALL_SPLIT][TRAIN_SPLIT:]
test_data = all_nn_data[ALL_SPLIT:]


print('Размер обучающей выборки (в страницах)', len(train_data))
print('Размер валидационной выборки (в страницах)', len(val_data))
print('Размер тестовой выборки (в страницах)', len(test_data))

# help_module.get_time(start_time, time(), "Получение позиций сущности в токенах")

train_dataset = WikiTextDataset(data=train_data, text_size=MAX_TEXT_SIZE, chunk_size=MAX_CHUNK_SIZE)
val_dataset = WikiTextDataset(data=val_data, text_size=MAX_TEXT_SIZE, chunk_size=MAX_CHUNK_SIZE)


def lr_scheduler(optim):
    return torch.optim.lr_scheduler.ReduceLROnPlateau(optim,
                                                      patience=2,
                                                      factor=0.5,
                                                      verbose=True)


# import gc
#
# gc.collect()
# torch.cuda.empty_cache()


(best_val_loss, best_nn_model) = train_eval_loop(model=nn,
                                                 train_dataset=train_dataset,
                                                 val_dataset=val_dataset,
                                                 criterion=F.binary_cross_entropy,
                                                 lr=0.001,
                                                 epoch_n=10,
                                                 batch_size=64,
                                                 early_stopping_patience=5,
                                                 max_batches_per_epoch_train=150,
                                                 max_batches_per_epoch_val=50,
                                                 lr_scheduler_ctor=lr_scheduler,
                                                 device=DEVICE,
                                                 shuffle_train=True)
torch.save(best_nn_model.state_dict(), f'nn_models/nn_model_{str(vocab_size)}_{str(time())}.pth')


device = torch.device(DEVICE)

# nn.load_state_dict(torch.load('./nn_models/nn_model_1621162343.2260022.pth'))
# nn.eval()
# nn.to(device)


from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
import utils

shuffle(all_nn_data)

dataset = WikiTextDataset(test_data, text_size=MAX_TEXT_SIZE, chunk_size=MAX_CHUNK_SIZE)
loader = DataLoader(dataset, batch_size=16, shuffle=False)

loss_results = []
loss_func = F.binary_cross_entropy

for batch_i, (batch_x, batch_y) in enumerate(loader):
    # print(batch_i)
    batch_x = utils.copy_data_to_device(batch_x, device)
    batch_y = utils.copy_data_to_device(batch_y, device)

    pred = nn(batch_x)

    loss = loss_func(pred, batch_y)
    loss_results.append(loss.item())
    # print(pred)
    # print(res)
    # print("-")

plt.plot(loss_results)
plt.ylabel('Loss values')
plt.show()

