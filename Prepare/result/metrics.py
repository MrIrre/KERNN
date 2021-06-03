# -*- coding: utf-8 -*-
from Prepare import constants
from time import time
from neural_network import MyNN
import numpy as np
import youtokentome as yttm
import torch
from torch.utils.data import DataLoader
from wiki_lazy_dataset import WikiTextLazyDataset
import utils
import matplotlib.pyplot as plt
from statistics import mean


DEVICE = 'cuda'
assert DEVICE in {'cpu', 'cuda'}
device = torch.device(DEVICE)

vocab_size = 15000
tokenizer = yttm.BPE(constants.BPE_MODEL_FILENAME + '_' + str(vocab_size))

filename = 'all_nn_data_15000_with_tf-idf-complete_1'
test_file_path = 'nn_data/result_data/test/' + filename + '_test_shuffled'
test_dataset = WikiTextLazyDataset(filepath=test_file_path)

nn = MyNN(vocab_size, layers_num=7, embedding_size=64)
print('Количество параметров', sum(np.product(t.shape) for t in nn.parameters()))

nn.to(device)
nn.load_state_dict(torch.load('./nn_models/nn_model_15000_tf-idf_1622579233.316039.pth'))

batch_size = 256
batch_count = test_dataset.line_count // batch_size
print(f'Количество батчей всего - {batch_count}')

loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

THRESHOLD = 0.25
# VALUE_TO_REPLACE = 0
# threshold_func = torch.nn.Threshold(THRESHOLD, VALUE_TO_REPLACE)

start_time = time()

stat_0 = np.zeros(10001)
stat_1 = np.zeros(10001)

for batch_i, (batch_x, batch_y) in enumerate(loader):
    if batch_i > batch_count:
        break

    batch_x = utils.copy_data_to_device(batch_x, device)
    y_true = batch_y.flatten()
    y_pred = nn(batch_x).to('cpu').flatten()

    for label, predict in zip(y_true, y_pred):
        label = label.item()
        predict = predict.item()
        if label == 0:
            stat_0[int(predict * 10000)] += 1
        else:
            stat_1[int(predict * 10000)] += 1

    if batch_i % 100 == 0:
        print(f'Пройдено {batch_i} батчей за {time() - start_time} секунд')

print(f'Пройдено {batch_i - 1} батчей за {time() - start_time} секунд')

tn = stat_0.cumsum()
fp = stat_1.cumsum()
fn = stat_0[::-1].cumsum()[::-1]
tp = stat_1[::-1].cumsum()[::-1]

print(f'True Positive = {tp}')
print(f'True Negative = {tn}')
print(f'False Positive = {fp}')
print(f'False Negative = {fn}')

precision = tp / (tp + fp)
recall = tp / (tp + fn)
f1_score = 2 * (recall * precision) / (recall + precision)

print(f'Precision = {precision}')
print(f'Recall = {recall}')
print(f'F1 Score = {f1_score}')

plt.plot(np.arange(10001) / 10000, precision)
# plt.xlabel('')
plt.ylabel('Precision')
plt.title('Precision')
plt.show()

plt.plot(np.arange(10001) / 10000, recall)
# plt.xlabel('')
plt.ylabel('Recall')
plt.title('Recall')
plt.show()

plt.plot(np.arange(10001) / 10000, f1_score)
# plt.xlabel('')
plt.ylabel('F1-Score')
plt.title('F1-Score')
plt.show()

plt.plot(precision, recall)
plt.xlabel('Precision')
plt.ylabel('Recall')
# plt.title('Precision/Recall')
plt.show()


