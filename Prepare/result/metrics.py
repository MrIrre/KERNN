# -*- coding: utf-8 -*-
from Prepare import constants
from neural_network import MyNN
import numpy as np
import youtokentome as yttm
import torch
from torch.utils.data import DataLoader
from wiki_lazy_dataset import WikiTextLazyDataset
from sklearn.metrics import f1_score, precision_score, recall_score
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
nn.load_state_dict(torch.load('./nn_models/nn_model_15000_tf-idf_1622552114.0006692.pth'))

batch_size = 256
batch_count = test_dataset.line_count // batch_size
print(f'Количество батчей всего - {batch_count}')

loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# THRESHOLD = 0.3
# VALUE_TO_REPLACE = 0
# threshold_func = torch.nn.Threshold(THRESHOLD, VALUE_TO_REPLACE)

for cur_threshold in np.arange(0.2, 0.3, 0.01):
    print(f'--------- Текущий Threshold = {cur_threshold} ---------')
    precision_values = []
    recall_values = []
    f1_score_values = []

    for batch_i, (batch_x, batch_y) in enumerate(loader):
        if batch_i > batch_count:
            break

        batch_x = utils.copy_data_to_device(batch_x, device)
        y_true = batch_y.flatten()

        pred = nn(batch_x)
        y_pred = (pred >= cur_threshold).float().to('cpu').flatten()

        cur_precision = precision_score(y_true, y_pred, average="binary")
        cur_recall = recall_score(y_true, y_pred, average="binary")
        cur_f1_score = f1_score(y_true, y_pred, average="binary")

        precision_values.append(cur_precision)
        recall_values.append(cur_recall)
        f1_score_values.append(cur_f1_score)

        if batch_i % 1000 == 0:
            print(f'Пройдено {batch_i} батчей')

    print(f'Пройдено {batch_i} батчей')
    print(f'Средний Precision = {mean(precision_values)}')
    print(f'Средний Recall = {mean(recall_values)}')
    print(f'Средний F1 Score = {mean(f1_score_values)}')

    plt.plot(precision_values, label='Precision')
    plt.plot(recall_values, label='Recall')
    plt.plot(f1_score_values, label='F1-Score')

    plt.xlabel('Batch')
    plt.ylabel('Metrics values')
    plt.title(f'Test metrics with threshold = {cur_threshold}')
    plt.legend()
    plt.figure(figsize=(19.8, 10.8))
    plt.show()

    print('------------------ Следующий Threshold ------------------')

