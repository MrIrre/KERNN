# -*- coding: utf-8 -*-
from Prepare import constants
from time import time as time
from neural_network import MyNN
import torch
import youtokentome as yttm


vocab_size = 15000

tokenizer = yttm.BPE(constants.BPE_MODEL_FILENAME + '_' + str(vocab_size))
vocab_size = len(tokenizer.vocab())
device = torch.device('cuda')
TEXT_CHUNK_SIZE = 200

nn = MyNN(vocab_size, embedding_size=64)

nn.load_state_dict(torch.load('./nn_models/nn_model_15000_tf-idf_1622470285.0365074.pth'))
nn.eval()
nn.to(device)

chunks = []
filename = 'temp'
encoding = 'utf-8'

try:
    with open(file='check_data/'+filename+'.txt', mode='r', encoding='utf-8') as f:
        text = f.read()
except UnicodeDecodeError:
    with open(file='check_data/'+filename+'.txt', mode='r', encoding='cp1251') as f:
        encoding = 'cp1251'
        text = f.read()

chunks.append(text)
tokenized_chunks = tokenizer.encode(chunks)
symbol_tokenized_chunks = list(map(lambda chunk: [tokenizer.id_to_subword(token) for token in chunk], tokenized_chunks))
symbol_tokenized_chunks = symbol_tokenized_chunks[0]


input_data = torch.tensor(tokenized_chunks)
input_data = input_data.to(device)
pred = nn.forward(input_data)
pred = pred[0]

torch.set_printoptions(profile='default', sci_mode=False)

# print(pred)

hue = 0
saturation = 100
lightness = 50

with open('check_data/results/'+filename+'_'+str(time())+'.html', mode='w', encoding=encoding) as resf:
    resf.write('<html>')

    for i in range(len(pred)):
        cur_pred = pred[i].item()
        token = symbol_tokenized_chunks[i].replace('\u2581', ' ')

        resf.write(f'<span style="color:hsl({hue}, {saturation}%, {cur_pred*100}%);">')
        resf.write(token)
        resf.write('</span>')

    resf.write('</html>')




