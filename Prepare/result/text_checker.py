import constants
from time import time as time
from neural_network import MyNN
import nlp_base
import torch
import youtokentome as yttm


tokenizer = yttm.BPE(constants.BPE_MODEL_FILENAME + '_22000')
vocab_size = len(tokenizer.vocab())
device = torch.device('cuda')
TEXT_CHUNK_SIZE = 200

nn = MyNN(vocab_size, max_in_length=104, embedding_size=64)

nn.load_state_dict(torch.load('./nn_models/nn_model_1621162343.2260022.pth'))
nn.eval()
nn.to(device)

chunks = []
filename = 'temp'
with open(file='check_data/'+filename+'.txt', mode='r', encoding='cp1251') as f:
    text = f.read()
    chunks.append(text)


tokenized_chunks = tokenizer.encode(chunks)
symbol_tokenized_chunks = list(map(lambda chunk: [tokenizer.id_to_subword(token) for token in chunk], tokenized_chunks))
symbol_tokenized_chunks = symbol_tokenized_chunks[0]


input_data = torch.tensor([tokenized_chunks])
input_data = input_data.to(device)
pred = nn.forward(input_data)
pred = pred[0][0]

torch.set_printoptions(profile='default', sci_mode=False)

print(pred)

hue = 0
saturation = 100
lightness = 50

with open('check_data/results/'+filename+'_'+str(time())+'.html', mode='w', encoding='cp1251') as resf:
    resf.write('<html>')

    for i in range(len(pred)):
        cur_pred = pred[i].item()
        token = symbol_tokenized_chunks[i].replace('\u2581', ' ')

        resf.write(f'<span style="color:hsl({hue}, {saturation}%, {cur_pred*100}%);">')
        resf.write(token)
        resf.write('</span>')

    resf.write('</html>')




