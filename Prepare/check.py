import youtokentome as yttm

from time import time

from Prepare.result import wiki_downloader, nlp_base, help_module

start_time = time()

page = wiki_downloader.download_page('Пушкин')
page_title = page.displaytitle
to_search = nlp_base.get_search_strings(page_title)
# print(f"SEARCHING IN TEXT -> {to_search}")
text = wiki_downloader.get_page_needed_text(page)


# text = 'Х' + text[1:]
# print(text)

# res_indexes = base.find_key_words(to_search, text)
#
# all_indexes = help_module.flatten(res_indexes.values())
#
# help_module.get_time(start_time, time(), "Поиск позиций ключевой сущности")

#
# for index in all_indexes:
#     begin = index[0]
#     end = index[1]
#
#     print(text[begin:end])

CHUNK_SIZE = 200

chunks = help_module.get_chunks(text, max_chunk_size=CHUNK_SIZE)

# text = 'Мама мыла раму. А.С.Пушкин'

BPE_MODEL_FILENAME = 'result/bpe_models/1_bpe.yttm'
TRAIN_TEXTS_FILENAME = 'result/bpe_models/1_bpe_train.txt'
# texts = [text]
chunks = chunks[0]

help_module.save_texts_to_file(chunks, TRAIN_TEXTS_FILENAME)
yttm.BPE.train(data=TRAIN_TEXTS_FILENAME, vocab_size=300, model=BPE_MODEL_FILENAME)
tokenizer = yttm.BPE(BPE_MODEL_FILENAME)
token_vocab = tokenizer.vocab()
# print(f"Словарь BPE токенов -> {token_vocab}")
# token_vocab_lengths = list(map(len, token_vocab))
# print(f"Длины BPE токенов в словаре -> {token_vocab_lengths}")


tokenized_chunks = tokenizer.encode(chunks)
help_module.get_time(start_time, time(), "Токенизация текста(chunks)")
symbol_tokenized_chunks = list(map(lambda chunk: [tokenizer.id_to_subword(token) for token in chunk], tokenized_chunks))
# print(f"Текст из токенов -> {symbol_tokenized_text}")
joined_symbol_tokenized_chunks = list(map(lambda chunk: ''.join(chunk), symbol_tokenized_chunks))
# print(f"Склеенный из токенов текст -> {joined_symbol_tokenized_text}")
space_replaced_joined_symbol_tokenized_chunks = list(map(lambda chunk: chunk.replace('▁', ' '), joined_symbol_tokenized_chunks))
# print(f"Склеенный из токенов текст, где заменены \"▁\" на пробелы -> {space_replaced_joined_symbol_tokenized_text}")


# cur_candidates = {word_info[0]: (word_info[1], word_info[2], word_info[3]) for word_info in map(base.get_lemma_info, to_search)}
# # print(cur_candidates)
# res_indexes = {key_word: [] for key_word in to_search}

# import pymystem3
# m = pymystem3.Mystem()
# starrrt = time()
# analyzed_text = m.analyze(text)
# help_module.get_time(starrrt, time(), "Анализ текста из токенов1")


cur_candidates = {word_info[0]: (word_info[1], word_info[2], word_info[3]) for word_info in map(nlp_base.get_lemma_info, to_search)}
# print(cur_candidates)
res_indexes = list(map(lambda chunk: nlp_base.find_key_words(to_search, chunk), space_replaced_joined_symbol_tokenized_chunks))
all_indexes = list(map(lambda indexes: help_module.flatten(indexes.values()), res_indexes))
# print(res_indexes)

z = zip(symbol_tokenized_chunks, all_indexes)

token_positions_in_tokenized_chunks = list(map(lambda info: nlp_base.find_keyword_token_positions_in_bpe(info[1], info[0]), z))

help_module.get_time(start_time, time(), "Получение позиций сущности в токенах")

# for info in zip(symbol_tokenized_chunks, token_positions_in_tokenized_chunks):
#     for i in range(len(info[0])):
#         if info[1][i] == 1:
#             print(info[0][i])



vocab_size = len(token_vocab)
# print(vocab_size)
# print(symbol_tokenized_text)
test_input = tokenized_chunks
# print(test_text)
test_answer = token_positions_in_tokenized_chunks
# print(test_answer)
# print(len(test_text))




nn = MyNN(vocab_size, max_in_length=CHUNK_SIZE)
dataset = WikiDataset(chunk_size=CHUNK_SIZE)

from Prepare.result.utils import train_eval_loop


def lr_scheduler(optim):
    return torch.optim.lr_scheduler.ReduceLROnPlateau(optim,
                                                      patience=5,
                                                      factor=0.5,
                                                      verbose=True)

import gc
gc.collect()
torch.cuda.empty_cache()

train_eval_loop(model=nn,
                train_dataset=dataset,
                val_dataset=dataset,
                criterion=F.binary_cross_entropy,
                lr=1e-1,
                epoch_n=200,
                batch_size=64,
                l2_reg_alpha=0,
                lr_scheduler_ctor=lr_scheduler,
                device='cuda',
                shuffle_train=False)


# from torch.utils.data import DataLoader
#
# dataset = WikiDataset(chunk_size=CHUNK_SIZE)
# loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
#
# for batch_x, batch_y in loader:
#     batch_x = batch_x.to('cuda')
#     batch_y = batch_y.to('cuda')
#
#     pred = nn(batch_x)
#     res = batch_y
#     print(res)
#     print("-")
