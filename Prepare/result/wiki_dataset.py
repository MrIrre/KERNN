import torch
from torch.utils.data import Dataset, IterableDataset


def ensure_length(chunk, out_len, pad_value):
    if len(chunk) < out_len:
        chunk = list(chunk) + [pad_value] * (out_len - len(chunk))
    else:
        chunk = chunk[:out_len]
    return chunk


def ensure_texts_length(text, text_out_len, chunk_out_len, pad_value, dtype):
    res = torch.full(size=(text_out_len, chunk_out_len), fill_value=pad_value, dtype=dtype)

    for chunk_i, chunk in enumerate(text):
        for token_i, token in enumerate(chunk):
            res[chunk_i, token_i] = token

    return res


# DATASET_TYPE = IterableDataset
DATASET_TYPE = Dataset


class WikiTextDataset(DATASET_TYPE):
    def __init__(self, data, text_size, chunk_size, pad_input_value=0, pad_answer_value=0):
        self.data = data
        self.text_size = text_size
        self.chunk_size = chunk_size
        self.pad_input_value = pad_input_value
        self.pad_answer_value = pad_answer_value

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        cur_item = self.data[item]
        cur_input, cur_answer = ensure_texts_length(cur_item[0], self.text_size, self.chunk_size, self.pad_input_value, torch.long), \
                                ensure_texts_length(cur_item[1], self.text_size, self.chunk_size, self.pad_answer_value, torch.float)

        return cur_input, cur_answer

    # def __getitem__(self, item):
    #     cur_item = self.data[item]
    #     cur_input, cur_answer = cur_item[0], cur_item[1]
    #
    #     cur_input = list(map(lambda chunk: ensure_length(chunk, self.chunk_size, self.pad_input_value), cur_input))
    #     cur_input = torch.tensor(cur_input, dtype=torch.long)
    #
    #     cur_answer = list(map(lambda chunk: ensure_length(chunk, self.chunk_size, self.pad_answer_value), cur_answer))
    #     cur_answer = torch.tensor(cur_answer, dtype=torch.long)
    #
    #     return cur_input, cur_answer

    # def __iter__(self):
    #     # data = [(input, answer)]
    #     # test_input - TextSize x ChunkSize (оба плавающие)
    #
    #     # for cur_input, cur_answer in self.data:
    #     test_input = self.data[0][0]
    #     test_answer = self.data[0][1]
    #
    #     for i in range(len(test_input)):
    #         length = len(test_input[i])
    #
    #         if length < self.chunk_size:
    #             yield torch.tensor(test_input[i] + [self.pad_input_value] * (self.chunk_size - length)), \
    #                   torch.tensor(test_answer[i] + [self.pad_answer_value] * (self.chunk_size - length))
    #         else:
    #             yield torch.tensor(test_input[i]), torch.tensor(test_answer[i])
    #
    #     return
    #
    #     # for page_title, page_text in self.wiki_page_iterator:
    #     #     entity_to_search = nlp_base.get_search_strings(page_title)
    #     #     tokenized_chunks, token_positions = nlp_base.get_nn_data(self.bpe_tokenizer, entity_to_search, page_text, self.chunk_size)
    #     #     test_input = tokenized_chunks
    #     #     test_answer = token_positions
    #     #
    #     #     for i in range(len(test_input)):
    #     #         length = len(test_input[i])
    #     #         if length < self.chunk_size:
    #     #             yield torch.tensor(test_input[i] + [self.pad_input_value] * (self.chunk_size - length)), \
    #     #                   torch.tensor(test_answer[i] + [self.pad_answer_value] * (self.chunk_size - length))
    #     #         else:
    #     #             yield torch.tensor(test_input[i]), torch.tensor(test_answer[i])
    #     #
    #     #     return
