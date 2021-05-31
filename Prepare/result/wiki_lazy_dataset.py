import torch
from torch.utils.data import Dataset, IterableDataset
import ujson


def ensure_length(chunk, out_len, pad_value):
    if len(chunk) < out_len:
        chunk = list(chunk) + [pad_value] * (out_len - len(chunk))
    else:
        chunk = chunk[:out_len]
    return chunk


DATASET_TYPE = IterableDataset
# DATASET_TYPE = Dataset


class WikiTextLazyDataset(DATASET_TYPE):
    def __init__(self, filepath, pad_input_value=0, pad_answer_value=0):
        print('WikiTextLazyDataset: open file...')
        self._file_stream = open(file=filepath, mode='r', encoding='utf-8')
        print('WikiTextLazyDataset: file is opened!')

        self.line_count = int(self._file_stream.readline())  # Первая строчка - количество чанков
        print(f'Lines - {self.line_count}')
        self.chunk_size = int(self._file_stream.readline())  # Вторая строчка - размер максимального чанка
        self.pad_input_value = pad_input_value
        self.pad_answer_value = pad_answer_value

    def __del__(self):
        if not self._file_stream.closed:
            print('WikiTextLazyDataset: close file...')
            self._file_stream.close()

        print('WikiTextLazyDataset: file is closed!')

    # def __len__(self):
    #     return len(self._line_offsets)
    #
    # def __getitem__(self, line):
    #     offset = self._line_offsets[line]
    #     self._file_stream.seek(offset, 0)  # Отсчитываем строку всегда от начала файла (whence=0)
    #     line = self._file_stream.readline()
    #
    #     cur_item = ujson.loads(line)
    #     input_raw = cur_item['input']
    #     answer_raw = cur_item['answer']
    #
    #     cur_input, cur_answer = ensure_texts_length(input_raw, self.text_size, self.chunk_size, self.pad_input_value, torch.long), \
    #                             ensure_texts_length(answer_raw, self.text_size, self.chunk_size, self.pad_answer_value, torch.float)
    #
    #     return cur_input, cur_answer

    def __iter__(self):
        # data = [(input, answer)]
        # test_input - TextSize x ChunkSize (оба плавающие)

        while True:
            self._file_stream.seek(0)

            # Пропускаем первые две строчки
            self._file_stream.readline()
            self._file_stream.readline()

            for line in self._file_stream:
                cur_json = ujson.loads(line)
                cur_input = cur_json[0]
                cur_answer = cur_json[1]
                length = len(cur_input)

                if length < self.chunk_size:
                    yield torch.tensor(cur_input + [self.pad_input_value] * (self.chunk_size - length), dtype=torch.long), \
                          torch.tensor(cur_answer + [self.pad_answer_value] * (self.chunk_size - length), dtype=torch.float)
                else:
                    yield torch.tensor(cur_input, dtype=torch.long), torch.tensor(cur_answer, dtype=torch.float)

        # for i in range(len(test_input)):
        #     length = len(test_input[i])
        #
        #     if length < self.chunk_size:
        #         yield torch.tensor(test_input[i] + [self.pad_input_value] * (self.chunk_size - length)), \
        #               torch.tensor(test_answer[i] + [self.pad_answer_value] * (self.chunk_size - length))
        #     else:
        #         yield torch.tensor(test_input[i]), torch.tensor(test_answer[i])
