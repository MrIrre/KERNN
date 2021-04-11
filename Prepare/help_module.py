import codecs
import functools
import operator


def get_chunks(text, max_chunk_size=200, step=None):
    chunks_start_indexes = []
    chunks = []

    if step is None:
        step = max_chunk_size // 2

    chunk_start_index = 0
    chunks_start_indexes.append(chunk_start_index)

    while len(text) > max_chunk_size:
        line_length = text[:max_chunk_size].rfind(' ')
        chunks.append(text[:line_length])

        chunk_start_index = text[:step].rfind(' ') + 1
        chunks_start_indexes.append(chunks_start_indexes[len(chunks_start_indexes) - 1] + chunk_start_index)
        text = text[chunk_start_index:]

    chunks.append(text)
    return chunks, chunks_start_indexes


def save_texts_to_file(texts, out_file):
    with codecs.open(filename=out_file, mode='w', encoding='utf-8') as outf:
        outf.write('\n'.join(texts))


def flatten(a):
    return functools.reduce(operator.iconcat, a, [])


def get_time(start_time, cur_time, action_name):
    print(action_name + f" -> {cur_time - start_time} секунд")


