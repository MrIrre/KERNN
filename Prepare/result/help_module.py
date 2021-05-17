import codecs
import functools
import operator


def save_texts_to_file(texts, out_file):
    with codecs.open(filename=out_file, mode='w', encoding='utf-8') as outf:
        outf.write('\n'.join(texts))


def flatten(a):
    return functools.reduce(operator.iconcat, a, [])


def get_time(start_time, cur_time, action_name):
    print(action_name + f" -> {cur_time - start_time} секунд")


