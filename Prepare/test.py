# -*- coding: utf-8 -*-
import re
import string
from Prepare import help_module

import pymorphy2
from pymorphy2.shapes import is_punctuation

pymorph = pymorphy2.MorphAnalyzer()

TOKEN_REGEX = re.compile(r'([^\w_-]|[+])', re.UNICODE)


def word_tokenize(text, _split=TOKEN_REGEX.split):
    return [t for t in _split(text) if t]


# def extract_words(s):
#     return [re.sub('^[{0}]+|[{0}]+$'.format(string.punctuation + '—«»'), '', w) for w in s.split()]


def lemmatize_word(word: str):
    if not is_punctuation(word) and not word.isspace():
        normal_form = pymorph.parse(word)[0].normal_form
        return word, normal_form

    return word, None

def extract_words(s):
    return re.findall(r'\b\S+\b', s)


# str = 'This is a string, with words!'
str = '''
Facebook написан на C++, PHP (HHVM).
[Ghbdtn|Privet]
[[Ghbdtn123|123Privet]]
(You Drive Me) Crazy
http://www.stackoverflow.ru/
"Hello!"
*!OLOLO*
"qwerty)
[HEHEHE]
«РУДА»
Hello — red
(G)I-DLE
'''

words = extract_words(str)
lemmas = list(filter(None, map(lambda word: lemmatize_word(word)[1], words)))

print(words)

print(lemmas)
