import pymystem3

mystem = pymystem3.Mystem()

titles = [
    'Пушкин, Александр Сергеевич',
    'Теистический эволюционизм',
    'Паскаль, Блез',
    'Арифмометр',
    'Эффект Матфея',
    'Си (язык программирования)',
    'Криштиану Роналду',
    'Де Бройль, Луи',
    'Объектно-ориентированный язык программирования'
]

import re

full_name_with_comma_regex = re.compile("^([а-яА-Я -]+), (\w+) ?(\w+)?$")

def get_search_strings(title):
    to_search = []

    word_infos = mystem.analyze(title)
    words = list(filter(lambda x: 'analysis' in x, word_infos))

    full_name_res = full_name_with_comma_regex.findall(title)

    if full_name_res:
        surname, first_name, patronymic = full_name_res[0]

        to_search.append(surname)
        to_search.append(first_name)
        to_search.append(f"{first_name} {surname}")
        to_search.append(f"{surname} {first_name}")

        if patronymic:
            to_search.append(f"{first_name} {patronymic}")
            to_search.append(f"{surname} {first_name} {patronymic}")
            to_search.append(f"{first_name} {patronymic} {surname}")
    else:
        print(words)

    return to_search

# for title in titles:
#     get_search_strings(title)


def analyze(text):
    return mystem.analyze(text)


def to_lemmas(text):
    return mystem.lemmatize(text)


def lemmatize(text):
    return ''.join(to_lemmas(text)[:-1])
