# -*- coding: utf-8 -*-
import json

count = 0
titles = {'Электронно-вычислительная машина', 'Компьютер', '(G)I-DLE', 'ЭВМ (группа)', 'BTS', 'Николай II',
          'Анастасия Николаевна', 'Mark I (танк)', 'Машина времени (группа)', 'Паспорт гражданина Российской Федерации',
          'Глазенап, Богдан Александрович фон', 'Католицизм', 'General Dynamics F-16 Fighting Falcon',
          'Город, в котором меня нет'}

with open(file='all-wiki-pages-json_1', mode='r', encoding='utf-8') as fp:
    with open(file='texts_for_extract_tests', mode='w', encoding='utf-8') as resfp:
        for line in fp:
            cur_json = json.loads(line)

            if cur_json['title'].strip() in titles and cur_json['text']:
                count += 1
                resfp.write(cur_json['title'])
                resfp.write('\n')
                resfp.write(cur_json['text'])
                resfp.write('\n')

                print(f"{count} штук записано")

print(f"{count} штук записано")
