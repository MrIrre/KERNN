# -*- coding: utf-8 -*-
import json

count = 0

with open(file='all-wiki-pages-json', mode='r', encoding='utf-8') as fp:
    with open(file='all-wiki-only-texts', mode='w', encoding='utf-8') as resfp:
        for line in fp:
            cur_json = json.loads(line)

            if cur_json['text']:
                count += 1
                resfp.write(cur_json['text'])
                resfp.write('\n')

            if count % 10000 == 0:
                print(f"{count} штук записано")

print(f"{count} штук записано")
