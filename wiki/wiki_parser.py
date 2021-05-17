# -*- coding: utf-8 -*-
import xml

# with open('wiki/ruwiki-latest-pages-articles.xml', 'r', encoding='utf-8') as f:
#     for i in range(150):
#         print(f.readline())
#
import sys

import re
import unicodedata
import xml.etree.ElementTree as etree


unaccentify = {
    '́': '',
    '̀': '',
   'ѝ': 'и',
   'о́': 'о'
}
unaccentify = {unicodedata.normalize('NFKC', i):j for i, j in unaccentify.items()}
pattern = re.compile('|'.join(unaccentify))
def unaccentify_text(text):
    def replacer(match):
        return unaccentify[match.group(0)]

    source = unicodedata.normalize('NFKC', text)
    return pattern.sub(replacer, source)


# def strip_tag_name(t):
#     idx = t.rfind("}")
#     if idx != -1:
#         t = t[idx + 1:]
#     return t
#
#
# events = ("start", "end")
#
# title = None
# value = None
# is_revision = False
# context = etree.iterparse('ruwiki-20210501-pages-meta-current.xml', events=events)
#
# for event, elem in context:
#     tag = elem.tag
#
#     assert tag is not None
#
#     if event == 'start':
#         tname = strip_tag_name(tag)
#
#         if elem.text is not None:
#             value = elem.text
#
#         if tname == 'revision':
#             is_revision = True
#     elif event == 'end':
#         if elem.text is not None:
#             value = elem.text
#
#         if value is not None:
#             if tname == 'title':
#                 title = unaccentify_text(value)
#                 # print(title)
#             elif tname == 'revision':
#                 is_revision = False
#             elif is_revision and title is not None and tname == 'text':
#                 try:
#                     print(f"Title - {title}")
#                 except UnicodeEncodeError as e:
#                     print(e)
#
#                 try:
#                     from wikiextractor import WikiExtractor
#
#                     WikiExtractor.main()
#
#                     print(f"Text - {res_value}")
#                 except UnicodeEncodeError as e:
#                     print(e)
#
#                 print('--------------------------------------------')
#                 title = None
#     else:
#         raise Exception(f"Event is {event}. WHAT?!")
#
#     elem.clear()

# context = etree.parse('wiki/result/AA/wiki_00')
# print(context)


import json

count = 0
# print("Using for loop")

# import locale
# print(locale.getpreferredencoding())

import json

with open(file='result-json-one-file/AA/wiki_00', mode='r', encoding='utf-8') as fp:
    for line in fp:
        cur_json = json.loads(line)

        if cur_json['text']:
            unaccentify_text(cur_json['text']).replace('­', '')
            count += 1

        if count % 10000 == 0:
            print(f"{count} штук записано")

print(f"{count} штук записано")



# import fileinput
#
# with fileinput.FileInput('all-wiki-texts', inplace=True, backup='.bak') as file:
#     for line in file:
#         print(line, end='')
