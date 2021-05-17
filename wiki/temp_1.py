# -*- coding: utf-8 -*-
import json
import html
import re
import unicodedata


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


FULL_NAME_WITH_COMMA_REGEX = re.compile("^([\w -]+), ([\w-]+) ?([\w-]+)?$")
USELESS_TEXT_REGEX = re.compile("^\W*$")

count = 0
text_not_found_count = 0
is_logged = False

with open(file='result-json-one-file/AA/wiki_00', mode='r', encoding='utf-8') as fp:
    with open(file='all-wiki-pages-json_1', mode='w', encoding='utf-8') as resfp:
        for line in fp:
            cur_json = json.loads(line)
            cur_json['text'] = unaccentify_text(html.unescape(cur_json['text'])).replace('­', '')

            if not USELESS_TEXT_REGEX.match(cur_json['text']):
                if is_logged:
                    is_logged = False

                cur_json['title'] = unaccentify_text(html.unescape(cur_json['title'])).replace('­', '')

                # print(cur_json['title'])
                count += 1
                resfp.write(json.dumps(cur_json))
                resfp.write('\n')
            else:
                # print(f"TEXT NOT FOUND - {cur_json['title']}")
                # print(cur_json)
                text_not_found_count += 1

            if count % 20000 == 0 and not is_logged:
                print(f"{count} штук записано")
                is_logged = True

print(f"{count} штук записано")
print(f"{text_not_found_count} штук НЕ записано")


# import json
# import re
# import html
# # —−―–
# r = re.compile('^[№/\w\s()@+\'`’‘".,!?&%‰*■^☺×§$·′=∞«»„“”°…-]+$')
#
# with open(file='result-json-one-file/AA/wiki_00', mode='r', encoding='utf-8') as fp:
#     for line in fp:
#         cur_json = json.loads(line)
#
#         if not r.match(html.unescape(cur_json['title'])):
#             try:
#                 print(cur_json['title'])
#                 print(unaccentify_text(cur_json['title']))
#             except UnicodeEncodeError as e:
#                 print(e)
#
#             print('---------')

