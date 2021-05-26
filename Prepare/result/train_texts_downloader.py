# -*- coding: utf-8 -*-
from wiki_downloader import WikiDownloader
from Prepare import constants

wiki_downloader = WikiDownloader()

with open(constants.TRAIN_TEXTS_FILENAME, "w+"):
    pass

# with open(constants.TRAIN_TEXTS_FILENAME, "a", encoding='utf-8') as f:
#     for name in constants.NAMES_TO_TRAIN:
#         try:
#             _, page_text = wiki_downloader.get_page(name)
#         except Exception as e:
#             print(f'{name} не скачался. Причина: {e}')
#             continue
#
#         f.write(page_text)
#         f.write('\n')
#         print(f'{name} записан')


# poets_page = wiki_downloader.get_members_from_category('Категория:Русские поэты')

with open(constants.TRAIN_TEXTS_FILENAME, "a", encoding='utf-8') as f:
    for page_title, page_text in wiki_downloader.get_members_from_category('Категория:Персоналии по алфавиту'):
        f.write(page_text)
        f.write('\n')
        print(f'{page_title} записан')
