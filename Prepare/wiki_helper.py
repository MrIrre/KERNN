from typing import List, Set
import base
import wikipediaapi
import re

wiki = wikipediaapi.Wikipedia(language='ru')


def download_page(page_name: str) -> wikipediaapi.WikipediaPage:
    page = wiki.page(page_name)
    print(f'GET PAGE WITH DISPLAY TITLE -> {page.displaytitle}')
    return page


ignored_sections = {'Справочная информация', 'Примечания', 'Литература', 'Ссылки', 'Библиография', 'Память'}
def get_page_needed_text(page: wikipediaapi.WikipediaPage) -> str:
    needed_root_sections = filter(lambda section: section.title not in ignored_sections, page.sections)
    section_texts = []
    visited = set()

    def dfs(cur_section):
        visited.add(cur_section.title)
        # print(f"VISIT - {cur_section.title}")
        text = cur_section.text

        if text:
            section_texts.append(text)

        for section in cur_section.sections:
            if section.title not in visited:
                dfs(section)

    for root_section in needed_root_sections:
        dfs(root_section)

    full_text = str.join('', section_texts)

    return full_text



