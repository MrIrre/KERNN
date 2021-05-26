# -*- coding: utf-8 -*-

from Prepare.constants import FULL_NAME_WITH_COMMA_REGEX
import wikipediaapi


class WikiDownloader:
    def __init__(self):
        self.wiki_api = wikipediaapi.Wikipedia(language='ru')
        self.ignored_sections = {'Справочная информация', 'Примечания', 'Литература', 'Ссылки', 'Библиография',
                                 'Память', 'Книги', 'Публикации и переводы', 'Семья', 'Награды и премии', 'Адреса', 'Труды'}

    def _download_page(self, page_title: str) -> wikipediaapi.WikipediaPage:
        page = self.wiki_api.page(page_title)
        # print(f'GET PAGE WITH DISPLAY TITLE -> {page.displaytitle}')
        return page

    def _get_page_needed_text(self, page: wikipediaapi.WikipediaPage) -> str:
        needed_root_sections = filter(lambda section: section.title not in self.ignored_sections, page.sections)
        section_texts = []
        visited = set()

        def dfs(cur_section):
            visited.add(cur_section.title)
            # print(f"VISIT - {cur_section.title}")
            text = cur_section.value

            if text:
                section_texts.append(text)

            for section in cur_section.sections:
                if section.title not in visited:
                    dfs(section)

        for root_section in needed_root_sections:
            dfs(root_section)

        full_text = '\n'.join(section_texts)

        return full_text

    def get_page(self, page_name):
        page = self._download_page(page_name)

        page_title = page.displaytitle
        page_text = self._get_page_needed_text(page)

        return page_title, page_text

    def get_members_from_category(self, category_name):
        category_page = self._download_page(category_name)

        for member in category_page.categorymembers:
            if FULL_NAME_WITH_COMMA_REGEX.match(member):
                page = category_page.categorymembers[member]

                if not list(filter(lambda section: section.title not in self.ignored_sections, page.sections)):
                    continue

                page_title = page.displaytitle
                page_text = self._get_page_needed_text(page)
                yield page_title, page_text


