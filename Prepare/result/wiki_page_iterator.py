import random


class WikiPageIterator:
    def __init__(self, downloader, page_names):
        self.counter = 0
        self.downloader = downloader
        self.page_names = page_names

        random.shuffle(self.page_names)

    def __iter__(self):
        return self

    def __next__(self):
        if self.counter >= len(self.page_names):
            self.counter = 0
            random.shuffle(self.page_names)

        current_page_name = self.page_names[self.counter]
        page_title, page_text = self.downloader.get_page(current_page_name)

        self.counter += 1

        return page_title, page_text
