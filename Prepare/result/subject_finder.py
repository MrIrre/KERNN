# -*- coding: utf-8 -*-
import spacy
from spacy import displacy
from spacy.matcher import Matcher

# nlp_base.get_search_strings("Пушкин, Александр")

articles = []

with open('../../wiki/texts_for_extract_tests', 'r', encoding='utf-8') as f:
    for _ in range(14):
        title = f.readline()[:-1]

        article_lines = []
        line = f.readline()

        while line != '\n' and line:
            article_lines.append(line)
            line = f.readline()

        text = ''.join(article_lines)

        articles.append((title, text))

nlp_ru = spacy.load("ru_core_news_lg")
matcher = Matcher(nlp_ru.vocab)

nlp_en = spacy.load("en_core_web_lg")

for article in articles:
    title, text = article[0], article[1]

    doc = nlp_ru(title)
    for ru_token in doc:
        print(ru_token.text, ru_token.dep_, ru_token.head.text, ru_token.head.pos_,
              [child for child in ru_token.children])
    print('-----')

    continue

    pattern = [{"TEXT": title}]
    matcher.add("CUR_PATTERN", [pattern])

    # displacy.serve(doc)

    matches = matcher(doc)

    for match_id, start, end in matches:
        doc_elem = doc[start:end]
        print(f"IN DOC: {doc_elem}")
        start_in_text = doc_elem.start_char
        end_in_text = doc_elem.end_char
        print(f"IN TEXT: {text[start_in_text:end_in_text]}")

    print('----------------')

    # matcher.remove("CUR_PATTERN")


# temp = 'Союз Советских Социалистических Республик'
# doc_en = nlp_en(temp)
# doc_ru = nlp_ru(temp)
# for ru_token in doc_ru:
#     print(ru_token.text, ru_token.dep_, ru_token.head.text, ru_token.head.pos_,
#           [child for child in ru_token.children])
#
# print("-")
#
# for en_token in doc_en:
#     print(en_token.text, en_token.dep_, en_token.head.text, en_token.head.pos_,
#           [child for child in en_token.children])
