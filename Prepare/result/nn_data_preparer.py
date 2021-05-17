# -*- coding: utf-8 -*-
import constants
import json
import nlp_base
import youtokentome as yttm


class ContinueOuterLoop(Exception):
    pass


vocab_size = 22000

bpe_tokenizer = yttm.BPE(constants.BPE_MODEL_FILENAME+'_'+str(vocab_size))
CHUNK_SIZE = 200

cur_index = 0
with_answer_in_text = 0
without_answer_in_text = 0

try:
    with open(file='../../wiki/all-full-name-pages-json', mode='r', encoding='utf-8') as f:
        with open(file='nn_data/all_nn_data'+'_'+str(vocab_size), mode='w', encoding='utf-8') as resf:
            for _ in range(cur_index):
                next(f)

            for line in f:
                cur_index += 1
                cur_json = json.loads(line)

                page_title = cur_json['title']
                page_text = cur_json['text']

                entity_to_search = nlp_base.get_search_strings(page_title)

                if not entity_to_search:
                    without_answer_in_text += 1
                    continue

                try:
                    tokenized_chunks, token_positions = nlp_base.get_nn_data(bpe_tokenizer, entity_to_search, page_text, CHUNK_SIZE)
                except Exception as e:
                    print('GET NN DATA EXCEPTION!!!!!')
                    print(e)
                    print(f'Line {cur_index} failed!!!!!')
                    print(f'Page Title - {page_title}')
                    print(f'Page Text - {page_text}')
                    print('-------------------------')
                    continue

                test_input = tokenized_chunks
                test_answer = token_positions

                try:
                    for chunk in test_answer:
                        if not any(chunk):
                            without_answer_in_text += 1
                            raise ContinueOuterLoop
                except ContinueOuterLoop:
                    continue

                json.dump({'input': test_input, 'answer': test_answer}, resf)
                resf.write('\n')

                with_answer_in_text += 1

                if with_answer_in_text % 1000 == 0:
                    print(f'With answer - {with_answer_in_text}')

except Exception as e:
    print(f'Cycle exception!')
    print(e)
    print(f'Line {cur_index} failed!!!!!')
    print(f'Page Title - {page_title}')
    print(f'Page Text - {page_text}')


print(f'With answer - {with_answer_in_text}')
print(f'Without answer - {without_answer_in_text}')
