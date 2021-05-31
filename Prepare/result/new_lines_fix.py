import os

file_with_newlines = 'bpe_models/all-wiki-only-texts_1'
file_with_newline_tokens = file_with_newlines + '_newlines_tokens'

if os.path.exists(file_with_newline_tokens):
    print(f"File {file_with_newline_tokens} is already exists")
    exit(1)

with open(file_with_newlines, 'r', encoding='utf-8') as f:
    with open(file_with_newline_tokens, 'w', encoding='utf-8') as resf:
        for line in f:
            resf.write(line.replace('\n', '<n>'))
