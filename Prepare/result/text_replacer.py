#!/usr/bin/env python3
import fileinput

cur_line = 0

line_index = 0
lines_to_replace = [
    909,
    16580,
    28733,
    126307,
    152490,
    168659,
    169025
]

new_lines = []
with open('nn_data/all_nn_data_15000_with_tf-idf_lolol', 'r') as f:
    new_lines = f.readlines()


with fileinput.FileInput('nn_data/all_nn_data_15000_with_tf-idf', inplace=True, backup='.bak') as file:
    for line in file:
        if cur_line == lines_to_replace[line_index]:
            line = new_lines[line_index]
            line_index += 1

        cur_line += 1
