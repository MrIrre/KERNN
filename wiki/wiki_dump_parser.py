import bz2
import json


def wikidata(filename):
    with bz2.open(filename, mode='rt') as f:
        f.read(2)  # skip first two bytes: "{\n"
        for line in f:
            try:
                yield json.loads(line.rstrip(',\n'))
            except json.decoder.JSONDecodeError:
                continue


for record in wikidata('clean_wiki_data/latest-all.json.bz2'):
    cur = json.dumps(record, ensure_ascii=False)
    print(cur)


print('END')
