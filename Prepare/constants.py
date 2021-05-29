# -*- coding: utf-8 -*-
import re

BPE_MODEL_FILENAME = 'bpe_models/all-wiki-only-texts_bpe_model_1'
TRAIN_TEXTS_FILENAME = 'result/bpe_models/all-wiki-only-texts_1'

FULL_NAME_WITH_COMMA_REGEX = re.compile("^([A-ZА-ЯЁ][\w -]+), ([A-ZА-ЯЁ][\w-]+)(?: ([A-ZА-ЯЁ][\w-]+))?$")
