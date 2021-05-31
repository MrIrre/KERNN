# -*- coding: utf-8 -*-
import re

BPE_MODEL_FILENAME = 'bpe_models/all-wiki-only-texts_bpe_model_complete'
TRAIN_TEXTS_FILENAME = 'bpe_models/all-wiki-only-texts_1_newlines_tokens'

FULL_NAME_WITH_COMMA_REGEX = re.compile("^([A-ZА-ЯЁ][\w -]+), ([A-ZА-ЯЁ][\w-]+)(?: ([A-ZА-ЯЁ][\w-]+))?$")
