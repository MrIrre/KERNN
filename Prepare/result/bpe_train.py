# -*- coding: utf-8 -*-
from Prepare import constants
import youtokentome as yttm

vocab_size = 50000
yttm.BPE.train(data=constants.TRAIN_TEXTS_FILENAME, vocab_size=vocab_size, model=constants.BPE_MODEL_FILENAME + '_' + str(vocab_size))

# tokenizer = yttm.BPE(constants.BPE_MODEL_FILENAME)
# print(f"Vocab Size = {tokenizer.vocab_size()}")
# print(f"Vocab -> {tokenizer.vocab()}")
