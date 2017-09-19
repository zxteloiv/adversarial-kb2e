# coding: utf-8

from __future__ import absolute_import
import chainer
import chainer.datasets
import numpy as np
import config
import logging
from vocabulary import Vocabulary


def load_corpus(filename, vocab_ent, vocab_rel):
    heads, rels, tails = [], [], []
    for i, (h, r, t) in enumerate(open_dataset(filename)):
        if config.USE_TINY_CORPUS_NUM > 0 and i / config.USE_TINY_CORPUS_NUM >= 1:
            break

        heads.append(np.array([vocab_ent(h)], dtype=np.int32))
        rels.append(np.array([vocab_rel(r)], dtype=np.int32))
        tails.append(np.array([vocab_ent(t)], dtype=np.int32))

    return chainer.datasets.TupleDataset(heads, rels, tails)


def open_dataset(filename):
    import importlib
    module_name = 'corpus.' + config.DATASET_IN_USE.lower()
    logging.debug('Trying to load the %s module' % module_name)
    dataset = importlib.import_module(module_name)
    return dataset.reader(open(filename))


def load_vocab():
    vocab_ent, vocab_rel = Vocabulary(), Vocabulary()
    vocab_ent.load(config.VOCAB_ENT_FILE)
    vocab_rel.load(config.VOCAB_REL_FILE)
    return vocab_ent, vocab_rel

