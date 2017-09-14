# coding: utf-8

from __future__ import absolute_import
import logging
from vocabulary import Vocabulary


def reader(generator):
    for i, l in enumerate(generator):
        h, r, t = l.rstrip().split('\t')
        yield h, r, t

        if i % 10000 == 0:
            logging.debug('%d examples are read' % (i + 1))
    logging.debug('%d examples are read' % (i + 1))


def build_vocab(filename):
    vocab_ent, vocab_rel = Vocabulary(), Vocabulary()
    for h, r, t in reader(open(filename)):
        vocab_ent.add_token(h)
        vocab_ent.add_token(t)
        vocab_rel.add_token(r)

    return vocab_ent, vocab_rel

if __name__ == "__main__":
    import config
    import sys
    if sys.argv[1] == '--debug':
        logging.getLogger().setLevel(logging.DEBUG)

    print "build vocab from file %s" % config.TRAIN_DATA
    vocab_ent, vocab_rel = build_vocab(config.TRAIN_DATA)
    vocab_ent.save(config.VOCAB_ENT_FILE)
    vocab_rel.save(config.VOCAB_REL_FILE)
    print "vocab built successfully at %s and %s" % (config.VOCAB_ENT_FILE, config.VOCAB_REL_FILE)



