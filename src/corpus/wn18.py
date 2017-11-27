# coding: utf-8

from __future__ import absolute_import
import logging
from vocabulary import Vocabulary
from .fb15k import reader

def main():
    import config
    import sys, os
    if len(sys.argv) > 1 and sys.argv[1] == '--debug':
        logging.getLogger().setLevel(logging.DEBUG)

    vocab_ent, vocab_rel = Vocabulary(), Vocabulary()

    print "build relation vocab from file %s" % config.TRAIN_DATA
    for _, r, _ in reader(open(config.TRAIN_DATA)):
        vocab_rel.add_token(r)

    # It is the definition file that contains the full set of synsets.
    synset_def = os.path.join(config.DATA_PATH,
                              filter(lambda f: 'definitions' in f, os.listdir(config.DATA_PATH))[0])

    print "build entity vocab from file %s" % synset_def
    for e, _, _ in reader(open(synset_def)):
        vocab_ent.add_token(e)

    vocab_ent.save(config.VOCAB_ENT_FILE)
    vocab_rel.save(config.VOCAB_REL_FILE)
    print "vocab built successfully at %s and %s" % (config.VOCAB_ENT_FILE, config.VOCAB_REL_FILE)

if __name__ == "__main__":
    import config
    if config.DATASET_IN_USE.lower() != 'WN18'.lower():
        print "The configured dataset is not WN18, stop building vocabulary"
    else:
        main()
