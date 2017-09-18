# coding: utf-8

from __future__ import absolute_import
import os, argparse, logging
import chainer
import numpy as np

import config, models
import corpus.dataset as mod_dataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('generator_model', help="load generator model snapshot")
    args = parser.parse_args()

    logging.getLogger().setLevel(logging.INFO)

    run_test(args)

def run_test(args):
    vocab_ent, vocab_rel = mod_dataset.load_vocab()
    logging.info('ent vocab size=%d, rel vocab size=%d' % (len(vocab_ent), len(vocab_rel)))

    train_data, valid_data, test_data = map(lambda f: mod_dataset.load_corpus(f, vocab_ent, vocab_rel),
                                            (config.TRAIN_DATA, config.VALID_DATA, config.TEST_DATA))

    logging.info('data loaded, size: train:valid:test=%d:%d:%d' % (len(train_data), len(valid_data), len(test_data)))

    g = models.Generator.create_generator(config.EMBED_SZ, vocab_ent, vocab_rel)
    chainer.serializers.load_npz(args.generator_model, g)

    token_fullset = vocab_ent.get_tokens()
    token_ids = set(filter(lambda x: x < 14954, (vocab_ent(t) for t in token_fullset)))
    avgrank, hits10 = 0, 0
    for i in xrange(len(train_data)):
        h, r, t = train_data[10]
        rank = get_score(h, r, t, g, token_ids)
        avgrank += rank
        hits10 += 1 if rank <= 10 else 0
        print rank
        break

        if i % 2 == 0:
            logging.info('%d testing data processed' % (i + 1))
    avgrank /= len(train_data)
    hits10 /= len(train_data)


def get_score(h, r, t, g, token_ids):
    h, r = map(lambda x: np.array([x], np.int32).reshape(-1, 1), (h, r))
    t_generated = g(h, r).data
    def scoring(g, t_prime):
        t_prime = np.array([t_prime], dtype=np.int32)
        score = np.linalg.norm(t_generated - g.embed_entity(t_prime).data)
        return score

    sorted_id = sorted(token_ids, key=lambda t_prime: scoring(g, t_prime))
    rank = sorted_id.index(t)

    return rank


if __name__ == "__main__":
    main()
