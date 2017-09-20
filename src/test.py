# coding: utf-8

from __future__ import absolute_import
import os, argparse, logging
import chainer
import numpy as np

import config, models
import corpus.dataset as mod_dataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--generator', help="load generator model snapshot", required=True)
    parser.add_argument('-d', '--discriminator')
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
    chainer.serializers.load_npz(args.generator, g)

    if args.discriminator:
        d = models.BilinearDiscriminator(config.EMBED_SZ)
        chainer.serializers.load_npz(args.discriminator, d)
    else:
        d = None

    token_ids = set(vocab_ent(t) for t in vocab_ent.get_tokens())
    avgrank, hits10 = 0, 0
    for i in xrange(len(train_data[:10])):
        h, r, t = train_data[i]
        rank = get_score(token_ids, h, r, t, g, d)
        avgrank += rank
        hits10 += 1 if rank <= 10 else 0

        if i % 2 == 0:
            logging.info('%d testing data processed' % (i + 1))

    avgrank /= len(train_data[:10])
    hits10 /= len(train_data[:10])
    print "avgrank:", avgrank, "hits@10:", hits10


def get_score(token_ids, h, r, t, g, d=None):
    h, r = map(lambda x: np.array([x], dtype=np.int32).reshape(-1, 1), (h, r))
    t_generated = g(h, r).data

    def scoring(t_prime):
        t_prime = np.array([t_prime], dtype=np.int32).reshape(-1, 1)
        if d is None:
            score = np.linalg.norm(t_generated - g.embed_entity(t_prime).data)
        else:
            h_emb, t_emb = g.embed_entity(h), g.embed_entity(t_prime)
            r_emb = g.embed_relation(r)

            score = d(h_emb, r_emb, t_emb).data[0][0]
            score += np.linalg.norm(t_generated - t_emb.data)

        return score

    sorted_id = sorted(token_ids, key=lambda t_prime: scoring(t_prime))
    rank = sorted_id.index(t)

    return rank


if __name__ == "__main__":
    main()
