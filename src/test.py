# coding: utf-8

from __future__ import absolute_import
import os, argparse, logging
import chainer
import numpy as np

import config, models
import corpus.dataset as mod_dataset

def main():
    parser = argparse.ArgumentParser()
    group_GAN = parser.add_argument_group('GAN setting')
    group_GAN.add_argument('-g', '--generator', help="load generator model snapshot")
    group_GAN.add_argument('-d', '--discriminator')
    group_TransE = parser.add_argument_group('TransE setting')
    group_TransE.add_argument('transE')
    args = parser.parse_args()

    vocab_ent, vocab_rel = mod_dataset.load_vocab()
    logging.info('ent vocab size=%d, rel vocab size=%d' % (len(vocab_ent), len(vocab_rel)))
    train_data, valid_data, test_data = map(lambda f: mod_dataset.load_corpus(f, vocab_ent, vocab_rel),
                                            (config.TRAIN_DATA, config.VALID_DATA, config.TEST_DATA))
    logging.info('data loaded, size: train:valid:test=%d:%d:%d' % (len(train_data), len(valid_data), len(test_data)))
    logging.getLogger().setLevel(logging.INFO)

    run_TransE_test(args, vocab_ent, vocab_rel, train_data, valid_data, test_data)

def run_TransE_test(args, vocab_ent, vocab_rel, train_data, valid_data, test_data):
    transE = models.TransE.create_transe(config.EMBED_SZ, vocab_ent, vocab_rel, config.TRANSE_GAMMA)
    chainer.serializers.load_npz(args.transE, transE)

    xp = np
    if config.DEVICE >= 0:
        chainer.cuda.get_device_from_id(config.DEVICE).use()
        transE.to_gpu(config.DEVICE)
        from chainer.cuda import cupy
        xp = cupy

    dataset = test_data
    data_iter = chainer.iterators.SerialIterator(dataset, batch_size=1, repeat=False, shuffle=False)

    candidate_t = xp.arange(1, len(vocab_ent) + 1, dtype=xp.int32)
    if config.DEVICE >= 0:
        candidate_t = chainer.dataset.to_device(config.DEVICE, candidate_t) # shape of (#entity_num, embedding_size)
    bsz = candidate_t.shape[0]
    ct_emb = transE.ent_emb(candidate_t).reshape(bsz, -1).data

    avgrank, hits10, count = 0, 0, 0
    for i, batch in enumerate(data_iter):
        h, r, t = batch[0] # each one is an array of shape (1, )
        if config.DEVICE >= 0:
            h = chainer.dataset.to_device(config.DEVICE, h)
            r = chainer.dataset.to_device(config.DEVICE, r)

        h_emb = transE.ent_emb(h).data # shape of (batchsz=1, embedding_size)
        r_emb = transE.rel_emb(r).data # and variable doesn't support broadcasting
        values = xp.linalg.norm(h_emb + r_emb - ct_emb, axis=1) # norm value vector with shape of (#entity_num, )

        scores = chainer.cuda.to_cpu(values) # cupy doesn't support argsort yet
        sorted_index = np.argsort(scores)
        rank = np.where(sorted_index == (t[0] - 1))[0][0] # tail ent id 1 ~ maxid, but array index: 0 ~ maxid-1
        avgrank += rank
        hits10 += 1 if rank < 10 else 0
        count += 1

        if i % 1000 == 0:
            logging.info('%d testing data processed, temp rank: %d, hits10: %d, avgrank: %f' % (
                count, avgrank, hits10, avgrank / count))

    avgrank /= count * 1.0
    hits10 /= count * 1.0
    print "avgrank:", avgrank, "hits@10:", hits10, "count:", count


def run_GAN_test(args, vocab_ent, vocab_rel, train_data, valid_data, test_data):
    g = models.Generator.create_generator(config.EMBED_SZ, vocab_ent, vocab_rel)
    chainer.serializers.load_npz(args.generator, g)

    if args.discriminator:
        d = models.Discriminator(config.EMBED_SZ)
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
