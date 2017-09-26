# coding: utf-8

from __future__ import absolute_import
import os, argparse, logging
import chainer
import numpy as np

import config, models
import corpus.dataset as mod_dataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('models', nargs='+')
    args = parser.parse_args()

    vocab_ent, vocab_rel = mod_dataset.load_vocab()
    logging.info('ent vocab size=%d, rel vocab size=%d' % (len(vocab_ent), len(vocab_rel)))
    train_data, valid_data, test_data = map(lambda f: mod_dataset.load_corpus(f, vocab_ent, vocab_rel),
                                            (config.TRAIN_DATA, config.VALID_DATA, config.TEST_DATA))
    logging.info('data loaded, size: train:valid:test=%d:%d:%d' % (len(train_data), len(valid_data), len(test_data)))
    logging.getLogger().setLevel(logging.INFO)

    # transE = models.TransE.create_transe(config.EMBED_SZ, vocab_ent, vocab_rel, config.TRANSE_GAMMA)
    # chainer.serializers.load_npz(args.models[0], transE)
    # run_ranking_test(TransE_Scorer, transE, vocab_ent, test_data)

    gen = models.HingeLossGen.create_hinge_gen(config.EMBED_SZ, vocab_ent, vocab_rel, config.TRANSE_GAMMA)
    chainer.serializers.load_npz(args.models[0], gen)
    run_ranking_test(HingeGen_Scorer, gen, vocab_ent, test_data)


class TransE_Scorer(object):
    def __init__(self, model, candidate_t):
        self.transE = model
        self.bsz = candidate_t.shape[0]
        self.ct_emb = self.transE.ent_emb(candidate_t).reshape(self.bsz, -1).data
        self.xp = np
        if config.DEVICE >= 0:
            chainer.cuda.get_device_from_id(config.DEVICE).use()
            from chainer.cuda import cupy
            self.xp = cupy

    def __call__(self, h, r):
        h_emb = self.transE.ent_emb(h).data # shape of (batchsz=1, embedding_size)
        r_emb = self.transE.rel_emb(r).data # and variable doesn't support broadcasting
        values = self.xp.linalg.norm(h_emb + r_emb - self.ct_emb, axis=1) # norm value vector with shape of (#entity_num, )

        scores = chainer.cuda.to_cpu(values) # cupy doesn't support argsort yet
        return scores

class HingeGen_Scorer(object):
    def __init__(self, model, candidate_t):
        self.model = model
        self.bsz = candidate_t.shape[0]
        self.ct_emb = self.model.ent_emb(candidate_t).reshape(self.bsz, -1).data
        self.xp = np
        if config.DEVICE >= 0:
            chainer.cuda.get_device_from_id(config.DEVICE).use()
            from chainer.cuda import cupy
            self.xp = cupy

    def __call__(self, h, r):
        t_tilde = self.model.run_gen(h, r).data
        values = self.xp.linalg.norm(t_tilde - self.ct_emb, axis=1)
        scores = chainer.cuda.to_cpu(values)
        return scores


def run_ranking_test(scorer_type, model, vocab_ent, test_data):
    xp = np
    if config.DEVICE >= 0:
        chainer.cuda.get_device_from_id(config.DEVICE).use()
        model.to_gpu(config.DEVICE)
        from chainer.cuda import cupy
        xp = cupy

    data_iter = chainer.iterators.SerialIterator(test_data, batch_size=1, repeat=False, shuffle=False)
    candidate_t = xp.arange(1, len(vocab_ent) + 1, dtype=xp.int32)
    if config.DEVICE >= 0:
        candidate_t = chainer.dataset.to_device(config.DEVICE, candidate_t) # shape of (#entity_num, embedding_size)
    scorer = scorer_type(model, candidate_t)

    avgrank, hits10, count = 0, 0, 0
    for i, batch in enumerate(data_iter):
        h, r, t = batch[0] # each one is an array of shape (1, )
        if config.DEVICE >= 0:
            h = chainer.dataset.to_device(config.DEVICE, h)
            r = chainer.dataset.to_device(config.DEVICE, r)

        scores = scorer(h, r)
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

if __name__ == "__main__":
    main()
