# coding: utf-8

from __future__ import absolute_import
import os, argparse, logging
import chainer
import chainer.functions as F
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

    xp = np
    if config.DEVICE >= 0:
        chainer.cuda.get_device_from_id(config.DEVICE).use()
        from chainer.cuda import cupy
        xp = cupy

    # # classical TransE
    # transE = models.TransE.create_transe(config.EMBED_SZ, vocab_ent, vocab_rel, config.TRANSE_GAMMA)
    # chainer.serializers.load_npz(args.models[0], transE)
    # scorer = TransE_Scorer(transE, xp)

    # # MLP generator + Hinge Loss as the same as TransE
    # gen = models.HingeLossGen.create_hinge_gen(config.EMBED_SZ, vocab_ent, vocab_rel, config.TRANSE_GAMMA)
    # chainer.serializers.load_npz(args.models[0], gen)
    # scorer = HingeGen_Scorer(gen, xp)

    # # GAN testing
    # gen = models.Generator.create_generator(config.EMBED_SZ, vocab_ent, vocab_rel)
    # chainer.serializers.load_npz(args.models[0], gen)
    # d = None
    # if len(args.models) > 1:
    #     d = models.Discriminator(config.EMBED_SZ)
    #     chainer.serializers.load_npz(args.models[1], d)
    #
    # if config.DEVICE >= 0:
    #     gen.to_gpu(config.DEVICE)
    #     if d is not None:
    #         d.to_gpu(config.DEVICE)
    # scorer = GAN_Scorer(gen, d, xp)

    # Experimental tesing
    generator = models.VarMLP([config.EMBED_SZ * 2, config.EMBED_SZ, config.EMBED_SZ])
    embeddings = models.Embeddings(config.EMBED_SZ, len(vocab_ent) + 1, len(vocab_rel) + 1)
    print args.models[0], args.models[1]
    chainer.serializers.load_npz(args.models[0], generator)
    chainer.serializers.load_npz(args.models[1], embeddings)
    scorer = Experimental_Scorer(generator, embeddings, xp)

    run_ranking_test(scorer, vocab_ent, test_data)


class TransE_Scorer(object):
    def __init__(self, model, xp):
        self.transE = model if config.DEVICE < 0 else model.to_gpu(config.DEVICE)
        self.xp = xp

    def set_candidate_t(self, candidate_t):
        self.bsz = candidate_t.shape[0]
        self.ct_emb = self.transE.ent_emb(candidate_t).reshape(self.bsz, -1).data

    def __call__(self, h, r):
        h_emb = self.transE.ent_emb(h).data # shape of (batchsz=1, embedding_size)
        r_emb = self.transE.rel_emb(r).data # and variable doesn't support broadcasting
        values = self.xp.linalg.norm(h_emb + r_emb - self.ct_emb, axis=1) # norm value vector with shape of (#entity_num, )

        scores = chainer.cuda.to_cpu(values) # cupy doesn't support argsort yet
        return scores

class HingeGen_Scorer(object):
    def __init__(self, model, xp):
        self.model = model if config.DEVICE < 0 else model.to_gpu(config.DEVICE)
        self.xp = xp

    def set_candidate_t(self, candidate_t):
        self.bsz = candidate_t.shape[0]
        self.ct_emb = self.model.gen.ent_emb(candidate_t).reshape(self.bsz, -1).data

    def __call__(self, h, r):
        t_tilde = self.model.run_gen(h, r).data
        values = self.xp.linalg.norm(t_tilde - self.ct_emb, axis=1)
        scores = chainer.cuda.to_cpu(values)
        return scores


class GAN_Scorer(object):
    def __init__(self, g, d, xp):
        self.g = g if config.DEVICE < 0 else g.to_gpu(config.DEVICE)
        self.d = d if config.DEVICE < 0 or d is None else d.to_gpu(config.DEVICE)
        self.xp = xp

    def set_candidate_t(self, candidate_t):
        self.bsz = candidate_t.shape[0]
        self.ct_emb = self.g.embed_entity(candidate_t)

    def __call__(self, h, r):
        return self.get_g_score(h, r)
        # return self.get_d_score(h, r)

    def get_d_score(self, h, r):
        h_emb = F.stack([self.g.embed_entity(h)] * self.bsz).reshape(self.bsz, -1)
        r_emb = F.stack([self.g.embed_relation(r)] * self.bsz).reshape(self.bsz, -1)
        values = self.d(h_emb, r_emb, self.ct_emb).reshape(-1).data
        scores = chainer.cuda.to_cpu(values)
        return scores

    def get_g_score(self, h, r):
        t_tilde = self.g(h, r)
        values = self.xp.linalg.norm(t_tilde.data - self.ct_emb.data, axis=1)
        scores = chainer.cuda.to_cpu(values)
        return scores


class Experimental_Scorer(object):
    def __init__(self, g, d, xp):
        self.g = g if config.DEVICE < 0 else g.to_gpu(config.DEVICE)
        self.d = d if config.DEVICE < 0 or d is None else d.to_gpu(config.DEVICE)
        self.xp = xp

    def set_candidate_t(self, candidate_t):
        self.bsz = candidate_t.shape[0]
        self.ct_emb = self.d.ent(candidate_t)

    def __call__(self, h, r):
        return self.get_g_score(h, r)

    def get_g_score(self, h, r):
        x = F.concat([self.d.ent(h), self.d.rel(r)])
        t_tilde = self.g(x)
        values = self.xp.linalg.norm(t_tilde.data - self.ct_emb.data, axis=1)
        scores = chainer.cuda.to_cpu(values)
        return scores


def run_ranking_test(scorer, vocab_ent, test_data):
    xp = scorer.xp

    data_iter = chainer.iterators.SerialIterator(test_data, batch_size=1, repeat=False, shuffle=False)
    candidate_t = xp.arange(1, len(vocab_ent) + 1, dtype=xp.int32)
    if config.DEVICE >= 0:
        candidate_t = chainer.dataset.to_device(config.DEVICE, candidate_t) # shape of (#entity_num, embedding_size)
    scorer.set_candidate_t(candidate_t)

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
            logging.info('%d testing data processed, temp rank: %d, hits10: %d, avgrank: %.4f' % (
                count, avgrank, hits10, avgrank / float(count)))

    avgrank /= count * 1.0
    hits10 /= count * 1.0
    print "avgrank:", avgrank, "hits@10:", hits10, "count:", count

if __name__ == "__main__":
    main()
