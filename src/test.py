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

    # MLE Scorer
    ent_num, rel_num = len(vocab_ent) + 1, len(vocab_rel) + 1
    generator = models.VarMLP([config.EMBED_SZ * 2, config.EMBED_SZ, config.EMBED_SZ, ent_num])
    embeddings = models.Embeddings(config.EMBED_SZ, ent_num, rel_num)
    chainer.serializers.load_npz(args.models[0], generator)
    chainer.serializers.load_npz(args.models[1], embeddings)
    if config.DEVICE >= 0:
        chainer.cuda.get_device_from_id(config.DEVICE).use()
        generator.to_gpu(config.DEVICE)
        embeddings.to_gpu(config.DEVICE)
    scorer = MLEGen_Scorer(generator, embeddings, xp)

    # # Experimental tesing
    # ent_num, rel_num = len(vocab_ent) + 1, len(vocab_rel) + 1
    # generator = models.VarMLP([config.EMBED_SZ * 2, config.EMBED_SZ, config.EMBED_SZ, ent_num])
    # embeddings = models.Embeddings(config.EMBED_SZ, ent_num, rel_num)
    # discriminator = models.VarMLP([config.EMBED_SZ * 3, config.EMBED_SZ, config.EMBED_SZ, 1])
    # chainer.serializers.load_npz(args.models[0], generator)
    # chainer.serializers.load_npz(args.models[1], discriminator)
    # chainer.serializers.load_npz(args.models[2], embeddings)
    #
    # if config.DEVICE >= 0:
    #     chainer.cuda.get_device_from_id(config.DEVICE).use()
    #     generator.to_gpu(config.DEVICE)
    #     discriminator.to_gpu(config.DEVICE)
    #     embeddings.to_gpu(config.DEVICE)
    #
    # print args.models[0], args.models[1], args.models[2]
    # scorer = Experimental_Scorer(generator, discriminator, embeddings, xp)

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


class MLEGen_Scorer(object):
    def __init__(self, g, e, xp):
        self.g = g if config.DEVICE < 0 else g.to_gpu(config.DEVICE)
        self.e = e if config.DEVICE < 0 or e is None else e.to_gpu(config.DEVICE)
        self.xp = xp

    def set_candidate_t(self, candidate_t):
        self.bsz = candidate_t.shape[0]
        self.ct_emb = self.e.ent(candidate_t)

    def __call__(self, h, r):
        h_raw = self.e.ent(h).reshape(h.shape[0], -1)   # (1, emb_sz)
        r_raw = self.e.rel(r).reshape(r.shape[0], -1)   # (1, emb_sz)
        logits = self.g(F.concat([h_raw, r_raw])).reshape(-1)
        value = -logits
        s = chainer.cuda.to_cpu(value.data)
        return s


class Experimental_Scorer(object):
    def __init__(self, g, d, e, xp):
        self.g = g if config.DEVICE < 0 else g.to_gpu(config.DEVICE)
        self.d = d if config.DEVICE < 0 or d is None else d.to_gpu(config.DEVICE)
        self.e = e if config.DEVICE < 0 or e is None else e.to_gpu(config.DEVICE)
        self.xp = xp

    def set_candidate_t(self, candidate_t):
        self.bsz = candidate_t.shape[0]
        self.ct_emb = self.e.ent(candidate_t)

    def __call__(self, h, r):
        h_raw = self.e.ent(h).reshape(h.shape[0], -1)   # (1, emb_sz)
        r_raw = self.e.rel(r).reshape(r.shape[0], -1)   # (1, emb_sz)
        d_value = self.get_d_score(h_raw, r_raw)
        # g_value = self.get_g_score(h_raw, r_raw)
        # value = - d_value - g_value
        # value = -g_value
        value = -d_value
        s = chainer.cuda.to_cpu(value.data)
        return s

    def get_g_score(self, h_emb, r_emb):
        logits = self.g(F.concat([h_emb, r_emb])).reshape(-1)
        # prob = F.softmax(logits)
        return logits

    def get_d_score(self, h_emb, r_emb):
        h_emb = F.broadcast_to(h_emb, self.ct_emb.shape)
        r_emb = F.broadcast_to(r_emb, self.ct_emb.shape)
        values = self.d(F.concat([h_emb, r_emb, self.ct_emb]))
        values = values.reshape(-1)
        return values


def run_ranking_test(scorer, vocab_ent, test_data):
    xp = scorer.xp

    data_iter = chainer.iterators.SerialIterator(test_data, batch_size=1, repeat=False, shuffle=False)
    candidate_t = xp.arange(0, len(vocab_ent) + 1, dtype=xp.int32)
    if config.DEVICE >= 0:
        candidate_t = chainer.dataset.to_device(config.DEVICE, candidate_t)  # shape of (#entity_num, embedding_size)
    scorer.set_candidate_t(candidate_t)

    avgrank, hits10, count = 0, 0, 0
    for i, batch in enumerate(data_iter):
        h, r, t = batch[0] # each one is an array of shape (1, )
        if config.DEVICE >= 0:
            h = chainer.dataset.to_device(config.DEVICE, h)
            r = chainer.dataset.to_device(config.DEVICE, r)

        scores = scorer(h, r)
        sorted_index = np.argsort(scores)
        rank = np.where(sorted_index == t[0])[0][0]  # tail ent id 1 ~ maxid, but the sorted index: 0 ~ maxid
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
