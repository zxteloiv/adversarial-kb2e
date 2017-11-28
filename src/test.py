# coding: utf-8

from __future__ import absolute_import
import os, argparse, logging, itertools
import chainer
import chainer.functions as F
import numpy as np

import config, models
import corpus.dataset as mod_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('models', nargs='+')
    parser.add_argument('--use-valid', '-v', action='store_true', help="use validation set, otherwise test")
    parser.add_argument('--setting', '-s', default='gan', choices=['gan', 'transe', 'mle'])
    parser.add_argument('--msg', '-m', help='a message to print')
    parser.add_argument('--alpha', '-a', default=.5, type=float, help='GAN: alpha * d_value + (1-alpha) * g_value')
    parser.add_argument('--full', action='store_true', help='do a full testing')
    parser.add_argument('--filter', action='store_true', help='do a full testing')
    parser.add_argument('--rel-cate', '-c', help="relation category tsv file")
    args = parser.parse_args()

    chainer.config.train = False
    full_testing = True if args.full else False

    if config.DATASET_IN_USE != "FB15k" and args.rel_cate:
        print "relation to category not available for datasets other than FB15k"
        return

    vocab_ent, vocab_rel = mod_dataset.load_vocab()
    ent_num, rel_num = len(vocab_ent) + 1, len(vocab_rel) + 1

    rel2cate = None
    if args.rel_cate:
        rel2cate = get_rel_cate(vocab_rel, args.rel_cate)

    logging.getLogger().setLevel(logging.INFO)
    if args.msg:
        logging.info("*==================*")
        logging.info(args.msg)

    logging.info('ent vocab size=%d, rel vocab size=%d' % (len(vocab_ent), len(vocab_rel)))
    valid_data, test_data = map(lambda f: mod_dataset.load_corpus(f, vocab_ent, vocab_rel),
                                (config.VALID_DATA, config.TEST_DATA))
    logging.info('data loaded, size: train:valid:test=-:%d:%d' % (len(valid_data), len(test_data)))

    factbase = {}
    if args.filter:
        train_data = mod_dataset.load_corpus(config.TRAIN_DATA, vocab_ent, vocab_rel)
        logging.info('training data loaded, size: train=%d' % len(train_data))
        for (h, r, t) in itertools.chain(train_data, valid_data, test_data):
            h, r, t = h[0], r[0], t[0]
            if (h, r) not in factbase:
                factbase[(h, r)] = np.zeros((ent_num,), dtype=np.int32)
            factbase[(h, r)][t] = 1

    xp = np
    if config.DEVICE >= 0:
        chainer.cuda.get_device_from_id(config.DEVICE).use()
        from chainer.cuda import cupy
        xp = cupy

    # classical TransE
    if args.setting == 'transe':
        transE = models.TransE(config.EMBED_SZ, ent_num, rel_num, config.MARGIN, config.TRANSE_NORM)
        chainer.serializers.load_npz(args.models[0], transE)
        scorer = TransE_Scorer(transE, xp)

    # GAN testing
    elif args.setting == 'gan':
        generator = models.Generator(config.EMBED_SZ, ent_num, rel_num, config.DROPOUT)
        # discriminator = models.Discriminator(config.EMBED_SZ, ent_num, rel_num, config.DROPOUT)
        discriminator = models.TransE(config.EMBED_SZ, ent_num, rel_num, config.MARGIN)
        chainer.serializers.load_npz(args.models[0], generator)
        chainer.serializers.load_npz(args.models[1], discriminator)
        if config.DEVICE >= 0:
            generator.to_gpu(config.DEVICE)
            discriminator.to_gpu(config.DEVICE)
        scorer = GAN_Scorer(generator, discriminator, xp, args.alpha)

    # MLE Scorer
    elif args.setting == 'mle':
        generator = models.VarMLP([config.EMBED_SZ * 2, config.EMBED_SZ, config.EMBED_SZ, ent_num], config.DROPOUT)
        embeddings = models.Embeddings(config.EMBED_SZ, ent_num, rel_num)
        chainer.serializers.load_npz(args.models[0], generator)
        chainer.serializers.load_npz(args.models[1], embeddings)
        if config.DEVICE >= 0:
            chainer.cuda.get_device_from_id(config.DEVICE).use()
            generator.to_gpu(config.DEVICE)
            embeddings.to_gpu(config.DEVICE)
        scorer = MLEGen_Scorer(generator, embeddings, xp)

    else:
        scorer = None

    if args.use_valid:  # use validation set
        print "validation"
        run_ranking_test(scorer, vocab_ent, valid_data, full_testing=full_testing, factbase=factbase, rel2cate=rel2cate)
    else:
        print "testing"
        run_ranking_test(scorer, vocab_ent, test_data, full_testing=full_testing, factbase=factbase, rel2cate=rel2cate)


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

        values = self.xp.linalg.norm(h_emb + r_emb - self.ct_emb, ord=config.TRANSE_NORM, axis=1)  # norm value vector with shape of (#entity_num, )

        scores = chainer.cuda.to_cpu(values) # cupy doesn't support argsort yet
        return scores


class GAN_Scorer(object):
    def __init__(self, g, d, xp, alpha=.5):
        self.g = g if config.DEVICE < 0 else g.to_gpu(config.DEVICE)
        self.d = d if config.DEVICE < 0 else d.to_gpu(config.DEVICE)
        self.xp = xp
        self.alpha = alpha

    def set_candidate_t(self, candidate_t):
        self.bsz = candidate_t.shape[0]
        self.ct = candidate_t

    def __call__(self, h, r):
        d_value = self.get_d_score(h, r)
        g_value = self.get_g_score(h, r)
        values = self.alpha * d_value + (1 - self.alpha) * (-g_value)
        # values = d_value
        # values = -g_value
        scores = chainer.cuda.to_cpu(values)
        return scores

    def get_d_score(self, h, r):
        h = F.broadcast_to(h, (self.bsz, 1))
        r = F.broadcast_to(r, (self.bsz, 1))
        values = self.d.dist(h, r, self.ct).reshape(-1).data
        return values

    def get_g_score(self, h, r):
        t_logits = self.g(h, r).reshape(-1)
        values = t_logits
        return values.data


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
        value = logits
        s = chainer.cuda.to_cpu(value.data)
        return s


def run_ranking_test(scorer, vocab_ent, test_data, full_testing=True, factbase=None, rel2cate=None):
    xp = scorer.xp

    data_iter = chainer.iterators.SerialIterator(test_data, batch_size=1, repeat=False, shuffle=False)
    candidate_t = xp.arange(0, len(vocab_ent) + 1, dtype=xp.int32)
    if config.DEVICE >= 0:
        candidate_t = chainer.dataset.to_device(config.DEVICE, candidate_t)  # shape of (#entity_num, embedding_size)
    scorer.set_candidate_t(candidate_t)

    avgrank, hits10, count = 0, 0, 0
    hits10bycate, countbycate = {}, {}
    for i, batch in enumerate(data_iter):
        try:
            h, r, t = batch[0] # each one is an array of shape (1, )
            if config.DEVICE >= 0:
                hg = chainer.dataset.to_device(config.DEVICE, h)
                rg = chainer.dataset.to_device(config.DEVICE, r)
            else:
                hg, rg = h, r

            scores = scorer(hg, rg)

            # filtered setting
            if factbase is not None and len(factbase) > 0:
                offsetvec = factbase.get((h[0], r[0]))
                offsetvec[t[0]] = 0     # make sure current triple is not affected by the filtering
                maxscore = np.max(scores)
                scores += offsetvec * maxscore

            sorted_index = np.argsort(scores)
            rank = np.where(sorted_index == t[0])[0][0]  # tail ent id 1 ~ maxid, but the sorted index: 0 ~ maxid
            avgrank += rank
            hits10 += 1 if rank < 10 else 0
            count += 1

            if rel2cate is not None:
                cate = rel2cate[r[0]]
                if cate not in hits10bycate:
                    hits10bycate[cate] = 0
                if cate not in countbycate:
                    countbycate[cate] = 0
                hits10bycate[cate] += 1 if rank < 10 else 0
                countbycate[cate] += 1

            if i % 1000 == 0:
                logging.info('%d testing data processed, temp rank: %d, hits10: %d, hits10p: %.4f, avgrank: %.4f' % (
                    count, avgrank, hits10, hits10 * 1.0 / count, avgrank / float(count)))
                if rel2cate is not None:
                    print ", ".join("{0} => {1:.4f}".format(cate, hits10bycate[cate] * 1.0 / countbycate[cate])
                             for cate in hits10bycate)

            if i / 1000 >= 10 and not full_testing:
                break
        except KeyboardInterrupt:
            break

    avgrank /= count * 1.0
    hits10 /= count * 1.0
    print "avgrank:", avgrank, "hits@10:", hits10, "count:", count
    for cate, hits in hits10bycate.iteritems():
        print cate, "=>", hits * 1.0 / countbycate[cate]



def get_rel_cate(vocab_rel, catefile):
    rel2cate = {}
    for l in open(catefile):
        r, cate = l.rstrip().split('\t')
        rel2cate[vocab_rel(r)] = cate

    return rel2cate


if __name__ == "__main__":
    main()
