# coding: utf-8

import chainer
import chainer.functions as F
import numpy as np
import models


class AbstractGANUpdater(chainer.training.StandardUpdater):
    def __init__(self, data_iter, opt_g, opt_d, device, d_epoch, g_epoch):
        super(AbstractGANUpdater, self).__init__(data_iter, {"opt_g": opt_g, "opt_d": opt_d}, device=device)
        self.d_epoch = d_epoch
        self.g_epoch = g_epoch
        self.d = opt_d.target
        self.g = opt_g.target
        self.xp = np
        if self.device >= 0:
            from chainer.cuda import cupy
            self.xp = cupy
        self.reports = {}

    def update_core(self):
        data_iter = self.get_iterator('main')
        self.reports = {}

        for epoch in xrange(self.d_epoch):
            batch = data_iter.__next__()
            tuple = self.converter(batch, self.device)

            self.update_d(*tuple)

        for epoch in xrange(self.g_epoch):
            batch = data_iter.__next__()
            tuple = self.converter(batch, self.device)

            self.update_g(*tuple)

        chainer.report(self.reports)

    def update_d(self, *args):
        raise NotImplementedError

    def update_g(self, *args):
        raise NotImplementedError

    def add_to_report(self, **kwargs):
        self.reports.update(kwargs)

    @staticmethod
    def get_report_list():
        return ['epoch', 'iteration', 'elapsed_time']


class GANUpdater(AbstractGANUpdater):
    def __init__(self, iterator, opt_g, opt_d, device, d_epoch=10, g_epoch=1, margin=10.,
                 greater_d_value_better=True, dist_based_g=False, dist_based_d=False, sample_num=10):
        super(GANUpdater, self).__init__(iterator, opt_g, opt_d, device, d_epoch=d_epoch, g_epoch=g_epoch)
        self.baseline = self.xp.array([0.]).astype('f')
        self.margin = margin
        self.greater_d_value_is_better = greater_d_value_better
        self.distance_based_g = dist_based_d
        self.distance_based_d = dist_based_g
        self.K = sample_num

    def update_d(self, h, r, t):
        bsz = h.shape[0]
        t_logits = self.g(h, r)     # batch * embedding(generator output)
        t_probs = F.softmax(t_logits)

        t_sample_neg = batch_multinomial(self.xp, t_probs, 1)       # (bsz, 1)
        t_sample_pos = batch_multinomial(self.xp, t_probs, self.K)  # (bsz, K)
        rank_pos = self.get_rank(h, r, t, t_sample_pos)
        rank_neg = self.get_rank(h, r, t_sample_neg, t)

        loss_d = -F.sum(rank_pos - rank_neg)

        self.d.cleargrads()
        loss_d.backward()
        self.get_optimizer('opt_d').update()
        self.add_to_report(loss_d=loss_d)

    def get_rank(self, h, r, t, ct, temperature=.2):
        """
        Rank a tail entity over a ct set

        :param h: head, (bsz, 1)
        :param r: relation, (bsz, 1)
        :param t: tail, (bsz, 1)
        :param ct: candidate tail, (bsz, K)
        :return: has shape (bsz, 1 + K)
        """
        bsz, K = ct.shape
        ts = F.concat([t, ct], axis=1)      # (bsz, 1 + K)
        hh = F.broadcast_to(h, ts.shape)    # (bsz, 1) -> (bsz, 1 + K)
        rr = F.broadcast_to(r, ts.shape)    # (bsz, 1) -> (bsz, 1 + K)

        hh = hh.reshape(-1, 1)              # (bsz, 1 + K) -> (bsz * (1 + K), 1)
        rr = rr.reshape(-1, 1)
        ts = ts.reshape(-1, 1)

        vals = self.d.dist(hh, rr, ts)     # (bsz * (1 + K), 1)
        vals = vals.reshape(bsz, -1)        # (bsz * (1 + K), 1) -> (bsz, 1 + K)
        if not self.greater_d_value_is_better:
            vals = -vals

        if self.distance_based_d:
            target_val = self.d.dist(h, r, t)  # (bsz, 1)
            if not self.greater_d_value_is_better:
                target_val = -target_val
            target_val = F.broadcast_to(target_val, vals.shape)     # (bsz, 1 + K)
            dist = F.sqrt(F.square(vals - target_val) + 1e-18)      # (bsz, 1 + K)
            vals = dist

        scores = F.softmax(vals / temperature, axis=1)
        scores = F.select_item(scores, self.xp.zeros(shape=(bsz,)).astype('i'))
        return scores

    def update_g(self, h, r, t):
        bsz = h.shape[0]
        t_logits = self.g(h, r)

        probs = batch_gumbel_softmax(self.xp, t_logits, 1.)

        sampleval = self.d.dist_one_hot_t(h, r, probs)  # (bsz, 1)
        targetval = self.d.dist(h, r, t)                # (bsz, 1)
        if not self.greater_d_value_is_better:
            sampleval = -sampleval
            targetval = -targetval

        if self.distance_based_g:
            sampledist = F.sqrt(F.square(sampleval - targetval) + 1e-15)  # (bsz, 1)
            targetdist = F.sqrt(F.square(targetval - targetval) + 1e-15)  # (bsz, 1)
            probs = F.softmax(F.concat([sampledist, targetdist], axis=1) / .1, axis=1)  # (bsz, 2)
            ranks = F.select_item(probs, self.xp.zeros(shape=(bsz,)).astype('i'))
        else:
            probs = F.softmax(F.concat([sampleval, targetval], axis=1) / .1, axis=1)    # (bsz, 2)
            ranks = F.select_item(probs, self.xp.zeros(shape=(bsz,)).astype('i'))

        loss_g = -F.sum(ranks)


        # t_probs = F.softmax(t_logits)
        # t_sample = batch_multinomial(self.xp, t_probs, 1)   # (bsz, 1)
        # reward = self.d.dist(h, r, t_sample)
        # eligibility = F.log(F.select_item(F.softmax(t_probs), t_sample.reshape(-1)) + 1e-12).reshape(bsz, -1)
        # loss_g = -F.sum(eligibility * (reward - F.broadcast_to(self.baseline, reward.shape)))
        # self.baseline = F.average(reward)  # a constant to be used in the next iteration

        self.g.cleargrads()
        loss_g.backward()
        self.get_optimizer('opt_g').update()

        self.add_to_report(loss_g=loss_g)

    @staticmethod
    def get_report_list():
        return ['epoch', 'iteration', 'loss_g', 'loss_d', 'elapsed_time']


class MLEGenUpdater(chainer.training.StandardUpdater):
    def __init__(self, data_iter, opt_g, opt_e, ent_num, device=-1):
        super(MLEGenUpdater, self).__init__(data_iter, opt_g, device=device)
        self.opt_e = opt_e
        self.opt_g = opt_g
        self.g = opt_g.target
        self.emb = opt_e.target
        self.ent_num = ent_num

    def update_core(self):
        data_iter = self.get_iterator('main')
        batch = data_iter.__next__()
        h, r, t = self.converter(batch, self.device)
        self.batch_update(h, r, t)

    def batch_update(self, h, r, t):
        xp = chainer.cuda.get_array_module(h)
        bsz = h.shape[0]
        h_emb = self.emb.ent(h).reshape(bsz, -1)    # (bsz, emb_sz)
        r_emb = self.emb.rel(r).reshape(bsz, -1)    # (bsz, emb_sz)
        logits = self.g(F.concat([h_emb, r_emb]))   # (bsz, V)

        loss = F.softmax_cross_entropy(logits, t.reshape(-1))   # (bsz, V)

        self.g.cleargrads()
        self.emb.cleargrads()
        loss.backward()
        self.get_optimizer('main').update()
        self.opt_e.update()

        chainer.report({'loss_g': loss})

    @staticmethod
    def get_report_list():
        return ['epoch', 'iteration', 'loss_g', 'elapsed_time']


class MLEGenNNSUpdater(MLEGenUpdater):
    def batch_update(self, h, r, t):
        xp = chainer.cuda.get_array_module(h)
        bsz = h.shape[0]
        h_emb = self.emb.ent(h).reshape(bsz, -1)    # (bsz, emb_sz)
        r_emb = self.emb.rel(r).reshape(bsz, -1)    # (bsz, emb_sz)
        logits = self.g(F.concat([h_emb, r_emb]))   # (bsz, V)

        probs = F.sigmoid(logits)   # (bsz, V)
        selected = F.select_item(probs, t.reshape(-1))
        loss = F.average(selected)

        self.g.cleargrads()
        self.emb.cleargrads()
        loss.backward()
        self.get_optimizer('main').update()
        self.opt_e.update()

        chainer.report({'loss_g': loss})


class MLEGenNSUpdater(MLEGenUpdater):
    """MLE Generator with negative sampling"""
    def batch_update(self, h, r, t):
        xp = chainer.cuda.get_array_module(h)
        bsz = h.shape[0]
        h_emb = self.emb.ent(h).reshape(bsz, -1)    # (bsz, emb_sz)
        r_emb = self.emb.rel(r).reshape(bsz, -1)    # (bsz, emb_sz)
        logits = self.g(F.concat([h_emb, r_emb]))   # (bsz, V)

        loss_pos = F.sigmoid(F.select_item(logits, t.reshape(-1)))

        t_corr = xp.random.randint(1, self.ent_num + 1, size=(bsz, 1)).astype('i')
        loss_neg_t = F.sigmoid(F.select_item(logits, t_corr.reshape(-1)))

        margin = 0.3 * xp.sign(xp.absolute(t - t_corr)).reshape(-1)
        loss = F.average(F.relu(loss_pos - loss_neg_t + margin))

        self.g.cleargrads()
        self.emb.cleargrads()
        loss.backward()
        self.get_optimizer('main').update()
        self.opt_e.update()

        chainer.report({'loss_g': loss})


class GANPretraining(chainer.training.StandardUpdater):
    def __init__(self, data_iter, opt_g, opt_d, ent_num, rel_num, margin=1., device=-1):
        super(GANPretraining, self).__init__(data_iter, opt_g, device=device)
        self.g = opt_g.target
        self.d = opt_d.target
        self.opt_g = opt_g
        self.opt_d = opt_d
        self.margin = margin
        self.ent_num = ent_num
        self.rel_num = rel_num

    def update_core(self):
        batch = self.get_iterator('main').__next__()
        h, r, t = self.converter(batch, self.device)
        bsz = h.shape[0]

        # train G with softmax-cross-entropy
        logits = self.g(h, r)
        loss_g = F.softmax_cross_entropy(logits, t.reshape(-1))
        self.g.cleargrads()
        loss_g.backward()
        self.opt_g.update()
        chainer.report({'loss_g': loss_g})

        # train D with hinge loss and random negative sampling
        loss_pos = self.d(h, r, t)
        xp = chainer.cuda.get_array_module(h)
        t_neg = xp.random.randint(0, self.ent_num, size=(bsz, 1))
        loss_neg = self.d(h, r, t_neg)
        margin = self.margin * xp.sign(xp.absolute(t - t_neg)).reshape(bsz, 1)
        loss_d = F.sum(F.relu(loss_neg - loss_pos + margin))
        self.d.cleargrads()
        loss_d.backward()
        self.opt_d.update()
        chainer.report({'loss_d': loss_d})

    @staticmethod
    def get_report_list():
        return ['epoch', 'iteration', 'loss_g', 'loss_d', 'elapsed_time']


def batch_multinomial(xp, batch_probs, size):
    """
    Sample the multinomial distributions given a batch of probabilities
    :param batch_probs: has shape (batch_size, V), a batch of probabilities
    :param size: int, the number of examples to sample every distribution
    :return: has shape (batch_size, size)
    """
    log_probs = F.log(batch_probs)                                      # (bsz, V)
    nums = xp.empty((size, log_probs.shape[0])).astype('int32')    # (K, bsz)
    for i in xrange(size):
        noise = xp.random.rand(*log_probs.shape).astype('f')   # bsz, V
        rand = F.argmax(log_probs - F.log(-F.log(noise)), axis=1)
        nums[i] = rand.data

    # # sampling the batch of distributions for K times directly will exhaust the GPU memory,
    # # thus still we choose to use sample K numbers each on each

    # K_log_probs = F.broadcast_to(log_probs, (size,) + log_probs.shape)  # (K, bsz, V)
    # noise = xp.random.rand(*K_log_probs.shape).astype('f')         # (K, bsz, V)
    # nums = F.argmax(K_log_probs - F.log(-F.log(noise)), axis=2)         # (K, bsz)
    # nums_t = F.transpose(nums)
    # del log_probs, K_log_probs, noise, nums
    # return nums_t
    # return xp.ones((batch_probs.shape[0], size), dtype=self.xp.int32)
    return F.transpose(nums)    # (K, bsz) -> (bsz, K)


def standard_gumbel(xp, shape, eps=1e-15):
    """sample a standard Gumbel distribution (Gumbel(0, 1))"""
    U = xp.random.rand(*shape).astype('f')
    samples = -xp.log(-xp.log(U + eps) + eps)
    return samples


def batch_gumbel_softmax(xp, logits, temperature=1., hard=False):
    samples = standard_gumbel(xp, logits.shape)
    y = logits + samples
    probs = F.softmax(y / temperature)
    return probs
