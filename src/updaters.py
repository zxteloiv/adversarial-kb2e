# coding: utf-8

import chainer
import chainer.functions as F
import numpy as np
import models


class AbstractGANUpdator(chainer.training.StandardUpdater):
    def __init__(self, data_iter, opt_g, opt_d, device, d_epoch, g_epoch):
        super(AbstractGANUpdator, self).__init__(data_iter, {"opt_g": opt_g, "opt_d": opt_d}, device=device)
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


class WGANUpdator(AbstractGANUpdator):
    def __init__(self, iterator, opt_g, opt_d, device, d_epoch=10, g_epoch=1, penalty_coeff=10):
        super(WGANUpdator, self).__init__(iterator, opt_g, opt_d, device, d_epoch=d_epoch, g_epoch=g_epoch)
        self.pen_coeff = penalty_coeff

    def update_d(self, h, r, t):
        h_emb = self.g.embed_entity(h)
        r_emb = self.g.embed_relation(r)
        t_emb = self.g.embed_entity(t) # batch * embedding

        t_tilde = self.g(h, r) # batch * embedding(generator output)

        # sampling
        epsilon = self.xp.random.uniform(0.0, 1.0, (h.shape[0], 1))
        t_hat = epsilon * t_emb + (1 - epsilon) * t_tilde
        delta = 1e-7
        t_hat_delta = t_hat + delta
        # t_hat_delta = (epsilon + delta) * t + (1 - epsilon - delta) * t_tilde

        delta_approx = F.sqrt(F.batch_l2_norm_squared(t_hat_delta - t_hat)).reshape(-1, 1)
        derivative = (self.d(h_emb, r_emb, t_hat_delta) - self.d(h_emb, r_emb, t_hat)) / delta_approx + delta
        penalty = (F.sqrt(F.batch_l2_norm_squared(derivative)) - 1) ** 2
        loss_penalty = F.average(penalty) * self.pen_coeff

        loss_gan = F.average(self.d(h_emb, r_emb, t_tilde) - self.d(h_emb, r_emb, t_emb))
        loss_d = loss_gan + loss_penalty

        self.d.cleargrads()
        loss_d.backward()
        self.get_optimizer('opt_d').update()
        ## weight clipping for lipschitz continuity
        # for name, param in self.d.namedparams():
        #     if param.data is None:
        #         continue
        #     param.data = xp.clip(param.data, -0.01, 0.01)
        # print loss_d.data

        self.add_to_report(loss_d=loss_d, loss_gan=loss_gan, loss_penalty=loss_penalty)

    def update_g(self, h, r, t):
        h_emb, r_emb = self.g.embed_entity(h), self.g.embed_relation(r)
        t_tilde = self.g(h, r)
        # loss_supervised = F.batch_l2_norm_squared(t_tilde - self.g.embed_entity(t)).reshape(-1, 1)
        loss_g = F.average(-self.d(h_emb, r_emb, t_tilde))
        self.g.cleargrads()
        loss_g.backward()
        self.get_optimizer('opt_g').update()
        self.add_to_report(loss_g=loss_g)

    @staticmethod
    def get_report_list():
        return ['epoch', 'iteration', 'loss_g', 'w-distance', 'penalty', 'elapsed_time']


class GANUpdater(AbstractGANUpdator):
    """the most basic GAN"""
    def __init__(self, data_iter, opt_g, opt_d, opt_eg, opt_ed, ent_num, device, d_epoch, g_epoch=5):
        super(GANUpdater, self).__init__(data_iter, opt_g, opt_d, device, d_epoch, g_epoch)
        self.opt_eg = opt_eg
        self.opt_ed = opt_ed
        self.g_emb = opt_eg.target
        self.d_emb = opt_ed.target
        self.ent_num = ent_num

    def update_d(self, h, r, t):
        bsz = h.shape[0]
        h_g_emb, t_g_emb = map(lambda x: self.g_emb.ent(x).reshape(bsz, -1), (h, t))
        r_g_emb = self.g_emb.rel(r).reshape(bsz, -1)
        h_d_emb, t_d_emb = map(lambda x: self.d_emb.ent(x).reshape(bsz, -1), (h, t))
        r_d_emb = self.d_emb.rel(r).reshape(bsz, -1)

        samples, _ = self.sample_g(h_g_emb, r_g_emb)
        samples_emb = self.d_emb.ent(samples).reshape(bsz, -1)

        # traditional GAN
        loss_real = -F.log(F.sigmoid(self.d(F.concat((h_d_emb, r_d_emb, t_d_emb)))) + 1e-9)
        loss_gen = -F.log(1 - F.sigmoid(self.d(F.concat((h_d_emb, r_d_emb, samples_emb)))) + 1e-9)
        loss_real = F.sum(loss_real)
        loss_gen = F.sum(loss_gen)
        loss_d = loss_real + loss_gen

        self.d.cleargrads()
        self.d_emb.cleargrads()
        loss_d.backward()
        self.get_optimizer('opt_d').update()
        self.opt_ed.update()
        self.add_to_report(loss_d=loss_d, loss_real=loss_real, loss_gen=loss_gen)

    def update_g(self, h, r, t):
        bsz = h.shape[0]
        h_g_emb = self.g_emb.ent(h).reshape(bsz, -1)
        r_g_emb = self.g_emb.rel(r).reshape(bsz, -1)
        h_d_emb = self.d_emb.ent(h).reshape(bsz, -1)
        r_d_emb = self.d_emb.rel(r).reshape(bsz, -1)

        samples, probs = self.sample_g(h_g_emb, r_g_emb)
        sample_probs = F.select_item(probs, samples).reshape(bsz, 1)

        samples_emb = self.d_emb.ent(samples)
        reward = F.log(1 - F.sigmoid(self.d(F.concat((h_d_emb, r_d_emb, samples_emb)))) + 1e-9)

        loss_g_adv = reward * F.log(sample_probs + 1e-9)
        loss_g_adv = F.sum(loss_g_adv)

        logits = self.g(F.concat((h_g_emb, r_g_emb)))
        loss_g_ce = F.softmax_cross_entropy(logits, t.reshape(-1), normalize=False) * 100

        loss_g = loss_g_adv + loss_g_ce

        self.g.cleargrads()
        self.g_emb.cleargrads()
        loss_g.backward()
        self.get_optimizer('opt_g').update()
        self.opt_eg.update()
        self.add_to_report(loss_g=loss_g, loss_g_adv=loss_g_adv, reward=F.sum(reward), loss_g_ce=loss_g_ce)

    def sample_g(self, h_emb, r_emb):
        bsz = h_emb.shape[0]
        logits = self.g(F.concat((h_emb, r_emb)))
        probs = F.softmax(logits, axis=1)
        samples = batch_multinomial(chainer.cuda.get_array_module(h_emb), probs, 1).reshape(bsz,)
        return samples, probs

    @staticmethod
    def get_report_list():
        return ['epoch', 'iteration', 'loss_g', 'loss_g_adv', 'reward', 'loss_g_ce',
                'loss_d', 'loss_real', 'loss_gen', 'elapsed_time']


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
    return nums  #F.transpose(nums)
