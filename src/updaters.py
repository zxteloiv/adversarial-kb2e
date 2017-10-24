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


class LSGANUpdater(AbstractGANUpdator):
    def __init__(self, data_iter, opt_g, opt_d, device, d_epoch, g_epoch=1, hinge_loss_weight=1, penalty_coeff=1):
        super(LSGANUpdater, self).__init__(data_iter, opt_g, opt_d, device, d_epoch, g_epoch)
        self.hinge_loss_weight = hinge_loss_weight
        self.penalty_coeff = penalty_coeff

    def update_d(self, h, r, t):
        t_tilde = self.g(h, r)

        # loss for real examples
        h_emb, t_emb = map(self.g.embed_entity, (h, t))
        r_emb = self.g.embed_relation(r)
        loss_real = self.d(h_emb, r_emb, t_emb)

        # hinge loss for generated loss and the distance margin
        hinge_loss = F.relu(F.batch_l2_norm_squared(t_tilde - t_emb).reshape(-1, 1)
                            + self.d(h_emb, r_emb, t_emb)
                            - self.d(h_emb, r_emb, t_tilde))
        hinge_loss *= self.hinge_loss_weight

        # Lipschitz Regularization for discriminator
        epsilon, delta = self.xp.random.uniform(0.0, 1.0, (h.shape[0], 1)), 1e-7
        t_hat = epsilon * t_emb + (1 - epsilon) * t_tilde
        t_hat_delta = t_hat + delta
        delta_approx = F.sqrt(F.batch_l2_norm_squared(t_hat_delta - t_hat)).reshape(-1, 1)
        derivative = (self.d(h_emb, r_emb, t_hat_delta) - self.d(h_emb, r_emb, t_hat)) / delta_approx + delta
        penalty = ((F.sqrt(F.batch_l2_norm_squared(derivative)) - 1) ** 2) * self.penalty_coeff

        loss_real = F.average(loss_real)
        hinge_loss = F.average(hinge_loss)
        loss_d_penalty = F.average(penalty)

        loss_d = loss_real + hinge_loss + loss_d_penalty
        self.d.cleargrads()
        loss_d.backward()
        self.get_optimizer('opt_d').update()
        self.add_to_report(loss_d=loss_d, loss_real=loss_real, hinge_loss=hinge_loss, d_penalty=loss_d_penalty)

    def update_g(self, h, r, t):
        # generator loss given a fixed discriminator
        h_emb = self.g.embed_entity(h)
        r_emb = self.g.embed_relation(r)
        t_tilde = self.g(h, r)
        loss_gen = F.average(self.d(h_emb, r_emb, t_tilde))

        # Lipschitz Regularization for generator
        epsilon, delta = self.xp.random.uniform(0.0, 1.0, (h.shape[0], 1)), 1e-7
        t_emb = self.g.embed_entity(t)
        t_hat = epsilon * t_emb + (1 - epsilon) * t_tilde
        t_hat_delta = t_hat + delta
        delta_approx = F.sqrt(F.batch_l2_norm_squared(t_hat_delta - t_hat)).reshape(-1, 1)
        derivative = (self.d(h_emb, r_emb, t_hat_delta) - self.d(h_emb, r_emb, t_hat)) / delta_approx + delta
        penalty = ((F.sqrt(F.batch_l2_norm_squared(derivative)) - 1) ** 2) * self.penalty_coeff
        loss_g_penalty = F.average(penalty)

        loss_g = loss_gen + loss_g_penalty
        self.g.cleargrads()
        loss_g.backward()
        self.get_optimizer('opt_g').update()
        self.add_to_report(loss_g=loss_g, loss_gen=loss_gen, g_penalty=loss_g_penalty)

    @staticmethod
    def get_report_list():
        return ['epoch', 'iteration', 'loss_g', 'loss_gen', 'g_penalty',
                'loss_d', 'loss_real', 'hinge_loss', 'd_penalty',
                'elapsed_time']


class FixedDGANUpdater(AbstractGANUpdator):
    """simple GAN but with fixed D, usually the D is a predefined and perfect classifier"""
    def __init__(self, data_iter, opt_g, opt_d, device, d_epoch, g_epoch=5):
        super(FixedDGANUpdater, self).__init__(data_iter, opt_g, opt_d, device, d_epoch, g_epoch)

    def update_d(self, *args):
        pass

    def update_g(self, h, r, t):
        # self.g.ent_emb.W.data = models.HingeLossGen.normalize_embedding(self.g.ent_emb.W.data)
        # self.g.rel_emb.W.data = models.HingeLossGen.normalize_embedding(self.g.rel_emb.W.data)

        h_emb = self.g.embed_entity(h)
        r_emb = self.g.embed_relation(r)
        t_emb = self.g.embed_entity(t)
        t_tilde = self.g(h, r)
        loss_g = F.sqrt(F.batch_l2_norm_squared(self.d(h_emb, r_emb, t_tilde)))
        loss_g = F.average(loss_g)

        # loss_sim = F.sqrt(F.batch_l2_norm_squared(self.d(h_emb, r_emb, t_tilde) - self.d(h_emb, r_emb, t_emb)) + 1e-7)
        # loss_sim = F.average(loss_sim)

        loss_sim = F.sqrt(F.batch_l2_norm_squared(t_tilde - t_emb) + 1e-7)
        loss_sim = F.average(loss_sim)

        loss = loss_g + loss_sim
        self.g.cleargrads()
        loss.backward()
        self.get_optimizer('opt_g').update()
        self.add_to_report(loss_g=loss_g, loss_sim=loss_sim)

    @staticmethod
    def get_report_list():
        return ['epoch', 'iteration', 'loss_g', 'loss_sim', 'elapsed_time']


class GANUpdater(AbstractGANUpdator):
    """the most basic GAN"""
    def __init__(self, data_iter, opt_g, opt_d, device, d_epoch, g_epoch=5):
        super(GANUpdater, self).__init__(data_iter, opt_g, opt_d, device, d_epoch, g_epoch)

    def update_d(self, h, r, t):
        h_emb, t_emb = map(self.g.embed_entity, (h, t))
        r_emb = self.g.embed_relation(r)
        t_tilde = self.g(h, r)

        # traditional GAN
        loss_real = -F.log(self.d(h_emb, r_emb, t_emb))
        loss_gen = -F.log(1 - self.d(h_emb, r_emb, t_tilde) + 1e-7)
        loss_real = F.sum(loss_real)
        loss_gen = F.sum(loss_gen)
        loss_d = loss_real + loss_gen

        self.d.cleargrads()
        loss_d.backward()
        self.get_optimizer('opt_d').update()
        self.add_to_report(loss_d=loss_d, loss_real=loss_real, loss_gen=loss_gen)

    def update_g(self, h, r, t):
        h_emb = self.g.embed_entity(h)
        r_emb = self.g.embed_relation(r)

        loss_g = F.log(1 - self.d(h_emb, r_emb, self.g(h, r)) + 1e-7)
        loss_g = F.sum(loss_g)

        self.g.cleargrads()
        loss_g.backward()
        self.get_optimizer('opt_g').update()
        self.add_to_report(loss_g=loss_g)

    @staticmethod
    def get_report_list():
        return ['epoch', 'iteration', 'loss_g','loss_d', 'loss_real', 'loss_gen', 'elapsed_time']


class LeastSquareGANUpdater(AbstractGANUpdator):
    def __init__(self, data_iter, opt_g, opt_d, device, d_epoch, g_epoch=5):
        super(LeastSquareGANUpdater, self).__init__(data_iter, opt_g, opt_d, device, d_epoch, g_epoch)

    def update_d(self, h, r, t):
        h_emb, t_emb = map(self.g.embed_entity, (h, t))
        r_emb = self.g.embed_relation(r)
        t_tilde = self.g(h, r)

        loss_real = F.batch_l2_norm_squared(self.d(h_emb, r_emb, t_emb) - 1) / 2
        loss_gen = F.batch_l2_norm_squared(self.d(h_emb, r_emb, t_tilde)) / 2
        loss_real = F.sum(loss_real)
        loss_gen = F.sum(loss_gen)
        loss_d = loss_real + loss_gen

        self.d.cleargrads()
        loss_d.backward()
        self.get_optimizer('opt_d').update()
        self.add_to_report(loss_d=loss_d, loss_real=loss_real, loss_gen=loss_gen)

    def update_g(self, h, r, t):
        h_emb = self.g.embed_entity(h)
        r_emb = self.g.embed_relation(r)

        loss_g = F.batch_l2_norm_squared(self.d(h_emb, r_emb, self.g(h, r)) - 1) / 2
        loss_g = F.sum(loss_g)

        self.g.cleargrads()
        loss_g.backward()
        self.get_optimizer('opt_g').update()
        self.add_to_report(loss_g=loss_g)

    @staticmethod
    def get_report_list():
        return ['epoch', 'iteration', 'loss_g','loss_d', 'loss_real', 'loss_gen', 'elapsed_time']


class ExperimentalGANUpdater(AbstractGANUpdator):
    def __init__(self, data_iter, opt_g, opt_d, opt_e, device, d_epoch, g_epoch, ent_num=None, sample_num=100):
        super(ExperimentalGANUpdater, self).__init__(data_iter, opt_g, opt_d, device, d_epoch, g_epoch)
        self.ent_num = self.d.ent.W.shape[0] if ent_num is None else ent_num
        self.opt_e = opt_e
        self.opt_g = opt_g
        self.opt_d = opt_d
        self.emb = opt_e.target
        self.sample_num = sample_num
        self.xp = np
        if self.device >= 0:
            from chainer.cuda import cupy
            self.xp = cupy

    def sample_g(self, h_raw, r_raw, sample_num=100):
        logits = self.g(F.concat([h_raw, r_raw]))                           # (bsz, V)
        prob = F.softmax(logits)                                            # (bsz, V)
        samples = self.batch_multinomial(prob, sample_num)
        return samples, prob

    def update_d(self, h, r, t):
        bsz = h.shape[0]
        h_raw, t_raw = map(lambda x: self.emb.ent(x).reshape(bsz, -1), (h, t))  # (bsz, emb_sz)
        r_raw = self.emb.rel(r).reshape(bsz, -1)                                # (bsz, emb_sz)

        loss_real = -F.sum(F.log(F.sigmoid(self.d(F.concat([h_raw, r_raw, t_raw])))))

        rand_ts, _ = self.sample_g(h_raw, r_raw, self.sample_num)           # (bsz, K), (bsz, V)
        rand_ts_emb = self.emb.ent(rand_ts)                                 # (bsz, K, emb_sz)
        rand_ts_emb = F.transpose(rand_ts_emb, axes=(1, 0, 2))              # (K, bsz, emb_sz)
        rand_ts_emb = rand_ts_emb.reshape(self.sample_num * bsz, -1)        # (K * bsz, emb_sz)

        h_emb = F.broadcast_to(h_raw, (self.sample_num, ) + h_raw.shape)    # (K, bsz, emb_sz)
        h_emb = h_emb.reshape(self.sample_num * bsz, -1)                    # (K * bsz, emb_sz)
        r_emb = F.broadcast_to(r_raw, (self.sample_num, ) + r_raw.shape)    # (K, bsz, emb_sz)
        r_emb = r_emb.reshape(self.sample_num * bsz, -1)                    # (K * bsz, emb_sz)

        loss_gen = F.log(1 - F.sigmoid(self.d(F.concat([h_emb, r_emb, rand_ts_emb]))) + 1e-7)   # (K * bsz, 1)
        loss_gen = -F.sum(loss_gen / self.sample_num)

        loss = loss_gen + loss_real
        self.d.cleargrads()
        self.emb.cleargrads()
        loss.backward()
        self.get_optimizer('opt_d').update()
        self.opt_e.update()
        self.add_to_report(loss_d=loss, loss_d_fake=loss_gen, loss_d_real=loss_real)

    def update_g(self, h, r, t):
        bsz = h.shape[0]
        h_raw, t_raw = map(lambda x: self.emb.ent(x).reshape(bsz, -1), (h, t))  # (bsz, V)
        r_raw = self.emb.rel(r).reshape(bsz, -1)                                # (bsz, V)

        rand_ts, probs = self.sample_g(h_raw, r_raw, self.sample_num)       # (bsz, K), (bsz, V)
        rand_ts_emb = self.emb.ent(rand_ts)                                 # (bsz, K, emb_sz)
        rand_ts_emb = F.transpose(rand_ts_emb, axes=(1, 0, 2))              # (K, bsz, emb_sz)
        rand_ts_emb = rand_ts_emb.reshape(self.sample_num * bsz, -1)        # (K * bsz, emb_sz)

        h_emb = F.broadcast_to(h_raw, (self.sample_num, ) + h_raw.shape)    # (K, bsz, emb_sz)
        h_emb = h_emb.reshape(self.sample_num * bsz, -1)                    # (K * bsz, emb_sz)
        r_emb = F.broadcast_to(r_raw, (self.sample_num, ) + r_raw.shape)    # (K, bsz, emb_sz)
        r_emb = r_emb.reshape(self.sample_num * bsz, -1)                    # (K * bsz, emb_sz)

        reward = -F.sigmoid(self.d(F.concat([h_emb, r_emb, rand_ts_emb])))  # (K * bsz, 1)

        rand_probs = self.xp.empty(rand_ts.shape, self.xp.float32)          # (bsz, K)
        for i in xrange(len(rand_probs)):
            rand_probs[i] = probs[i][rand_ts.data[i]].data
        rand_probs = F.transpose(rand_probs).reshape(-1, 1)                 # (K, bsz) -> (K * bsz, 1)
        log_rand_probs = F.log(rand_probs)

        loss_g = F.sum(log_rand_probs * reward) / self.sample_num

        self.g.cleargrads()
        loss_g.backward()
        self.get_optimizer('opt_g').update()
        self.add_to_report(loss_g=loss_g)

    def batch_multinomial(self, batch_probs, size):
        """
        Sample the multinomial distributions given a batch of probabilities
        :param batch_probs: has shape (batch_size, V), a batch of probabilities
        :param size: int, the number of examples to sample every distribution
        :return: has shape (batch_size, size)
        """
        log_probs = F.log(batch_probs)                                      # (bsz, V)
        nums = self.xp.empty((size, log_probs.shape[0])).astype('int32')    # (K, bsz)
        for i in xrange(size):
            noise = self.xp.random.rand(*log_probs.shape).astype('f')   # bsz, V
            rand = F.argmax(log_probs - F.log(-F.log(noise)), axis=1)
            nums[i] = rand.data

        # # sampling the batch of distributions for K times directly will exhaust the GPU memory,
        # # thus still we choose to use sample K numbers each on each

        # K_log_probs = F.broadcast_to(log_probs, (size,) + log_probs.shape)  # (K, bsz, V)
        # noise = self.xp.random.rand(*K_log_probs.shape).astype('f')         # (K, bsz, V)
        # nums = F.argmax(K_log_probs - F.log(-F.log(noise)), axis=2)         # (K, bsz)
        # nums_t = F.transpose(nums)
        # del log_probs, K_log_probs, noise, nums
        # return nums_t
        # return self.xp.ones((batch_probs.shape[0], size), dtype=self.xp.int32)
        return F.transpose(nums)

    @staticmethod
    def get_report_list():
        return ['epoch', 'iteration', 'loss_g', 'loss_d', 'loss_d_fake', 'loss_d_real', 'elapsed_time']


class MLEGenUpdater(chainer.training.StandardUpdater):
    def __init__(self, data_iter, opt_g, opt_e, device):
        super(MLEGenUpdater, self).__init__(data_iter, opt_g, device=device)
        self.opt_e = opt_e
        self.opt_g = opt_g
        self.g = opt_g.target
        self.emb = opt_e.target

    def update_core(self):
        data_iter = self.get_iterator('main')
        batch = data_iter.__next__()
        h, r, t = self.converter(batch, self.device)
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

    def get_report_list(self):
        return ['epoch', 'iteration', 'loss_g', 'elapsed_time']


