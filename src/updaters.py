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
        super(WGANUpdator, self).__init__(iterator, opt_g, opt_d, device, d_epoch=10, g_epoch=1)
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
    def __init__(self, data_iter, opt_g, opt_d, device, d_epoch, g_epoch, margin):
        super(ExperimentalGANUpdater, self).__init__(data_iter, opt_g, opt_d, device, d_epoch, g_epoch)
        self.margin = margin

    def update_d(self, h, r, t):
        h_emb, t_emb = map(lambda x: self.d.ent(x).reshape(h.shape[0], -1), (h, t))
        r_emb = self.d.rel(r).reshape(h.shape[0], -1)

        def distance(x, y):
            return F.sqrt(F.batch_l2_norm_squared(x - y) + 1e-7).reshape(-1, 1)

        t_tilde_emb = self.g(F.concat((h_emb, r_emb)))
        loss_real = distance(h_emb + r_emb, t_emb)
        loss_gen = distance(h_emb + r_emb, t_tilde_emb)

        loss = F.relu(self.margin + loss_real - loss_gen)
        loss = F.average(loss)

        self.d.cleargrads()
        loss.backward()
        self.get_optimizer('opt_d').update()
        self.add_to_report(loss_d=loss)

    def update_g(self, h, r, t):
        h_emb, t_emb = map(lambda x: self.d.ent(x).reshape(h.shape[0], -1), (h, t))
        r_emb = self.d.rel(r).reshape(h.shape[0], -1)
        t_tilde_emb = self.g(F.concat((h_emb, r_emb)))

        loss_g = F.sqrt(F.batch_l2_norm_squared(t_tilde_emb - t_emb) + 1e-7).reshape(-1, 1)
        loss_g = F.average(loss_g)

        self.g.cleargrads()
        loss_g.backward()
        self.get_optimizer('opt_g').update()
        self.add_to_report(loss_g=loss_g)

    @staticmethod
    def get_report_list():
        return ['epoch', 'iteration', 'loss_g', 'loss_d', 'elapsed_time']





