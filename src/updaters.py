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
    def __init__(self, data_iter, opt_g, opt_d, device, d_epoch, g_epoch, margin=1, ent_num=None):
        super(ExperimentalGANUpdater, self).__init__(data_iter, opt_g, opt_d, device, d_epoch, g_epoch)
        self.margin = margin
        self.ent_num = self.d.ent.W.shape[0] if ent_num is None else ent_num

    def update_d(self, h, r, t):
        bsz = h.shape[0]
        h_emb, t_emb = map(lambda x: self.d.ent(x).reshape(bsz, -1), (h, t))
        r_emb = self.d.rel(r).reshape(h.shape[0], -1)

        def distance(x, y):
            return F.sqrt(F.batch_l2_norm_squared(x - y) + 1e-7).reshape(-1, 1)

        t_tilde_emb = self.g(F.concat((h_emb, r_emb)))
        loss_d_gen = F.relu(self.margin + distance(h_emb + r_emb, t_emb) - distance(h_emb + r_emb, t_tilde_emb))

        half = bsz / 2
        h_corrupted = self.xp.random.randint(1, self.ent_num + 1, size=(half, 1))
        t_corrupted = self.xp.random.randint(1, self.ent_num + 1, size=(bsz - half, 1))
        h_corrupted_emb = self.d.ent(h_corrupted).reshape(half, -1)
        t_corrupted_emb = self.d.ent(t_corrupted).reshape(bsz - half, -1)

        dis_neg_h = distance(h_corrupted_emb + r_emb[:half], t_emb[:half])
        dis_neg_t = distance((h_emb + r_emb)[half:], t_corrupted_emb)
        dis_neg = F.concat([dis_neg_h, dis_neg_t], axis=0)  # 1:1 size for corrupted heads and tails

        loss_d_neg = F.relu(self.margin * 2 + distance(h_emb + r_emb, t_emb) - dis_neg)

        loss_d_gen = F.average(loss_d_gen)
        loss_d_neg = F.average(loss_d_neg)
        loss = loss_d_gen + loss_d_neg

        self.d.cleargrads()
        loss.backward()
        self.get_optimizer('opt_d').update()
        self.add_to_report(loss_d=loss, loss_d_gen=loss_d_gen, loss_d_neg=loss_d_neg)

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
        return ['epoch', 'iteration', 'loss_g', 'loss_d', 'loss_d_gen', 'loss_d_neg', 'elapsed_time']


class AdvEmbUpdater(chainer.training.StandardUpdater):
    def __init__(self, data_iter, opt_emb, opt_ent, opt_rel, opt_c, device, ent_num, rel_num, d_epoch=5, g_epoch=1):
        super(AdvEmbUpdater, self).__init__(data_iter,
                                            {'opt_emb': opt_emb, 'opt_ent':opt_ent, 'opt_rel':opt_rel, 'opt_c':opt_c},
                                            device=device)
        self.ent_num = ent_num
        self.rel_num = rel_num
        self.d_epoch = d_epoch
        self.g_epoch = g_epoch
        self.emb = opt_emb.target
        self.g_ent = opt_ent.target
        self.g_rel = opt_rel.target
        self.critic = opt_c.target
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
            example = self.converter(batch, self.device)

            self.update_d(*example)

        for epoch in xrange(self.g_epoch):
            batch = data_iter.__next__()
            example = self.converter(batch, self.device)

            self.update_g(*example)

        chainer.report(self.reports)

    def update_d(self, *args):
        h, r, t = args
        bsz = h.shape[0]
        h_emb, t_emb = map(lambda x: self.emb.ent(x).reshape(bsz, -1), [h, t])
        r_emb = self.emb.rel(r).reshape(bsz, -1)

        half = bsz / 2
        h_neg = self.xp.random.randint(1, self.ent_num + 1, size=(half, 1))
        t_neg = self.xp.random.randint(1, self.ent_num + 1, size=(bsz - half, 1))
        h_neg_emb, t_neg_emb = map(lambda x, y: self.emb.ent(x).reshape(y, -1), [h_neg, t_neg], [half, bsz - half])
        h_neg_emb = F.concat([h_neg_emb, h_emb[half:]], axis=0)
        t_neg_emb = F.concat([t_emb[:half], t_neg_emb], axis=0)

        loss_pos = self.critic(F.concat([self.g_ent(h_emb), self.g_rel(r_emb), self.g_ent(t_emb)]))
        loss_pos = self.distance(loss_pos, 1)

        loss_neg = self.critic(F.concat([self.g_ent(h_neg_emb), self.g_rel(r_emb), self.g_ent(t_neg_emb)]))
        loss_neg = self.distance(loss_neg, -1)

        loss_c = F.average(loss_pos + loss_neg)

        self.critic.cleargrads()
        loss_c.backward()
        self.get_optimizer('opt_c').update()
        self.add_to_report(loss_c=loss_c)

    def update_g(self, *args):
        h, r, t = args
        bsz = h.shape[0]
        h_emb, t_emb = map(lambda x: self.emb.ent(x).reshape(bsz, -1), [h, t])
        r_emb = self.emb.rel(r).reshape(bsz, -1)

        half = bsz / 2
        h_neg = self.xp.random.randint(1, self.ent_num + 1, size=(half, 1))
        t_neg = self.xp.random.randint(1, self.ent_num + 1, size=(bsz - half, 1))
        h_neg_emb, t_neg_emb = map(lambda x, y: self.emb.ent(x).reshape(y, -1), [h_neg, t_neg], [half, bsz - half])
        h_neg_emb = F.concat([h_neg_emb, h_emb[half:]], axis=0)
        t_neg_emb = F.concat([t_emb[:half], t_neg_emb], axis=0)

        loss_g = self.critic(F.concat([self.g_ent(h_neg_emb), self.g_rel(r_emb), self.g_ent(t_neg_emb)]))
        loss_g = F.average(F.batch_l2_norm_squared(loss_g - 1))

        self.g_ent.cleargrads()
        self.g_rel.cleargrads()
        loss_g.backward()
        self.get_optimizer('opt_ent').update()
        self.get_optimizer('opt_rel').update()
        self.add_to_report(loss_g=loss_g)

    @staticmethod
    def distance(x, y, eps=1e-7):
        return F.sqrt(F.batch_l2_norm_squared(x - y) + eps)

    @staticmethod
    def get_report_list():
        return ['epoch', 'iteration', 'loss_g', 'loss_c', 'elapsed_time']

    def add_to_report(self, **kwargs):
        self.reports.update(kwargs)







