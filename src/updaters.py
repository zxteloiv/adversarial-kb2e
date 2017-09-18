# coding: utf-8

import chainer
import chainer.functions as F

class WGANUpdator(chainer.training.StandardUpdater):
    def __init__(self, iterator, opt_g, opt_d, device, d_epoch, penalty_coeff=10):
        super(WGANUpdator, self).__init__(iterator, {"opt_g": opt_g, "opt_d": opt_d}, device=device)
        self.d_epoch = d_epoch
        self.device = device
        self.g = opt_g.target
        self.d = opt_d.target
        self.pen_coeff = penalty_coeff

        self.debug = None

    def update_core(self):
        iterator = self.get_iterator('main')
        # xp = np

        # train the discriminator
        for epoch in xrange(self.d_epoch):
            batch = iterator.__next__()
            h, r, t = self.converter(batch, self.device)
            xp = chainer.cuda.get_array_module(h) # either numpy or xp based on device

            h = h.astype(xp.int32, copy=False) # batch * 1
            r = r.astype(xp.int32, copy=False) # batch * 1
            t_org = t.astype(xp.int32, copy=False) # batch * 1
            t = self.g.embed_entity(t_org).reshape(h.shape[0], -1) # batch * 1 * embedding -> batch * embedding
            t_tilde = self.g(h, r) # batch * embedding(generator output)

            epsilon = xp.random.uniform(0.0, 1.0, (h.shape[0], 1))
            t_hat = epsilon * t + (1 - epsilon) * t_tilde
            delta = 1e-7
            # t_hat_delta = (epsilon + delta) * t + (1 - epsilon - delta) * t_tilde
            t_hat_delta = t_hat + delta

            delta_approx = F.sqrt(F.batch_l2_norm_squared(t_hat_delta - t_hat)).reshape(-1, 1)
            derivative = (self.d(t_hat_delta) - self.d(t_hat)) / delta_approx + delta
            penalty = (F.sqrt(F.batch_l2_norm_squared(derivative)) - 1) ** 2

            loss_gan = F.average(self.d(t_tilde) - self.d(t))
            loss_penalty = F.average(penalty) * self.pen_coeff
            loss_d = loss_gan + loss_penalty

            self.d.cleargrads()
            loss_d.backward()
            self.get_optimizer('opt_d').update()

        batch = iterator.__next__()
        h, r, t = self.converter(batch, self.device)
        xp = chainer.cuda.get_array_module(h) # either numpy or xp based on device
        h = h.astype(xp.int32, copy=False)
        r = r.astype(xp.int32, copy=False)
        t_tilde = self.g(h, r)
        loss_g = F.average(-self.d(t_tilde))
        self.g.cleargrads()
        loss_g.backward()
        self.get_optimizer('opt_g').update()

        chainer.report({"loss_g": loss_g, "w-distance": -loss_gan, "penalty": loss_penalty})



