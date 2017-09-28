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

            h_emb = self.g.embed_entity(h)
            r_emb = self.g.embed_relation(r)
            t_emb = self.g.embed_entity(t) # batch * embedding

            t_tilde = self.g(h, r) # batch * embedding(generator output)

            # sampling
            epsilon = xp.random.uniform(0.0, 1.0, (h.shape[0], 1))
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

        batch = iterator.__next__()
        h, r, t = self.converter(batch, self.device)
        xp = chainer.cuda.get_array_module(h) # either numpy or xp based on device
        h_emb, r_emb = self.g.embed_entity(h), self.g.embed_relation(r)
        t_tilde = self.g(h, r)
        # loss_supervised = F.batch_l2_norm_squared(t_tilde - self.g.embed_entity(t)).reshape(-1, 1)
        loss_g = F.average(-self.d(h_emb, r_emb, t_tilde))
        self.g.cleargrads()
        loss_g.backward()
        self.get_optimizer('opt_g').update()

        # loss_penalty = 0
        chainer.report({"loss_g": loss_g, "w-distance": -loss_gan, "penalty": loss_penalty})


class LSGANUpdater(chainer.training.StandardUpdater):
    def __init__(self, data_iter, opt_g, opt_d, device, d_epoch, hinge_loss_weight=1):
        super(LSGANUpdater, self).__init__(data_iter, {"opt_g": opt_g, "opt_d": opt_d}, device=device)
        self.d_epoch = d_epoch
        self.g = opt_g.target
        self.d = opt_d.target
        self.hinge_loss_weight = hinge_loss_weight

    def update_core(self):
        data_iter = self.get_iterator('main')

        # train the discriminator
        for epoch in xrange(self.d_epoch):
            batch = data_iter.__next__()
            h, r, t = self.converter(batch, self.device)
            t_tilde = self.g(h, r)

            # TODO: check if the embedding are also updated after one discriminator iteration
            h_emb, t_emb = map(self.g.embed_entity, (h, t))
            r_emb = self.g.embed_relation(r)
            supervision_loss = self.d(h_emb, r_emb, t_emb)

            hinge_loss = F.relu(F.batch_l2_norm_squared(t_tilde - t_emb).reshape(-1, 1)
                                + self.d(h_emb, r_emb, t_emb)
                                - self.d(h_emb, r_emb, t_tilde))
            hinge_loss *= self.hinge_loss_weight

            supervision_loss = F.sum(supervision_loss)
            hinge_loss = F.sum(hinge_loss)
            loss_d = supervision_loss + hinge_loss
            self.d.cleargrads()
            loss_d.backward()
            self.get_optimizer('opt_d').update()

        batch = data_iter.__next__()
        h, r, _ = self.converter(batch, self.device)
        h_emb = self.g.embed_entity(h)
        r_emb = self.g.embed_relation(r)
        loss_g = F.sum(self.d(h_emb, r_emb, self.g(h, r)))
        self.g.cleargrads()
        loss_g.backward()
        self.get_optimizer('opt_g').update()

        chainer.report({"loss_g": loss_g, "loss_d": loss_d,
                        "hinge_loss_d": hinge_loss, "supervision_loss": supervision_loss})






