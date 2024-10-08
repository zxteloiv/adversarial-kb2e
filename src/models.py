# coding: utf-8

from __future__ import absolute_import
import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np
import math


class VarMLP(chainer.ChainList):
    def __init__(self, layer_dims, dropout=0.1, activate=None):
        super(VarMLP, self).__init__()
        for i in xrange(len(layer_dims) - 1):
            l = L.Linear(layer_dims[i], layer_dims[i + 1])
            self.add_link(l)
        self.dropout = dropout
        self.activate = activate if activate is not None else F.tanh

    def __call__(self, x):
        hidden = x
        for i, link in enumerate(self.children()):
            if i < len(self) - 1:
                hidden = self.activate(link(hidden))
                hidden = F.dropout(hidden, ratio=self.dropout)
            else:
                hidden = link(hidden)
        return hidden


class Embeddings(chainer.Chain):
    def __init__(self, emb_sz, ent_num, rel_num):
        random_range = 6 / math.sqrt(emb_sz)
        initial_ent_w = np.random.uniform(-random_range, random_range, (ent_num, emb_sz))
        initial_rel_w = np.random.uniform(-random_range, random_range, (rel_num, emb_sz))

        super(Embeddings, self).__init__()
        with self.init_scope():
            self.ent = L.EmbedID(ent_num, emb_sz, initialW=initial_ent_w)
            self.rel = L.EmbedID(rel_num, emb_sz, initialW=initial_rel_w)


class AdaptiveHighwayLayer(chainer.Chain):
    def __init__(self, in_dim, out_dim):
        super(AdaptiveHighwayLayer, self).__init__()
        with self.init_scope():
            self.affine_link = L.Linear(in_dim, out_dim)
            self.trans_gate_link = L.Linear(in_dim, out_dim, initial_bias=-np.ones((out_dim,)).astype('i'))

        self.in_dim = in_dim
        self.out_dim = out_dim

    def __call__(self, x):
        hidden = F.tanh(self.affine_link(x))
        if self.in_dim == self.out_dim:
            gate = F.sigmoid(self.trans_gate_link(x))
            out = hidden * gate + x * (1 - gate)
        else:
            out = hidden

        return out


class HighwayNetwork(chainer.ChainList):
    def __init__(self, layer_dims, dropout=0.0):
        super(HighwayNetwork, self).__init__()
        for i in xrange(len(layer_dims) - 2):
            link = AdaptiveHighwayLayer(layer_dims[i], layer_dims[i + 1])
            self.add_link(link)

        # the last layer must be a simple linear mapping without activation
        last_link = L.Linear(layer_dims[-2], layer_dims[-1])
        self.add_link(last_link)

        self.dropout = dropout

    def __call__(self, x):
        hidden = x
        for i, link in enumerate(self.children()):
            if i < len(self) - 1:
                hidden = link(hidden)
                hidden = F.dropout(hidden, ratio=self.dropout)
            else:
                hidden = link(hidden)
        return hidden


class Generator(chainer.Chain):
    def __init__(self, emb_sz, ent_num, rel_num, dropout=0.2):
        # generator = models.VarMLP([config.EMBED_SZ * 2, config.EMBED_SZ, config.EMBED_SZ, ent_num], config.DROPOUT)
        super(Generator, self).__init__()
        with self.init_scope():
            self.mlp = VarMLP([emb_sz * 2, emb_sz, emb_sz, ent_num], dropout)
            self.emb = Embeddings(emb_sz, ent_num, rel_num)

        self.ent_num = ent_num
        self.rel_num = rel_num
        self.dropout = dropout

    def __call__(self, h, r):
        bsz = h.shape[0]
        h_emb = self.emb.ent(h).reshape(bsz, -1)
        r_emb = self.emb.rel(r).reshape(bsz, -1)

        logits = self.mlp(F.concat([h_emb, r_emb]))
        return logits


class Discriminator(chainer.Chain):
    def __init__(self, emb_sz, ent_num, rel_num, dropout=0.2):
        super(Discriminator, self).__init__()
        with self.init_scope():
            # for h, r, and t
            self.mlp = VarMLP([emb_sz * 3, emb_sz, emb_sz, 1], dropout)
            self.emb = Embeddings(emb_sz, ent_num, rel_num)

        self.ent_num = ent_num
        self.rel_num = rel_num
        self.dropout = dropout

    def __call__(self, h, r, t):
        bsz = h.shape[0]
        h_emb = self.emb.ent(h).reshape(bsz, -1)
        r_emb = self.emb.ent(r).reshape(bsz, -1)
        t_emb = self.emb.ent(t).reshape(bsz, -1)
        x = F.concat((h_emb, r_emb, t_emb))
        return F.sigmoid(self.mlp(x))

    def input_t_as_onehot(self, h, r, t_one_hot):
        bsz = h.shape[0]
        h_emb = self.emb.ent(h).reshape(bsz, -1)
        r_emb = self.emb.ent(r).reshape(bsz, -1)
        t_emb = F.matmul(t_one_hot, self.emb.ent.W)
        x = F.concat((h_emb, r_emb, t_emb))
        return F.sigmoid(self.mlp(x))


class TransE(chainer.Chain):
    def __init__(self, emb_sz, ent_num, rel_num, margin, norm=1):
        random_range = 6 / math.sqrt(emb_sz)
        initial_ent_w = np.random.uniform(-random_range, random_range, (ent_num, emb_sz))
        initial_rel_w = np.random.uniform(-random_range, random_range, (rel_num, emb_sz))

        super(TransE, self).__init__()
        with self.init_scope():
            self.ent_emb = L.EmbedID(ent_num, emb_sz, initial_ent_w)
            self.rel_emb = L.EmbedID(rel_num, emb_sz, initial_rel_w)

        self.ent_num = ent_num
        self.rel_num = rel_num
        self.margin = margin
        self.norm = norm
        self.rel_emb.W.data = self.normalize_embedding(self.rel_emb.W.data)

    @staticmethod
    def normalize_embedding(x, eps=1e-7, axis=1):
        xp = chainer.cuda.get_array_module(x)
        norm = xp.linalg.norm(x, axis=axis) + eps
        norm = xp.expand_dims(norm, axis=axis)
        return x / norm

    def dist(self, h, r, t):
        bsz = h.shape[0]
        xp = chainer.cuda.get_array_module(h)
        h_emb = self.ent_emb(h).reshape(bsz, -1)
        t_emb = self.ent_emb(t).reshape(bsz, -1)
        r_emb = self.rel_emb(r).reshape(bsz, -1)

        if self.norm == 1:
            dis_pos = F.sum(F.absolute(h_emb + r_emb - t_emb), axis=1)
        else:
            dis_pos = F.sqrt(F.batch_l2_norm_squared(h_emb + r_emb - t_emb))

        return dis_pos.reshape(bsz, -1)

    def dist_one_hot_t(self, h, r, t_one_hot):
        bsz = h.shape[0]
        xp = chainer.cuda.get_array_module(h)
        h_emb = self.ent_emb(h).reshape(bsz, -1)
        r_emb = self.rel_emb(r).reshape(bsz, -1)
        t_emb = F.matmul(t_one_hot, self.ent_emb.W)

        if self.norm == 1:
            dis_pos = F.sum(F.absolute(h_emb + r_emb - t_emb), axis=1)
        else:
            dis_pos = F.sqrt(F.batch_l2_norm_squared(h_emb + r_emb - t_emb))

        return dis_pos.reshape(bsz, -1)

    def __call__(self, h, r, t):
        self.ent_emb.W.data = self.normalize_embedding(self.ent_emb.W.data)

        bsz = h.shape[0]
        xp = chainer.cuda.get_array_module(h)
        h_emb = self.ent_emb(h).reshape(bsz, -1)
        t_emb = self.ent_emb(t).reshape(bsz, -1)
        r_emb = self.rel_emb(r).reshape(bsz, -1)

        half = bsz / 2
        h_corr = xp.random.randint(1, self.ent_num, size=(half, 1)).astype('i')
        t_corr = xp.random.randint(1, self.ent_num, size=(bsz - half, 1)).astype('i')
        h_neg = F.concat([h_corr, h[half:]], axis=0)
        t_neg = F.concat([t[:half], t_corr], axis=0)
        h_neg_emb = self.ent_emb(h_neg).reshape(bsz, -1)
        t_neg_emb = self.ent_emb(t_neg).reshape(bsz, -1)

        if self.norm == 1:
            # L1 norm
            dis_pos = F.sum(F.absolute(h_emb + r_emb - t_emb), axis=1)
            dis_neg = F.sum(F.absolute(h_neg_emb + r_emb - t_neg_emb), axis=1)
        else:
            # L2 norm
            dis_pos = F.sqrt(F.batch_l2_norm_squared(h_emb + r_emb - t_emb))
            dis_neg = F.sqrt(F.batch_l2_norm_squared(h_neg_emb + r_emb - t_neg_emb))

        margin = self.margin * xp.sign(xp.absolute((h - h_neg).data) + xp.absolute((t - t_neg).data)).reshape(-1)

        loss = F.sum(F.relu(margin + dis_pos - dis_neg))
        chainer.report({'loss': loss, 'loss_pos': F.sum(dis_pos), 'loss_neg': F.sum(dis_neg)})
        return loss

    @staticmethod
    def get_report_list():
        return ['epoch', 'iteration', 'loss', 'loss_pos', 'loss_neg', 'elapsed_time']


class TransENNG(TransE):
    """TransE without Negative Sampling"""
    def __call__(self, h, r, t):
        self.ent_emb.W.data = self.normalize_embedding(self.ent_emb.W.data)

        bsz = h.shape[0]
        xp = chainer.cuda.get_array_module(h)
        h_emb = self.ent_emb(h).reshape(bsz, -1)
        t_emb = self.ent_emb(t).reshape(bsz, -1)
        r_emb = self.rel_emb(r).reshape(bsz, -1)

        if self.norm == 1:
            # L1 norm
            dis_pos = F.sum(F.absolute(h_emb + r_emb - t_emb), axis=1)
        else:
            # L2 norm
            dis_pos = F.sqrt(F.batch_l2_norm_squared(h_emb + r_emb - t_emb))

        loss = F.sum(dis_pos)
        chainer.report({'loss': loss, 'loss_pos': F.sum(dis_pos), 'loss_neg': 0})
        return loss


class BatchLinear(chainer.Link):
    def __init__(self, rel_num, emb_sz):
        super(BatchLinear, self).__init__()
        initialM = np.zeros(shape=(rel_num, emb_sz, emb_sz), dtype=np.float32)
        for i in xrange(len(initialM)):
            initialM[i] = np.identity(emb_sz)

        with self.init_scope():
            self.M = chainer.Parameter(initializer=initialM, shape=(rel_num, emb_sz, emb_sz))

    def __call__(self, batch_rel_idx, x):
        bsz = x.shape[0]
        batch_rel_idx = chainer.as_variable(batch_rel_idx.reshape(-1))
        M_r = F.get_item(self.M, batch_rel_idx.data)
        y = F.batch_matmul(M_r, x)
        return y.reshape(bsz, -1)


class TransR(chainer.Chain):
    def __init__(self, emb_sz, ent_num, rel_num, margin=1, norm=1):
        super(TransR, self).__init__()
        with self.init_scope():
            self.bl = BatchLinear(rel_num, emb_sz)
            self.emb = Embeddings(emb_sz, ent_num, rel_num)

        self.emb_sz = emb_sz
        self.ent_num = ent_num
        self.rel_num = rel_num
        self.margin = margin
        self.norm = norm

    def __call__(self, h, r, t):
        bsz = h.shape[0]
        xp = chainer.cuda.get_array_module(h)
        h_emb = self.emb.ent(h).reshape(bsz, -1)
        t_emb = self.emb.ent(t).reshape(bsz, -1)
        r_emb = self.emb.rel(r).reshape(bsz, -1)
        Mr_h = self.bl(r, h_emb)
        Mr_t = self.bl(r, t_emb)

        half = bsz / 2
        h_corr = xp.random.randint(0, self.ent_num, size=(half, 1)).astype('i')
        t_corr = xp.random.randint(0, self.ent_num, size=(bsz - half, 1)).astype('i')
        h_neg = F.concat([h_corr, h[half:]], axis=0)
        t_neg = F.concat([t[:half], t_corr], axis=0)
        h_neg_emb = self.emb.ent(h_neg).reshape(bsz, -1)
        t_neg_emb = self.emb.ent(t_neg).reshape(bsz, -1)
        Mr_h_neg = self.bl(r, h_neg_emb)
        Mr_t_neg = self.bl(r, t_neg_emb)

        if self.norm == 1:
            # L1 norm
            dis_pos = F.sum(F.absolute(Mr_h + r_emb - Mr_t), axis=1)
            dis_neg = F.sum(F.absolute(Mr_h_neg + r_emb - Mr_t_neg), axis=1)
        else:
            # L2 norm
            dis_pos = F.sqrt(F.batch_l2_norm_squared(h_emb + r_emb - t_emb))
            dis_neg = F.sqrt(F.batch_l2_norm_squared(Mr_h_neg + r_emb - Mr_t_neg))

        margin = self.margin * xp.sign(xp.absolute((h - h_neg).data) + xp.absolute((t - t_neg).data)).reshape(-1)

        loss = F.sum(F.relu(margin + dis_pos - dis_neg))
        chainer.report({'loss': loss, 'loss_pos': F.sum(dis_pos), 'loss_neg': F.sum(dis_neg)})
        return loss

    @staticmethod
    def get_report_list():
        return ['epoch', 'iteration', 'loss', 'loss_pos', 'loss_neg', 'elapsed_time']

    def dist(self, h, r, t):
        bsz = h.shape[0]
        xp = chainer.cuda.get_array_module(h)
        h_emb = self.emb.ent(h).reshape(bsz, -1)
        t_emb = self.emb.ent(t).reshape(bsz, -1)
        r_emb = self.emb.rel(r).reshape(bsz, -1)
        Mr_h = self.bl(r, h_emb)
        Mr_t = self.bl(r, t_emb)
        if self.norm == 1:
            # L1 norm
            dis_pos = F.sum(F.absolute(Mr_h + r_emb - Mr_t), axis=1)
        else:
            # L2 norm
            dis_pos = F.sqrt(F.batch_l2_norm_squared(h_emb + r_emb - t_emb))

        return dis_pos.reshape(bsz, -1)


if __name__ == "__main__":
    print "WARNING: debug mode for models only, do no rely on this executable script directly"

    rel_num, ent_num, emb_sz, bsz = 10, 100, 7, 3
    rels = chainer.Variable(np.random.randint(rel_num, size=(bsz, 1)).astype('i').reshape(-1, 1))

    emb = L.EmbedID(ent_num, emb_sz)
    l = BatchLinear(rel_num, emb_sz, emb_sz)
    x = np.random.randint(ent_num, size=(bsz,)).astype('i')
    x_emb = emb(x)

    y = l(rels, x_emb)
    print y

