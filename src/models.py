# coding: utf-8

from __future__ import absolute_import
import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np
import math


class MLP(chainer.Chain):
    def __init__(self, in_dim, out_dim):
        super(MLP, self).__init__()
        with self.init_scope():
            self.l1 = L.Linear(in_dim, in_dim)
            self.l2 = L.Linear(in_dim, in_dim)
            self.l3 = L.Linear(in_dim, in_dim)
            self.l4 = L.Linear(in_dim, out_dim)

    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        h3 = F.relu(self.l3(h2))
        h4 = self.l4(h3)
        return h4


class VarMLP(chainer.ChainList):
    def __init__(self, layer_dims, dropout=0.9):
        super(VarMLP, self).__init__()
        for i in xrange(len(layer_dims) - 1):
            l = L.Linear(layer_dims[i], layer_dims[i + 1])
            self.add_link(l)
        self.dropout = dropout

    def __call__(self, x):
        hidden = x
        for i, link in enumerate(self.children()):
            if i < len(self) - 1:
                hidden = F.tanh(link(hidden))
                hidden = F.dropout(hidden, ratio=self.dropout)
            else:
                hidden = link(hidden)
        return hidden


class Embeddings(chainer.Chain):
    def __init__(self, emb_sz, ent_num, rel_num):
        super(Embeddings, self).__init__(
            ent=L.EmbedID(ent_num, emb_sz),
            rel=L.EmbedID(rel_num, emb_sz)
        )


class ResidualBlock(chainer.Chain):
    def __init__(self, in_dim):
        super(ResidualBlock, self).__init__()
        with self.init_scope():
            self.l1 = L.Linear(in_dim)
            self.l2 = L.Linear(in_dim)

        self.in_dim = in_dim

    def __call__(self, x):
        h1 = F.tanh(self.l1(x))
        h2 = self.l2(h1)
        h2 = F.selu(h2 + x)
        return h2


class ResidualGenerator(chainer.ChainList):
    def __init__(self, layer_dims, dropout=0.0):
        super(ResidualGenerator, self).__init__()
        for i in xrange(len(layer_dims) - 2):
            if layer_dims[i] == layer_dims[i + 1]:
                link = ResidualBlock(layer_dims[i])
                self.add_link(link)
            else:
                link = L.Linear(layer_dims[i], layer_dims[i + 1])
                self.add_link(link)

        # the last layer must be a simple linear mapping without activation
        last_link = L.Linear(layer_dims[-2], layer_dims[-1])
        self.add_link(last_link)

        self.dropout = dropout

    def __call__(self, x):
        hidden = x
        for i, link in enumerate(self.children()):
            if i + 1 == len(self):  # last layer
                hidden = link(hidden)
            elif isinstance(link, L.Linear):
                hidden = link(hidden)
                hidden = F.tanh(hidden)
                hidden = F.dropout(hidden, ratio=self.dropout)
            elif isinstance(link, ResidualBlock):
                hidden = link(hidden)
                hidden = F.dropout(hidden, ratio=self.dropout)
            else:
                hidden = link(hidden)
        return hidden


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
    def __init__(self, in_dim, ent_num, rel_num, dropout=0.2):
        # generator = models.VarMLP([config.EMBED_SZ * 2, config.EMBED_SZ, config.EMBED_SZ, ent_num], config.DROPOUT)
        super(Generator, self).__init__()
        with self.init_scope():
            self.l1 = L.Linear(in_dim * 2, in_dim)
            self.l2 = L.Linear(in_dim, in_dim)
            self.l3 = L.Linear(in_dim, in_dim)

        self.ent_num = ent_num
        self.rel_num = rel_num
        self.dropout = dropout

    def __call__(self, h_emb, r_emb):
        h1 = F.tanh(self.l1(F.concat([h_emb, r_emb])))
        h1 = F.dropout(h1, self.dropout)
        h2 = F.dropout(F.tanh(self.l2(h1)), self.dropout)
        h3 = self.l3(h2)
        return h3


class Discriminator(chainer.Chain):
    def __init__(self, in_dim):
        super(Discriminator, self).__init__(
            # for h, r, and t
            mlp=VarMLP([in_dim * 3, in_dim, in_dim, in_dim / 2, in_dim / 2, 1]),
        )
        for link in self.mlp:
            link.W.data = TransE.normalize_embedding(link.W.data, axis=0)

    def __call__(self, h_emb, r_emb, t_emb):
        x = F.concat((h_emb, r_emb, t_emb))
        return F.sigmoid(self.mlp(x))


class NTN(chainer.Chain):
    def __init__(self, emb_sz, ent_num, rel_num, out_sz):
        super(NTN, self).__init__()
        with self.init_scope():
            self.linear = L.Linear(out_sz, 1, nobias=True)
            self.ntn = L.Bilinear(emb_sz, emb_sz, out_sz)
            self.emb = Embeddings(emb_sz, ent_num, rel_num)

        self.ent_num = ent_num
        self.rel_num = rel_num
        self.out_sz = out_sz

    def __call__(self, h, t, r):
        bsz = h.shape[0]
        xp = chainer.cuda.get_array_module(h)

        h_emb = self.emb.ent(h).reshape(bsz, -1)
        t_emb = self.emb.ent(t).reshape(bsz, -1)
        pos_score = self.scorer(h_emb, t_emb, r)

        half = bsz / 2
        h_neg = xp.random.randint(0, self.ent_num, size=(half, 1))
        t_neg = xp.random.randint(0, self.ent_num, size=(bsz - half, 1))
        h_neg = F.concat([h_neg, h[half:]])
        t_neg = F.concat([t[:half], t_neg])

        h_neg_emb = self.emb.ent(h_neg).reshape(bsz, -1)
        t_neg_emb = self.emb.ent(t_neg).reshape(bsz, -1)
        neg_score = self.scorer(h_neg_emb, t_neg_emb, r)

        loss = F.average(F.relu(pos_score - neg_score + 1))

        chainer.report({'loss': loss, 'loss_pos': F.sum(pos_score), 'loss_neg': F.sum(neg_score)})
        return loss

    def scorer(self, h_emb, t_emb, r):
        h1 = F.tanh(self.ntn(h_emb, t_emb))
        h2 = self.linear(h1)
        return h2

    @staticmethod
    def get_report_list():
        return ['epoch', 'iteration', 'loss', 'loss_pos', 'loss_neg', 'elapsed_time']


class TransE(chainer.Chain):
    def __init__(self, emb_sz, ent_num, rel_num, margin, norm=1):
        random_range = 6 / math.sqrt(emb_sz)
        initial_ent_W = np.random.uniform(-random_range, random_range, (ent_num, emb_sz))
        initial_rel_W = np.random.uniform(-random_range, random_range, (rel_num, emb_sz))

        super(TransE, self).__init__(
            ent_emb=L.EmbedID(ent_num, emb_sz, initial_ent_W),
            rel_emb=L.EmbedID(rel_num, emb_sz, initial_rel_W),
        )

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

    def __call__(self, h, r, t):
        self.ent_emb.W.data = self.normalize_embedding(self.ent_emb.W.data)

        bsz = h.shape[0]
        h = self.ent_emb(h).reshape(bsz, -1)
        t = self.ent_emb(t).reshape(bsz, -1)
        r = self.rel_emb(r).reshape(bsz, -1)
        xp = chainer.cuda.get_array_module(h)

        half = bsz / 2
        h_corrupted = xp.random.randint(1, self.ent_num + 1, size=(half, 1))
        t_corrupted = xp.random.randint(1, self.ent_num + 1, size=(bsz - half, 1))
        h_corrupted = self.ent_emb(h_corrupted).reshape(half, -1)
        t_corrupted = self.ent_emb(t_corrupted).reshape(bsz - half, -1)

        if self.norm == 1:
            # L1 norm
            dis_pos = F.sum(F.absolute(h + r - t), axis=1)
            dis_neg_h = F.sum(F.absolute(h_corrupted + r[:half] - t[:half]), axis=1)
            dis_neg_t = F.sum(F.absolute(h[half:] + r[half:] - t_corrupted), axis=1)
        else:
            # L2 norm
            dis_pos = F.sqrt(F.batch_l2_norm_squared(h + r - t))
            dis_neg_h = F.sqrt(F.batch_l2_norm_squared(h_corrupted + r[:half] - t[:half]))
            dis_neg_t = F.sqrt(F.batch_l2_norm_squared(h[half:] + r[half:] - t_corrupted))

        dis_neg = F.concat([dis_neg_h, dis_neg_t], axis=0)  # 1:1 size for corrupted heads and tails

        loss = F.average(F.relu(self.margin + dis_pos - dis_neg))
        chainer.report({'loss': loss})
        return loss

    @staticmethod
    def create_transe(emb_sz, vocab_ent, vocab_rel, gamma, norm=1):
        m = TransE(emb_sz, len(vocab_ent) + 1, len(vocab_rel) + 1, gamma, norm)
        return m
