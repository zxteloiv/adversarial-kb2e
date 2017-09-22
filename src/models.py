# coding: utf-8

from __future__ import absolute_import
import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np
import math

class MLP(chainer.Chain):
    def __init__(self, in_dim, out_dim):
        super(MLP, self).__init__(
            l1=L.Linear(in_dim, in_dim),
            l2=L.Linear(in_dim, in_dim),
            l3=L.Linear(in_dim, in_dim),
            l4=L.Linear(in_dim, out_dim),
        )

    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        h3 = F.relu(self.l3(h2))
        h4 = self.l4(h3)
        return h4


class Generator(chainer.Chain):
    def __init__(self, in_dim, ent_num, rel_num):
        super(Generator, self).__init__(
            ent_emb=L.EmbedID(ent_num, in_dim),
            rel_emb=L.EmbedID(rel_num, in_dim),
            mlp1=MLP(in_dim * 2, in_dim),
            mlp2=MLP(in_dim, in_dim),
        )

    def __call__(self, h, r):
        h_emb = self.ent_emb(h).reshape(h.shape[0], -1)
        r_emb = self.rel_emb(r).reshape(h.shape[0], -1)
        x = F.concat((h_emb, r_emb))
        x1 = F.relu(self.mlp1(x))
        x2 = self.mlp2(x1)
        return x2

    def embed_entity(self, e):
        return self.ent_emb(e).reshape(e.shape[0], -1)

    def embed_relation(self, r):
        return self.rel_emb(r).reshape(r.shape[0], -1)

    @staticmethod
    def create_generator(emb_sz, vocab_ent, vocab_rel):
        # embedding link starts from 0, but token id starts from 1,
        # thus make a spare embedding for dummy id 0
        g = Generator(emb_sz, len(vocab_ent) + 1, len(vocab_rel) + 1)
        return g

class TransE(chainer.Chain):
    def __init__(self, emb_sz, ent_num, rel_num, margin):
        random_range = 6 / math.sqrt(emb_sz)
        # initial_ent_W = np.random.uniform(-random_range, random_range, (ent_num, emb_sz))
        # initial_rel_W = np.random.uniform(-random_range, random_range, (rel_num, emb_sz))

        super(TransE, self).__init__(
            ent_emb=L.EmbedID(ent_num, emb_sz),
            rel_emb=L.EmbedID(rel_num, emb_sz),
            # ent_emb=L.EmbedID(ent_num, emb_sz, initial_ent_W),
            # rel_emb=L.EmbedID(rel_num, emb_sz, initial_rel_W),
        )

        self.ent_num = ent_num
        self.rel_num = rel_num
        self.margin = margin
        xp = chainer.cuda.get_array_module(self.ent_emb)
        # self.rel_emb.W = F.normalize(self.rel_emb.W, eps=1e-7)

    def __call__(self, h, r, t):
        # self.ent_emb.W = F.normalize(self.ent_emb.W, eps=1e-7)

        bsz = h.shape[0]
        h = self.ent_emb(h).reshape(bsz, -1)
        t = self.ent_emb(t).reshape(bsz, -1)
        r = self.rel_emb(r).reshape(bsz, -1)
        xp = chainer.cuda.get_array_module(h)

        h_corrupted = xp.random.randint(1, self.ent_num + 1, size=(h.shape[0], 1))
        t_corrupted = xp.random.randint(1, self.ent_num + 1, size=(h.shape[0], 1))
        h_corrupted = self.ent_emb(h_corrupted).reshape(bsz, -1)
        t_corrupted = self.ent_emb(t_corrupted).reshape(bsz, -1)

        dis_pos = F.sqrt(F.batch_l2_norm_squared(h + r - t))
        dis_neg = F.sqrt(F.batch_l2_norm_squared(h_corrupted + r - t_corrupted))

        loss = F.sum(F.relu(self.margin + dis_pos - dis_neg))
        chainer.report({'loss': loss})
        return loss

    @staticmethod
    def create_transe(emb_sz, vocab_ent, vocab_rel, gamma):
        m = TransE(emb_sz, len(vocab_ent) + 1, len(vocab_rel) + 1, gamma)
        return m

class Discriminator(chainer.Chain):
    def __init__(self, in_dim):
        super(Discriminator, self).__init__(
            # for h, r, and t
            mlp1=MLP(in_dim * 3, in_dim),
            mlp2=MLP(in_dim, 1)
        )

    def __call__(self, h_emb, r_emb, t_emb):
        x = F.concat((h_emb, r_emb, t_emb))
        h1 = F.relu(self.mlp1(x))
        h2 = self.mlp2(h1)
        return h2

class BilinearDiscriminator(chainer.Chain):
    def __init__(self, in_dim):
        super(BilinearDiscriminator, self).__init__(
            bl=L.Bilinear(in_dim * 3, in_dim * 3, 1)
        )

    def __call__(self, h_emb, r_emb, t_emb):
        x = F.concat((h_emb, r_emb, t_emb)) # batch * embedding
        return self.bl(x, x)




