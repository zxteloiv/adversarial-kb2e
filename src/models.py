# coding: utf-8

from __future__ import absolute_import
import chainer
import chainer.functions as F
import chainer.links as L

class MLP(chainer.Chain):
    def __init__(self, in_dim, out_dim):
        super(MLP, self).__init__(
            l1=L.Linear(in_dim, in_dim),
            l2=L.Linear(in_dim, in_dim),
            l3=L.Linear(in_dim, in_dim),
            l4=L.Linear(in_dim, out_dim)
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
            mlp=MLP(in_dim * 2, in_dim),
        )

    def __call__(self, h, r):
        h_emb = self.ent_emb(h).reshape(h.shape[0], -1)
        r_emb = self.rel_emb(r).reshape(h.shape[0], -1)
        x = F.concat((h_emb, r_emb))
        return self.mlp(x)

    def embed_entity(self, e):
        return self.ent_emb(e)

class Discriminator(chainer.Chain):
    def __init__(self, in_dim):
        super(Discriminator, self).__init__(
            mlp=MLP(in_dim, 1)
        )

    def __call__(self, x):
        return self.mlp(x)


