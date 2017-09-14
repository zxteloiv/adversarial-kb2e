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
    def __init__(self, in_dim, out_dim):
        super(Generator, self).__init__()
        pass

    def __call__(self, x):
        return x
