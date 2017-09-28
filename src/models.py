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


class VarMLP(chainer.ChainList):
    def __init__(self, layer_dims):
        super(VarMLP, self).__init__()
        for i in xrange(len(layer_dims) -1):
            l = L.Linear(layer_dims[i], layer_dims[i+1])
            self.add_link(l)

    def __call__(self, x):
        hidden = x
        for i, link in enumerate(self.children()):
            hidden = selu(link(hidden)) if i < len(self) - 1 else link(hidden)
        return hidden


class Generator(chainer.Chain):
    def __init__(self, in_dim, ent_num, rel_num):
        super(Generator, self).__init__(
            ent_emb=L.EmbedID(ent_num, in_dim),
            rel_emb=L.EmbedID(rel_num, in_dim),
            mlp=VarMLP([in_dim * 2, in_dim, in_dim])
        )

        self.ent_num = ent_num
        self.rel_num = rel_num

    def __call__(self, *args, **kwargs):
        h, r = args
        h_emb = self.ent_emb(h).reshape(h.shape[0], -1)
        r_emb = self.rel_emb(r).reshape(h.shape[0], -1)
        x = F.concat((h_emb, r_emb))
        x = self.mlp(x)
        return x

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


class PretrainedGenerator(Generator):
    def __init__(self, in_dim, ent_num, rel_num):
        super(PretrainedGenerator, self).__init__(in_dim, ent_num, rel_num)

    def __call__(self, *args, **kwargs):
        h, r, t = args
        self.ent_emb.W.data = HingeLossGen.normalize_embedding(self.ent_emb.W.data)

        xp = chainer.cuda.get_array_module(h)
        bsz = h.shape[0]
        h = self.ent_emb(h).reshape(bsz, -1)
        r = self.rel_emb(r).reshape(bsz, -1)
        t = self.ent_emb(t).reshape(bsz, -1)

        half = bsz / 2
        h_corrupted = xp.random.randint(1, self.ent_num + 1, size=(half, 1))
        t_corrupted = xp.random.randint(1, self.ent_num + 1, size=(bsz - half, 1))
        h_corrupted = self.ent_emb(h_corrupted).reshape(half, -1)
        t_corrupted = self.ent_emb(t_corrupted).reshape(bsz - half, -1)

        t_tilde = self.mlp(F.concat([h, r]))
        t_tilde_head_currupted = self.mlp(F.concat([h_corrupted, r[:half]]))

        # L2 norm
        dis_pos = F.sqrt(F.batch_l2_norm_squared(t_tilde - t))
        dis_neg_h = F.sqrt(F.batch_l2_norm_squared(t_tilde_head_currupted - t[:half]))
        dis_neg_t = F.sqrt(F.batch_l2_norm_squared(t_tilde[half:] - t_corrupted))

        dis_neg = F.concat([dis_neg_h, dis_neg_t], axis=0) # 1:1 size for corrupted heads and tails

        loss = F.sum(F.relu(1 + dis_pos - dis_neg)) # margin fixed
        chainer.report({'loss': loss})
        return loss

    @staticmethod
    def create_generator(emb_sz, vocab_ent, vocab_rel):
        # embedding link starts from 0, but token id starts from 1,
        # thus make a spare embedding for dummy id 0
        g = PretrainedGenerator(emb_sz, len(vocab_ent) + 1, len(vocab_rel) + 1)
        return g


class Discriminator(chainer.Chain):
    def __init__(self, in_dim):
        super(Discriminator, self).__init__(
            # for h, r, and t
            mlp=VarMLP([in_dim * 3, in_dim, in_dim, 1]),
        )
        for link in self.mlp:
            link.W.data = HingeLossGen.normalize_embedding(link.W.data, axis=0)

    def __call__(self, h_emb, r_emb, t_emb):
        x = F.concat((h_emb, r_emb, t_emb))
        return F.sigmoid(self.mlp(x))

class BilinearDiscriminator(chainer.Chain):
    def __init__(self, in_dim):
        super(BilinearDiscriminator, self).__init__(
            bl=L.Bilinear(in_dim * 3, in_dim * 3, 1)
        )

    def __call__(self, h_emb, r_emb, t_emb):
        x = F.concat((h_emb, r_emb, t_emb)) # batch * embedding
        return self.bl(x, x)


class HingeLossGen(chainer.Chain):
    def __init__(self, emb_sz, ent_num, rel_num, margin, norm=1):
        super(HingeLossGen, self).__init__(
            ent_emb=L.EmbedID(ent_num, emb_sz),
            rel_emb=L.EmbedID(rel_num, emb_sz),
            gen=VarMLP([emb_sz * 2, emb_sz, emb_sz])
        )

        self.emb_sz = emb_sz
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

        xp = chainer.cuda.get_array_module(h)
        bsz = h.shape[0]
        h = self.ent_emb(h).reshape(bsz, -1)
        r = self.rel_emb(r).reshape(bsz, -1)
        t = self.ent_emb(t).reshape(bsz, -1)

        half = bsz / 2
        h_corrupted = xp.random.randint(1, self.ent_num + 1, size=(half, 1))
        t_corrupted = xp.random.randint(1, self.ent_num + 1, size=(bsz - half, 1))
        h_corrupted = self.ent_emb(h_corrupted).reshape(half, -1)
        t_corrupted = self.ent_emb(t_corrupted).reshape(bsz - half, -1)

        t_tilde = self.gen(F.concat([h, r]))
        t_tilde_head_currupted = self.gen(F.concat([h_corrupted, r[:half]]))

        if self.norm == 1:
            # L1 norm
            dis_pos = F.sum(F.absolute(t_tilde - t), axis=1)
            dis_neg_h = F.sum(F.absolute(t_tilde_head_currupted - t[:half]), axis=1)
            dis_neg_t = F.sum(F.absolute(t_tilde[half:] - t_corrupted), axis=1)
        else:
            # L2 norm
            dis_pos = F.sqrt(F.batch_l2_norm_squared(t_tilde - t))
            dis_neg_h = F.sqrt(F.batch_l2_norm_squared(t_tilde_head_currupted - t[:half]))
            dis_neg_t = F.sqrt(F.batch_l2_norm_squared(t_tilde[half:] - t_corrupted))

        dis_neg = F.concat([dis_neg_h, dis_neg_t], axis=0) # 1:1 size for corrupted heads and tails

        loss = F.sum(F.relu(self.margin + dis_pos - dis_neg))
        chainer.report({'loss': loss})
        return loss

    def run_gen(self, h, r):
        xp = chainer.cuda.get_array_module(h)
        bsz = h.shape[0]
        h = self.ent_emb(h).reshape(bsz, -1)
        r = self.rel_emb(r).reshape(bsz, -1)
        t_tilde = self.gen(F.concat([h, r]))
        return t_tilde

    @staticmethod
    def create_hinge_gen(emb_sz, vocab_ent, vocab_rel, gamma, norm=1):
        m = HingeLossGen(emb_sz, len(vocab_ent) + 1, len(vocab_rel) + 1, gamma, norm)
        return m


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

    def normalize_embedding(self, x, eps=1e-7, axis=1):
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

        dis_neg = F.concat([dis_neg_h, dis_neg_t], axis=0) # 1:1 size for corrupted heads and tails

        loss = F.average(F.relu(self.margin + dis_pos - dis_neg))
        chainer.report({'loss': loss})
        return loss

    @staticmethod
    def create_transe(emb_sz, vocab_ent, vocab_rel, gamma, norm=1):
        m = TransE(emb_sz, len(vocab_ent) + 1, len(vocab_rel) + 1, gamma, norm)
        return m


# copied selu source code from chainer 3.0.0rc
from chainer.functions.activation import elu


def selu(x,
         alpha=1.6732632423543772848170429916717,
         scale=1.0507009873554804934193349852946):
    """Scaled Exponential Linear Unit function.
    For parameters :math:`\\alpha` and :math:`\\lambda`, it is expressed as
    .. math::
        f(x) = \\lambda \\left \\{ \\begin{array}{ll}
        x & {\\rm if}~ x \\ge 0 \\\\
        \\alpha (\\exp(x) - 1) & {\\rm if}~ x < 0,
        \\end{array} \\right.
    See: https://arxiv.org/abs/1706.02515
    Args:
        x (:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`):
            Input variable. A :math:`(s_1, s_2, ..., s_N)`-shaped float array.
        alpha (float): Parameter :math:`\\alpha`.
        scale (float): Parameter :math:`\\lambda`.
    Returns:
        ~chainer.Variable: Output variable. A
        :math:`(s_1, s_2, ..., s_N)`-shaped float array.
    """
    return scale * elu.elu(x, alpha=alpha)



