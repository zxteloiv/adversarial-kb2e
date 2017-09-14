# coding: utf-8

from __future__ import absolute_import
import chainer
import config
import corpus.dataset

def main():
    vocab_ent, vocab_rel = corpus.dataset.load_vocab()
    dataset = map(lambda x: corpus.dataset.load_corpus(x, vocab_ent, vocab_rel), (config.TRAIN_DATA, config.VALID_DATA))
    train_iter, valid_iter = map(lambda x: chainer.iterators.SerialIterator(x, batch_size=config.BATCH_SZ), dataset)

    for batch in train_iter:
        print batch
        break

    pass

def train(model, train_iter, val_iter):
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)


if __name__ == "__main__":
    main()