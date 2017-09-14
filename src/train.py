# coding: utf-8

from __future__ import absolute_import
import os, datetime
import chainer
import config
import corpus.dataset
import models
import updaters

def main():
    vocab_ent, vocab_rel = corpus.dataset.load_vocab()
    dataset = map(lambda x: corpus.dataset.load_corpus(x, vocab_ent, vocab_rel), (config.TRAIN_DATA, config.VALID_DATA))
    train_iter, valid_iter = map(lambda x: chainer.iterators.SerialIterator(x, batch_size=config.BATCH_SZ), dataset)

    generator = models.Generator(10, 10)
    opt_g = chainer.optimizers.Adam()
    opt_g.setup(generator)
    discriminator = models.MLP(10, 10)
    opt_d = chainer.optimizers.Adam()
    opt_d.setup(discriminator)

    updater = updaters.WGANUpdator(train_iter, opt_g, opt_d, device=config.DEVICE, d_epoch=config.OPT_D_EPOCH)

    trainer = chainer.training.Trainer(updater, (config.EPOCH_NUM, 'epoch'),
                                       out=os.path.join(config.MODEL_PATH,
                                                        datetime.datetime.now().strftime('result-%Y-%m-%d-%H-%M-%S')))

    trainer.extend(chainer.training.extensions.LogReport())
    trainer.extend(chainer.training.extensions.ProgressBar(update_interval=100))
    trainer.extend(chainer.training.extensions.snapshot())
    trainer.run()

if __name__ == "__main__":
    main()