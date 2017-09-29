# coding: utf-8

from __future__ import absolute_import
import os, datetime, sys
import chainer
import chainer.training.extensions as extensions
import config
import corpus.dataset as mod_dataset
import models
import updaters

def main():
    vocab_ent, vocab_rel = mod_dataset.load_vocab()
    dataset = map(lambda x: mod_dataset.load_corpus(x, vocab_ent, vocab_rel), (config.TRAIN_DATA, config.VALID_DATA))
    train_iter, valid_iter = map(lambda x: chainer.iterators.SerialIterator(x, batch_size=config.BATCH_SZ), dataset)

    # trainer = WGAN_setting(vocab_ent, vocab_rel, train_iter, valid_iter)
    # trainer = TransE_setting(vocab_ent, vocab_rel, train_iter, valid_iter)
    # trainer = HingeGenerator_setting(vocab_ent, vocab_rel, train_iter, valid_iter)
    # trainer = LSGAN_Pretraining_setting(vocab_ent, vocab_rel, train_iter, valid_iter)
    # trainer = LSGAN_setting(vocab_ent, vocab_rel, train_iter, valid_iter)
    trainer = GAN_setting(vocab_ent, vocab_rel, train_iter, valid_iter)
    trainer.run()


def HingeGenerator_setting(vocab_ent, vocab_rel, train_iter, valid_iter):
    hinge_gen = models.HingeLossGen.create_hinge_gen(config.EMBED_SZ, vocab_ent, vocab_rel,
                                                           config.TRANSE_GAMMA, config.TRANSE_NORM)
    if len(sys.argv) > 1:
        chainer.serializers.load_npz(sys.argv[1], hinge_gen)

    if config.DEVICE >= 0:
        chainer.cuda.get_device_from_id(config.DEVICE).use()
        hinge_gen.to_gpu(config.DEVICE)

    opt = chainer.optimizers.SGD(config.SGD_LR)
    opt.setup(hinge_gen)
    updater = chainer.training.StandardUpdater(train_iter, opt, device=config.DEVICE)
    trainer = chainer.training.Trainer(updater, (config.EPOCH_NUM, 'epoch'), out=get_trainer_out_path())
    trainer.extend(extensions.LogReport(trigger=(1, 'iteration')))
    trainer.extend(extensions.PrintReport(['epoch', 'iteration', 'loss', 'elapsed_time']))
    trainer.extend(extensions.snapshot_object(hinge_gen, 'gen_iter_{.updater.iteration}'), trigger=(1000, 'iteration'))
    return trainer


def TransE_setting(vocab_ent, vocab_rel, train_iter, valid_iter):
    transE = models.TransE.create_transe(config.EMBED_SZ, vocab_ent, vocab_rel, config.TRANSE_GAMMA, config.TRANSE_NORM)
    if len(sys.argv) > 1:
        chainer.serializers.load_npz(sys.argv[1], transE)

    if config.DEVICE >= 0:
        chainer.cuda.get_device_from_id(config.DEVICE).use()
        transE.to_gpu(config.DEVICE)

    opt = chainer.optimizers.SGD(config.SGD_LR)
    opt.setup(transE)
    updater = chainer.training.StandardUpdater(train_iter, opt, device=config.DEVICE)
    trainer = chainer.training.Trainer(updater, (config.EPOCH_NUM, 'epoch'), out=get_trainer_out_path())
    trainer.extend(extensions.LogReport(trigger=(1, 'iteration')))
    trainer.extend(extensions.PrintReport(['epoch', 'iteration', 'loss', 'elapsed_time']))
    trainer.extend(extensions.snapshot_object(transE, 'transE_iter_{.updater.iteration}'), trigger=(1000, 'iteration'))
    return trainer

def LSGAN_Pretraining_setting(vocab_ent, vocab_rel, train_iter, valid_iter):
    gen = models.PretrainedGenerator.create_generator(config.EMBED_SZ, vocab_ent, vocab_rel)
    if len(sys.argv) > 1:
        chainer.serializers.load_npz(sys.argv[1], gen)

    if config.DEVICE >= 0:
        chainer.cuda.get_device_from_id(config.DEVICE).use()
        gen.to_gpu(config.DEVICE)

    opt = chainer.optimizers.SGD(config.SGD_LR)
    opt.setup(gen)
    updater = chainer.training.StandardUpdater(train_iter, opt, device=config.DEVICE)
    trainer = chainer.training.Trainer(updater, (config.EPOCH_NUM, 'epoch'), out=get_trainer_out_path())
    trainer.extend(extensions.LogReport(trigger=(1, 'iteration')))
    trainer.extend(extensions.PrintReport(['epoch', 'iteration', 'loss', 'elapsed_time']))
    trainer.extend(extensions.snapshot_object(gen, 'gen_iter_{.updater.iteration}'), trigger=(1000, 'iteration'))
    return trainer


def GAN_setting(vocab_ent, vocab_rel, train_iter, valid_iter):

    generator = models.Generator.create_generator(config.EMBED_SZ, vocab_ent, vocab_rel)
    discriminator = models.Discriminator(config.EMBED_SZ)
    if len(sys.argv) > 1:
        chainer.serializers.load_npz(sys.argv[1], generator)
    if len(sys.argv) > 2:
        chainer.serializers.load_npz(sys.argv[2], discriminator)

    if config.DEVICE >= 0:
        chainer.cuda.get_device_from_id(config.DEVICE).use()
        generator.to_gpu(config.DEVICE)
        discriminator.to_gpu(config.DEVICE)

    opt_g = chainer.optimizers.Adam(1e-3, 0.5)
    opt_d = chainer.optimizers.Adam(1e-3, 0.5)
    opt_g.setup(generator)
    opt_d.setup(discriminator)
    # opt_g.add_hook(chainer.optimizer.WeightDecay(config.WEIGHT_DECAY), 'hook_g_weight')
    # opt_d.add_hook(chainer.optimizer.WeightDecay(config.WEIGHT_DECAY), 'hook_d_weight')

    updater = updaters.GANUpdater(train_iter, opt_g, opt_d, config.DEVICE, config.OPT_D_EPOCH)
    trainer = chainer.training.Trainer(updater, (config.EPOCH_NUM, 'epoch'), out=get_trainer_out_path())
    trainer.extend(extensions.LogReport(trigger=(1, 'iteration')))
    trainer.extend(extensions.PrintReport(['epoch', 'iteration', 'loss_g','loss_d',
                                           'loss_real', 'loss_gen', 'elapsed_time']))
    trainer.extend(extensions.snapshot_object(generator, 'gen_iter_{.updater.iteration}'), trigger=(100, 'iteration'))
    trainer.extend(extensions.snapshot_object(discriminator, 'd_iter_{.updater.iteration}'), trigger=(100, 'iteration'))

    return trainer


def LSGAN_setting(vocab_ent, vocab_rel, train_iter, valid_iter):

    generator = models.Generator.create_generator(config.EMBED_SZ, vocab_ent, vocab_rel)
    discriminator = models.Discriminator(config.EMBED_SZ)
    if len(sys.argv) > 1:
        chainer.serializers.load_npz(sys.argv[1], generator)
    if len(sys.argv) > 2:
        chainer.serializers.load_npz(sys.argv[2], discriminator)

    if config.DEVICE >= 0:
        chainer.cuda.get_device_from_id(config.DEVICE).use()
        generator.to_gpu(config.DEVICE)
        discriminator.to_gpu(config.DEVICE)

    opt_g = chainer.optimizers.Adam(1e-3, 0.5)
    opt_d = chainer.optimizers.Adam(1e-3, 0.5)
    opt_g.setup(generator)
    opt_d.setup(discriminator)
    # opt_g.add_hook(chainer.optimizer.WeightDecay(config.WEIGHT_DECAY), 'hook_g_weight')
    # opt_d.add_hook(chainer.optimizer.WeightDecay(config.WEIGHT_DECAY), 'hook_d_weight')

    updater = updaters.LSGANUpdater(train_iter, opt_g, opt_d, config.DEVICE, config.OPT_D_EPOCH, config.HINGE_LOSS_WEIGHT)
    trainer = chainer.training.Trainer(updater, (config.EPOCH_NUM, 'epoch'), out=get_trainer_out_path())
    trainer.extend(extensions.LogReport(trigger=(1, 'iteration')))
    trainer.extend(extensions.PrintReport(['epoch', 'iteration', 'loss_g', 'loss_d',
                                           'supervision_loss', 'hinge_loss_d', 'elapsed_time']))
    trainer.extend(extensions.snapshot_object(generator, 'gen_iter_{.updater.iteration}'), trigger=(100, 'iteration'))
    trainer.extend(extensions.snapshot_object(discriminator, 'd_iter_{.updater.iteration}'), trigger=(100, 'iteration'))

    return trainer


def WGAN_setting(vocab_ent, vocab_rel, train_iter, valid_iter):

    generator = models.Generator.create_generator(config.EMBED_SZ, vocab_ent, vocab_rel)
    discriminator = models.Discriminator(config.EMBED_SZ)
    if config.DEVICE >= 0:
        chainer.cuda.get_device_from_id(config.DEVICE).use()
        generator.to_gpu(config.DEVICE)
        discriminator.to_gpu(config.DEVICE)

    opt_g = chainer.optimizers.Adam(1e-4, 0.5, 0.9)
    opt_d = chainer.optimizers.Adam(1e-4, 0.5, 0.9)
    # opt_g = chainer.optimizers.RMSprop()
    # opt_d = chainer.optimizers.RMSprop()
    opt_g.setup(generator)
    opt_d.setup(discriminator)

    updater = updaters.WGANUpdator(train_iter, opt_g, opt_d,
                                   device=config.DEVICE, d_epoch=config.OPT_D_EPOCH, penalty_coeff=config.PENALTY_COEFF)

    trainer = chainer.training.Trainer(updater, (config.EPOCH_NUM, 'epoch'), out=get_trainer_out_path())

    trainer.extend(extensions.LogReport(trigger=(1, 'iteration')))
    trainer.extend(extensions.PrintReport(['epoch', 'iteration', 'loss_g', 'w-distance', 'penalty', 'elapsed_time']))
    trainer.extend(extensions.snapshot_object(generator, 'gen_iter_{.updater.iteration}'), trigger=(100, 'iteration'))
    trainer.extend(extensions.snapshot_object(discriminator, 'd_iter_{.updater.iteration}'), trigger=(100, 'iteration'))

    return trainer

def get_trainer_out_path():
    return os.path.join(config.MODEL_PATH, datetime.datetime.now().strftime('result-%Y-%m-%d-%H-%M-%S'))

if __name__ == "__main__":
    main()