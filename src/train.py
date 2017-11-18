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

    ent_num, rel_num = len(vocab_ent) + 1, len(vocab_rel) + 1

    # trainer = adversarial_trainer(ent_num, rel_num, train_iter, valid_iter)
    trainer = mle_trainer(ent_num, rel_num, train_iter, valid_iter)

    # Exp 1. & 2. TransE with and without negative sampling
    # model = models.TransE(config.EMBED_SZ, ent_num, rel_num, config.TRANSE_MARGIN, config.TRANSE_NORM)
    # model = models.TransENNG(config.EMBED_SZ, ent_num, rel_num, config.TRANSE_MARGIN, config.TRANSE_NORM)
    # trainer = standard_trainer(model, train_iter, valid_iter)
    trainer.run()


def adversarial_trainer(ent_num, rel_num, train_iter, valid_iter):
    generator = models.VarMLP([config.EMBED_SZ * 2, config.EMBED_SZ, config.EMBED_SZ, ent_num])
    discriminator = models.VarMLP([config.EMBED_SZ * 3, config.EMBED_SZ, config.EMBED_SZ, 1])
    g_embedding = models.Embeddings(config.EMBED_SZ, ent_num, rel_num)
    d_embedding = models.Embeddings(config.EMBED_SZ, ent_num, rel_num)
    if len(sys.argv) > 1:
        chainer.serializers.load_npz(sys.argv[1], generator)
    if len(sys.argv) > 2:
        chainer.serializers.load_npz(sys.argv[2], discriminator)
    if len(sys.argv) > 3:
        chainer.serializers.load_npz(sys.argv[3], g_embedding)
    if len(sys.argv) > 4:
        chainer.serializers.load_npz(sys.argv[4], d_embedding)

    if config.DEVICE >= 0:
        chainer.cuda.get_device_from_id(config.DEVICE).use()
        generator.to_gpu(config.DEVICE)
        discriminator.to_gpu(config.DEVICE)
        g_embedding.to_gpu(config.DEVICE)
        d_embedding.to_gpu(config.DEVICE)

    opt_g = chainer.optimizers.Adam(config.ADAM_ALPHA, config.ADAM_BETA1)
    opt_d = chainer.optimizers.Adam(config.ADAM_ALPHA, config.ADAM_BETA1)
    opt_eg = chainer.optimizers.Adam(config.ADAM_ALPHA, config.ADAM_BETA1)
    opt_ed = chainer.optimizers.Adam(config.ADAM_ALPHA, config.ADAM_BETA1)
    opt_g.setup(generator)
    opt_d.setup(discriminator)
    opt_eg.setup(g_embedding)
    opt_ed.setup(d_embedding)

    # updater = updaters.WGANUpdator(train_iter, opt_g, opt_d, device=config.DEVICE, d_epoch=config.OPT_D_EPOCH,
    #                                g_epoch=config.OPT_G_EPOCH, penalty_coeff=config.PENALTY_COEFF)
    updater = updaters.GANUpdater(train_iter, opt_g, opt_d, opt_eg, opt_ed, ent_num,
                                  config.DEVICE, config.OPT_D_EPOCH, config.OPT_G_EPOCH)

    trainer = chainer.training.Trainer(updater, config.TRAINING_LIMIT, out=get_trainer_out_path())
    trainer.extend(extensions.LogReport(trigger=(1, 'iteration')))
    trainer.extend(extensions.PrintReport(updater.get_report_list()))
    trainer.extend(extensions.snapshot_object(generator, 'g_iter_{.updater.iteration}'),
                   trigger=(config.SAVE_ITER_INTERVAL, 'iteration'))
    trainer.extend(extensions.snapshot_object(discriminator, 'd_iter_{.updater.iteration}'),
                   trigger=(config.SAVE_ITER_INTERVAL, 'iteration'))
    trainer.extend(extensions.snapshot_object(g_embedding, 'eg_iter_{.updater.iteration}'),
                   trigger=(config.SAVE_ITER_INTERVAL, 'iteration'))
    trainer.extend(extensions.snapshot_object(d_embedding, 'ed_iter_{.updater.iteration}'),
                   trigger=(config.SAVE_ITER_INTERVAL, 'iteration'))

    return trainer


def mle_trainer(ent_num, rel_num, train_iter, valid_iter):
    # generator = models.VarMLP([config.EMBED_SZ * 2, config.EMBED_SZ, config.EMBED_SZ, ent_num], config.DROPOUT)
    generator = models.Generator(config.EMBED_SZ, ent_num, rel_num, config.DROPOUT)
    embeddings = models.Embeddings(config.EMBED_SZ, ent_num, rel_num)
    if len(sys.argv) > 1:
        chainer.serializers.load_npz(sys.argv[1], generator)
    if len(sys.argv) > 2:
        chainer.serializers.load_npz(sys.argv[2], embeddings)

    if config.DEVICE >= 0:
        chainer.cuda.get_device_from_id(config.DEVICE).use()
        generator.to_gpu(config.DEVICE)
        embeddings.to_gpu(config.DEVICE)

    opt_g = chainer.optimizers.Adam(config.ADAM_ALPHA, config.ADAM_BETA1)
    opt_e = chainer.optimizers.Adam(config.ADAM_ALPHA, config.ADAM_BETA1)
    opt_g.setup(generator)
    opt_e.setup(embeddings)

    # updater = updaters.MLEGenUpdater(train_iter, opt_g, opt_e, ent_num, config.DEVICE)
    # updater = updaters.MLEGenPointwiseUpdater(train_iter, opt_g, opt_e, ent_num, config.DEVICE)
    updater = updaters.MLEGenNNUpdater(train_iter, opt_g, opt_e, ent_num, config.DEVICE)

    trainer = chainer.training.Trainer(updater, config.TRAINING_LIMIT, out=get_trainer_out_path())
    trainer.extend(extensions.LogReport(trigger=(1, 'iteration')))
    trainer.extend(extensions.PrintReport(updater.get_report_list()))
    trainer.extend(extensions.snapshot_object(generator, 'g_iter_{.updater.iteration}'),
                   trigger=(config.SAVE_ITER_INTERVAL, 'iteration'))
    trainer.extend(extensions.snapshot_object(embeddings, 'e_iter_{.updater.iteration}'),
                   trigger=(config.SAVE_ITER_INTERVAL, 'iteration'))

    return trainer


def standard_trainer(model, train_iter, valid_iter, opt=None):
    if len(sys.argv) > 1:
        chainer.serializers.load_npz(sys.argv[1], model)

    if config.DEVICE >= 0:
        chainer.cuda.get_device_from_id(config.DEVICE).use()
        model.to_gpu(config.DEVICE)

    opt_g = opt if opt is not None else chainer.optimizers.Adam(config.ADAM_ALPHA, config.ADAM_BETA1)
    opt_g.setup(model)

    updater = chainer.training.StandardUpdater(train_iter, opt_g, device=config.DEVICE)

    trainer = chainer.training.Trainer(updater, config.TRAINING_LIMIT, out=get_trainer_out_path())
    trainer.extend(extensions.LogReport(trigger=(1, 'iteration')))
    trainer.extend(extensions.PrintReport(model.get_report_list()))
    trainer.extend(extensions.snapshot_object(model, 'm_iter_{.updater.iteration}'),
                   trigger=(config.SAVE_ITER_INTERVAL, 'iteration'))

    return trainer


def get_trainer_out_path():
    return os.path.join(config.MODEL_PATH, datetime.datetime.now().strftime('result-%Y-%m-%d-%H-%M-%S'))

if __name__ == "__main__":
    main()
