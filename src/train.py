# coding: utf-8

from __future__ import absolute_import
import os, datetime, sys, re
import chainer
import chainer.training.extensions as extensions
import config
import corpus.dataset as mod_dataset
import models
import updaters


def main(argv):
    vocab_ent, vocab_rel = mod_dataset.load_vocab()
    dataset = map(lambda x: mod_dataset.load_corpus(x, vocab_ent, vocab_rel), (config.TRAIN_DATA, config.VALID_DATA))
    train_iter, valid_iter = map(lambda x: chainer.iterators.SerialIterator(x, batch_size=config.BATCH_SZ), dataset)

    ent_num, rel_num = len(vocab_ent) + 1, len(vocab_rel) + 1

    trainer = adversarial_trainer(argv, ent_num, rel_num, train_iter, valid_iter)
    # trainer = mle_trainer(argv, ent_num, rel_num, train_iter, valid_iter)

    # Exp 1. & 2. TransE with and without negative sampling
    # model = models.TransE(config.EMBED_SZ, ent_num, rel_num, config.MARGIN, config.TRANSE_NORM)
    # model = models.TransENNG(config.EMBED_SZ, ent_num, rel_num, config.MARGIN, config.TRANSE_NORM)
    # trainer = standard_trainer(argv, model, train_iter, valid_iter)
    dump_conf(trainer)
    try:
        trainer.run()
    except KeyboardInterrupt:
        pass


def adversarial_trainer(argv, ent_num, rel_num, train_iter, valid_iter):
    generator = models.Generator(config.EMBED_SZ, ent_num, rel_num, config.DROPOUT)
    # discriminator = models.Discriminator(config.EMBED_SZ, ent_num, rel_num, config.DROPOUT)
    discriminator = models.TransE(config.EMBED_SZ, ent_num, rel_num, config.MARGIN)
    if len(argv) > 1:
        chainer.serializers.load_npz(argv[1], generator)
    if len(argv) > 2:
        chainer.serializers.load_npz(argv[2], discriminator)

    if config.DEVICE >= 0:
        chainer.cuda.get_device_from_id(config.DEVICE).use()
        generator.to_gpu(config.DEVICE)
        discriminator.to_gpu(config.DEVICE)

    opt_g = chainer.optimizers.Adam(config.ADAM_ALPHA)
    opt_d = chainer.optimizers.Adam(config.ADAM_ALPHA)
    opt_g.setup(generator)
    opt_d.setup(discriminator)
    opt_d.add_hook(chainer.optimizer.WeightDecay(config.WEIGHT_DECAY))
    # opt_d.add_hook(chainer.optimizer.GradientHardClipping(-config.GRADIENT_CLIP, config.GRADIENT_CLIP))
    opt_g.add_hook(chainer.optimizer.WeightDecay(config.WEIGHT_DECAY))

    updater = updaters.GANUpdater(train_iter, opt_g, opt_d, device=config.DEVICE,
                                  d_epoch=config.OPT_D_EPOCH, g_epoch=config.OPT_G_EPOCH, margin=config.MARGIN,
                                  greater_d_value_better=isinstance(discriminator, models.Discriminator),
                                  dist_based_g=config.DISTANCE_BASED_G, dist_based_d=config.DISTANCE_BASED_D)
    # updater = updaters.GANPretraining(train_iter, opt_g, opt_d, ent_num, rel_num, config.MARGIN, config.DEVICE)

    trainer = chainer.training.Trainer(updater, config.TRAINING_LIMIT, out=get_trainer_out_path())
    trainer.extend(extensions.LogReport(trigger=(1, 'iteration')))
    trainer.extend(extensions.PrintReport(updater.get_report_list()))
    trainer.extend(extensions.snapshot_object(generator, 'g_iter_{.updater.iteration}'),
                   trigger=(config.SAVE_ITER_INTERVAL, 'iteration'))
    trainer.extend(extensions.snapshot_object(discriminator, 'd_iter_{.updater.iteration}'),
                   trigger=(config.SAVE_ITER_INTERVAL, 'iteration'))

    return trainer


def mle_trainer(argv, ent_num, rel_num, train_iter, valid_iter):
    generator = models.VarMLP([config.EMBED_SZ * 2, config.EMBED_SZ, config.EMBED_SZ, ent_num], config.DROPOUT)
    embeddings = models.Embeddings(config.EMBED_SZ, ent_num, rel_num)
    if len(argv) > 1:
        chainer.serializers.load_npz(argv[1], generator)
    if len(argv) > 2:
        chainer.serializers.load_npz(argv[2], embeddings)

    if config.DEVICE >= 0:
        chainer.cuda.get_device_from_id(config.DEVICE).use()
        generator.to_gpu(config.DEVICE)
        embeddings.to_gpu(config.DEVICE)

    opt_g = chainer.optimizers.Adam(config.ADAM_ALPHA, config.ADAM_BETA1)
    opt_e = chainer.optimizers.Adam(config.ADAM_ALPHA, config.ADAM_BETA1)
    opt_g.setup(generator)
    opt_e.setup(embeddings)

    # updater = updaters.MLEGenUpdater(train_iter, opt_g, opt_e, ent_num, config.DEVICE)
    updater = updaters.MLEGenNNSUpdater(train_iter, opt_g, opt_e, ent_num, config.DEVICE)
    # updater = updaters.MLEGenNSUpdater(train_iter, opt_g, opt_e, ent_num, config.DEVICE)

    trainer = chainer.training.Trainer(updater, config.TRAINING_LIMIT, out=get_trainer_out_path())
    trainer.extend(extensions.LogReport(trigger=(1, 'iteration')))
    trainer.extend(extensions.PrintReport(updater.get_report_list()))
    trainer.extend(extensions.snapshot_object(generator, 'g_iter_{.updater.iteration}'),
                   trigger=(config.SAVE_ITER_INTERVAL, 'iteration'))
    trainer.extend(extensions.snapshot_object(embeddings, 'e_iter_{.updater.iteration}'),
                   trigger=(config.SAVE_ITER_INTERVAL, 'iteration'))

    return trainer


def standard_trainer(argv, model, train_iter, valid_iter, opt=None):
    if len(argv) > 1:
        chainer.serializers.load_npz(argv[1], model)

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


def dump_conf(trainer):
    if not os.path.exists(trainer.out):
        os.mkdir(trainer.out, 0775)
    conf_dump = os.path.join(trainer.out, 'conf')
    with open(conf_dump, 'w') as f:
        for k in dir(config):
            if re.match('^[A-Z]', k):
                f.write("{0} = {1}\n".format(k, getattr(config, k)))


if __name__ == "__main__":
    main(sys.argv)
