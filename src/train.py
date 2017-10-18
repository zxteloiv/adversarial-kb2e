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

    # trainer = TransE_setting(vocab_ent, vocab_rel, train_iter, valid_iter)
    # trainer = HingeGenerator_setting(vocab_ent, vocab_rel, train_iter, valid_iter)
    # trainer = GAN_Pretraining_setting(vocab_ent, vocab_rel, train_iter, valid_iter)
    # trainer = GAN_setting(vocab_ent, vocab_rel, train_iter, valid_iter)
    trainer = ExperimentalGAN_setting(vocab_ent, vocab_rel, train_iter, valid_iter)
    # trainer = AdversarialEmbedding_setting(vocab_ent, vocab_rel, train_iter, valid_iter)
    trainer.run()


def HingeGenerator_setting(vocab_ent, vocab_rel, train_iter, valid_iter):
    hinge_gen = models.HingeLossGen.create_hinge_gen(config.EMBED_SZ, vocab_ent, vocab_rel,
                                                           config.TRANSE_GAMMA, config.TRANSE_NORM)
    if len(sys.argv) > 1:
        chainer.serializers.load_npz(sys.argv[1], hinge_gen)

    if config.DEVICE >= 0:
        chainer.cuda.get_device_from_id(config.DEVICE).use()
        hinge_gen.to_gpu(config.DEVICE)

    # opt = chainer.optimizers.SGD(config.SGD_LR)
    opt = chainer.optimizers.Adam(config.ADAM_ALPHA, config.ADAM_BETA1)
    opt.setup(hinge_gen)
    opt.add_hook(chainer.optimizer.WeightDecay(config.WEIGHT_DECAY), 'weight_decay')
    updater = chainer.training.StandardUpdater(train_iter, opt, device=config.DEVICE)
    trainer = chainer.training.Trainer(updater, config.TRAINING_LIMIT, out=get_trainer_out_path())
    trainer.extend(extensions.LogReport(trigger=(1, 'iteration')))
    trainer.extend(extensions.PrintReport(['epoch', 'iteration', 'loss', 'elapsed_time']))
    trainer.extend(extensions.snapshot_object(hinge_gen, 'gen_iter_{.updater.iteration}'),
                   trigger=(config.SAVE_ITER_INTERVAL, 'iteration'))
    return trainer


def TransE_setting(vocab_ent, vocab_rel, train_iter, valid_iter):
    transE = models.TransE.create_transe(config.EMBED_SZ, vocab_ent, vocab_rel, config.TRANSE_GAMMA, config.TRANSE_NORM)
    if len(sys.argv) > 1:
        chainer.serializers.load_npz(sys.argv[1], transE)

    if config.DEVICE >= 0:
        chainer.cuda.get_device_from_id(config.DEVICE).use()
        transE.to_gpu(config.DEVICE)

    # opt = chainer.optimizers.SGD(config.SGD_LR)
    opt = chainer.optimizers.Adam(config.ADAM_ALPHA, config.ADAM_BETA1)
    opt.setup(transE)
    updater = chainer.training.StandardUpdater(train_iter, opt, device=config.DEVICE)
    trainer = chainer.training.Trainer(updater, config.TRAINING_LIMIT, out=get_trainer_out_path())
    trainer.extend(extensions.LogReport(trigger=(1, 'iteration')))
    trainer.extend(extensions.PrintReport(['epoch', 'iteration', 'loss', 'elapsed_time']))
    trainer.extend(extensions.snapshot_object(transE, 'transE_iter_{.updater.iteration}'),
                   trigger=(config.SAVE_ITER_INTERVAL, 'iteration'))
    return trainer


def GAN_Pretraining_setting(vocab_ent, vocab_rel, train_iter, valid_iter):
    gen = models.PretrainedGenerator.create_generator(config.EMBED_SZ, vocab_ent, vocab_rel)
    if len(sys.argv) > 1:
        chainer.serializers.load_npz(sys.argv[1], gen)

    if config.DEVICE >= 0:
        chainer.cuda.get_device_from_id(config.DEVICE).use()
        gen.to_gpu(config.DEVICE)

    opt = chainer.optimizers.SGD(config.SGD_LR)
    opt.setup(gen)
    updater = chainer.training.StandardUpdater(train_iter, opt, device=config.DEVICE)
    trainer = chainer.training.Trainer(updater, config.TRAINING_LIMIT, out=get_trainer_out_path())
    trainer.extend(extensions.LogReport(trigger=(1, 'iteration')))
    trainer.extend(extensions.PrintReport(['epoch', 'iteration', 'loss', 'elapsed_time']))
    trainer.extend(extensions.snapshot_object(gen, 'gen_iter_{.updater.iteration}'),
                   trigger=(config.SAVE_ITER_INTERVAL, 'iteration'))
    return trainer


def GAN_setting(vocab_ent, vocab_rel, train_iter, valid_iter):
    generator = models.Generator.create_generator(config.EMBED_SZ, vocab_ent, vocab_rel)
    discriminator = models.NonparametricDiscriminator(config.EMBED_SZ)
    if len(sys.argv) > 1:
        chainer.serializers.load_npz(sys.argv[1], generator)
    if len(sys.argv) > 2:
        chainer.serializers.load_npz(sys.argv[2], discriminator)

    if config.DEVICE >= 0:
        chainer.cuda.get_device_from_id(config.DEVICE).use()
        generator.to_gpu(config.DEVICE)
        discriminator.to_gpu(config.DEVICE)

    opt_g = chainer.optimizers.Adam(config.ADAM_ALPHA, config.ADAM_BETA1)
    opt_d = chainer.optimizers.Adam(config.ADAM_ALPHA, config.ADAM_BETA1)
    opt_g.setup(generator)
    opt_d.setup(discriminator)

    # updater = updaters.WGANUpdator(train_iter, opt_g, opt_d, device=config.DEVICE, d_epoch=config.OPT_D_EPOCH,
    #                                g_epoch=config.OPT_G_EPOCH, penalty_coeff=config.PENALTY_COEFF)
    # updater = updaters.LSGANUpdater(train_iter, opt_g, opt_d, config.DEVICE, config.OPT_D_EPOCH, config.OPT_G_EPOCH,
    #                                 config.HINGE_LOSS_WEIGHT)
    # updater = updaters.LeastSquareGANUpdater(train_iter, opt_g, opt_d, config.DEVICE,
    #                                          config.OPT_D_EPOCH, config.OPT_G_EPOCH)
    # updater = updaters.GANUpdater(train_iter, opt_g, opt_d, config.DEVICE, config.OPT_D_EPOCH, config.OPT_G_EPOCH)
    updater = updaters.FixedDGANUpdater(train_iter, opt_g, opt_d, config.DEVICE, config.OPT_D_EPOCH, config.OPT_G_EPOCH)

    trainer = chainer.training.Trainer(updater, config.TRAINING_LIMIT, out=get_trainer_out_path())
    trainer.extend(extensions.LogReport(trigger=(1, 'iteration')))
    trainer.extend(extensions.PrintReport(updater.get_report_list()))
    trainer.extend(extensions.snapshot_object(generator, 'gen_iter_{.updater.iteration}'),
                   trigger=(config.SAVE_ITER_INTERVAL, 'iteration'))
    trainer.extend(extensions.snapshot_object(discriminator, 'd_iter_{.updater.iteration}'),
                   trigger=(config.SAVE_ITER_INTERVAL, 'iteration'))

    return trainer


def ExperimentalGAN_setting(vocab_ent, vocab_rel, train_iter, valid_iter):
    ent_num, rel_num = len(vocab_ent) + 1, len(vocab_rel) + 1
    generator = models.VarMLP([config.EMBED_SZ * 2, config.EMBED_SZ, config.EMBED_SZ, ent_num])
    embeddings = models.Embeddings(config.EMBED_SZ, ent_num, rel_num)
    discriminator = models.VarMLP([config.EMBED_SZ * 3, config.EMBED_SZ, config.EMBED_SZ, 1])
    if len(sys.argv) > 1:
        chainer.serializers.load_npz(sys.argv[1], generator)
    if len(sys.argv) > 2:
        chainer.serializers.load_npz(sys.argv[2], discriminator)
    if len(sys.argv) > 3:
        chainer.serializers.load_npz(sys.argv[3], embeddings)

    if config.DEVICE >= 0:
        chainer.cuda.get_device_from_id(config.DEVICE).use()
        generator.to_gpu(config.DEVICE)
        discriminator.to_gpu(config.DEVICE)
        embeddings.to_gpu(config.DEVICE)

    opt_g = chainer.optimizers.Adam(config.ADAM_ALPHA, config.ADAM_BETA1)
    opt_d = chainer.optimizers.Adam(config.ADAM_ALPHA, config.ADAM_BETA1)
    opt_e = chainer.optimizers.Adam(config.ADAM_ALPHA, config.ADAM_BETA1)
    opt_g.setup(generator)
    opt_d.setup(discriminator)
    opt_e.setup(embeddings)

    updater = updaters.ExperimentalGANUpdater(train_iter, opt_g, opt_d, opt_e, config.DEVICE,
                                              config.OPT_D_EPOCH, config.OPT_G_EPOCH, ent_num)

    trainer = chainer.training.Trainer(updater, config.TRAINING_LIMIT, out=get_trainer_out_path())
    trainer.extend(extensions.LogReport(trigger=(1, 'iteration')))
    trainer.extend(extensions.PrintReport(updater.get_report_list()))
    trainer.extend(extensions.snapshot_object(generator, 'g_iter_{.updater.iteration}'),
                   trigger=(config.SAVE_ITER_INTERVAL, 'iteration'))
    trainer.extend(extensions.snapshot_object(discriminator, 'd_iter_{.updater.iteration}'),
                   trigger=(config.SAVE_ITER_INTERVAL, 'iteration'))
    trainer.extend(extensions.snapshot_object(embeddings, 'e_iter_{.updater.iteration}'),
                   trigger=(config.SAVE_ITER_INTERVAL, 'iteration'))

    return trainer


def AdversarialEmbedding_setting(vocab_ent, vocab_rel, train_iter, valid_iter):
    embeddings = models.Embeddings(config.EMBED_SZ, len(vocab_ent) + 1, len(vocab_rel) + 1)
    gen_ent = models.VarMLP([config.EMBED_SZ, config.EMBED_SZ, config.EMBED_SZ, config.EMBED_SZ])
    gen_rel = models.VarMLP([config.EMBED_SZ, config.EMBED_SZ, config.EMBED_SZ, config.EMBED_SZ])
    critic = models.VarMLP([config.EMBED_SZ * 3, config.EMBED_SZ, config.EMBED_SZ, 1])

    for i, m in enumerate([embeddings, gen_ent, gen_rel, critic]):
        if len(sys.argv) > i + 1:  # train.py(argv[0]) emb(argv[1]) ent(argv[2]) rel(argv[3]) critic(argv[4])
            chainer.serializers.load_npz(sys.argv[i + 1], m)

    if config.DEVICE >= 0:
        chainer.cuda.get_device_from_id(config.DEVICE).use()
        for m in [embeddings, gen_ent, gen_rel, critic]:
            m.to_gpu(config.DEVICE)

    opts = [chainer.optimizers.Adam(config.ADAM_ALPHA, config.ADAM_BETA1) for i in xrange(4)]
    map(lambda o, m: o.setup(m), opts, [embeddings, gen_ent, gen_rel, critic])

    updater = updaters.AdvEmbUpdater(train_iter, *opts,
                                     device=config.DEVICE,
                                     ent_num=len(vocab_ent) + 1, rel_num=len(vocab_rel) + 1,
                                     d_epoch=config.OPT_D_EPOCH, g_epoch=config.OPT_G_EPOCH)

    trainer = chainer.training.Trainer(updater, config.TRAINING_LIMIT, out=get_trainer_out_path())
    trainer.extend(extensions.LogReport(trigger=(1, 'iteration')))
    trainer.extend(extensions.PrintReport(updater.get_report_list()))
    trainer.extend(extensions.snapshot_object(embeddings, 'emb_iter_{.updater.iteration}'),
                   trigger=(config.SAVE_ITER_INTERVAL, 'iteration'))
    trainer.extend(extensions.snapshot_object(critic, 'c_iter_{.updater.iteration}'),
                   trigger=(config.SAVE_ITER_INTERVAL, 'iteration'))
    trainer.extend(extensions.snapshot_object(gen_ent, 'g_ent_iter_{.updater.iteration}'),
                   trigger=(config.SAVE_ITER_INTERVAL, 'iteration'))
    trainer.extend(extensions.snapshot_object(gen_rel, 'g_rel_iter_{.updater.iteration}'),
                   trigger=(config.SAVE_ITER_INTERVAL, 'iteration'))

    return trainer


def get_trainer_out_path():
    return os.path.join(config.MODEL_PATH, datetime.datetime.now().strftime('result-%Y-%m-%d-%H-%M-%S'))

if __name__ == "__main__":
    main()
