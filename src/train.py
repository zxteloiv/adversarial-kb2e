# coding: utf-8

from __future__ import absolute_import
import os, datetime, sys, re, argparse, logging, itertools

import chainer
import chainer.training.extensions as extensions
import config
import corpus.dataset as mod_dataset
import models as mod_models
import updaters


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--setting', '-s', choices=['transe', 'transr', 'adv_e', 'generator'], default='transe',
                        help="choose the desired trainer (training mechanism) to use")
    parser.add_argument('models', nargs='*', help='pretrained models for the same setting')

    parser.add_argument('--other-pretrained',
                        help='heterogeneously pretrained model from perhaps other settings, say TransE, when needed')
    return parser


def main(parser, argv):
    """
    main entry, used to do training pipeline
    :param parser:
    :param argv:
    :return:
    """
    args = parser.parse_args(argv)

    vocab_ent, vocab_rel = mod_dataset.load_vocab()
    dataset = map(lambda x: mod_dataset.load_corpus(x, vocab_ent, vocab_rel), (config.TRAIN_DATA, config.VALID_DATA))
    train_iter, valid_iter = map(lambda x: chainer.iterators.SerialIterator(x, batch_size=config.BATCH_SZ), dataset)

    ent_num, rel_num = len(vocab_ent) + 1, len(vocab_rel) + 1

    trainer = prepare(args, ent_num, rel_num, train_iter, valid_iter)

    dump_conf(trainer, args)
    try:
        trainer.run()
    except KeyboardInterrupt:
        pass


def prepare(args, ent_num, rel_num, train_iter, valid_iter):
    """get trainer under different settings"""

    if args.setting == 'transe':
        model = mod_models.TransE(config.EMBED_SZ, ent_num, rel_num, config.MARGIN, config.TRANSE_NORM)
        if args.models:
            chainer.serializers.load_npz(args.models[0], model)
        opts = build_optimizers(model)
        updater = chainer.training.StandardUpdater(train_iter, opts[0], device=config.DEVICE)
        # report list is provided by model if the updater used is not customized
        trainer = build_trainer((model,), ('m',), updater, model.get_report_list())

    elif args.setting == 'transe_nns':
        model = mod_models.TransENNG(config.EMBED_SZ, ent_num, rel_num, config.MARGIN, config.TRANSE_NORM)
        if args.models:
            chainer.serializers.load_npz(args.models[0], model)
        opts = build_optimizers(model)
        updater = chainer.training.StandardUpdater(train_iter, opts[0], device=config.DEVICE)
        trainer = build_trainer((model,), ('m',), updater, model.get_report_list())

    elif args.setting == 'transr':
        model = mod_models.TransR(config.EMBED_SZ, ent_num, rel_num, config.MARGIN, config.TRANSE_NORM)
        if args.models:
            chainer.serializers.load_npz(args.models[0], model)
        if args.other_pretrained:
            transE = mod_models.TransE(config.EMBED_SZ, ent_num, rel_num, config.MARGIN, config.TRANSE_NORM)
            chainer.serializers.load_npz(args.other_pretrained, transE)
            model.emb.ent.W.data = transE.ent_emb.W.data
            model.emb.rel.W.data = transE.rel_emb.W.data
        opts = build_optimizers(model)
        updater = chainer.training.StandardUpdater(train_iter, opts[0], device=config.DEVICE)
        trainer = build_trainer((model,), ('m',), updater, model.get_report_list())

    elif args.setting == 'generator':
        generator = mod_models.Generator(config.EMBED_SZ, ent_num, rel_num, config.DROPOUT)
        if args.models:
            chainer.serializers.load_npz(args.models[0], generator)
        opts = build_optimizers(generator)
        updater = updaters.GANPretraining(train_iter, opts[0], opts[0], ent_num, rel_num, config.MARGIN, config.DEVICE)
        trainer = build_trainer((generator,), ('g',), updater, updater.get_report_list())

    elif args.setting == 'generator_ns':
        generator = mod_models.VarMLP([config.EMBED_SZ * 2, config.EMBED_SZ, config.EMBED_SZ, ent_num], config.DROPOUT)
        embeddings = mod_models.Embeddings(config.EMBED_SZ, ent_num, rel_num)
        opts = build_optimizers(generator, embeddings)
        if args.models:
            chainer.serializers.load_npz(args.models[0], generator)
            chainer.serializers.load_npz(args.models[1], embeddings)
        updater = updaters.MLEGenNSUpdater(train_iter, opts[0], opts[1], ent_num, config.DEVICE)
        trainer = build_trainer((generator, embeddings), ('g', 'e'), updater, updater.get_report_list())

    elif args.setting == 'generator_nns':
        generator = mod_models.VarMLP([config.EMBED_SZ * 2, config.EMBED_SZ, config.EMBED_SZ, ent_num], config.DROPOUT)
        embeddings = mod_models.Embeddings(config.EMBED_SZ, ent_num, rel_num)
        if args.models:
            chainer.serializers.load_npz(args.models[0], generator)
            chainer.serializers.load_npz(args.models[1], embeddings)
        opts = build_optimizers(generator, embeddings)
        updater = updaters.MLEGenNNSUpdater(train_iter, opts[0], opts[1], ent_num, config.DEVICE)
        trainer = build_trainer((generator, embeddings), ('g', 'e'), updater, updater.get_report_list())

    elif args.trainer == 'adv_e':
        generator = mod_models.Generator(config.EMBED_SZ, ent_num, rel_num, config.DROPOUT)
        # discriminator = mod_models.Discriminator(config.EMBED_SZ, ent_num, rel_num, config.DROPOUT)
        discriminator = mod_models.TransE(config.EMBED_SZ, ent_num, rel_num, config.MARGIN)
        models = (generator, discriminator)
        for dump, model in itertools.izip(args.model, models):
            chainer.serializers.load_npz(dump, model)
        opts = build_optimizers(*models)
        updater = updaters.GANUpdater(train_iter, opts[0], opts[1], device=config.DEVICE,
                                      d_epoch=config.OPT_D_EPOCH, g_epoch=config.OPT_G_EPOCH, margin=config.MARGIN,
                                      greater_d_value_better=isinstance(discriminator, mod_models.Discriminator),
                                      dist_based_g=config.DISTANCE_BASED_G, dist_based_d=config.DISTANCE_BASED_D)
        trainer = build_trainer((generator, discriminator), ('g', 'd'), updater, updater.get_report_list())

    else:
        return None

    return trainer


def build_optimizers(*models):
    opts = []
    for m in models:
        opt = chainer.optimizers.Adam(config.ADAM_ALPHA, config.ADAM_BETA1)
        opt.setup(m)
        opt.add_hook(chainer.optimizer.WeightDecay(config.WEIGHT_DECAY))
        opts.append(opt)

    return opts


def build_trainer(models, model_prefixes, updater, report_list):
    if config.DEVICE >= 0:
        chainer.cuda.get_device_from_id(config.DEVICE).use()
        for m in models:
            m.to_gpu(config.DEVICE)

    trainer = chainer.training.Trainer(updater, config.TRAINING_LIMIT,
                                       out=os.path.join(config.MODEL_PATH,
                                                        datetime.datetime.now().strftime('result-%Y-%m-%d-%H-%M-%S')))

    trainer.extend(extensions.LogReport(trigger=(1, 'iteration')))
    trainer.extend(extensions.PrintReport(report_list))
    for (m, p) in itertools.izip(models, model_prefixes):
        trainer.extend(extensions.snapshot_object(m, p + '_iter_{.updater.iteration}'),
                       trigger=(config.SAVE_ITER_INTERVAL, 'iteration'))

    return trainer


def dump_conf(trainer, args):
    if not os.path.exists(trainer.out):
        os.mkdir(trainer.out, 0775)
    conf_dump = os.path.join(trainer.out, 'conf')
    with open(conf_dump, 'w') as f:
        for k in dir(config):
            if re.match('^[A-Z]', k):
                f.write("{0} = {1}\n".format(k, getattr(config, k)))

        for k in vars(args).iterkeys():
            f.write("{0} = {1}\n".format(k, getattr(args, k)))


if __name__ == "__main__":
    parser = get_args_parser()

    main(parser, sys.argv[1:])
