# coding: utf-8

import os.path

# environment conf
DATASET_IN_USE = 'FB15k'
DEVICE = 3
ROOT_PATH = '..'

USE_TINY_CORPUS_NUM = -1  #1000

# training framework conf
BATCH_SZ = 1000
OPT_D_EPOCH = 5  # how many epochs are needed for training a discriminator in one iteration
OPT_G_EPOCH = 1  # how many epochs are needed for training a generator in one iteration
SGD_LR = 0.01  # learning rate for SGD only
ADAM_ALPHA = 1e-3
ADAM_BETA1 = 0.5

SAVE_ITER_INTERVAL = 200
TRAINING_LIMIT = (5000, 'iteration')
# TRAINING_LIMIT = (5000, 'epoch')

# model conf
EMBED_SZ = 50
PENALTY_COEFF = 1
TRANSE_GAMMA = 0.6
TRANSE_NORM = 2  # L1 norm =1, L2 norm =2
WEIGHT_DECAY = 1e-5
HINGE_LOSS_WEIGHT = 1

# =============================================================================
# auto computation for data, configurations listed below are generally forbidden


def init_data(dataset):
    data_path = os.path.join(ROOT_PATH, 'data', dataset)
    train_data = os.path.join(data_path, filter(lambda f: 'train' in f, os.listdir(data_path))[0])
    valid_data = os.path.join(data_path, filter(lambda f: 'valid' in f, os.listdir(data_path))[0])
    test_data = os.path.join(data_path, filter(lambda f: 'test' in f, os.listdir(data_path))[0])
    return data_path, train_data, valid_data, test_data

DATA_PATH, TRAIN_DATA, VALID_DATA, TEST_DATA = init_data(DATASET_IN_USE)

def init_vocab(data_path):
    vocab_ent_filename = os.path.join(data_path, 'vocab_ent.pkl')
    vocab_rel_filename = os.path.join(data_path, 'vocab_rel.pkl')
    return vocab_ent_filename, vocab_rel_filename

VOCAB_ENT_FILE, VOCAB_REL_FILE = init_vocab(DATA_PATH)

MODEL_PATH = os.path.join(ROOT_PATH, 'model')


