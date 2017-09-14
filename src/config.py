# coding: utf-8

import os.path

# environment conf
DATASET_IN_USE = 'FB15k'
DEVICE = -1
ROOT_PATH = '..'

# training framework conf
BATCH_SZ = 100
EPOCH_NUM = 40
OPT_D_EPOCH = 40 # how many epochs are needed for training a discriminator in one epoch for the generator


# model conf
EMBED_SZ = 50

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


