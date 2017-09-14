# coding: utf-8

import os.path

DATASET_IN_USE = 'FB15k'
DEVICE = -1

# =============================================================================
# auto computation for data, configurations listed below are generally forbidden


def init_conf(dataset):
    data_path = os.path.join('..', 'data', dataset)
    train_data = filter(lambda f: 'train' in f, os.listdir(data_path))[0]
    valid_data = filter(lambda f: 'valid' in f, os.listdir(data_path))[0]
    test_data = filter(lambda f: 'test' in f, os.listdir(data_path))[0]
    return data_path, train_data, valid_data, test_data

DATA_PATH, TRAIN_DATA, VALID_DATA, TEST_DATA = init_conf(DATASET_IN_USE)

