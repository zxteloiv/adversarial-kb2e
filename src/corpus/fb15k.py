# coding: utf-8

import logging

def reader(generator):
    for l in generator:
        h, r, t = l.rstrip().split('\t')
        yield h, r, t




