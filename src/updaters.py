# coding: utf-8

import chainer

class WGANUpdator(chainer.training.StandardUpdater):
    def __init__(self, iterator, opt_g, opt_d, device, d_epoch):
        super(WGANUpdator, self).__init__(iterator, {"opt_g": opt_g, "opt_d": opt_d}, device=device)
        self.d_epoch = d_epoch
        self.device = device

    def update_core(self):
        iterator = self.get_iterator('main')

        batch = iterator.__next__()
        #print "update iteration %d" % self.iteration
        import time
        time.sleep(0.01)
