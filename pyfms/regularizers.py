from pytensor import tensor as T

from . import core

class L1(core.Regularizer):
    def __init__(self, beta_w0 = 0.0, beta_w1 = 0.0, beta_v = 0.0):
        self.beta_w0 = beta_w0
        self.beta_w1 = beta_w1
        self.beta_v = beta_v

    def regularize(self, loss, w0, w1, v):
        penalty = (self.beta_w0 * abs(w0)) \
                  + (self.beta_w1 * T.mean(abs(w1))) \
                  + (self.beta_v * T.mean(abs(v)))
        return loss + penalty


class L2(core.Regularizer):
    def __init__(self, beta_w0 = 0.0, beta_w1 = 0.0, beta_v = 0.0):
        self.beta_w0 = beta_w0
        self.beta_w1 = beta_w1
        self.beta_v = beta_v

    def regularize(self, loss, w0, w1, v):
        penalty = (self.beta_w0 * (w0 ** 2)) \
                  + (self.beta_w1 * T.mean(w1 ** 2)) \
                  + (self.beta_v * T.mean(v ** 2))
        return loss + penalty
