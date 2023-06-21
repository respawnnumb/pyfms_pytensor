from pytensor import tensor as T
import pymc as pm

from . import core

class Linear(core.Transformer):
    def transform(self, y_hat):
        return y_hat


class Sigmoid(core.Transformer):
    def transform(self, y_hat):
        #return T.nnet.sigmoid(y_hat)
        return pm.math.sigmoid(y_hat)
