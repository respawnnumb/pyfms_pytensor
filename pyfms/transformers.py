from pytensor import tensor as T

from . import core

class Linear(core.Transformer):
    def transform(self, y_hat):
        return y_hat


class Sigmoid(core.Transformer):
    def transform(self, y_hat):
        #return T.nnet.sigmoid(y_hat)
        return return 1 / (1 + T.exp(-y_hat))
