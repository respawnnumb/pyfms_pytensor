from pytensor import tensor as T

from . import core

class SquaredError(core.Error):
    def apply(self, y, y_hat):
        return (y - y_hat)**2


class BinaryCrossEntropy(core.Error):
    def apply(self, y, y_hat):
        #return T.nnet.binary_crossentropy(y_hat, y)
        return -(y_hat * T.log(y) + (1.0 - y_hat) * T.log(1.0 - y))
