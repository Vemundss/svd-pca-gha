import numpy as np


class GHA(object):
    """
    Code with inspiration form:
    https://github.com/matwey/python-gha
    """

    def __init__(self, n_input, n_features, lr=1000, loc=0.0, scale=0.1, lr_incr=10):
        self.n_input = n_input
        self.n_features = n_features
        self.lr = lr
        self.lr_incr = lr_incr

        self._W = np.random.normal(
            loc=loc, scale=scale / n_input, size=(n_features, n_input)
        )
        self.cma = 0
        self.cma_n = 0

    def forward(self, x, mean_shift=False):
        return np.einsum(
            "...ij,...j->...i", self._W, (x - self.cma if mean_shift else x)
        )

    def train_step(self, x):
        # calculate cumulative moving average
        self.cma = (np.mean(x, axis=0) + self.cma_n * self.cma) / (self.cma_n + 1)
        self.cma_n += 1

        # forward zero-mean input (x)
        # => Cov(X,X) = E[(X - E[X])(X - E[X])^T] = E[XX^T]
        y = self.forward(x - self.cma)

        # outer products (retain batch dim)
        y_yT = np.einsum("...i,...j->...ij", y, y)
        y_xT = np.einsum("...i,...j->...ij", y, x)

        # lower-triangular
        LT_Y = np.tril(y_yT)

        # dot product
        LT_YC = np.einsum("...ij,...jk->...ik", LT_Y, self._W)

        dW = np.mean((y_xT - LT_YC), axis=0) / self.lr
        print(dW)
        self._W += dW

        # update lr
        self.lr += self.lr_incr
        return y

    def inverse(self, y, mean_shift=False):
        # einsum is ordered alphabetically, so ji is a transpose
        return np.einsum("...ji,...j->...i", self._W, y) + (
            self.cma if mean_shift else 0
        )
