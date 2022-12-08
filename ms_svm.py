import statistics

import numpy as np


class Kernel:
    def __call__(self, a, b):
        ...


class PolynomialKernel(Kernel):

    def __init__(self, degree):
        assert degree >= 1
        self.degree = degree

    def __call__(self, a, b):
        # (a*b + 1) ** D
        v = (a @ b + 1) ** self.degree
        return v


class RbfKernel(Kernel):

    def __init__(self, sigma):
        assert sigma >= 0
        self.s = sigma

    def __call__(self, a, b):
        # exp(-||a-b||^2/s^2)
        v = np.exp(- ((a - b) @ (a - b)) / self.s ** 2)
        return v


class Svm:
    def __init__(self, kernel):
        self.b = 0
        self.x_train = np.array([])
        self.y_train = np.array([])
        self.memo = None
        self.a = np.array([])

        self.ker = kernel

        self.speed = [2e-3, 1e-3, 5e-4, 5e-5, 1e-6]
        self.C = 10
        self.iterations_per_stage = 1000

    def __call__(self, x: np.ndarray):
        if len(x.shape) == 1:
            return np.sign(
                sum(a_i * y_i * self.ker(x, x_i) for a_i, x_i, y_i in zip(self.a, self.x_train, self.y_train)) - self.b)
        else:
            return np.array([self(x[i]) for i in range(x.shape[0])])

    def predict(self, x: np.ndarray):
        return self(x)

    def fit(self, x: np.ndarray, y: np.ndarray, caching=True, log=True):

        assert x.shape[0] == y.shape[0]

        N = x.shape[0]

        self.x_train = x
        self.y_train = y
        self.a = np.ones((N, 1))

        if caching:
            # we keep in memo
            # y[i] * y[j] * k(x[i], x[j])
            self.memo = np.array([[y[i] * y[j] * self.ker(x[i], x[j]) for i in range(N)] for j in range(N)])

        for stage in range(len(self.speed)):
            if log:
                print("Stage {0} | {1} Loss: {2}".format(1 + stage, len(self.speed), self.loss_f()))

            for iteration in range(self.iterations_per_stage):
                # we minimize L = <d> - <s> + C*<r>, where
                # <d> := 1/2 Sum(y_i*y_j * a_i*a_j * k(x_i, x_j), i = 1..N, j = 1..N)   -- double sum
                # <s> := Sum(a_i, i = 1..N)                                             -- single sum
                # <r> := |Sum(y_i * a_i, i = 1..N)|                                     -- restriction

                if not caching:
                    ddda = np.array([(y[i] * y[i] * self.a[i] * self.ker(x[i], x[i]) +
                                      sum(y[i] * y[j] * self.a[j] * self.ker(x[i], x[j])
                                          for j in range(N))) / 2
                                     for i in range(N)])
                else:
                    ddda = np.array([(self.a[i] * self.memo[i][i] +
                                      self.memo[i] @ self.a) / 2
                                     for i in range(N)])

                dsda = np.ones_like(self.a)

                drda = np.sign(y.transpose() @ self.a) * y

                # final gradient of the loss function
                dLda = ddda - dsda + self.C * drda

                # applying the gradient
                self.a -= self.speed[stage] * dLda

        if log:
            print("Final Loss: {}".format(self.loss_f()))

        # finally, computing b
        self.b = statistics.median(abs(y[i] - sum(self.a[j] * y[j] * self.ker(x[i], x[j]) for j in range(N)))
                                   for i in range(N))
        # and freeing memory
        self.memo = None

    def loss_f(self):
        N = self.x_train.shape[0]

        d = sum(
            self.y_train[i] * self.y_train[j] * self.a[i] * self.a[j] * self.ker(self.x_train[i], self.x_train[j]) for i
            in range(N) for j in range(N)) / 2
        s = sum(self.a[i] for i in range(N))
        ri = sum(self.y_train[i] * self.a[i] for i in range(N))
        r = abs(ri)

        return d - s + self.C * r
