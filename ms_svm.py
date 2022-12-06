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
        v = (a @ b + 4) ** self.degree
        return v


class RbfKernel(Kernel):

    def __init__(self, sigma):
        assert sigma >= 0
        self.s = sigma

    def __call__(self, a, b):
        # exp(-||a-b||^2/s^2)
        v = np.exp(- ((a-b) @ (a-b)) / self.s**2)
        return v


class Svm:
    def __init__(self, kernel):
        self.b = 0
        self.x_train = np.array([])
        self.y_train = np.array([])
        self.a = np.array([])

        self.ker = kernel

        self.speed = 1e-6
        self.e_s_n = 100     # early stopping

    def __call__(self, x: np.ndarray):
        if len(x.shape) == 1:
            return sum(a_i*y_i*self.ker(x, x_i) for a_i, x_i, y_i in zip(self.a, self.x_train, self.y_train)) + self.b
        else:
            return np.array([self(x[i]) for i in range(x.shape[0])])

    def fit(self, x: np.ndarray, y: np.ndarray):

        assert x.shape[0] == y.shape[0]

        N = x.shape[0]

        if type(self.ker) == PolynomialKernel:
            C = 100
            self.speed = 10**(-3-self.ker.degree)
        elif type(self.ker) == RbfKernel:
            C = 1e1
            self.speed = 1e-4
        else:
            C = 1e1
            self.speed = 1e-4

        self.x_train = x
        self.y_train = y
        self.a = np.ones(N)/2

        n = 0
        prev_L = np.inf
        prev_a = self.a
        while n < self.e_s_n:
            # we minimize L = <d> - <s> + C*<r>, where
            # <d> := 1/2 Sum(y_i*y_j * a_i*a_j * k(x_i, x_j), i = 1..N, j = 1..N)   -- double sum
            # <s> := Sum(a_i, i = 1..N)                                             -- single sum
            # <r> := |Sum(y_i * a_i, i = 1..N)|                                     -- restriction

            d = sum(y[i] * y[j] * self.a[i] * self.a[j] * self.ker(x[i], x[j]) for i in range(N) for j in range(N))/2
            s = sum(self.a[i] for i in range(N))
            ri = sum(y[i] * self.a[i] for i in range(N))
            r = abs(ri)

            L = d - s + C*r

            if L + 1e-9 < prev_L:
                prev_L = L
                prev_a = self.a
                n = 0
            else:
                n += 1

            ddda = np.array([(
                    y[i]*y[i]*self.a[i]*self.ker(x[i], x[i]) +
                    sum(y[i]*y[j]*self.a[j]*self.ker(x[i], x[j])
                        for j in range(N))) / 2
                for i in range(N)])

            dsda = np.ones_like(self.a)

            drda = np.sign(ri) * np.array([y[i] for i in range(N)])

            # final gradient of the loss function
            dLda = ddda - dsda + C*drda

            # applying the gradient
            self.a -= self.speed * dLda

        self.a = prev_a

        # finally, computing b
        self.b = statistics.median(abs(y[i] - sum(self.a[j]*y[j]*self.ker(x[i], x[j]) for j in range(N)))
                                   for i in range(N))


