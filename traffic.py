import os
import time
import numpy as np

import bottleneck as bn
import math
import random

import mip
import logging

def linear_fun(x, x1, y1, x2, y2):
    """
    linear function (x1, y1) and (x2, y2)
    """
    slope = (y2 - y1) / (x2 - x1)

    return slope * (x - x1) + y1

def log_fun(x, x1, y1, x2, y2):
    """
    logarithmic function (x1, y1) and (x2, y2)
    """
    a = (y2 - y1) / (math.log(x2) - math.log(x1))
    b = y1 - a * math.log(x1)

    return a * np.log(x) + b

def mono_fun(x, x1, y1, x2, y2):
    """
    construct a monotonoically increaseing function
    between (x1, y1) and (x2, y2)
    """
    funs = {
        "linear_fun": linear_fun,
        "log_fun": log_fun
    }

    rnd_name = random.choice(list(funs.keys()))
    rnd_fun = funs[rnd_name]

    return rnd_fun(x, x1, y1, x2, y2)

def noiser(x, n, std_dev):
    """
    Add the noise to the function.
    """
    return x + np.random.normal(0.0, std_dev, n)

class TrafficPara(object):
    """
    Online Traffic Model: parameters builder
    """
    def __init__(self, N, M, eps, seed = 1337) -> None:
        np.random.seed(seed)
        self._N = N
        self._M = M
        self._eps = eps
        self._pij = np.random.rand(N, M)
        self._sj = np.random.randint(0, N, M)
        self._sj = self._sj/np.sum(self._sj) + eps
    
    @property
    def N(self):
        return self._N
    
    @property
    def M(self):
        return self._M

    @property
    def eps(self):
        return self._eps
    
    @property
    def pij(self):
        return self._pij

    @property
    def sj(self):
        return self._sj


class TrafficDual(object):
    """
    """
    def __init__(self, para : TrafficPara, filename='app.log', level = logging.INFO) -> None:
        self.N = para.N
        self.M = para.M
        self.pij = para.pij
        self.sj = para.sj
        self.lamb = np.random.rand(self.M)

        self.logger = logging.getLogger(filename)
        self.logger.setLevel(level)
        ch = logging.StreamHandler()
        ch.setLevel(level)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

    def func(self, lamb):
        """Lagrangian function
        """
        varphi = self.pij - lamb
        # partition, Partition array so that the 
        # first 3 elements (indices 0, 1, 2) are the 
        # smallest 3 elements (note, as in this example, 
        # that the smallest 3 elements may not be sorted):
        varphi = -bn.partition(-varphi, 1, -1)[:, 0]
        return np.sum(varphi) * self.N * np.sum(lamb * self.sj)

    def dfunc(self, lamb):
        """derivative of Lagrangian
        """
        x = np.zeros((self.N, self.M))
        varphi = self.pij - lamb
        varphi_idx = bn.argpartition(-varphi, 1, -1)
        for i in range(self.N):
            for j in range(1):
                x[i, varphi_idx[i, j]] = 1
        return -np.sum(x, axis=0) + self.sj * self.N

    @property
    def x(self):
        x = np.zeros((self.N, self.M))
        varphi = self.pij - self.lamb
        varphi_idx = bn.argpartition(-varphi, 1, -1)
        for i in range(self.N):
            for j in range(1):
                x[i, varphi_idx[i,j]] = 1
        return x

    @property
    def objective_value(self):
        x = np.zeros((self.N, self.M))
        varphi = self.pij - self.lamb
        varphi_idx = bn.argpartition(-varphi, 1, -1)
        for i in range(self.N):
            for j in range(1):
                x[i, varphi_idx[i,j]] = 1
        return np.sum(self.pij * x)

    def checkConstraint(self):
        """check the violation the major constraint"""
        return np.sum(self.x, axis=0) - self.sj * self.N

    def checkAbsConstraint(self):
        """check the magnitude of violation of the constraints"""
        val = np.sum(self.x, axis=0) - self.sj * self.N
        return np.sum(val[val > 0])/self.N

    def optimize(self, optimizer_name = "adam", tolx=1e-5, tolf=1e-5, nitermax = 10000):

        optimizers = {
            "adam": self.adam
        }
        optimizer = optimizers[optimizer_name]
        self.lamb = optimizer(tolx, tolf, nitermax)

    def adam(self, tolx=1e-4, tolf=1e-4, nitermax = 10000):
        """
        Adam algorithm to find the optimal value
        """
        self.logger.info(f"Iteration Begins!")

        theta0 = self.lamb

        start = time.time()

        alpha, beta1, beta2, eps = 0.001, 0.9, 0.999, 1e-8
        beta1powert, beta2powert = 1.0, 1.0

        niter = 0
    
        theta_old = theta0 
        ndim = len(theta0)
        mold = np.zeros(ndim)
        vold = np.zeros(ndim)

        fold = self.func(theta0)

        while niter < nitermax:

            self.logger.debug(f"Iteration: {niter}")
            niter += 1

            g = self.dfunc(theta_old)
            mnew = beta1 * mold + (1-beta1)*g
            vnew = beta2 * vold + (1-beta2)*g*g
        
            beta1powert *= beta1
            beta2powert *= beta2
        
            mhat = mnew/(1 - beta1powert)
            vhat = vnew/(1 - beta2powert)
        
            theta_new = theta_old - alpha * mhat / (np.sqrt(vhat) + eps)

            self.logger.debug(f"theta_old: {theta_old}")
            self.logger.debug(f"theta_new: {theta_new}")

            theta_new[theta_new<0.0] = 0.0

            if np.sqrt(np.inner(theta_new - theta_old, theta_new - theta_old)) < tolx:

                end = time.time()
                self.logger.info(f"Exit from gradient")
                self.logger.info(f"Running time: {end - start}")
                return theta_new

            self.logger.debug(f"fold: {fold}")
            fnew = self.func(theta_new)
            self.logger.debug(f"fnew: {fnew}")

            if np.abs(fold - fnew) < tolf:
                end = time.time()
                self.logger.info(f"Exit from function")
                self.logger.info(f"Running time: {end - start}")
                return theta_new

            theta_old = theta_new
            fold = fnew
            mold = mnew
            vold = vnew

            self.logger.debug(f"{niter}th iteration \t theta: {theta_old} \
                obj func: {theta_new} \t grad: {g}")

        self.logger.warning("EXCEED THE MAXIMUM ITERATION NUMBERS!")
        end = time.time()
        self.logger.warning(f"Running time : {end - start}")

        return theta_new


class TrafficMIP(object):
    """MIP solver
    """
    def __init__(self, para : TrafficPara, filename='mip.log', level = logging.INFO,
    maxseconds=300) -> None:
        self.N = para.N
        self.M = para.M
        self.pij = para.pij
        self.sj = para.sj

        self.maxseconds = maxseconds

        self.logger = logging.getLogger(filename)
        self.logger.setLevel(level)
        ch = logging.StreamHandler()
        ch.setLevel(level)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

        I = range(self.N)
        V = range(self.M)

        self.model = mip.Model()

        self.x = [[self.model.add_var(var_type = mip.BINARY) for j in V] for i in I]

        self.model.objective = mip.maximize(mip.xsum(self.pij[i][j] * self.x[i][j] for i in I for j in V))

        # local constraints, only 3 benefits recommended for 
        for i in I:
            self.model += mip.xsum(self.x[i][j] for j in V) == 1
    
        # global constraints, the coupons are limited by numbers
        for j in V:
            self.model += mip.xsum(self.x[i][j] for i in I) <= self.sj[j] * self.N

        self.model.optimize(max_seconds=maxseconds)

    def checkAbsConstraint(self):
        """check the magnitude of violation of the constraints
        """
        val = []
        for j in range(self.M):
            sumval = 0
            for i in range(self.N):
                sumval += self.x[i][j].x
            val.append(sumval)
        val = np.array(val)
        val -= self.sj * self.N
        return np.sum(val[val > 0])/self.N


def optimality(q, qs):
    return 1 - np.abs(q - qs)/qs


if __name__ == "__main__":

    para = TrafficPara(1000, 10, 0.01, 13)
    model_dual = TrafficDual(para)
    model_dual.optimize()
    print(model_dual.checkAbsConstraint())

    model_mip  = TrafficMIP(para, maxseconds=30)
    print(model_mip.checkAbsConstraint())

    q  = model_dual.objective_value
    qs = model_mip.model.objective_value
    
    print(optimality(q, qs))




