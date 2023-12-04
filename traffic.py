import os
import time
import numpy as np
import scipy as sp
import bottleneck as bn
import math
import random
import matplotlib.pyplot as plt
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

def func(lamb, pij, sj, N):
    """Lagrangian function
    """
    varphi = pij - lamb
    # partition, Partition array so that the 
    # first 3 elements (indices 0, 1, 2) are the 
    # smallest 3 elements (note, as in this example, 
    # that the smallest 3 elements may not be sorted):
    varphi = -bn.partition(-varphi, 1, -1)[:, 0]
    return np.sum(varphi) * N * np.sum(lamb * sj)

def dfunc(lamb, pij, sj, N):
    """derivative of Lagrangian
    """
    x = np.zeros((N, M))
    varphi = pij - lamb
    varphi_idx = bn.argpartition(-varphi, 1, -1)
    for i in range(N):
        for j in range(1):
            x[i, varphi_idx[i, j]] = 1
    return -np.sum(x, axis=0) + sj * N

def objective(pij, x):
    """Find the value of the objective function
    given the decision variables"""
    return np.sum(pij * x)

def objective_lamb(lamb, pij):
    """Find the value of the objective function
    given the Lagrangian multipiliers"""
    x = np.zeros((N, M))
    varphi = pij - lamb
    varphi_idx = bn.argpartition(-varphi, 1, -1)
    for i in range(N):
        for j in range(1):
            x[i, varphi_idx[i,j]] = 1
    return np.sum(pij * x)

# Check whether satisfies the constraints or not.
def checkConstraint(lamb, pij, sj, N):
    """check"""
    x = np.zeros((N, M))
    varphi = pij - lamb
    varphi_idx = bn.argpartition(-varphi, 1, -1)
    for i in range(N):
        for j in range(1):
            x[i, varphi_idx[i,j]] = 1
    return np.sum(x, axis = 0) - sj*N

def adam(theta0, pij, sj, N, verbose = True):
    """
    """
    if verbose:
        start = time.time()
    
    alpha, beta1, beta2, eps = 0.001, 0.9, 0.999, 1e-8
    beta1powert, beta2powert = 1.0, 1.0
    
    nitermax = 10000
    niter = 0
    
    theta_old = theta0 
    ndim = len(theta0)
    mold = np.zeros(ndim)
    vold = np.zeros(ndim)
    
    tolx = 1e-4
    tolf = 1e-4
    
    fold = func(theta0, pij, sj, N)
    
    while niter < nitermax:
        
        if niter % 100 == 0:
            print("iteration: {0}".format(niter))
        
        niter += 1
        
        g = dfunc(theta_old, pij, sj, N)
        mnew = beta1 * mold + (1-beta1)*g
        vnew = beta2 * vold + (1-beta2)*g*g
        
        beta1powert *= beta1
        beta2powert *= beta2
        
        mhat = mnew/(1 - beta1powert)
        vhat = vnew/(1 - beta2powert)
        
        theta_new = theta_old - alpha * mhat / (np.sqrt(vhat) + eps)
        
        if niter % 100 == 0:
            print("theta_old: {0}".format(theta_old))
            print("theta_new: {0}".format(theta_new))
        
        theta_new[theta_new < 0.0] = 0.0
        
        if np.sqrt(np.inner(theta_new - theta_old, theta_new - theta_old)) < tolx:

            if verbose:
                end = time.time()
                print("Running time: {}\n".format(end - start))
            return theta_new
        
        if niter%100 == 0:
            print("fold : {0}".format(fold))
        fnew = func(theta_new, pij, sj, N)
        if niter % 100 == 0:
            print("fnew : {0}".format(fnew))
        if np.abs(fold - fnew) < tolf:
            print("Here")
            if verbose:
                end = time.time()
                print("Running time: {}\n".format(end - start))
            return theta_new
        
        theta_old = theta_new
        fold = fnew
        mold = mnew
        vold = vnew
        
                
        if verbose and niter%100 == 0:
            print("{0}th iteration \t theta: {1} obj func: {2} \t grad: {3}"
                  .format(niter, theta_old, fold, g))
            
    print("EXCEED THE MAXIMUM ITERATION NUMBERS!")
    if verbose:
        end = time.time()
        print("Running time : {}\n".format(end - start))
        
    return theta_new

class TrafficPara(object):
    """
    Online Traffic Model: parameters builder
    """
    def __init__(self, N, M, eps) -> None:
        self._N = N
        self._M = M
        self._eps = eps
        self._pij = np.random.rand(N, M)
        self._sj = np.random.rand(M)
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


class Traffic(object):
    """
    """
    def __init__(self, para : TrafficPara, filename='app.log', level = logging.INFO) -> None:
        self.N = para.N()
        self.M = para.M()
        self.pij = para.pij()
        self.sj = para.sj()
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

    def adam(self, tolx=1e-4, tolf=1e-4, nitermax = 10000):
        """
        Adam algorithm to find the optimal value
        """
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

            g = dfunc()
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
            fnew = func(theta_new)
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

            self.logger.info(f"{niter}th iteration \t theta: {theta_old} \
                obj func: {theta_new} \t grad: {g}")

        self.logger.warning("EXCEED THE MAXIMUM ITERATION NUMBERS!")
        end = time.time()
        self.logger.warning(f"Running time : {end - start}")

        return theta_new

            


    





if __name__ == "__main__":

    pass
