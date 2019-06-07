# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 09:39:27 2019

LM least square error 
examples in 

https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html

@author: fuyin
"""
import numpy as np
from scipy.sparse import lil_matrix
from scipy.optimize import least_squares

def fun_rosenbrock(x):
    return np.array([10 * (x[1] - x[0]**2), (1 - x[0])])

def jac_rosenbrock(x):
    return np.array([
        [-20 * x[0], 10],
        [-1, 0]])

def fun_broyden(x):
    f = (3 - x) * x + 1
    f[1:] -= x[:-1]
    f[:-1] -= 2 * x[1:]
    return f

def sparsity_broyden(n):
    sparsity = lil_matrix((n, n), dtype=int)
    i = np.arange(n)
    sparsity[i, i] = 1
    i = np.arange(1, n)
    sparsity[i, i - 1] = 1
    i = np.arange(n - 1)
    sparsity[i, i + 1] = 1
    return sparsity

def gen_data(t, a, b, c, noise = 0, n_outliers = 0, random_state = 0):
    y = a + b * np.exp(t*c)
    
    rnd = np.random.RandomState(random_state)
    error = noise * rnd.randn(t.size)
    outliers = rnd.randint(0, t.size, n_outliers)
    error[outliers] *= 10
    
    return y +error

def fun_residual(x, t, y):
    return x[0] + x[1]*np.exp(t*x[2]) - y

if __name__ == "__main__":
    x0_rosenbrock = np.array([2, 2])
    r = fun_rosenbrock(x0_rosenbrock)
    res_1 = least_squares(fun_rosenbrock, x0_rosenbrock)
    print('x = {} cost = {} optimality = {}'.format(res_1.x, res_1.cost, res_1.optimality))
    res_1 = least_squares(fun_rosenbrock, x0_rosenbrock, jac_rosenbrock, bounds=([-np.inf, 1.5], np.inf))
    print('x = {} cost = {} optimality = {}'.format(res_1.x, res_1.cost, res_1.optimality))
    n = 10 #100000
    x0_broyden = -np.ones(n) 
    res_3 = least_squares(fun_broyden, x0_broyden, jac_sparsity=sparsity_broyden(n))
    print('res3 cost = {} optimality = {}'.format(res_3.cost, res_3.optimality))
    
    a, b, c, t_min, t_max = 0.5, 2.0, -1, 0, 10
    n_pts = 15
    t_train = np.linspace(t_min, t_max, n_pts)
    y_train = gen_data(t_train, a, b, c, noise = 0.1, n_outliers = 3)
    
    #%% least square fit 
    x0 = np.array([1.0, 1.0, 1.0])
    res_lsq = least_squares(fun_residual, x0, args = (t_train, y_train))
    res_soft_l1 = least_squares(fun_residual, x0, loss='soft_l1', f_scale = 0.1, args = (t_train, y_train))
    res_log = least_squares(fun_residual, x0, loss = 'cauchy', f_scale = 0.1, args = (t_train, y_train))
    
    #%% 
    t_test = np.linspace(t_min, t_max, n_pts * 10)
    y_true = gen_data(t_test, a, b, c)
    y_lsq = gen_data(t_test, *res_lsq.x)
    y_soft_l1 = gen_data(t_test, *res_soft_l1.x)
    y_log = gen_data(t_test, *res_log.x)
    
    import matplotlib.pyplot as plt
    plt.plot(t_train, y_train, 'o')
    plt.plot(t_test, y_true, 'k', linewidth=2, label='true')
    plt.plot(t_test, y_lsq, label='linear loss')
    plt.plot(t_test, y_soft_l1, label='soft_l1 loss')
    plt.plot(t_test, y_log, label='cauchy loss')
    plt.xlabel("t")
    plt.ylabel("y")
    plt.legend()
    plt.show()

    
    
    