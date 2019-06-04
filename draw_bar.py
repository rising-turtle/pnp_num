# -*- coding: utf-8 -*-
"""
Created on Fri May 31 13:41:07 2019

@author: fuyin
"""

import matplotlib.pyplot as plt
import numpy as np

def draw_mean_std(x, mu, std, title = None):
    """
    draw x, mu std 
    """        


def example():
    # example data
    # x = np.arange(0.1, 4, 0.5)
    x = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
    y = np.exp(-x)
    
    # example variable error bar values
    yerr = 0.1 + 0.2*np.sqrt(x)
    xerr = 0.1 + yerr
    
    # First illustrate basic pyplot interface, using defaults where possible.
    plt.figure()
    plt.errorbar(x, y, xerr=0.2, yerr=0.4)
    plt.title("Simplest errorbars, 0.2 in x, 0.4 in y")
    
    # Now switch to a more OO interface to exercise more features.
    fig, axs = plt.subplots(nrows=2, ncols=2, sharex=True)
    ax = axs[0,0]
    ax.errorbar(x, y, yerr=yerr, fmt='o')
    ax.set_title('Vert. symmetric')
    
    # With 4 subplots, reduce the number of axis ticks to avoid crowding.
    ax.locator_params(nbins=4)
    
    ax = axs[0,1]
    ax.errorbar(x, y, xerr=xerr, fmt='o')
    ax.set_title('Hor. symmetric')
    
    ax = axs[1,0]
    ax.errorbar(x, y, yerr=[yerr, 2*yerr], xerr=[xerr, 2*xerr], fmt='--o')
    ax.set_title('H, V asymmetric')
    
    ax = axs[1,1]
    ax.set_yscale('log')
    # Here we have to be careful to keep all y values positive:
    ylower = np.maximum(1e-2, y - yerr)
    yerr_lower = y - ylower
    
    ax.errorbar(x, y, yerr=[yerr_lower, 2*yerr], xerr=xerr,
                fmt='o', ecolor='g', capthick=2)
    ax.set_title('Mixed sym., log y')
    
    fig.suptitle('Variable errorbars')
    
    plt.show()

    
if __name__=="__main__":
    
#    x = np.array([1, 2, 3, 4, 5])
#    y = np.power(x, 2) # Effectively y = x**2
#    e = np.array([1.5, 2.6, 3.7, 4.6, 5.5])
#
#    plt.errorbar(x, y, e, uplims=True, lolims=True, marker='o')
#    # plt.errorbar(x, y, e, linestyle='None', marker='o')
#    # plt.plot(x,y, 'r-')
#    
#    plt.show()   
    
    example()
    # example data
#    x = np.arange(0.1, 4, 0.5)
#    y = np.exp(-x)
#    # example error bar values that vary with x-position
#    error = 0.1 + 0.2 * x
#    # error bar values w/ different -/+ errors
#    lower_error = 0.4 * error
#    upper_error = error
#    asymmetric_error = [lower_error, upper_error]
#    
#    fig, (ax0, ax1) = plt.subplots(nrows=2, sharex=True)
#    ax0.errorbar(x, y, yerr=error, fmt='-o')
#    ax0.set_title('variable, symmetric error')
#    
#    ax1.errorbar(x, y, xerr=asymmetric_error, fmt='o')
#    ax1.set_title('variable, asymmetric error')
#    ax1.set_yscale('log')
#    plt.show()