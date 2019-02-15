#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 12:13:30 2019

@author: lagerwer
"""
import numpy as np
import scipy.interpolate as sp
import pylab
import time
import gc
from odl.contrib import fom
from tqdm import tqdm
from numba import autojit, prange, jit
pylab.rcParams.update({'font.size': 16})
# %%
def comp_MSE(x_rec, x_ref):
    return  np.linalg.norm(x_rec - x_ref) ** 2 / (2 * np.linalg.norm(x_ref)**2)

def comp_AV_VAR(array):
    AV = np.mean(array)
    VAR = np.sum((array - AV) ** 2)/ (np.size(array) - 1)
    
    return AV, VAR

@autojit
def loop_x_interp(pix, nL, x0, x1, DP):
    x_in = np.zeros((nL[0], nL[1], pix ** 2))
    for npix in prange(pix ** 2):
        CS = sp.RectBivariateSpline(x0[0], x0[1], DP[:, :, npix])
        x_in[:, :, npix] = CS(x1[0], x1[1])
    
    return x_in.reshape((nL[0], nL[1], pix, pix))

@autojit
def loop_comp_QM(obj):
    MSE_in = np.zeros(obj.nL)
    MSE_app = np.zeros(obj.nL)
    SSIM_in = np.zeros(obj.nL)
    SSIM_app = np.zeros(obj.nL)
    for i in prange(obj.nL[0]):
        for j in prange(obj.nL[1]):
            MSE_in[i, j] = comp_MSE(obj.x_in[i, j, :, :], obj.RC.meth.IP.f)
            MSE_app[i, j] = comp_MSE(obj.x_in[i, j, :, :], obj.RC.x[i][j])            
            SSIM_in[i, j] = fom.ssim(obj.x_in[i, j, :, :], obj.RC.meth.IP.f)
            SSIM_app[i, j] = fom.ssim(obj.x_in[i, j, :, :], obj.RC.x[i][j])
    return  MSE_app, SSIM_in, SSIM_app, MSE_in


class recon_2param_class:
    def __init__(self, method_class, lam_fine):
        self.meth = method_class
        self.nLi = len(lam_fine)
        self.LF = []
        self.nL = []
        for i in range(self.nLi):
            self.LF += [lam_fine[i]]
            self.nL += [len(lam_fine[i])]
        self.pix = self.meth.IP.pix
        
    def load(self, load_path):
        self.x = []
        x_array = np.load(load_path + '_x_list.npy')
        self.MSE = np.load(load_path + '_MSE.npy')
        self.SSIM = np.load(load_path + '_SSIM.npy')
        for i in range(self.nL[0]):
            self.x += [[]]
            for j in range(self.nL[1]):
                self.x[i] += [self.meth.IP.reco_space.element(
                        x_array[i, j, :, :])]
        del x_array
        gc.collect()
        self.max_SSIM = np.argmax(self.SSIM)
        self.min_MSE = np.argmin(self.MSE)
    # %% Function to compute all the recosntructions on the fine grid
    def compute(self, niter):
        self.niter = niter
        self.x = []
        self.MSE = np.zeros([*self.nL])
        self.SSIM = np.zeros([*self.nL])
        # this is hardcoded for 2 different lambda, don't know how to automate
        # this for variable number of lambda
        for i in tqdm(range(self.nL[0])):
            self.x += [[]]
            for j in (range(self.nL[1])):
                self.x[i] += [self.meth.do(self.niter,
                              [self.LF[0][i], self.LF[1][j]])]
                self.MSE[i, j] = comp_MSE(self.x[i][j], self.meth.IP.f)
                self.SSIM[i, j] = fom.ssim(self.x[i][j], self.meth.IP.f)
        self.max_SSIM = np.argmax(self.SSIM)
        self.min_MSE = np.argmin(self.MSE)
    
    # %% Think about how to generalize this, atm we need equidistantly spaced
    # lam in log or lin space
    # ! ! ! TODO ! ! ! Make list for multiple instances
class interp_2param_class:
    def __init__(self, interp_points, RC):
        self.RC = RC
        self.pix = self.RC.pix
        self.LF = self.RC.LF
        self.nL = self.RC.nL
        self.nLi = self.RC.nLi
        self.nP = []
        self.points = []
        x0 = []
        x1 = []
        for j in range(self.nLi):
            self.nP += [interp_points[j]]
            self.points += [[i for i in range(0, self.nL[j],
                                        -(-self.nL[j] // (self.nP[j] - 1)))]]
            self.points[j] += [self.nL[j] - 1]
            x0 += [np.log10(self.LF[j][self.points[j]])]
            x1 += [np.log10(self.LF[j])]

        
        self.DP = np.zeros([*self.nP, self.pix ** 2])
        i = 0
        self.in_points = np.ones(self.nL, dtype=bool)
        for p0 in self.points[0]:
            j = 0
            for p1 in self.points[1]:
                self.DP[i, j, :] = np.ravel(RC.x[p0][p1])
                self.in_points[p0, p1] = False
                j += 1
            i += 1
        # Take the data points we want to use for the interpolation
        t = time.time()
        self.x_in = loop_x_interp(self.pix, self.nL, x0, x1, self.DP)
        print(time.time() - t)

        t = time.time()
 #       
        self.MSE_in, self.MSE_app, self.SSIM_in, self.SSIM_app = loop_comp_QM(
                self)
        print(time.time() -t)
        gc.collect()
        self.max_SSIM_in = np.argmax(self.SSIM_in)
        self.min_MSE_in = np.argmin(self.MSE_in)
 

# %% Show a reconstruction 
    def show_rec(self, rec):
        fig, (ax1, ax2, ax3) = pylab.subplots(1, 3,  figsize=[27, 9.2])   
        if type(rec) == str:
            if rec == 'MSE':
                n = self.RC.min_MSE
                fig.suptitle('#points = ' + str(self.nP) +
                             ', Minimum MSE reconstructions')
            elif rec == 'SSIM':
                n = self.RC.max_SSIM
                fig.suptitle('#points = ' + str(self.nP) + 
                             ', Maximum SSIM reconstructions')
        elif type(rec) == int:
            n = rec
            fig.suptitle('Reconstructions with lam=' + str(self.LF[n]))
        else:
            raise TypeError('Input not understood... options are:'+
                            'integer, str(MSE), str(SSIM)')
                 
        im1 = ax1.imshow(np.rot90(self.RC.x[n]))
        ax1.set_title(self.RC.meth.method + ' rec, lam=' + str(self.LF[n]))
        fig.colorbar(im1, ax=(ax1), shrink=.7)
        
        im2 = ax2.imshow(np.rot90(self.x_in[n]))
        ax2.set_title(self.RC.meth.method + ' rec spline, lam=' +
                      str(self.LF[n]))
        fig.colorbar(im2, ax=(ax2), shrink=.7)
        
        im3 = ax3.imshow(np.rot90(self.RC.x[n]-self.x_in[n]))
        ax3.set_title('Diff, lam=' + str(self.LF[n]))
        fig.colorbar(im3, ax=(ax3), shrink=.7)
        

