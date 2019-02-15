#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 12:06:33 2019

@author: lagerwer
"""

import numpy as np
import scipy.interpolate as sp
import pylab
import gc
from odl.contrib import fom
from tqdm import tqdm
import tabulate
pylab.rcParams.update({'font.size': 16})
# %%
def comp_MSE(x_rec, x_ref):
    return  np.linalg.norm(x_rec - x_ref) ** 2 / (2 * np.linalg.norm(x_ref) ** 2)

def comp_AV_VAR(array):
    AV = np.mean(array)
    VAR = np.sum((array - AV) ** 2)/ (np.size(array) - 1)
    
    return AV, VAR

class recon_class:
    def __init__(self, method_class, lam_fine):
        self.meth = method_class
        self.LF = lam_fine
        self.nL = np.size(self.LF)
        self.pix = self.meth.IP.pix
        
            
    # %%
    def load(self, load_path):
        self.x = []
        x_array = np.load(load_path + '_x_list.npy')
        self.MSE = np.load(load_path + '_MSE.npy')
        self.SSIM = np.load(load_path + '_SSIM.npy')
        if np.size(self.LF) != np.shape(x_array)[0]:
            min_L = np.log10(self.LF[0])
            max_L = np.log10(self.LF[-1])
            if (max_L - min_L) * 100 != (np.size(self.LF) - 1):
                raise ValueError('lam fine has not the correct number of' + \
                                 'points')
            if min_L != -4:
                x_array = x_array[int(100 * (min_L + 4)):]
                self.SSIM = self.SSIM[int(100 * (min_L + 4)):]
                self.MSE = self.MSE[int(100 * (min_L + 4)):]
            if max_L != 2:
                x_array = x_array[:int((max_L - 2) * 100)]
                self.SSIM = self.SSIM[:int((max_L - 2) * 100)]
                self.MSE = self.MSE[:int((max_L - 2) * 100)]
        for i in tqdm(range(self.nL)):
            self.x += [self.meth.IP.reco_space.element(x_array[i, :, :])]
        del x_array
        gc.collect()
        self.max_SSIM = np.argmax(self.SSIM)
        self.min_MSE = np.argmin(self.MSE)
        
    # %% Function to compute all the recosntructions on the fine grid
    def compute(self, niter):
        self.niter = niter
        self.x = []
        self.MSE = np.zeros(self.nL)
        self.SSIM = np.zeros(self.nL)
        for i in tqdm(range(self.nL)):
            self.x += [self.meth.do(self.niter, self.LF[i])]
            self.MSE[i] = comp_MSE(self.x[i], self.meth.IP.f)
            self.SSIM[i] = fom.ssim(self.x[i], self.meth.IP.f)
        self.max_SSIM = np.argmax(self.SSIM)
        self.min_MSE = np.argmin(self.MSE)
    
    # %% Think about how to generalize this, atm we need equidistantly spaced
    # lam in log or lin space
class interp_class:
    def __init__(self, interp_points, RC, spl):
        self.RC = RC
        self.spl = spl
        self.pix = self.RC.pix
        self.LF = self.RC.LF
        self.nL = self.RC.nL
        self.nP = interp_points
        # You should be able to define a non-uniform x0

        # Equally space the first nP-1 points, start at 0
        self.points = [i for i in range(0, self.nL,
                                        -(-self.nL // (self.nP - 1)))]
        # Last point needs to be the endpoint
        self.points += [self.nL - 1]
        x0 = np.log10(self.LF[self.points])
        x1 = np.log10(self.LF)
        
        # Take the data points we want to use for the interpolation
        self.DP = np.concatenate([[np.ravel(self.RC.x[p]),] 
                                    for p in self.points])
        if self.spl == 'linear':
            self.SplineObj = sp.interp1d(x0, self.DP, kind='slinear', axis=0)
        elif self.spl == 'quadratic':
            self.SplineObj =  sp.interp1d(x0, self.DP, kind='quadratic', axis=0)
        elif self.spl == 'cubic':
            self.SplineObj =  sp.CubicSpline(x0, self.DP, bc_type='clamped')
        DIP = self.SplineObj(x1)
        self.x_in = []
        self.MSE_in = np.zeros(self.nL)
        self.MSE_app = np.zeros(self.nL)
        self.SSIM_in = np.zeros(self.nL)
        self.SSIM_app = np.zeros(self.nL)
        for i in range(self.nL):
            self.x_in += [self.RC.meth.IP.reco_space.element(np.reshape(
                                            DIP[i, :], (self.pix, self.pix)))]
            self.MSE_in[i] = comp_MSE(self.x_in[i], self.RC.meth.IP.f)
            self.MSE_app[i] = comp_MSE(self.x_in[i], self.RC.x[i])            
            self.SSIM_in[i] = fom.ssim(self.x_in[i], self.RC.meth.IP.f)
            self.SSIM_app[i] = fom.ssim(self.x_in[i], self.RC.x[i])
            
        self.in_points = np.nonzero(self.MSE_app)
        self.max_SSIM_in = np.argmax(self.SSIM_in)
        self.min_MSE_in = np.argmin(self.MSE_in)
        self.interesting_stuff()
        # %%
    def interesting_stuff(self):
        self.IS = {'MSE_app' :{}, 'SSIM_app': {}}
        self.IS['MSE_app']['AV'], self.IS['MSE_app']['VAR'] = comp_AV_VAR(
                self.MSE_app[self.in_points])
        self.IS['MSE_app']['max'] = [np.max(self.MSE_app),
               'lam = ' + str(self.LF[np.argmax(self.MSE_app)])]
        self.IS['SSIM_app']['AV'], self.IS['SSIM_app']['VAR'] = comp_AV_VAR(
                self.SSIM_app[self.in_points])
        self.IS['SSIM_app']['min'] = [np.min(self.SSIM_app), 
               'lam = ' + str(self.LF[np.argmin(self.SSIM_app)])]

    def print_stat_approx(self):
        headers = ['#interp points = ' + str(self.nP), 'MSE(x_spl, x_rec)',
                   'SSIM(x_spl, x_rec)']
        AV = ['MEAN', self.IS['MSE_app']['AV'], 
              self.IS['SSIM_app']['AV']]
        VAR = ['VAR', self.IS['MSE_app']['VAR'], 
              self.IS['SSIM_app']['VAR']]
        extr = ['Worst case', self.IS['MSE_app']['max'][1], 
                self.IS['SSIM_app']['min'][1]]
        extr2 =['', self.IS['MSE_app']['max'][0], 
                self.IS['SSIM_app']['min'][0]]
        print(tabulate.tabulate([AV, VAR, extr, extr2], headers,
                                tablefmt='fancy_grid', floatfmt=".4f"))
        
    def print_optim_lambda(self):
        print('Number of interpolation points used: ' + str(self.nP))
        print('')
        headers = ['#interp points = ' + str(self.nP), 'MSE(x_i, x_GT)',
                   'SSIM(x_i, x_GT)']
        opt_rec = ['Optimal lambda, x_i=x_rec', 'lam=10^({:.4f})'.format(
                np.log10(self.LF[self.RC.min_MSE])),
                 'lam=10^({:.4f})'.format(np.log10(self.LF[self.RC.max_SSIM]))]
        opt_QM_rec = ['QM for optimal lambda', 
                      self.RC.MSE[self.RC.min_MSE],
                      self.RC.SSIM[self.RC.max_SSIM]]
        opt_spl = ['Optimal lambda, x_i=x_spl', 'lam=10^({:.4f})'.format(
                np.log10(self.LF[self.min_MSE_in])),
                 'lam=10^({:.4f})'.format(np.log10(self.LF[self.max_SSIM_in]))]
        opt_QM_spl = ['QM for optimal interpolated lambda', 
                      self.RC.MSE[self.min_MSE_in],
                      self.RC.SSIM[self.max_SSIM_in]]

        print(tabulate.tabulate([opt_rec, opt_QM_rec, opt_spl, opt_QM_spl],
                            headers, tablefmt='fancy_grid', floatfmt=".4f"))
# %% Function to compare the approximation to the        
    def show_pixelwise_interp(self, **kwargs):
        if 'npicks' in kwargs:
            npicks = kwargs['npicks']
            coords = np.arange(np.size(self.RC.x[0]))
            picks = list(np.random.choice(coords, size=npicks, replace=False))
        elif 'pixels' in kwargs:
            if type(kwargs['pixels']) is not list:
                raise TypeError('The input for pixels should be a list')
            picks = kwargs['pixels']
        else:
            print('You did not specify what you wanted, showing 10 random'+
                  'pixels')
            coords = np.arange(np.size(self.RC.x[0]))
            picks = list(np.random.choice(coords, size=10, replace=False))
        x_r = np.zeros(self.nL)
        x_i = np.zeros(self.nL)
        for p in picks:
            for i in range(self.nL):
                x_r[i] = np.ravel(self.RC.x[i])[p]
                x_i[i] = np.ravel(self.x_in[i])[p]
            pylab.figure()
            pylab.plot(self.LF, x_r, lw=3., label='pixel value')
            pylab.plot(self.LF, x_i, ls=':', lw=2, label='interp value')
            pylab.legend()
            pylab.xscale('log')
            pylab.title('#points = ' + str(self.nP) + ', Pixel ' + str(p))
        return picks
            
# %% Show a reconstruction 
    def show_rec(self, rec, **kwargs):
        fig, (ax1, ax2, ax3) = pylab.subplots(1, 3,  figsize=[27, 9.2])   
        if type(rec) == str:
            if rec == 'MSE':
                n = self.RC.min_MSE
                fig.suptitle('$N_{ip}$ = ' + str(self.nP) +
                             ', Minimum MSE reconstructions')
            elif rec == 'SSIM':
                n = self.RC.max_SSIM
                fig.suptitle('$N_{ip}$ = ' + str(self.nP) + 
                             ', Maximum SSIM reconstructions')
        elif type(rec) == int:
            n = rec
            if 'title' in kwargs:
                fig.suptitle(kwargs['title'])
            else:
                fig.suptitle('Reconstructions with lam=' + str(self.LF[n]))
        else:
            raise TypeError('Input not understood... options are:'+
                            'integer, str(MSE), str(SSIM)')
                 
        im1 = ax1.imshow(np.rot90(self.RC.x[n]))
        ax1.set_title(self.RC.meth.method + ' rec, lam=10^' + str(self.LF[n]))
        fig.colorbar(im1, ax=(ax1), shrink=.7)
        
        im2 = ax2.imshow(np.rot90(self.x_in[n]))
        ax2.set_title(self.RC.meth.method + ' rec spline, lam=' +
                      str(self.LF[n]))
        fig.colorbar(im2, ax=(ax2), shrink=.7)
        
        im3 = ax3.imshow(np.rot90(self.RC.x[n]-self.x_in[n]))
        ax3.set_title('Diff, lam=' + str(self.LF[n]))
        fig.colorbar(im3, ax=(ax3), shrink=.7)
        if 'save_path' in kwargs:
            pylab.savefig(kwargs['save_path'] + '.eps')

# %%
    def show_approx_curve(self, QM):
        pylab.figure()
        if QM == 'MSE':
            QM_array = self.MSE_app
            pylab.title('#points = ' + str(self.nP) + 
                        ', MSE(x_(interp,lam) x_lam)')
        else:
            QM_array = self.SSIM_app
            pylab.title('#points = ' + str(self.nP) + 
                        ', SSIM(x_(interp,lam), x_lam)')            
        pylab.plot(self.LF, QM_array, lw=3)
        pylab.xscale('log')
       
    def show_optim_curve(self, QM):
        pylab.figure()
        if QM == 'MSE':
            QM_array1 = self.RC.MSE
            lab1 = 'MSE'
            QM_array2 = self.MSE_in
            lab2 = 'MSE_interp'
            pylab.title('#points = ' + str(self.nP) + 
                        ', MSE(x_(interp,lam) x_GT)')
        else:
            QM_array1 = self.RC.SSIM
            lab1 = 'SSIM'
            QM_array2 = self.SSIM_in
            lab2 = 'SSIM_interp'
            pylab.title('#points = ' + str(self.nP) + 
                        ', SSIM(x_(interp,lam), x_GT)')            
        pylab.plot(self.LF, QM_array1, label=lab1, lw=3)
        pylab.plot(self.LF, QM_array2, label=lab2, lw=2, ls='-')
        pylab.xscale('log')

# %%
# %% Interp class no reference reconstructions
# %%
class recon_class_noref:
    def __init__(self, method_class, lam):
        self.meth = method_class
        self.lam = lam
        self.nL = np.size(self.lam)
        self.pix = self.meth.IP.pix
        
            
    # %% Function to compute all the recosntructions on the fine grid
    def compute(self, niter):
        self.niter = niter
        self.x = np.zeros((self.nL, self.pix, self.pix))

        for i in tqdm(range(self.nL)):
            self.x[i, :, :] = self.meth.do(self.niter, self.lam[i])

        
# %%        
class interp_class_noref:
    def __init__(self, pix, lam, rc_array, lam_fine, spl):
        self.pix = pix
        self.lam = lam
        self.lam_fine = lam_fine
        self.nL = len(lam_fine)
        self.rc_array = rc_array
        self.spl = spl
        x0 = np.log10(lam)
        x1 = np.log10(lam_fine)

        
        # Take the data points we want to use for the interpolation
        self.DP = np.reshape(self.rc_array, (len(lam), self.pix ** 2))
        if self.spl == 'linear':
            self.SplineObj = sp.interp1d(x0, self.DP, kind='slinear', axis=0)
        elif self.spl == 'quadratic':
            self.SplineObj =  sp.interp1d(x0, self.DP, kind='quadratic', axis=0)
        elif self.spl == 'cubic':
            self.SplineObj =  sp.CubicSpline(x0, self.DP, bc_type='clamped')
        
        self.x_in = self.SplineObj(x1)
        self.x_in = np.reshape(self.x_in, (self.nL, self.pix, self.pix))

