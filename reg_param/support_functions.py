#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 12:18:37 2019

@author: lagerwer
"""

import numpy as np
import pylab
import tabulate
import pickle
import gc
from odl.contrib import fom
import scipy.interpolate as sp
pylab.rc('text', usetex=True)
pylab.rcParams.update({'font.size': 30})
# %%
def show_rec(IC, rec, **kwargs):
    clim = [np.min(IC.RC.meth.IP.f), np.max(IC.RC.meth.IP.f)]
    fig, (ax1, ax2, ax3) = pylab.subplots(1, 3,  figsize=[27, 9.2])   
    if type(rec) == str:
        if rec == 'MSE':
            n = IC.RC.min_MSE
            fig.suptitle('$N_{ip}$ = ' + str(IC.nP) +
                         ', Minimum MSE reconstruction and approximation')
        elif rec == 'SSIM':
            n = IC.RC.max_SSIM
            fig.suptitle('$N_{ip}$ = ' + str(IC.nP) + 
                         ', Maximum SSIM reconstruction and approximation')
    elif type(rec) == int:
        n = rec
        if 'title' in kwargs:
            fig.suptitle(kwargs['title'])
        else:
            fig.suptitle(r'Reconstruction and approximation for ' + 
                         '$\lambda=10^{}'.format('{' +
                                      str(np.log10(IC.LF[n])) + '}$'))
    else:
        raise TypeError('Input not understood... options are:'+
                        'integer, str(MSE), str(SSIM)')
             
    im1 = ax1.imshow(np.rot90(IC.RC.x[n]), clim=clim)
    ax1.set_title(r'${\mathbf{x}}$' + '$^\lambda_{}$'.format(
                                                '{' + IC.RC.meth.method + '}'))
#    ax1.set_clip_box([0, .18])
    fig.colorbar(im1, ax=(ax1), shrink=.7)
    
    
    im2 = ax2.imshow(np.rot90(IC.x_in[n]), clim=clim)
#    ax2.set_clip_box([0, .18])
    ax2.set_title(r'$\tilde{\mathbf{x}}$' + '$^\lambda_{}$'.format(
                                                '{' + IC.RC.meth.method + '}'))
    fig.colorbar(im2, ax=(ax2), shrink=.7)
    
    im3 = ax3.imshow(np.rot90(IC.RC.x[n]-IC.x_in[n]))
    ax3.set_title(r'$\mathbf{x}$' + '$^\lambda_{}$'.format(
                          '{' + IC.RC.meth.method + '}') + \
                          r'- $\tilde{\mathbf{x}}$' + '$^\lambda_{}$'.format(
                                  '{' +IC.RC.meth.method + '}'))
    fig.colorbar(im3, ax=(ax3), shrink=.7)
    if 'save_path' in kwargs:
        pylab.savefig(kwargs['save_path'] + '.eps', bbox='tight')


def compare_QM_curves(lam, rec_obj, I_list, QM, **kwargs):
    pylab.figure(figsize=[12, 9.2])
    if QM == 'MSE':
        rec_QM = rec_obj.MSE
        pylab.title(r'MSE($\tilde{\mathbf{x}}^\lambda$, $\mathbf{x}_{GT}$)')
    elif QM == 'SSIM':
        rec_QM = rec_obj.SSIM
        pylab.title(r'SSIM($\tilde{\mathbf{x}}^\lambda$, $\mathbf{x}_{GT}$)')
    pylab.plot(lam, rec_QM, lw=3, label=r'${\mathbf{x}}^\lambda$')
    for io in I_list:
        if QM == 'MSE':
            pylab.plot(lam, io.MSE_in, lw=2, ls='--',
                   label=r'$\tilde{\mathbf{x}}^\lambda$, $N_{{ip}}$ = ' + str(io.nP))
        elif QM == 'SSIM':
            pylab.plot(lam, io.SSIM_in, lw=2, ls='--',
                   label=r'$\tilde{\mathbf{x}}^\lambda$, $N_{{ip}}$ = ' + str(io.nP))
    pylab.xscale('log')
    pylab.xlabel(r'$\lambda$')
    pylab.legend()
    if 'save_path' in kwargs:
        pylab.savefig(kwargs['save_path'] + QM + '_curves.eps')

    
    
# %%
def compare_QM_approx_curves(lam, I_list, **kwargs):
    fig, (ax1, ax2) = pylab.subplots(1, 2,  figsize=[18, 9.2])

    ax1.set_title(r'MSE($\tilde{\mathbf{x}}^\lambda$, $\mathbf{x}^\lambda$)')
    ax2.set_title(r'SSIM($\tilde{\mathbf{x}}^\lambda$, $\mathbf{x}^\lambda$)')
    for io in I_list:

        ax1.plot(lam, io.MSE_app, lw=2, ls='-',
                   label=r'$N_{{ip}}$ = ' + str(io.nP))
        ax2.plot(lam, io.SSIM_app, lw=2, ls='-',
                   label=r'$N_{{ip}}$ = ' + str(io.nP))
    ax1.set_xscale('log')
    ax1.set_xlabel(r'$\lambda$')
    ax1.legend()
    ax2.set_xscale('log')
    ax2.set_xlabel(r'$\lambda$')
    ax2.legend()
    if 'save_path' in kwargs:
        pylab.savefig(kwargs['save_path'] + 'app_curves.eps')


# %%
def compare_approx_pixelwise(lam, pix_list, recs, I_list, **kwargs):
    nL = len(lam)
    x_r = np.zeros(nL)
    x_i = np.zeros((nL, len(I_list)))
    for p in pix_list:
        pylab.figure(figsize=[12, 9.2])
        x_GT = np.ones(nL) * np.ravel(recs.meth.IP.f)[p]
        pylab.title('Pixel, p = '+ str(p))
        for i in range(nL):
            x_r[i] = np.ravel(recs.x[i])[p]
            for j in range(len(I_list)):
                x_i[i, j] = np.ravel(I_list[j].x_in[i])[p]
        pylab.plot(lam, x_GT, lw=1, ls=':', label=r'$(\mathbf{x}_{GT})_p$',
                   color='C7')
        pylab.plot(lam, x_r, lw=3., label=r'$(\mathbf{x}$' + 
                        '$^\lambda_{})_p$'.format('{' + recs.meth.method + '}'))
        for j in range(len(I_list)):
            pylab.plot(lam, x_i[:, j], ls='--', lw=2, label=r'($\tilde{\mathbf{x}}$' 
                       + '$^\lambda_{})_p,$'.format('{' + recs.meth.method + '}') +
                       ' $N_{ip}$ = ' + str(I_list[j].nP))
        pylab.legend()
        pylab.xscale('log')
        if 'save_path' in kwargs:
            pylab.savefig(kwargs['save_path'] + 'pix' + str(p) + '.eps')


# %%
def stats_QM(I_list, **kwargs):
    fig, (ax1, ax2) = pylab.subplots(1, 2,  figsize=[18, 9.2])
    p_list = np.zeros(len(I_list))
    AV_MSE_list = np.zeros(len(I_list))
    SV_MSE_list = np.zeros(len(I_list))
    AV_SSIM_list = np.zeros(len(I_list))
    SV_SSIM_list = np.zeros(len(I_list))
    i = 0
    for io in I_list:
        p_list[i] = io.nP
        AV_MSE_list[i] = io.IS['MSE_app']['AV']
        SV_MSE_list[i] = io.IS['MSE_app']['VAR'] ** (1/2)
        AV_SSIM_list[i] = io.IS['SSIM_app']['AV']
        SV_SSIM_list[i] = io.IS['SSIM_app']['VAR'] ** (1/2)
        i += 1
    ax1.errorbar(p_list, AV_MSE_list, SV_MSE_list, lw=3, capsize=5)
    ax1.set_title(r'${E}$[MSE($\tilde{\mathbf{x}}^\lambda$, $\mathbf{x}^\lambda$)]')
    ax1.set_xlabel(' number of points used for interpolation')
    ax2.errorbar(p_list, AV_SSIM_list, SV_SSIM_list, lw= 3, capsize=5)
    ax2.set_title(r'${E}$[SSIM($\tilde{\mathbf{x}}^\lambda$, $\mathbf{x}^\lambda$)]')
    ax2.set_xlabel('number of points used for interpolation')
    if 'save_path' in kwargs:
        fig.savefig(kwargs['save_path'] + 'QM_stats.eps')

# %%
def curvature_splines(x, y=None, error=0.1):
    """Calculate the signed curvature of a 2D curve at each point
    using interpolating splines.
    Parameters
    ----------
    x,y: numpy.array(dtype=float) shape (n_points, )
         or
         y=None and
         x is a numpy.array(dtype=complex) shape (n_points, )
         In the second case the curve is represented as a np.array
         of complex numbers.
    error : float
        The admisible error when interpolating the splines
    Returns
    -------
    curvature: numpy.array shape (n_points, )
    Note: This is 2-3x slower (1.8 ms for 2000 points) than `curvature_gradient`
    but more accurate, especially at the borders.
    """

    # handle list of complex case
    if y is None:
        x, y = x.real, x.imag

    t = np.arange(x.shape[0])
    std = error * np.ones_like(x)

    fx = sp.UnivariateSpline(t, x, k=4, w=1 / np.sqrt(std))
    fy = sp.UnivariateSpline(t, y, k=4, w=1 / np.sqrt(std))

    xˈ = fx.derivative(1)(t)
    xˈˈ = fx.derivative(2)(t)
    yˈ = fy.derivative(1)(t)
    yˈˈ = fy.derivative(2)(t)
    curvature = (xˈ* yˈˈ - yˈ* xˈˈ) / np.power(xˈ** 2 + yˈ** 2, 3 / 2)
    return curvature

def show_DP_LC(lam, recs, I_list, f, **kwargs):
    DF = np.zeros((1 + len(I_list), recs.nL))
    Reg = np.zeros((1 + len(I_list), recs.nL))
    DF_func = recs.meth.DF_norm * recs.meth.IP.FP
    reg_func = recs.meth.reg_norm * recs.meth.Grad
    eps = DF_func(f)
    for i in range(recs.nL):
        DF[0, i] = DF_func(recs.x[i])
        Reg[0, i] = reg_func(recs.x[i])
        for j in range(len(I_list)):
            DF[j + 1, i] = DF_func(I_list[j].x_in[i])
            Reg[j + 1, i] = reg_func(I_list[j].x_in[i])
    colors = ['C1', 'C2', 'C3', 'C4']
    fig, axes = pylab.subplots(2, 2,  figsize=[30, 16])
    axes[0, 0].set_ylabel(r'$D(W\mathbf{x}^\lambda_{TV}, \mathbf{y})$')
    axes[0, 0].set_xscale('log')
    axes[0, 0].set_xlabel(r'$\lambda$')
    axes[0, 0].plot(lam, DF[0, :], lw=3)
    axes[0, 0].plot(lam, eps*np.ones(len(lam)), lw=1, label=r'$\epsilon$',
        color='C7')
    for j in range(len(I_list)):
        axes[0, 0].plot(lam, DF[j + 1, :], lw=2, ls='--', color=colors[j])
#        axes[0, 0].plot(lam, DF[j + 5, :], lw=2, ls=':', color=colors[j])
    axes[0, 0].legend()
    
    axes[0,1].set_ylabel(r'$TV(\mathbf{x}^\lambda_{TV})$')
    axes[0,1].set_xscale('log')
    axes[0,1].set_xlabel('$\lambda$')
    axes[0,1].plot(lam, Reg[0, :], lw=3)
    for j in range(len(I_list)):
        axes[0,1].plot(lam, Reg[j + 1, :], lw=2, ls='--', color=colors[j])
#        axes[0,1].plot(lam, Reg[j + 5, :], lw=2, ls=':', color=colors[j])
    
    LDF = np.log(DF)
    LReg = np.log(Reg)
    
    axes[1,0].set_ylabel(r'log$(R(\mathbf{x}^\lambda_{TV}))$')
    axes[1,0].set_xlabel(r'log$(D(W\mathbf{x}^\lambda_{TV}, \mathbf{y}))$')
    axes[1,0].plot(LDF[0, :], LReg[0, :], lw=3)
    for j in range(len(I_list)):
        axes[1,0].plot(LDF[j + 1], LReg[j + 1, :], lw=2, ls='--',
            color=colors[j])
#        axes[1,0].plot(LDF[j + 5], LReg[j + 5, :], lw=2, ls=':',
#            color=colors[j])

    
    axes[1,1].set_ylabel('Curvature, $\kappa(\lambda)$')
    axes[1,1].set_xscale('log')
    axes[1,1].set_xlabel('$\lambda$')
    curv = curvature_splines(LDF[0, :], LReg[0, :])
    axes[1,1].plot(lam, curv, lw=3, label=r'Reconstructions')
    for j in range(len(I_list)):
        curv = curvature_splines(LDF[j + 1, :], LReg[j + 1, :])
        axes[1,1].plot(lam, curv, lw=2, ls='--', color=colors[j],
            label=r'Pixel-wise interpolation, $N_{ip}$=' + str(I_list[j].nP))
#        curv = curvature_splines(LDF[j + 5, :], LReg[j + 5, :])
#        axes[1,1].plot(lam, curv, lw=2, ls=':', color=colors[j],
#            label=r'Direct interpolation, $N_{ip}$=' + str(I_list[j].nP))
#    axes[1,1].set_ylim([-25, 500])
    
    handles, labels = axes[1,1].get_legend_handles_labels()
    fig.subplots_adjust(left=0.05, right=0.97)
    fig.legend(handles, labels)
    if 'save_path' in kwargs:
        fig.savefig(kwargs['save_path'] + 'L_curve.eps')
    
    
# %%
def loop_DP(DF, rec_list, delta):
    i = 0
    delta_t = delta + 1
    while delta_t > delta:
        i -= 1
        delta_t = DF(rec_list[i])
    return delta_t, i
        
def discr_princ(lam, recs, I_list, **kwargs):
    DF_sqr = recs.meth.DF_norm * recs.meth.IP.FP
    delta = DF_sqr(recs.meth.IP.f)
    DP_recs, DPit_recs = loop_DP(DF_sqr, recs.x, delta)
    DPit_spl, DP_spl = np.zeros(len(I_list), dtype=int), np.zeros(len(I_list))
    headers = ['', 'Recs']
    lam_DP = ['lambda=', lam[DPit_recs]]
    DP = ['D(Wx_DP, y)', DP_recs]
    
    for i in range(len(I_list)):
        DP_spl[i], DPit_spl[i] = loop_DP(DF_sqr, I_list[i].x_in, delta)
        headers += ['N_ip=' + str(I_list[i].nP)]
        lam_DP += [lam[DPit_spl[i]]]
        DP += [DP_spl[i]]
        
    print(tabulate.tabulate([DP, lam_DP], headers,
                                tablefmt='fancy_grid', floatfmt=".4f"))
   
# %%
def save_object(obj, filename):
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

def save_recs(pix, lam, recs, save_path):
    x_list = np.zeros((len(lam), pix, pix))
    for i in range(len(lam)):
        x_list[i, :, :] = np.asarray(recs.x[i])
    np.save(save_path + '_x_list', x_list)
    np.save(save_path + '_MSE', recs.MSE)
    np.save(save_path + '_SSIM', recs.SSIM)
    del x_list
    gc.collect()
   
    
def save_recs_2D(pix, lam, recs, save_path):
    x_list = np.zeros((len(lam[0]), len(lam[1]), pix, pix))
    for i in range(len(lam[0])):
        for j in range(len(lam[1])):
            x_list[i, j, :, :] = np.asarray(recs.x[i][j])
    np.save(save_path + '_x_list', x_list)
    np.save(save_path + '_MSE', recs.MSE)
    np.save(save_path + '_SSIM', recs.SSIM)
    del x_list
    gc.collect()


# %%

def show_QM_2param(lam, recs, I_list):
    MSE_l = []
    SSIM_l = []
    tit_MSE = []
    tit_SSIM = []
    pylab.rcParams.update({'font.size': 20})
    for i in range(len(I_list)):
        MSE_l += [I_list[i].MSE_app]
        tit_MSE += [r'rMSE$(\tilde{\mathbf{x}}_{TGV}^{\vec{\lambda}}, ' + 
            r'\mathbf{x}_{TGV}^{\vec{\lambda}})$, $N_{ip,1}=' +  
            str(I_list[i].nP[0]) + '$, $N_{ip,2}=' + str(I_list[i].nP[1]) +'$']
        SSIM_l += [I_list[i].SSIM_app]
        tit_SSIM += [r'rSSIM$(\tilde{\mathbf{x}}_{TGV}^{\vec{\lambda}}, ' + 
            r'\mathbf{x}_{TGV}^{\vec{\lambda}})$, $N_{ip,1}=' +  
            str(I_list[i].nP[0]) + '$, $N_{ip,2}=' + str(I_list[i].nP[1]) +'$']
    MSE_l += [recs.MSE]
    SSIM_l += [recs.SSIM]
    tit_MSE += [r'rMSE$({\mathbf{x}}_{TGV}^{\vec{\lambda}}, \mathbf{x}_{GT})$']
    tit_SSIM += [r'rSSIM$({\mathbf{x}}_{TGV}^{\vec{\lambda}}, \mathbf{x}_{GT})$']

    fig, axes = pylab.subplots(len(I_list), 3, figsize=(40, 12))
    
    for i in range(len(I_list) + 1):
        im = axes[0, i].imshow(np.rot90(MSE_l[i]), extent=[-4, 2, -2, 2])
        axes[0, i].set_title(tit_MSE[i])
        axes[0, i].set_xlabel('log$_{10}(\lambda_1)$')
        axes[0, i].set_ylabel('log$_{10}(\lambda_2)$')
        fig.colorbar(im, ax=axes[0, i], shrink=.9)
        im = axes[1, i].imshow(np.rot90(SSIM_l[i]), extent=[-4, 2, -2, 2])    
        axes[1, i].set_title(tit_SSIM[i])
        fig.colorbar(im, ax=axes[1, i], shrink=.9)
        axes[1, i].set_xlabel('log$_{10}(\lambda_1)$')
        axes[1, i].set_ylabel('log$_{10}(\lambda_2)$')
            
        
def show_rec_2param(IC, recs):
    clim = [np.min(IC.RC.meth.IP.f), np.max(IC.RC.meth.IP.f)]
    S_WCR = np.unravel_index(np.argmin(IC.SSIM_app), np.shape(IC.SSIM_app))
    M_WCR = np.unravel_index(np.argmax(IC.MSE_app), np.shape(IC.MSE_app))
    
    x_r = recs.x[S_WCR[0]][S_WCR[1]]
    x_i = IC.x_in[S_WCR[0], S_WCR[1], :, :]
    
    pylab.rc('text', usetex=True)
    pylab.rcParams.update({'font.size': 20})
    clim=[0.0, 1]
    fig, axes = pylab.subplots(2, 3, figsize=[30, 16])
    
    im = axes[0,0].imshow(np.rot90(x_r), clim=clim)
    axes[0,0].set_title(r'Minimum MSE, ${\mathbf{x}}$' + '$^\lambda_{TV}$')
    fig.colorbar(im, ax=(axes[0,0]), shrink=.9)
    axes[0,0].set_xticks([], [])
    axes[0,0].set_xlabel('x')
    axes[0,0].set_yticks([], [])
    axes[0,0].set_ylabel('y')
    
    im = axes[0,1].imshow(np.rot90(x_i), clim=clim)
    axes[0,1].set_title(r'$\tilde{\mathbf{x}}$' + '$^\lambda_{TV}$, ' + \
       '$N_{ip,1}=' + str(IC.nP[0]) + '$, $N_{ip,2}=' + str(IC.nP[1]) +'$')
    fig.colorbar(im, ax=(axes[0,1]), shrink=.9)
    axes[0,1].set_xticks([], [])
    axes[0,1].set_xlabel('x')
    axes[0,1].set_yticks([], [])
    axes[0,1].set_ylabel('y')
    
    im = axes[0,2].imshow(np.rot90(x_r-x_i))
    axes[0,2].set_title(r'$\mathbf{x}^\lambda_{TV}- \tilde{\mathbf{x}}^\lambda_{TV}$')
    fig.colorbar(im, ax=(axes[0,2]), shrink=.9)
    axes[0,2].set_xticks([], [])
    axes[0,2].set_xlabel('x')
    axes[0,2].set_yticks([], [])
    axes[0,2].set_ylabel('y')
            
    x_r = recs.x[M_WCR[0]][M_WCR[1]]
    x_i = IC.x_in[M_WCR[0], M_WCR[1], :, :]
    im = axes[1,0].imshow(np.rot90(x_r), clim=clim)
    axes[1,0].set_title(r'Maximum SSIM, ${\mathbf{x}}$' + '$^\lambda_{TV}$')
    fig.colorbar(im, ax=(axes[1,0]), shrink=.9)
    axes[1,0].set_xticks([], [])
    axes[1,0].set_xlabel('x')
    axes[1,0].set_yticks([], [])
    axes[1,0].set_ylabel('y')
    
    im = axes[1,1].imshow(np.rot90(x_i), clim=clim)
    axes[1,1].set_title(r'$\tilde{\mathbf{x}}$' + '$^\lambda_{TV}$, ' + \
       '$N_{ip,1}=' + str(IC.nP[0]) + '$, $N_{ip,2}=' + str(IC.nP[1]) +'$')
    fig.colorbar(im, ax=(axes[1,1]), shrink=.9)
    axes[1,1].set_xticks([], [])
    axes[1,1].set_xlabel('x')
    axes[1,1].set_yticks([], [])
    axes[1,1].set_ylabel('y')
    
    im = axes[1,2].imshow(np.rot90(x_r-x_i))
    axes[1,2].set_title(r'$\mathbf{x}^\lambda_{TV}- \tilde{\mathbf{x}}^\lambda_{TV}$')
    fig.colorbar(im, ax=(axes[1,2]), shrink=.9)
    axes[1,2].set_xticks([], [])
    axes[1,2].set_xlabel('x')
    axes[1,2].set_yticks([], [])
    axes[1,2].set_ylabel('y')
    fig.subplots_adjust(left=0.05)
        
        
    
        
        
        
        
        
        
        
        
        
        
        
        
        
    
    