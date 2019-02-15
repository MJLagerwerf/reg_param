#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 12:24:17 2019

@author: lagerwer
"""

import numpy as np
import reg_param as rp
from tqdm import tqdm
from odl.contrib import fom
import os
import pylab

# %%
# Phantom type
PH = 'FORBILD' 

# Number of pixels in the x and y direction
pix = 256

# Number of projection angles
angles = 180 # 64

# Type of noise and its level
noise = ['Gaussian', 0.1] # None

# Number of iterations for PDHG
niter = 500

# Reconstruction method
method = 'TV' # 'Sob'

# Create the fine grid of regularization parameters
l0, lNip, Ns = -4, 2, 201 
lam = np.logspace(l0, lNip, Ns)

# Number of interpolation points
Nip_list = [6, 11, 16, 21]

# Type of interpolation
Tspl = 'cubic'


# %% Make a path for saving the reconstructions to speed up later runs
# If you change something above, you should recompute the recons
recompute = True
path = 'recons'
if not os.path.exists(path):
    os.makedirs(path)
save_path = path + '/' + method 
# %% Define the problem and create the method
IP = rp.problem_definition_class(PH, pix, angles, noise)
MC = rp.method_class(IP, method)

# %% Create the reconstructions 
recs = rp.recon_class(MC, lam)

if recompute:
    recs.compute(niter)
    rp.save_recs(pix, lam, recs, save_path)
    
if os.path.exists(save_path + '_x_list.npy'):
    if not hasattr(recs, 'x'):
        recs.load(save_path)
else:
    recs.compute(niter)
    rp.save_recs(pix, lam, recs, save_path)


# %% Create interpolations for varying number of interpolation points
IC = []
for Nip in Nip_list: 
    IC += [rp.interp_class(Nip, recs, Tspl)]

# %% Show some results
pylab.close('all')
# Show the MSE and SSIM curve of the interpolations and the reconstructions 
# w.r.t ground truth
rp.compare_QM_curves(lam, recs, IC, 'MSE')
rp.compare_QM_curves(lam, recs, IC, 'SSIM')

# Show the MSE and SSIM curve of the interpolations w.r.t. reconstructions
rp.compare_QM_approx_curves(lam, IC)

# Show some pixelwise approximations
pix_list = [pix ** 2 // 8 + pix // 2, pix ** 2 // 6 + pix // 2,
            pix ** 2 // 3 + pix // 3, pix ** 2 * 2 // 3 + pix // 2,
            pix * 3 // 4 * (pix + 3)]
rp.compare_approx_pixelwise(lam, pix_list, recs, IC)

# Show optimal MSE and SSIM reconstructions and approximations for Nip = 6
rp.show_rec(IC[0], 'MSE')
rp.show_rec(IC[0], 'SSIM')
# Show the approximations and reconstructions for lam[44] 
rp.show_rec(IC[0], 44)

# %%
# SHow L-curve and DP curves
rp.show_DP_LC(lam, recs, IC, IP.f)


