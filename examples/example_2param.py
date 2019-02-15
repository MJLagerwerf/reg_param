#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 10:00:30 2019

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
PH = 'TGV' 

# Number of pixels in the x and y direction
pix = 64

# Number of projection angles
angles = 90 # 64

# Type of noise and its level
noise = ['Gaussian', 0.1] # None

# Number of iterations for PDHG
niter = 500

# Reconstruction method
method = 'TGV' # 'Sob'

# Create the fine grid of regularization parameters
lam1 = np.logspace(-4, 2, 31)
lam2 = np.logspace(-2, 2, 16)
lam = [lam1, lam2]
# Number of interpolation points
Nip_list = [[5, 5], [8, 5]]



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
recs = rp.recon_2param_class(MC, lam)

if recompute:
    recs.compute(niter)
    rp.save_recs_2D(pix, lam, recs, save_path)
    
if os.path.exists(save_path + '_x_list.npy'):
    if not hasattr(recs, 'x'):
        recs.load(save_path)
else:
    recs.compute(niter)
    rp.save_recs_2D(pix, lam, recs, save_path)


# %% Create interpolations for varying number of interpolation points
IC = []
for Nip in Nip_list: 
    IC += [rp.interp_2param_class(Nip, recs)]

# %% Show some results
pylab.close('all')

# Show QM figures
rp.show_QM_2param(lam, recs, IC)

## Show the MSE and SSIM curve of the interpolations and the reconstructions 
rp.show_rec_2param(IC[1], recs)
