#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 12:03:57 2019

@author: lagerwer
"""

import odl 
import numpy as np
import scipy.interpolate as sp
from odl.contrib import fom

# %%
class method_class:
    def __init__(self, inv_prob, method, tau=0.1):
        # Save the input variables
        self.IP = inv_prob
        self.method = method
        if self.method == 'TV':
            # %%
            # Initialize gradient operator
            self.Grad = odl.Gradient(self.IP.reco_space)
            
            # Column vector of two operators
            self.op = odl.BroadcastOperator(self.IP.FP, self.Grad)
            
            # Do not use the g functional, set it to zero.
            self.G_func = odl.solvers.ZeroFunctional(self.op.domain)
            
            # Create functionals for the dual variable
            
            # l2-squared data matching
            self.DF_norm = odl.solvers.L2NormSquared(
                    self.IP.FP.range).translated(self.IP.g)
            
            # Estimated operator norm, add 10 percent to ensure 
            # ||K||_2^2 * sigma * tau < 1
            self.FP_norm = odl.power_method_opnorm(self.IP.FP)
            self.Grad_norm = odl.power_method_opnorm(self.Grad)
            # You can approximate this one by adding FP_norm and Grad_norm
            self.op_norm = 1.1 * odl.power_method_opnorm(self.op)
            
            # Step size for the primal variable
            self.tau = tau / self.op_norm 
            # Step size for the dual variable
            self.sigma = .99 / (self.tau * self.op_norm ** 2 ) 
            
            # Optionally pass callback to the solver to display intermediate 
            # results
            lam_scale = self.FP_norm / self.Grad_norm
            # Isotropic TV-regularization i.e. the l1-norm
            self.reg_norm = lam_scale * odl.solvers.L1Norm(self.Grad.range)
            
        elif self.method == 'Sob':
        # %%
            # Initialize gradient operator
            self.Grad = odl.Gradient(self.IP.reco_space)
            
            # Column vector of two operators
            self.op = odl.BroadcastOperator(self.IP.FP, self.Grad)
            
            # Do not use the g functional, set it to zero.
            self.G_func = odl.solvers.ZeroFunctional(self.op.domain)
            
            # Create functionals for the dual variable
            
            # l2-squared data matching
            self.DF_norm = odl.solvers.L2NormSquared(
                    self.IP.FP.range).translated(self.IP.g)
            
            # Estimated operator norm, add 10 percent to ensure 
            # ||K||_2^2 * sigma * tau < 1
            self.FP_norm = odl.power_method_opnorm(self.IP.FP)
            self.Grad_norm = odl.power_method_opnorm(self.Grad)
            # You can approximate this one by adding FP_norm and Grad_norm
            self.op_norm = 1.1 * odl.power_method_opnorm(self.op)
            
            # Step size for the primal variable
            self.tau = tau / self.op_norm 
            # Step size for the dual variable
            self.sigma = .99 / (self.tau * self.op_norm ** 2 ) 
            
            # Optionally pass callback to the solver to display intermediate 
            # results
            lam_scale = self.FP_norm / self.Grad_norm
            # Isotropic TV-regularization i.e. the l1-norm
            self.reg_norm = lam_scale * odl.solvers.L2NormSquared(
                                self.Grad.range)
# %%            
        elif self.method == 'TGV':
#            print('under construction')
            self.Grad = odl.Gradient(self.IP.reco_space, method='forward',
                             pad_mode='symmetric')
            V = self.Grad.range
            
            self.Dx = odl.PartialDerivative(self.IP.reco_space, 0,
                                       method='backward', pad_mode='symmetric')
            self.Dy = odl.PartialDerivative(self.IP.reco_space, 1,
                                       method='backward', pad_mode='symmetric')
            
            # Create symmetrized operator and weighted space.
            # TODO: As the weighted space is currently not supported in ODL we find a
            # workaround.
            # W = odl.ProductSpace(U, 3, weighting=[1, 1, 2])
            # sym_gradient = odl.operator.ProductSpaceOperator(
            #    [[Dx, 0], [0, Dy], [0.5*Dy, 0.5*Dx]], range=W)
            self.E = odl.operator.ProductSpaceOperator(
                [[self.Dx, 0], [0, self.Dy], [0.5 * self.Dy, 0.5 * self.Dx],
                 [0.5 * self.Dy, 0.5 * self.Dx]])
            W = self.E.range
            
            # Create the domain of the problem, given by the reconstruction space and the
            # range of the gradient on the reconstruction space.
            domain = odl.ProductSpace(self.IP.reco_space, V)
            
            # Column vector of three operators defined as:
            # 1. Computes ``Ax``
            # 2. Computes ``Gx - y``
            # 3. Computes ``Ey``
            self.op = odl.BroadcastOperator(
                self.IP.FP * odl.ComponentProjection(domain, 0),
                odl.ReductionOperator(self.Grad, odl.ScalingOperator(V, -1)),
                self.E * odl.ComponentProjection(domain, 1))
            
            # Do not use the f functional, set it to zero.
            self.G_func = odl.solvers.ZeroFunctional(domain)
            
            # l2-squared data matching
            self.DF_norm = odl.solvers.L2NormSquared(
                    self.IP.FP.range).translated(self.IP.g)
            
            # Estimated operator norm, add 10 percent to ensure 
            # ||K||_2^2 * sigma * tau < 1
            self.FP_norm = odl.power_method_opnorm(self.IP.FP)
            self.Grad_norm = odl.power_method_opnorm(self.Grad)
            self.E_norm = odl.power_method_opnorm(self.E)
            # You can approximate this one by adding FP_norm and Grad_norm
            self.op_norm = 1.1 * odl.power_method_opnorm(self.op)
            # Estimated operator norm, add 10 percent to ensure 
            # ||K||_2^2 * sigma * tau < 1
            # Step size for the primal variable
            self.tau = tau / self.op_norm 
            # Step size for the dual variable
            self.sigma = .99 / (self.tau * self.op_norm ** 2 ) 
            
            # parameters
            C1 = self.FP_norm / self.Grad_norm
            C2 = self.Grad_norm / self.E_norm
            # The l1-norms scaled by regularization paramters
            self.reg_norm1 = C1 * odl.solvers.L1Norm(V)
            self.reg_norm2 = C2 * odl.solvers.L1Norm(W)
            
        else:
            raise ValueError('Method not implemented: ({})'
                             ''.format(self.method))
# %%
    def plug_in_lam(self, lam):
        if self.method == 'TV':
            # Isotropic TV-regularization i.e. the l1-norm
            print
            l1_norm_lam = lam * self.reg_norm
            # Combine functionals, order must correspond to the operator K
            self.F_func = odl.solvers.SeparableSum(self.DF_norm,
                                                   l1_norm_lam)
        elif self.method == 'Sob':
            # Isotropic TV-regularization i.e. the l1-norm
            l2_norm_lam = lam * self.reg_norm
            # Combine functionals, order must correspond to the operator K
            self.F_func = odl.solvers.SeparableSum(self.DF_norm,
                                                   l2_norm_lam)
        elif self.method == 'TGV':
            if type(lam) is not list:
                raise ValueError('Wrong format for lambda. Should be: lam =' + 
                                 ' [lam1, lam2]')
            # Isotropic TV-regularization i.e. the l1-norm
            l1_norm1_lam = lam[0] * self.reg_norm1
            l1_norm2_lam = lam[1] * lam[0] * self.reg_norm2
            
            # Combine functionals, order must correspond to the operator K
            # --- Select solver parameters and solve using PDHG --- #
            self.F_func = odl.solvers.SeparableSum(self.DF_norm,
                                                   l1_norm1_lam, l1_norm2_lam)
        else:
            raise ValueError('Method not implemented: ({})'
                             ''.format(self.method))
                
    def do(self, niter, lam, **kwargs):
        self.plug_in_lam(lam)
        self.x = self.op.domain.zero()
        if 'callback' in kwargs:    
            odl.solvers.pdhg(self.x, self.G_func, self.F_func, self.op,
                             niter=niter, tau=self.tau, sigma=self.sigma,
                             callback=kwargs['callback'])
        else:
            odl.solvers.pdhg(self.x, self.G_func, self.F_func, self.op,
                             niter=niter, tau=self.tau, sigma=self.sigma)
        if len(self.x) == 2:
            return self.x[0].copy()
        else:
            return self.x.copy()
    
    # Needs x_relax and y for initialization to really be the same as 
    # starting where you left off
    def do_warm_start(self, x, niter, lam=False, tau=1, **kwargs):
        if lam is not False:
            self.plug_in_lam(lam)
        x_start = x.copy()
        if 'callback' in kwargs:
            odl.solvers.pdhg(x_start, self.G_func, self.F_func, self.op,
                             niter=niter, tau=self.tau * tau, sigma=self.sigma / tau,
                             callback=kwargs['callback'])
        else:
            odl.solvers.pdhg(x_start, self.G_func, self.F_func, self.op,
                             niter=niter, tau=self.tau * tau, sigma=self.sigma / tau)
        if len(x_start) == 2:
            return x_start[0].copy()
        else:
            return x_start.copy()