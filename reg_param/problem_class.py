#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 11:39:02 2019

@author: lagerwer
"""

import odl 
import numpy as np
import scipy.interpolate as sp
import gc
from odl.contrib import fom
# %%
class problem_definition_class:
    def __init__(self, PH, pix, angles, noise, **kwargs):
        # %% Save all the input variables in the class
        if 'experimental' in kwargs:
            self.g = np.load('/export/scratch2/lagerwer/reg_param/' + \
                             'experimental/g_noisy_3DC_2DC.npy')
            self.pix = 1024

            self.angles = np.shape(self.g)[0]
            self.dpix = np.shape(self.g)[1]
            pix_size = 0.00748
            self.detecsize = self.dpix * pix_size / 2
            self.volumesize = self.detecsize * (13.4) / (21.4 + 13.4)
            self.angle_partition = odl.uniform_partition(0, 2 * np.pi,
                                                         self.angles)

            self.det_partition = odl.uniform_partition(-self.detecsize,
                                                   self.detecsize, self.dpix)
            self.reco_space = odl.uniform_discr(
                    (-self.volumesize, -self.volumesize),
                    (self.volumesize, self.volumesize), (self.pix, self.pix))
            self.geometry = odl.tomo.FanFlatGeometry(self.angle_partition,
                                                        self.det_partition,
                                                        src_radius=13.4,
                                                        det_radius=21.4)
            self.FP = odl.tomo.RayTransform(self.reco_space, self.geometry)
            # %%
        else:
            self.PH = PH
            self.pix = pix
            pixels = [pix, pix]
            pixels_up = [2 * pix, 2 * pix]
            self.dpix = 2 * pix
            self.angles = angles
            self.noise = noise
            self.angle_partition = odl.uniform_partition(0, np.pi, angles)
            if 'load_data_g' in kwargs:
                self.reco_space, self.f = self.phantom_creation(pixels, **kwargs)
                self.create_data(pixels, self.reco_space, self.f, **kwargs)
            else:
                reco_space_up, f_up = self.phantom_creation(pixels_up, **kwargs)
                self.create_data(pixels_up, reco_space_up, f_up, **kwargs)
                reco_space_up, f_up = None, None
                gc.collect()
                self.reco_space, self.f = self.phantom_creation(pixels, **kwargs)
            self.det_partition = odl.uniform_partition(-self.detecsize,
                                                   self.detecsize, self.dpix)
            self.geometry = odl.tomo.Parallel2dGeometry(self.angle_partition,
                                                        self.det_partition)
            self.FP = odl.tomo.RayTransform(self.reco_space, self.geometry)
# %%
    def phantom_creation(self, pixels, **kwargs):
        if self.PH == 'Shepp-Logan':
            self.volumesize = np.array([4, 4])
            self.detecsize = 2 * self.volumesize[0]
            reco_space = odl.uniform_discr(min_pt=-self.volumesize,
                                           max_pt=self.volumesize,
                                           shape=pixels,
                                           dtype='float32')
            
            f = odl.phantom.shepp_logan(reco_space, True) / 0.2 * 0.182
            return reco_space, f
        
        elif self.PH == 'FORBILD':
            self.volumesize = np.array([15, 15])
            self.detecsize = 2 * self.volumesize[0]
            reco_space = odl.uniform_discr(min_pt=-self.volumesize,
                                           max_pt=self.volumesize,
                                           shape=pixels,
                                           dtype='float32')
            
            f = odl.phantom.forbild(reco_space) * 0.1
            return reco_space, f
        elif self.PH == 'TGV':
            self.volumesize = np.array([20, 20])
            self.detecsize = 2 * self.volumesize[0]
            reco_space = odl.uniform_discr(min_pt=-self.volumesize,
                                           max_pt=self.volumesize,
                                           shape=pixels,
                                           dtype='float32')
            
            f = odl.phantom.tgv_phantom(reco_space)
            return reco_space, f
        else:
            raise ValueError('Phantom not implemented: ({})'
                             ''.format(self.PH))
        

    def create_data(self, pixels_up, reco_space_up, f_up, **kwargs):
        dpix_up = 2 * pixels_up[0]

        # Make a flat detector space
        det_partition = odl.uniform_partition(-self.detecsize,
                                               self.detecsize, dpix_up)
        # Create data_space_up and data_space
        data_space = odl.uniform_discr((0, -self.detecsize),
                                       (2 * np.pi, self.detecsize),
                                       [self.angles, self.dpix],
                                       dtype='float32')
        data_space_up = odl.uniform_discr((0, -self.detecsize),
                                       (2 * np.pi, self.detecsize), 
                                       [self.angles, dpix_up],
                                       dtype='float32')
        # Create geometry
        geometry = odl.tomo.Parallel2dGeometry(self.angle_partition,
                                               det_partition)

        FP = odl.tomo.RayTransform(reco_space_up, geometry)
        
        resamp = odl.Resampling(data_space_up, data_space)

        if 'load_data_g' in kwargs:
            if type(kwargs['load_data_g']) == str: 
                self.g = data_space.element(np.load(kwargs['load_data_g']))
            else:
                self.g = data_space.element(kwargs['load_data_g'])
        else:
            self.g = resamp(FP(f_up))
            if self.noise == None:
                pass
            elif self.noise[0] == 'Gaussian':
                self.g += data_space.element(
                        odl.phantom.white_noise(resamp.range) * \
                        np.mean(self.g) * self.noise[1])
            elif self.noise[0] == 'Poisson':
                # 2**8 seems to be the minimal accepted I_0
                self.g = data_space.element(
                        self.add_poisson_noise(self.noise[1]))
            else:
                raise ValueError('unknown `noise type` ({})'
                                 ''.format(self.noise[0]))
        
        
    def add_poisson_noise(self, I_0, seed=None):
        seed_old = np.random.get_state()
        np.random.seed(seed)
        data = np.asarray(self.g.copy())
        Iclean = (I_0 * np.exp(-data))
        data = None
        Inoise = np.random.poisson(Iclean)
        Iclean = None
        np.random.set_state(seed_old)
        return (-np.log(Inoise / I_0))