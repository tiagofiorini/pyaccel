#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 08:32:47 2023
@author: Tiago Silva - IFUSP
"""

# version
version = 0.1

# Importing essential modules
import numpy as np
import os as os
import matplotlib as mpl
import matplotlib.pyplot as plt

# Importing essential functions


# Defining physical constants
NAvogadro            = 6.02214086E23
Proton_mass_kg       = 1.672621777E-27  # kg
Proton_mass_MeV      = 938.2720813      # MeV/c2
Electron_mass_kg     = 9.10938291E-31   # kg
Electron_mass_MeV    = 0.510998910      # MeV/c2
Atomic_mass_unit_kg  = 1.660538921E-27  # kg
Atomic_mass_unit_MeV = 931.494095       # MeV/c2
Spedd_of_light       = 299792458        # m/s
Planck_constant      = 6.626070040E-34  # J.s
Planck_constant_2pi  = 6.626070040E-34 / (2*np.pi) # J.s
Electron_charge      = 1.602176565E-19  # C
Bohr_radius          = 5.2917721092E-11 # m
Bohr_velocity        = 2.1876912633E6   # m/s


print(
"""___________________________________________________
_-=   * PyAccel Module imported successfuly   * =-_
_-=   *             Version: %1.2f             * =-_
___________________________________________________""" % version)


# Definition of the CreateElement class: Use it to define elements
class InitilazeTransportLine ( object ):
    """ Class InitilazeTransportLine is used to start the project of a beam transport line
    examples:\n
    TransportLine = InitilazeTransportLine()\n
    """

#####################        
# Private variables #
#####################
    S = np.array([0.0])
    M = np.zeros((6,6,1))
    num = 0.0
    M[:,:,0] = np.eye(6)
    V0 = np.array([])
    V = np.array([])
    Nelement = np.array([0], dtype=int)
    Name_element =[ 'Source' ]
    
#####################        
# !Initialization!  #
#####################
    def __init__ ( self , sx = 1.0, sxp = 1.0, sy = 1.0, syp = 1.0, sz = 1.0, sdppp0 = 1e-3 , m0 = Proton_mass_kg, e0 = Electron_charge, E0 = 1.0, num = 1000, XYdistribution = 'uniform'):
        self.num = num
        if XYdistribution == 'uniform':
            r = np.sqrt(np.random.rand(1,num))
        elif XYdistribution == 'gaussian':
            r = np.random.randn(1,num)
        theta = np.random.rand(1,num)*2*np.pi

        self.V0 = np.random.randn(6,num) #*2.0 - 1.0
        self.V0[0,:] = (r*np.cos(theta))[:]*sx
        self.V0[1,:] *= sxp
        self.V0[2,:] = (r*np.sin(theta))[:]*sy
        self.V0[3,:] *= syp
        self.V0[4,:] *= sz
        self.V0[5,:] *= sdppp0
        self.V = np.copy(self.V0)
        
    def Propagate(self, N=0):
        self.V = np.copy(self.V0)
        for e in range(N+1):
            for p in range(self.num):
                self.V[:,p] = self.M[:,:,e] @ self.V[:,p]
        
    def DeleteElement(self, N):
        if N != 0:
            self.Name_element.pop(N)
            self.Nelement = np.delete(self.Nelement,N)
            self.M = np.delete(self.M,N,axis=2)
        else:
            print("It is not possible to remove the beam source.")
        
    def ListElements (self):
        print('Listing elements in the transport line:')
        for i in self.Nelement:
            print('[%d] - %s'%(i,self.Name_element[i]))
       
    def PlotPhaseSpace (self, xlim = 0.0, xplim = 0.0, ylim = 0.0, yplim = 0.0, zlim = 0.0, dppp0lim = 0.0):
        plt.figure(figsize=(12,4))
        plt.subplot(131)
        plt.scatter(self.V[0,:],self.V[1,:], s = 1.0, c = 'k')
        plt.xlabel('x [mm]')
        plt.ylabel('xp [mrad]')
        if xlim != 0.0 or xplim != 0.0:
            plt.xlim(xmin=-xlim, xmax=xlim)
            plt.ylim(ymin=-xplim,ymax=xplim)
        plt.subplot(132)
        plt.scatter(self.V[2,:],self.V[3,:], s = 1.0, c = 'k')
        plt.xlabel('y [mm]')
        plt.ylabel('yp [mrad]')
        if ylim != 0.0 or yplim != 0.0:
            plt.xlim(xmin=-ylim, xmax=ylim)
            plt.ylim(ymin=-yplim,ymax=yplim)
        plt.subplot(133)
        plt.scatter(self.V[4,:],1e3*self.V[5,:], s = 1.0, c = 'k')
        plt.xlabel('z [mm]')
        plt.ylabel('dp/p0 [1e-3]')
        if zlim != 0.0 or dppp0lim != 0.0:
            plt.xlim(xmin=-zlim, xmax=zlim)
            plt.ylim(ymin=-dppp0lim,ymax=dppp0lim)
        plt.tight_layout()

    def PlotBeamSpot(self, Rscale, MomentumColorScale = False, realisticPlot = False ):
        plt.figure(figsize=(3,3))
        if MomentumColorScale == False and realisticPlot == False:
            plt.scatter(self.V[0,:],self.V[2,:], s = 1.0, c = 'k')
        elif MomentumColorScale == True and realisticPlot == False:
            plt.scatter(self.V[0,:],self.V[2,:], s = 1.0, c = self.V[5,:], cmap='jet')
        else:
            plt.hist2d(self.V[0,:], self.V[2,:], bins=(100, 100), range = [[-Rscale, Rscale], [-Rscale, Rscale]],
                       norm=mpl.colors.LogNorm(), cmap=mpl.cm.gray)
        plt.axis('equal')
        plt.xlim(xmin=-Rscale, xmax=Rscale)
        plt.ylim(ymin=-Rscale,ymax=Rscale)
        plt.xlabel('x [mm]')
        plt.ylabel('y [mm]')
        plt.tight_layout()

    def AddDriftSpace (self, L):
        self.Nelement = np.append(self.Nelement, self.Nelement[-1]+1)
        self.Name_element.append('Drift Space')
        T = np.zeros((6,6,1))
        T[:,:,0] = np.eye(6)
        T[0,1,0] = L
        T[2,3,0] = L
        self.M = np.append(self.M,T,axis=2)
    
    def AddThinLens(self, f, setXdivergence = False, setYdivergence = False):
        self.Nelement = np.append(self.Nelement, self.Nelement[-1]+1)
        self.Name_element.append('Thin Lens')
        T = np.zeros((6,6,1))
        T[:,:,0] = np.eye(6)
        
        if setXdivergence == True:
            T[1,0,0] = 1/f
        else:
            T[1,0,0] = -1/f
            
        if setYdivergence == True:
            T[3,2,0] = 1/f
        else:
            T[3,2,0] = -1/f
        self.M = np.append(self.M,T,axis=2)

    def AddBendingMagnetX(self, BendingRadius, BendingAngle ):
        self.Nelement = np.append(self.Nelement, self.Nelement[-1]+1)
        self.Name_element.append('Bending magnet in X')
        T = np.zeros((6,6,1))
        T[:,:,0] = np.eye(6)
        T[0,0,0] = np.cos(BendingAngle)
        T[0,1,0] = BendingRadius*np.sin(BendingAngle)
        T[0,5,0] = BendingRadius*(1-np.cos(BendingAngle))
        T[1,1,0] = np.cos(BendingAngle)
        T[1,1,0] = -1*np.sin(BendingAngle)/BendingRadius
        T[1,5,0] = np.sin(BendingAngle)
        T[2,3,0] = BendingAngle*BendingRadius
        self.M = np.append(self.M,T,axis=2)
        
        
        
        
        
        
        
        
        
        
        
        
        