# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 17:37:15 2020

@author: John Rabaey
@title: Optimizer.py
"""

from numpy import genfromtxt, column_stack, savetxt, interp, linspace, pi, arccos, cos, average, log, fmod
from scipy.integrate import quad as integral
from random import random
import matplotlib.pyplot as plt

# Choose which data files to load & where to save.
material = "BPSG"

# A note about units: 
#  E     is given in MeV
#  dQ/dx is given in fC / micron
#  x     is gven in microns
#  l     is given in microns (l = mean free path of boron in B203)

# Width of the layers. We ignore edge effects for the time being
w_BPSG_list = [ .01, .3, .6, 1, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0 ]
w_Si   = .03
E0_Li  = .84
E0_He  = 1.47
l_neutron = 235

layers = 20

EPSILON = .01 # minimum charge deposit for a particle to count as interacting

# Number of trials
nTrials = 200000

####### READ FROM DIGITIZED TI PLOT DATA
# Read in digitized dQdx vs E plot data; convert to functions.
# First Helium
dQdx_He_csv = genfromtxt(f"./data/dQdx-He-{material}.csv", delimiter=",")
range_He_csv = genfromtxt(f"./data/range-He-{material}.csv", delimiter=",")
E_data_dQdx_He = dQdx_He_csv[:,0]
dQdx_data_He = dQdx_He_csv[:,1]
E_data_range_He = range_He_csv[:,0]
range_data_He = range_He_csv[:,1]
def dQdx_He_E(E):
    return interp(E, E_data_dQdx_He, dQdx_data_He)
def distance_He(E):
    return interp(E, E_data_range_He, range_data_He)

# Next Lithium4
dQdx_Li_csv = genfromtxt(f"./data/dQdx-Li-{material}.csv", delimiter=",")
range_Li_csv = genfromtxt(f"./data/range-Li-{material}.csv", delimiter=",")
E_data_dQdx_Li = dQdx_Li_csv[:,0]
dQdx_data_Li = dQdx_Li_csv[:,1]
E_data_range_Li = range_Li_csv[:,0]
range_data_Li = range_Li_csv[:,1]
def dQdx_Li_E(E):
    return interp(E, E_data_dQdx_Li, dQdx_data_Li)
def distance_Li(E):
    return interp(E, E_data_range_Li, range_data_Li)
# Find dQ/dx as a function of x
# First Helium
E_list = linspace(E0_He,0,100)
dQdx_list_He = dQdx_He_E(E_list)
range_list_He = [distance_He(E0_He) - distance_He(E) for E in E_list]
def dQdx_He_x(x):
    if x < 0 or x > distance_He(E0_He):
        return 0
    return interp(x, range_list_He, dQdx_list_He)
# Next Lithium
dQdx_list_Li = dQdx_Li_E(E_list)
range_list_Li = [distance_Li(E0_Li) - distance_Li(E) for E in E_list]
def dQdx_Li_x(x):
    if x < 0 or x > distance_Li(E0_Li):
        return 0
    return interp(x, range_list_Li, dQdx_list_Li)

# Write a function to calculate the charge lost over a certain distance.
#  r1 and r2 are the distances from the beginning of the track.
def Q_deposited_He(r1, r2):
    return integral(dQdx_He_x, r1, r2)[0]
def Q_deposited_Li(r1, r2):
    return integral(dQdx_Li_x, r1, r2)[0]

# To speed up the process, save the integral as a set of points and use linear interpolation to come back to it.
dist_list_Q_dep_He = linspace(0, distance_He(E0_He), 100)
dist_list_Q_dep_Li = linspace(0, distance_Li(E0_Li), 100)
integrated_Q_dep_He_list = [ Q_deposited_He(0, x) for x in dist_list_Q_dep_He ]
integrated_Q_dep_Li_list = [ Q_deposited_Li(0, x) for x in dist_list_Q_dep_Li ]

def Q_deposited_He_fast(r1,r2):
    return interp(r2, dist_list_Q_dep_He, integrated_Q_dep_He_list) - interp(r1, dist_list_Q_dep_He, integrated_Q_dep_He_list)
def Q_deposited_Li_fast(r1,r2):
    return interp(r2, dist_list_Q_dep_Li, integrated_Q_dep_Li_list) - interp(r1, dist_list_Q_dep_Li, integrated_Q_dep_Li_list)

def pickANeutronPathLength():
    return -l_neutron * log(random())

interaction_prob_list = []
for i in range(len(w_BPSG_list)):
    w_BPSG = w_BPSG_list[i]
    Q_dep_He_list = []
    Q_dep_Li_list = []
    interaction_count_He = 0
    interaction_count_Li = 0
    for j in range(nTrials):
        Q_dep_He = 0
        Q_dep_Li = 0
        # Generate z0, the point of fission capture, in microns from the side of the BPSG layer farther from the ONO layer.
        z0 = pickANeutronPathLength()
        if z0 < w_BPSG * layers:
            z0 = fmod(z0, w_BPSG)
            # Generate angles in a uniform spherical distribution
            #  phi is the azimuthal angle; let phi = pi/2 define a plane parallel to the plane of the chip
            theta = random() * pi * 2
            phi = arccos(1 - 2 * random())
            
            # Calculate
            r1 = (w_BPSG - z0) / cos(phi)
            r2 = (w_BPSG + w_Si - z0) / cos(phi)
                    
            Q_dep_He = Q_deposited_He_fast(r1, r2)
            Q_dep_Li = Q_deposited_Li_fast(-r1, -r2)
        Q_dep_He_list.append(Q_dep_He)
        Q_dep_Li_list.append(Q_dep_Li)
        interaction_count_He += 1 if abs(Q_dep_He) > EPSILON else 0
        interaction_count_Li += 1 if abs(Q_dep_Li) > EPSILON else 0
    
    # save the interaction probability.
    interaction_prob_list.append((interaction_count_He + interaction_count_Li) / nTrials)

plt.figure()
plt.title(f"Efficiency of a chip with {layers} die{'s' if layers > 1 else ''}")
plt.xlabel(f"width of {material} layer ($\\mu m$)")
plt.ylabel(f"probability of a bit flip")
plt.plot(w_BPSG_list, interaction_prob_list, 'k')
plt.grid()
plt.savefig(f"results/layers/efficiency{layers}dyes.svg")
savetxt(f"results/layers/efficiency{layers}dyes.csv", 
        column_stack([w_BPSG_list, interaction_prob_list]), 
        delimiter=',',
        header='# width, total interaction probability'
        )

plt.show()
