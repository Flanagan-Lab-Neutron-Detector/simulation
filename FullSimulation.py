# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 18:04:03 2020

@author: John Rabaey
@title: BuffonsNeedleFullSimulation.py
"""

from numpy import genfromtxt, column_stack, savetxt, interp, linspace, pi, arccos, cos, average
from scipy.integrate import quad as integral
from random import random
import matplotlib.pyplot as plt

# Choose which data files to load & where to save.
material = "BPSG"

# A note about units: 
#  E     is given in MeV
#  dQ/dx is given in fC / micron
#  x     is gven in microns

# Width of the layers. We ignore edge effects for the time being
w_BPSG_list = [ .01, .3, .5, 1, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0 ]
w_Si   = .03
E0_Li  = .84
E0_He  = 1.47

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

# Next Lithium
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

# plot distance vs dQ/dx
dist_list = linspace(0,6,100)
plt.figure()
plt.title("Charge deposited as a function of distance travelled")
plt.plot(dist_list, [dQdx_Li_x(x) for x in dist_list], 'b', label="Li")
plt.plot(dist_list, [dQdx_He_x(x) for x in dist_list], 'r', label="He")
plt.grid()
plt.legend()
plt.xlabel("x ($\\mu m$)")
plt.ylabel("$\\frac{dQ}{dx}$ ($\\frac{fC}{\\mu m}$)")
plt.savefig(f"results/{material} plots/charge-distance.svg")
savetxt(f"results/{material} data/charge-distance.csv", 
        column_stack([ dist_list, [dQdx_Li_x(x) for x in dist_list], [dQdx_He_x(x) for x in dist_list] ]), 
        delimiter=',',
        header='# dist, Li dQdx, He dQdx'
        )

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

plt.figure()
plt.title("Integrated charge deposit")
plt.plot(dist_list, Q_deposited_Li_fast(0, dist_list), 'b', label="Li")
plt.plot(dist_list, Q_deposited_He_fast(0, dist_list), 'r', label="He")
plt.grid()
plt.legend()
plt.xlabel("x ($\\mu m$)")
plt.ylabel("Q (fC)")
plt.savefig(f"results/{material} plots/integrated-charge-distance.svg")
savetxt(f"results/{material} data/integrated-charge-distance.csv", 
        column_stack([dist_list, Q_deposited_Li_fast(0, dist_list), Q_deposited_He_fast(0, dist_list)]), 
        delimiter=',',
        header='# dist, Q Li, Q He'
        )

Q_avg_He_list = []
Q_sum_He_list = []
interaction_prob_He_list = []
normalized_interaction_count_He_list = [] # normalized to 1 particle per micron BPSG
Q_avg_Li_list = []
Q_sum_Li_list = []
interaction_prob_Li_list = []
normalized_interaction_count_Li_list = []
for i in range(len(w_BPSG_list)):
    w_BPSG = w_BPSG_list[i]
    Q_dep_He_list = []
    Q_dep_Li_list = []
    interaction_count_He = 0
    interaction_count_Li = 0
    for j in range(nTrials):
        # Generate angles in a uniform spherical distribution, x0 in the range of the thickness of the BPSG
        #  phi is the azimuthal angle; let phi = pi/2 define a plane parallel to the plane of the chip
        theta = random() * pi * 2
        phi = arccos(1 - 2 * random())
        z0 = random() * w_BPSG
        
        # Calculate
        r1 = (w_BPSG - z0) / cos(phi)
        r2 = (w_BPSG + w_Si - z0) / cos(phi)
            
        Q_dep_He = Q_deposited_He_fast(r1, r2)
        Q_dep_Li = Q_deposited_Li_fast(-r1, -r2)
        Q_dep_He_list.append(Q_dep_He)
        Q_dep_Li_list.append(Q_dep_Li)
        interaction_count_He += 1 if abs(Q_dep_He) > EPSILON else 0
        interaction_count_Li += 1 if abs(Q_dep_Li) > EPSILON else 0
    
    plt.figure()
    plt.hist([he + li for he, li in zip(Q_dep_He_list, Q_dep_Li_list)], normed=1, bins=20, range=[0,3])
    plt.title(f"Charge deposit distribution for BPSG width={w_BPSG}$\mu$m")
    plt.xlabel("Charge deposited (fC)")

    # save the average Q deposited for this w_BPSG value
    Q_avg_He_list.append(average(Q_dep_He_list))
    Q_sum_He_list.append(sum(Q_dep_He_list) * w_BPSG / nTrials)
    interaction_prob_He_list.append(interaction_count_He / nTrials)
    normalized_interaction_count_He_list.append(interaction_count_He * w_BPSG / nTrials)
    Q_avg_Li_list.append(average(Q_dep_Li_list))
    Q_sum_Li_list.append(sum(Q_dep_Li_list) * w_BPSG / nTrials)
    interaction_prob_Li_list.append(interaction_count_Li / nTrials)
    normalized_interaction_count_Li_list.append(interaction_count_Li * w_BPSG / nTrials)

plt.figure()
plt.title(f"Average charge deposition as a function of {material} layer width")
plt.xlabel(f"width of {material} layer ($\\mu m$)")
plt.ylabel("average charge deposition (fC)")
plt.plot(w_BPSG_list, Q_avg_Li_list, 'b', label="Li")
plt.plot(w_BPSG_list, Q_avg_He_list, 'r', label="He")
plt.legend()
plt.grid()
plt.savefig(f"results/{material} plots/average-charge-depo.svg")
savetxt(f"results/{material} data/average-charge-depo.csv", 
        column_stack([w_BPSG_list, Q_avg_Li_list, Q_avg_He_list]), 
        delimiter=',',
        header='# width, Li avg charge, He avg charge'
        )

plt.figure()
plt.title(f"Total charge deposition as a function of {material} layer width")
plt.xlabel(f"width of {material} layer ($\\mu m$)")
plt.ylabel("total charge deposition (fC)")
plt.plot(w_BPSG_list, Q_sum_Li_list, 'b', label="Li")
plt.plot(w_BPSG_list, Q_sum_He_list, 'r', label="He")
plt.plot(w_BPSG_list, [ h + l for h,l in zip(Q_sum_He_list, Q_sum_Li_list) ], 'k', label="Total")
plt.legend()
plt.grid()
plt.savefig(f"results/{material} plots/total-charge-depo.svg")
savetxt(f"results/{material} data/total-charge-depo.csv", 
        column_stack([w_BPSG_list, Q_sum_Li_list, Q_sum_He_list]), 
        delimiter=',',
        header='# width, Li total charge, He total charge'
        )

plt.figure()
plt.title("Probability that a particle will interact in the ONO layer")
plt.xlabel(f"width of {material} layer ($\\mu m$)")
plt.ylabel("probability")
plt.plot(w_BPSG_list, interaction_prob_Li_list, 'b', label="Li")
plt.plot(w_BPSG_list, interaction_prob_He_list, 'r', label="He")
plt.plot(w_BPSG_list, [ h + l for h,l in zip(interaction_prob_He_list, interaction_prob_Li_list) ], 'k', label="Total")
plt.legend()
plt.grid()
plt.savefig(f"results/{material} plots/interaction-prob.svg")
savetxt(f"results/{material} data/interaction-prob.csv", 
        column_stack([w_BPSG_list, interaction_prob_Li_list, interaction_prob_He_list]), 
        delimiter=',',
        header='# width, Li interaction prob, He interaction prob'
        )

plt.figure()
plt.title(f"Interactions in the ONO layer")
plt.xlabel(f"width of {material} layer ($\\mu m$)")
plt.ylabel(f"number of interactions")
plt.plot(w_BPSG_list, normalized_interaction_count_Li_list, 'b', label="Li")
plt.plot(w_BPSG_list, normalized_interaction_count_He_list, 'r', label="He")
plt.plot(w_BPSG_list, [ h + l for h,l in zip(normalized_interaction_count_He_list, normalized_interaction_count_Li_list) ], 'k', label="Total")
plt.legend()
plt.grid()
plt.savefig(f"results/{material} plots/normalized-interaction-count.svg")
savetxt(f"results/{material} data/normalized-interaction-count.csv", 
        column_stack([w_BPSG_list, normalized_interaction_count_Li_list, normalized_interaction_count_He_list]), 
        delimiter=',',
        header='# width, Li interaction count, He interaction count'
        )

# Simulate particles at varying angles
#  use a few different distances
for dist in [0,.5,1,3]:
    phi_list = linspace(0,pi/2,100)
    Q_dep_list_He = []
    Q_dep_list_Li = []
    for phi in phi_list:
        r1 = dist / cos(phi)
        r2 = (dist + w_Si) / cos(phi)
        Q_dep_He = Q_deposited_He_fast(r1, r2)
        Q_dep_list_He.append(Q_dep_He)
        Q_dep_Li = Q_deposited_Li_fast(r1, r2)
        Q_dep_list_Li.append(Q_dep_Li)
    plt.figure()
    plt.title("Charge deposited as a function of angle; distance = %.2f $\mu m$" % (dist))
    plt.plot([pi/2 - phi for phi in phi_list], Q_dep_list_He, 'r', label="He")
    plt.plot([pi/2 - phi for phi in phi_list], Q_dep_list_Li, 'b', label="Li")
    plt.xlabel("$\\theta$ (rad)")
    plt.ylabel("Charge deposited in the ONO layer (fC)")
    plt.grid()
    plt.legend()
    plt.savefig(f"results/{material} plots/charge-vs-angle-{dist}.svg")
    savetxt(f"results/{material} data/charge-vs-angle-{dist}.csv", 
        column_stack([phi_list, Q_dep_list_Li, Q_dep_list_He]), 
        delimiter=',',
        header='# phi, charge Li, charge He'
        )

# Simulate particles at varying initial energies
#  assume particle trajectory is perpendicular to the plane of the chip
E_list_He = linspace(0,E0_He,100)
Q_dep_list_He = [ Q_deposited_He(distance_He(E0_He) - distance_He(E), distance_He(E0_He) - distance_He(E) + w_Si) for E in E_list_He ]

E_list_Li = linspace(0,E0_Li,100)
Q_dep_list_Li = [ Q_deposited_Li(distance_Li(E0_Li) - distance_Li(E), distance_Li(E0_Li) - distance_Li(E) + w_Si) for E in E_list_Li ]

plt.figure()
plt.title("Charge deposited as a function of entrance energy")
plt.plot(E_list_He, Q_dep_list_He, 'r', label="He")
plt.plot(E_list_Li, Q_dep_list_Li, 'b', label="Li")
plt.xlabel("Energy of particle entering ONO layer (MeV)")
plt.ylabel("Charge deposited in ONO layer (fC)")
plt.grid()
plt.legend()
plt.savefig(f"results/{material} plots/charge-vs-energy.svg")
savetxt(f"results/{material} data/charge-vs-energy.csv", 
    column_stack([E_list_Li, Q_dep_list_Li, E_list_He, Q_dep_list_He]), 
    delimiter=',',
    header='# E Li, Q Li, E He, Q He',
    )

plt.show()
