#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15 11:31:02 2022

@author: yasaman
"""

import uproot
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.colors as colors


def lafferty_whyatt(edges, gamma):
     ex = 1 - gamma
     rmin = edges[:-1]
     rmax = edges[1:]
     return ((rmax - rmin) * ex / (rmax**ex - rmin**ex))**(1 / gamma) 

with uproot.open('/Users/yasaman/AMS02/data/Niko/LeptonAnalysis_EffectiveAcceptance_Average (1).root') as file:
    print(list(file))
    ref_acceptance = file['McEffectiveAcceptance_Electrons'].values()


acceptance  = np.loadtxt('/Users/yasaman/AMS02/data/acceptance_After_Id.txt')
acceptance = acceptance*1e4


E = np.array([0.5,0.65,0.82,1.01,1.22,1.46,1.72,2.0,2.31,2.65,3.0,3.36,3.73,
                             4.12,4.54,5.0,5.49,6.0,6.54,7.1,7.69,8.3,8.95,9.62,10.32,11.04,
                             11.8,12.59,13.41,14.25,15.14,16.05,17.0,17.98,18.99,20.04,21.13,
                             22.25,23.42,24.62,25.9,27.25,28.68,30.21,31.82,33.53,35.36,37.31,
                             39.39,41.61,44.0,46.57,49.33,52.33,55.58,59.13,63.02,67.3,72.05,
                             77.37,83.36,90.19,98.08,107.3,118.4,132.1,148.8,169.9,197.7,237.2,
                             290.0,370.0,500.0,700.0,1000.0])

x= lafferty_whyatt(E, 3)


figure = plt.figure(figsize=(12, 10))
ax1, ax2 = figure.subplots(2,1,gridspec_kw={'height_ratios': [3,1]})

l1=ax1.scatter(ref_acceptance[0],ref_acceptance[1], label="Niko's data", color='k')
l2=ax1.scatter(x, acceptance, label= "my data",color='m') 
ax1.set_xscale("log")   
# ax1.set_yscale("log")
ax1.set_ylabel(r"$\rm A_{\rm eff}/(cm^2 \; sr)$",fontsize=20, fontweight='bold')
# ax1.legend(fontsize=15)

ax2.grid()
l3=ax2.scatter(x, acceptance/ref_acceptance[1][1:-1], label="my data/ Niko's data", color='r')
l4=ax2.hlines(1,0.5,1000,linestyle='--', linewidth=3, color='k', label="y=1")
ax2.set_xlabel("Energy/Gev",fontsize = 20,fontweight = 'bold')
ax2.set_ylabel("acceptance ratio",fontsize = 17,fontweight = 'bold')

ax2.set_xscale("log")
ax2.set_ylim(0.9,1.5)
figure.legend([l1, l2, l3, l4], labels=["Niko's data","My data","My data/Niko's data","y=1"], bbox_to_anchor=(0.4, 0.5))
plt.subplots_adjust(wspace=0, hspace=0)  
 
plt.show()
figure.savefig("/Users/yasaman/AMS02/plots/acceptance.pdf" , dpi=250)  
       
# plt.scatter(acceptance[0], acceptance[1]) 
