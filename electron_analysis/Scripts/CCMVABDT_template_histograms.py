#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 16:17:13 2023

@author: yasaman
"""

import multiprocessing as mp
import os

import numpy as np
import awkward as ak
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.colors as colors


with np.load(os.path.join("/Users/yasaman/AMS02/plots/newpass8/CCMVABDT_template_ISS.npz")) as result_file:
    print(list(result_file))
    CCMVABDT_binn_edges = result_file["var1_binning"]  
    Energy_binning = result_file["var2_binning"] 
    CCMVBDT_el_template_st = result_file["eEvents_st"]
    CCMVBDT_p_template_st = result_file["pEvents_st"]
    CCMVBDT_ccp_template_st = result_file["ccpEvents_st"]
    
CCMVBDT_binn = (CCMVABDT_binn_edges[:-1]+CCMVABDT_binn_edges[1:])/2

with np.load(os.path.join("/Users/yasaman/AMS02/plots/newpass8/CCMVABDT_template_MC.npz")) as result_file:
    CCMVBDT_cce_template_st = result_file["cc_eEvents_st"]

i=0
for binn in range(len(Energy_binning) -1):
    i=i+1
    
    e1 = round(Energy_binning[binn],2)
    e2= round(Energy_binning[binn +1],2)
    
    figure = plt.figure(figsize=(12, 10))
    plot = figure.subplots(1, 1)
    plot.set_xlabel(r'$\Lambda_{\rm CC}$',fontsize = 26,fontweight = 'bold')
    plot.set_ylabel("Single Track Events", fontsize = 26,fontweight = 'bold')
    plot.set_yscale("log")
    plot.tick_params(axis='both', which="major",direction='in', length=14, width=1.5, labelsize=25)
    plot.tick_params(axis='both', which="minor",direction='in', length=8, width=1.5, labelsize=25)
    plt.rcParams['xtick.top'] = True
    plt.rcParams['ytick.right'] = True
    plt.minorticks_on()
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    # plot.set_xlim(0,2.2)
    # plot.set_ylim(10**-5,0.5)
    for axis in ['top','bottom','left','right']:
        plot.spines[axis].set_linewidth(2)
    
    plot.step(np.concatenate(([CCMVABDT_binn_edges[0]], CCMVABDT_binn_edges)), np.concatenate(([0], CCMVBDT_el_template_st[:,binn], [0])), where="post",linewidth=2,label="electron (ISS)")
    #plt.fill_between(CCMVABDT_binn_edges, np.concatenate(([0], CCMVBDT_el_template_st[:,binn])),step="pre",alpha=0.4,color='none',hatch='',edgecolor='b')
    
    plot.step(np.concatenate(([CCMVABDT_binn_edges[0]], CCMVABDT_binn_edges)), np.concatenate(([0], CCMVBDT_p_template_st[:,binn], [0])), where="post",linewidth=2,label="proton(ISS)")
    #plt.fill_between(CCMVABDT_binn_edges, np.concatenate(([0], CCMVBDT_p_template_st[:,binn])),step="pre",alpha=0.4,color='none',hatch='',edgecolor='r')
    
    plot.step(np.concatenate(([CCMVABDT_binn_edges[0]], CCMVABDT_binn_edges)), np.concatenate(([0], CCMVBDT_ccp_template_st[:,binn], [0])), where="post",linewidth=2,label="ccproton (ISS)")
    #plt.fill_between(CCMVABDT_binn_edges, np.concatenate(([0], CCMVBDT_ccp_template_st[:,binn])),step="pre",alpha=0.4,color='none',hatch='',edgecolor='g')
    
    plot.step(np.concatenate(([CCMVABDT_binn_edges[0]], CCMVABDT_binn_edges)), np.concatenate(([0], CCMVBDT_cce_template_st[:,binn], [0])), where="post",linewidth=2,label="ccelectron (MC)")
    #plt.fill_between(CCMVABDT_binn_edges, np.concatenate(([0], CCMVBDT_cce_template_st[:,binn])),step="pre",alpha=0.4,color='none',hatch='',edgecolor='k')
    
    plot.errorbar([],[],fmt= ' ', label= "Energy = "+ "[" +str(e1)+" , "+ str(e2) +"]"+" "+r"$\rm GeV$")
    plt.legend(fontsize=10)
    
    figure.savefig("/Users/yasaman/AMS02/plots/newpass8/CCMVABDT_template/CCMVABDT_template_single_"+str(i)+".pdf" , dpi=250)
    plt.close(figure)
    
    