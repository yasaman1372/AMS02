#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 16:09:10 2023

@author: yasaman
"""

import multiprocessing as mp
import os

import numpy as np
import awkward as ak
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.colors as colors


TRD_Estimator_tmplate_electron = np.loadtxt("/Users/yasaman/AMS02/plots/template_fit/TRD_Estimator_tmplate_electron.txt")
TRD_Estimator_tmplate_ccproton = np.loadtxt("/Users/yasaman/AMS02/plots/template_fit/TRD_Estimator_tmplate_ccproton.txt")

with np.load(os.path.join("/Users/yasaman/AMS02/data/templatefit_sample/pass6_new/", "Edependent_results_identification_test.npz")) as result_file:
    TRDEstimator_binn_edges = result_file["var1_binning"]
    Energy_binning = result_file["var2_binning"] 

with np.load(os.path.join("/Users/yasaman/AMS02/data/templatefit_sample/ElectronCCMVBDTTemplateFitMC6", "CCMVABDT_template_MC.npz")) as result_file:
    CCMVABDT_binn_edges = result_file["var1_binning"]
    cce_template = result_file["cc_eEvents"]
    
with np.load(os.path.join("/Users/yasaman/AMS02/data/templatefit_sample/ElectronCCMVBDTTemplateFitISS6", "CCMVABDT_template_ISS.npz")) as result_file:
    CCMVABDT_binn_edges = result_file["var1_binning"]  
    CCMVBDT_el_template = result_file["eEvents"]
    CCMVBDT_p_template = result_file["pEvents"]
    CCMVBDT_ccp_template = result_file["ccpEvents"]
    
TRD_binn = (TRDEstimator_binn_edges[:-1]+TRDEstimator_binn_edges[1:])/2 
CCMVBDT_binn = (CCMVABDT_binn_edges[:-1]+CCMVABDT_binn_edges[1:])/2


print(CCMVBDT_binn)
print(TRD_binn)
 
i=0
for binn in range(len(Energy_binning) -1):
    i=i+1
    
    e1 = round(Energy_binning[binn],2)
    e2= round(Energy_binning[binn +1],2)
    
    print(CCMVBDT_el_template[:,binn])
    el_template =  TRD_Estimator_tmplate_electron[:,binn][:,None] * CCMVBDT_el_template[:,binn] [None,:]  
    print(TRD_Estimator_tmplate_electron.shape)
    print(CCMVBDT_el_template.shape)
    print(el_template.shape)
    
    # el_template_hist, el_template_edges = np.histogramdd((TRD_Estimator_tmplate_electron[:,binn],CCMVBDT_el_template[:,binn]), bins= (TRDEstimator_binn_edges,CCMVABDT_binn_edges))
    # el_template = el_template + el_template_hist    
    
    figure = plt.figure(figsize=(12, 10))
    plot = figure.subplots(1, 1)
    plot.set_xlabel(r'$Z \times \Lambda_{\rm TRD}$',fontsize = 26)
    plot.set_ylabel(r'$\Lambda_{\rm CC}$', fontsize = 26)
    plot.tick_params(axis='both', which="major",direction='in', length=12, width=1.5, labelsize=20)
    plot.tick_params(axis='both', which="minor",direction='in', length=7, width=1.5)
    plt.rcParams['xtick.top'] = True
    plt.rcParams['ytick.right'] = True
    plt.minorticks_on()
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    for axis in ['top','bottom','left','right']:
        plot.spines[axis].set_linewidth(2)   
        
    el_template = np.ma.masked_where(el_template < 1, el_template).transpose()    
    mesh = plot.pcolormesh(-TRD_binn, CCMVBDT_binn , el_template, norm=colors.LogNorm(), cmap =plt.cm.get_cmap('jet'))
    cbar = plt.colorbar(mesh, ax=plot, aspect=10)
    cbar.set_label("Events",fontsize=25)
    
    plot.errorbar([],[],fmt= ' ', label= "Energy = "+ "[" +str(e1)+" , "+ str(e2) +"]"+" "+r"$\rm GeV$")
        
    cbar.ax.tick_params(which="major",direction='in', length=7, width=1.5,labelsize=20)
    cbar.ax.tick_params(which="minor",direction='in')  
    plt.xlim(-2,2)
    plt.legend(fontsize=25)
    
    figure.savefig("/Users/yasaman/AMS02/plots/2Dtemplates"+str(i)+".pdf" , dpi=250)
    plt.close(figure)
