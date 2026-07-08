#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 11 12:11:58 2022

@author: yasaman
"""
import matplotlib.pyplot as plt
import numpy as np
import os

filename="/Users/yasaman/AMS02/data/templatefit_sample/pass6_new/"
param_values=np.loadtxt("/Users/yasaman/AMS02/plots/template_fit/values.txt",comments='#')
error_values=np.loadtxt("/Users/yasaman/AMS02/plots/template_fit/errors.txt",comments='#')

param_name=np.array([r'$N_{e}^{e}$', r'$N_{e}^{ccp}$',r'$\sigma_{G,e}$',r'$\delta m$', r'$\sigma_{N,e}$', r'$\mu_{N,e}$', r'$\tau_{N,e}$',
            r'$\sigma_{N,ccp}$', r'$\mu_{N,cpp}$', r'$\tau_{N,ccp}$', r'$\alpha_{e}$',r'$N_{ccp}^{e}$',r'$N_{ccp}^{ccp}$'])



with np.load(os.path.join(filename, "Edependent_results.npz")) as result_file:
    TRD_binning = result_file["var1_binning"]
    Energy_binning = result_file["var2_binning"]   
    eEvents = result_file["eEvents"]
    ccpEvents = result_file["ccpEvents"]

y= (Energy_binning[1:] + Energy_binning[:-1])/2 

for i in range(len(param_name)):   

    figure = plt.figure(figsize=(12, 10))
    plot = figure.subplots(1, 1)
    plot.errorbar(y,param_values[:,i],error_values[:,i],fmt='.',marker='o', mfc='w',markersize=10,color='r')
    plot.set_xlabel("Ecal Energy/ Gev",fontsize = 26,fontweight = 'bold')
    plot.set_ylabel(param_name[i], fontsize = 26,fontweight = 'bold')
    plot.set_xscale("log")
    # plt.yticks(np.arange(-0.2,0.4,0.1))
    # plot.set_ylim(-0.2,0.3)
    #plot.set_xlim(0.5,50)
        
    plot.tick_params(axis='both', which="major",direction='in', length=14, width=1.5, labelsize=25)
    plot.tick_params(axis='both', which="minor",direction='in', length=8, width=1.5, labelsize=25)
    # plot.xaxis.set_tick_params(labeltop='on')

    for axis in ['top','bottom','left','right']:
        plot.spines[axis].set_linewidth(2)
        
    plt.rcParams['xtick.top'] = True
    plt.rcParams['ytick.right'] = True
    plt.minorticks_on()
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
        
    figure.savefig("/Users/yasaman/AMS02/plots/template_fit/"+param_name[i]+".pdf" , dpi=250)
    plt.close(figure)
    
    
figure = plt.figure(figsize=(12, 10))
plot = figure.subplots(1, 1)
plot.scatter(y,param_values[:,5]+param_values[:,3],facecolor='none',edgecolor='r',s=70)
plot.set_xlabel("Ecal Energy/ Gev",fontsize = 26,fontweight = 'bold')
plot.set_ylabel(r'$\mu_{G,e}$', fontsize = 26,fontweight = 'bold')
plot.set_xscale("log")
plt.yticks(np.arange(0,1,0.2))
plot.set_ylim(0,0.8)  
  
plot.tick_params(axis='both', which="major",direction='in', length=14, width=1.5, labelsize=25)
plot.tick_params(axis='both', which="minor",direction='in', length=8, width=1.5, labelsize=25)

for axis in ['top','bottom','left','right']:
    plot.spines[axis].set_linewidth(2)
    
plt.rcParams['xtick.top'] = True
plt.rcParams['ytick.right'] = True
plt.minorticks_on()
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
    
figure.savefig("/Users/yasaman/AMS02/plots/template_fit/"+r'$\mu_{G,e}$' , dpi=250)
plt.close(figure)   

figure = plt.figure(figsize=(12, 10))
plot = figure.subplots(1, 1)
plot.scatter(y,param_values[:,5]+param_values[:,8], facecolor='none',edgecolor='r',s=70)
plot.set_xlabel("Ecal Energy/ Gev",fontsize = 26,fontweight = 'bold')
plot.set_ylabel(r'$\mu_{N,cpp}$', fontsize = 26,fontweight = 'bold')
plot.set_xscale("log")
plt.yticks(np.arange(0,2,0.5))
plot.set_ylim(0,1.1)  
  
plot.tick_params(axis='both', which="major",direction='in', length=14, width=1.5, labelsize=25)
plot.tick_params(axis='both', which="minor",direction='in', length=8, width=1.5, labelsize=25)

for axis in ['top','bottom','left','right']:
    plot.spines[axis].set_linewidth(2)
    
plt.rcParams['xtick.top'] = True
plt.rcParams['ytick.right'] = True
plt.minorticks_on()
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
    
figure.savefig("/Users/yasaman/AMS02/plots/template_fit/"+r'$\mu_{N,cpp}$' , dpi=250)
plt.close(figure) 