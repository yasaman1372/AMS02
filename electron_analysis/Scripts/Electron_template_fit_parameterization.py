#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 14:32:59 2022

@author: yasaman
"""
import os
import numpy as np
import awkward as ak
import matplotlib.pyplot as plt
from iminuit import Minuit
from iminuit.cost import ExtendedBinnedNLL, LeastSquares

filename="/Users/yasaman/AMS02/data/templatefit_sample/pass6_new/"
param_values=np.loadtxt("/Users/yasaman/AMS02/plots/template_fit/values.txt",comments='#')
error_values=np.loadtxt("/Users/yasaman/AMS02/plots/template_fit/errors.txt",comments='#')

param_name=np.array([r'$N_{e}^{e}$', r'$N_{e}^{ccp}$',r'$\sigma_{G,e}$',r'$\delta m$', r'$\sigma_{N,e}$', r'$\mu_{N,e}$', r'$\tau_{N,e}$',
            r'$\sigma_{N,ccp}$', r'$\delta m{p}$', r'$\tau_{N,ccp}$', r'$\alpha_{e}$',r'$N_{ccp}^{e}$',r'$N_{ccp}^{ccp}$'])


def model(x,a,b,c,d,e,f):
    
    # i =np.maximum((np.log(x)-b),1e-6)
    x=np.log(x)
    return a*x**5 + b*x**4 + c*x**3 + d*x**2 +e*x +f
    
    #return (a*((np.log(x)-b)**c) + d)*(x<=2) + (e*np.log(x)**2 +r*np.log(x) + s)*(x>2)


with np.load(os.path.join(filename, "Edependent_results.npz")) as result_file:
    TRD_binning = result_file["var1_binning"]
    Energy_binning = result_file["var2_binning"]   
    eEvents = result_file["eEvents"]
    ccpEvents = result_file["ccpEvents"]
    

param= np.zeros((len(param_values),6))
print(param[0])

print(param_values[0,:].shape)
i=0
#x= (Energy_binning[1:] + Energy_binning[:-1])/2 

guess =[0,-0.004,0.03,-0.08,0.5,0.1] 




  
for i in range(13):  
    x= (Energy_binning[1:] + Energy_binning[:-1])/2 
    y = param_values[:,i]
    yerr = error_values[:,i]
    y=y[yerr >0]
    x= x[yerr >0]	    
    yerr = yerr[yerr >0]
    liklihood = LeastSquares(x,y,yerr,model)
    m = Minuit(liklihood, *guess)

    m.migrad()
    print(m)
    param[i]=m.values
    
   

    figure = plt.figure(figsize=(12, 10))
    plot = figure.subplots(1, 1)
    plot.set_xlabel("Ecal Energy/ Gev",fontsize = 26,fontweight = 'bold')
    plot.set_ylabel(param_name[i], fontsize = 26,fontweight = 'bold')
    plot.set_xscale("log")
    plot.tick_params(axis='both', which="major",direction='in', length=14, width=1.5, labelsize=25)
    plot.tick_params(axis='both', which="minor",direction='in', length=8, width=1.5, labelsize=25)
# plot.set_ylim(0,1)
    plot.errorbar(x,y,yerr,fmt='.',marker='o', mfc='None',markersize=10,color='r')
    plt.plot(x,model(x,*m.values),color="k",linestyle="--",label="fit")
#plt.plot(x,model(x,*guess),color="m",linestyle="-.",label="guess")

    chi=m.fval/(len(x)-m.nfit)
    fit_info = [
    f"$\\chi^2$ / $n_\\mathrm{{dof}}$ = {chi:.1f}",]
    for p, v, e in zip(m.parameters, m.values, m.errors):
        fit_info.append(f"{p} = ${v:.3f} \\pm {e:.3f}$")
    
    figure.legend(title="\n".join(fit_info), bbox_to_anchor=(0.3, 0.6))    

    figure.savefig("/Users/yasaman/AMS02/plots/template_fit/"+param_name[i]+"_Parameterized.pdf" , dpi=250)
np.savetxt("/Users/yasaman/AMS02/plots/template_fit/parameterization.txt", param) 
# plt.close(figure)