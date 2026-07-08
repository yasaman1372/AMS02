#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 10:13:21 2022

@author: yasaman
"""

from iminuit import Minuit
import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit
from iminuit.cost import ExtendedBinnedNLL, LeastSquares
from iminuit.util import describe
import os
from models import *
from matplotlib.patches import Rectangle
from numpy import genfromtxt


el_count_ref = genfromtxt('/Users/yasaman/AMS02/data/Niko/ElectronNumber.csv', delimiter=',')

param_name=[r'$N_{e}^{e}$', r'$N_{e}^{ccp}$',r'$\sigma_{G,e}$',r'$\delta m$', r'$\sigma_{N,e}$', r'$\mu_{N,e}$', r'$\tau_{N,e}$',
            r'$\sigma_{N,ccp}$', r'$\mu_{N,cpp}$', r'$\tau_{N,ccp}$', r'$\alpha_{e}$',r'$N_{ccp}^{e}$',r'$N_{ccp}^{ccp}$']

#inival= np.loadtxt("/Users/yasaman/AMS02/plots/template_fit/values.txt")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--title", default="ISS", help="title of the input data (e.g. ISS)")
    parser.add_argument("--filename", default= "/Users/yasaman/AMS02/data/templatefit_sample/pass6_new/",help="Path to root file to read tree from. Can also be path to a file with a list of root files or a pattern of root files (e.g. results/ExampleAnalysisTree*.root)")
    #parser.add_argument("filename",help="Path to root file to read tree from. Can also be path to a file with a list of root files or a pattern of root files (e.g. results/ExampleAnalysisTree*.root)")
    args = parser.parse_args()
    font_size='30'


# we open the electoron sample and charged confused proton sample + Energy and TRD binning
    with np.load(os.path.join(args.filename, "Edependent_results_identification_test.npz")) as result_file:
        TRD_binning = result_file["var1_binning"]
        Energy_binning = result_file["var2_binning"]   
        eEvents = result_file["eEvents"]
        ccpEvents = result_file["ccpEvents"]
        
    # with np.load("/Users/yasaman/AMS02/data/templatefit_sample/pass6_new/Edependent_results.npz") as result_file:
   
    #     eEvents_NoEcalCut = result_file["eEvents"]
        # pEvents = result_file["pEvents"]
        
      
     
    X =  (TRD_binning[1:] + TRD_binning[:-1])/2   
     
    lowlim=10
    upplim=-40
    
    # we romove bins with high residuals
    TRD_binning = TRD_binning[lowlim:upplim]
      
    x= (TRD_binning[1:] + TRD_binning[:-1])/2 #TRD binning
    y= (Energy_binning[1:] + Energy_binning[:-1])/2 #Energy binning
    

    param_num = 13
    
    
    
    param_values = np.zeros((len(y),param_num)) # make empty list to store the parameters values in each energy range
    param_errors = np.zeros((len(y), param_num))# make empty list to store the parameters errors in each energy range
    
    el_chi= np.zeros(len(y)) 
    ccp_chi= np.zeros(len(y))
    el_count= np.zeros(len(y))
    cpp_count=np.zeros(len(y))
    el_count_error= np.zeros(len(y))
    cpp_count_error=np.zeros(len(y))
    
 
    guess=dict(n=20000 , npr =1000, s1=0.06, delta_m=-0.1, s2 =0.1, m2=0.35, t=0.2, sp =0.15,
                  delta_mp =0.45, tp = 0.1 , alpha =0.99 , ncc =200 , npcc =100)
    dof = len(x)- len(guess)
    
    #guess=dict(n=20000 , npr =2, s1=0.06, delta_m=-0.1, s2 =0.1, m2=0.35, t=0.2, sp =0.15,
                  #delta_mp =0.6, tp = 0.1 , alpha =0.99 , ncc =200 , npcc =100)
    
    electron_guess = {variable: guess[variable] for variable in guess if variable in describe(cumulative_model)}
    CCproton_guess = {variable: guess[variable] for variable in guess if variable in describe(cumulative_model_CC)}
    counter=0
    central_guess = None
    for binn in list(range(len(y)//2, len(y))) + list(range(len(y)//2-1, -1, -1)): 
        counter = counter+1
        #the itteration begins at the center of the energy bins and moves to the end then goes back from the center to the beginning. 
        
        # print(binn)
        
        if binn == len(y) // 2 - 1:
            guess = central_guess
            # when we go one step back from the ceneter the energi bins, guess values will be loaded from the saved valuses of previous step.
        events_electron= eEvents[: , binn][lowlim:upplim]
        #events_electron_NoEcalCut = eEvents_NoEcalCut[: , binn][lowlim:upplim]
        events_CCproton= ccpEvents[: , binn][lowlim:upplim]
        # events= np.delete(events, index)
        #print(eEvents[: , binn].shape)
        liklihood_electron = ExtendedBinnedNLL(events_electron,TRD_binning,cumulative_model)
        liklihood_CCproton = ExtendedBinnedNLL(events_CCproton,TRD_binning,cumulative_model_CC)
        liklihood=  liklihood_electron + liklihood_CCproton 
        minuit = Minuit(liklihood, **guess)
        minuit.limits["tp"]=(-0.25,0.25)
        minuit.limits["alpha"]=(0,1)
        minuit.limits["s1"]=(0.05,0.2)
        minuit.limits["s2"]=(0,0.2)
        minuit.limits["sp"]=(0,0.3)
        minuit.limits["t"]=(0,1)
        minuit.limits["delta_mp"]=(0,1)
        #minuit.limits["m1"]=(0.2,0.8)
        minuit.limits["delta_m"]=(-0.005,0)
        minuit.limits["m2"]=(0,1)
        minuit.limits["npr"]=(0,10000)
        minuit.limits["n"]=(0,None)
        minuit.limits["ncc"]=(0,20000)
        minuit.limits["npcc"]=(0,None)
        
        if y[binn] > 10:
            minuit.values['alpha']=1
            minuit.fixed['alpha']=True
            #minuit.fixed["m1"]=True
            minuit.fixed["s1"]=True            
        
        minuit.migrad()
        print(binn)
        print(minuit.valid)
        #print(minuit)
        
        electron_values = minuit.values[:-2]
        CCproton_values = minuit.values[2:]
        
        
        param_values[binn,:] = minuit.values
        param_errors[binn,:] = minuit.errors
        
        index=np.where(events_electron>=1)
        el_res= chisquare(events_electron[index],Com_model(x,*electron_values)[index])
        el_chi[binn] = np.sum(el_res)/dof
        
        ccp_res= chisquare(events_CCproton[index],Com_model_CC(x,*CCproton_values)[index])
        ccp_chi[binn] = np.sum(ccp_res)/dof
        
        if central_guess is None:
            central_guess = dict(zip(minuit.parameters, minuit.values))
        
        n=minuit.values[0]
        npr=minuit.values[1]
        s1=minuit.values[2]
        delta_m=minuit.values[3]
        s2=minuit.values[4]
        m2=minuit.values[5]
        t=minuit.values[6]
        sp=minuit.values[7]
        delta_mp=minuit.values[8]
        tp=minuit.values[9]
        alpha=minuit.values[10]
        ncc=minuit.values[11]
        npcc=minuit.values[12] 
        
        particle_counts, particle_counts_errors = El_template_fit_function(x,n,npr,s1,delta_m,s2,m2,t,sp,delta_mp,tp,alpha,TRD_binning,events_electron)
        el_count[binn] = particle_counts[0]
        el_count_error[binn] = particle_counts_errors[0]
        cpp_count[binn] = particle_counts[1]
        cpp_count_error[binn] = particle_counts_errors[1]
        
        #print('electron number:',particle_counts,events_electron.sum())
        
        
# ########## we plot the fit function

        figure = plt.figure(figsize=(12, 10))
        plot = figure.subplots(1, 1)
        a = round(Energy_binning[binn],2)
        b = round(Energy_binning[binn +1],2)
        #plot.set_title("electron sample"+","+" "+args.title +","+" " +"Energy range = "+ "[" +str(a)+" , "+ str(b) +"]"+" "+r"$\rm Gev$", fontsize = 20)
        plot.set_xlabel(r'$\Lambda_{\rm TRD}$',fontsize = 26,fontweight = 'bold')
        plot.set_ylabel("Events", fontsize = 26, fontweight = 'bold')
        plot.set_yscale("log")
        #plot.set_ylim(0.1,max(events_electron)*1.1)
        plot.tick_params(axis='both', which="major",direction='in', length=14, width=1.5, labelsize=25)
        plot.tick_params(axis='both', which="minor",direction='in', length=8, width=1.5, labelsize=25)
        # plot.xaxis.set_tick_params(labeltop='on')

        for axis in ['top','bottom','left','right']:
            plot.spines[axis].set_linewidth(2)
        
       
        #plot.plot(x,Com_model(x,**electron_guess),label='guess')
        #print(Com_model(X,**electron_guess))
        plot.plot(X,n*electron_model(X,alpha,s1,delta_m,s2,m2,t),label="electron",color='c')
        #plot.plot(X,n*electron_model_Gaus(X,alpha,s1,delta_m,s2,m2,t),label="electron_Gaus",color="g")
        #plot.plot(X,n*electron_model_Nov(X,alpha,s1,delta_m,s2,m2,t),label="electron_Nov",color="m")
        plt.fill_between(X,n*electron_model(X,alpha,s1,delta_m,s2,m2,t),facecolor='c',alpha=0.2)
        plot.plot(X,npr*CCproton_model(X,sp,delta_mp,tp,m2),label="proton",color='deeppink')
        plt.fill_between(X,npr*CCproton_model(X,sp,delta_mp,tp,m2),facecolor="deeppink",alpha=0.2)
        plot.plot(X,Com_model(X,*electron_values), label= 'fit result',linestyle='dashed',color='orange',linewidth=5)
        #plot.plot(X,Com_model(X,**electron_guess), label= 'guess',linestyle='dashed',color='blue',linewidth=5)
        plot.errorbar(x, events_electron, np.sqrt(events_electron), fmt=".", markersize=16,label="data",color='k')
        plot.errorbar([],[],fmt=' ' ,label=r'$\chi^2/dof=$'+str(round(el_chi[binn],2)))
        plot.errorbar([],[],fmt= ' ', label= "Energy = "+ "[" +str(a)+" , "+ str(b) +"]"+" "+r"$\rm Gev$")
        
        #legend_properties = {'weight':'bold'}
        plot.legend(title='ISS data',title_fontsize=15,fontsize=15,frameon=False,loc='upper right')
       
        plot.set_xlim(0,1.8)
        plot.set_ylim(bottom=0.6)
        plt.rcParams['xtick.top'] = True
        plt.rcParams['ytick.right'] = True
        plt.minorticks_on()
        plt.rcParams["font.weight"] = "bold"
        plt.rcParams["axes.labelweight"] = "bold"
        
        
        
        figure.savefig("/Users/yasaman/AMS02/plots/template_fit/"+"electron_histogram_"+str(binn)+".pdf" , dpi=250)
        plt.close(figure) 
        
        figure = plt.figure(figsize=(12, 10))
        plot = figure.subplots(1, 1)
        a = round(Energy_binning[binn],2)
        b = round(Energy_binning[binn +1],2)
        #plot.set_title("electron sample"+","+" "+args.title +","+" " +"Energy range = "+ "[" +str(a)+" , "+ str(b) +"]"+" "+r"$\rm Gev$", fontsize = 20)
        plot.set_xlabel(r'$\Lambda_{\rm TRD}$',fontsize = 26,fontweight = 'bold')
        plot.set_ylabel("Events", fontsize = 26, fontweight = 'bold')
        plot.set_yscale("log")
        #plot.set_ylim(0.1,max(events_electron)*1.1)
        plot.tick_params(axis='both', which="major",direction='in', length=14, width=1.5, labelsize=25)
        plot.tick_params(axis='both', which="minor",direction='in', length=8, width=1.5, labelsize=25)
        # plot.xaxis.set_tick_params(labeltop='on')

        for axis in ['top','bottom','left','right']:
            plot.spines[axis].set_linewidth(2)
        
       # print("cc proton guess", CCproton_guess)
        #plot.plot(x,Com_model_CC(x,**ccproton_guess),label='guess')   
        plot.plot(X, ncc*electron_model(X,alpha,s1,delta_m,s2,m2,t),label="electron",color='c')
        plt.fill_between(X,ncc*electron_model(X,alpha,s1,delta_m,s2,m2,t),facecolor='c',alpha=0.2)
        plot.plot(X,npcc*CCproton_model(X,sp,delta_mp,tp,m2),label="proton",color='deeppink')
        plt.fill_between(X,npcc*CCproton_model(X,sp,delta_mp,tp,m2),facecolor="deeppink",alpha=0.2)
        plot.plot(X,Com_model_CC(X,*CCproton_values), label="fit result",linestyle='dashed',color='orange',linewidth=5)
        #plot.plot(X,Com_model_CC(X,**CCproton_guess), label="        guess",linestyle='dashed',color='blue',linewidth=5)
        plot.errorbar(x, events_CCproton, np.sqrt(events_CCproton), fmt=".", markersize=16,label="data",color='k')
        plot.errorbar([],[],fmt=' ',label="\t"+r'$\chi^2/dof=$'+str(round(ccp_chi[binn],2)))
        plot.errorbar([],[],fmt=' ',label="Energy = "+ "[" +str(a)+" , "+ str(b) +"]"+" "+r"$\rm Gev$")

        plot.legend(title='ISS data',title_fontsize=14,fontsize=14,frameon=False, loc='upper right')

        # plot.set_xlim(0,1.8)
        plot.set_ylim(bottom=0.6)
        plt.rcParams['xtick.top'] = True
        plt.rcParams['ytick.right'] = True
        plt.minorticks_on()
        plt.rcParams["font.weight"] = "bold"
        plt.rcParams["axes.labelweight"] = "bold"
        
        figure.savefig("/Users/yasaman/AMS02/plots/template_fit/"+"ccproton_histogram_"+str(binn)+".pdf" , dpi=250)
        plt.close(figure)
        
        #counter +=1
        
# ######### we plot the residuals       

        figure = plt.figure(figsize=(12, 6.15))
        plot = figure.subplots(1, 1)
        plot.scatter(np.arange(len(x))[index],el_res,marker='o')
        plot.set_xlabel("bin number",fontsize = font_size)
        plot.set_ylabel('residuals', fontsize = font_size)
        # plot.set_xscale("log")
        plot.set_yscale("log")
        plt.hlines(10,0,len(x),linestyle='dashed')
            
        figure.savefig("/Users/yasaman/AMS02/plots/template_fit/el_residuals_" + str(binn)+ ".pdf" , dpi=250)
        plt.close(figure)        
        
#we use the result of previous binn for the next binn        
            
        val=minuit.values
        
        guess=dict(n=val[0] , npr =val[1], s1=val[2], delta_m=val[3], s2 =val[4], m2=val[5], t=val[6], sp =val[7],
                   delta_mp =val[8], tp = val[9] , alpha = val[10] , ncc = val[11] , npcc = val[12]) 

    
        
# ########## now we plot and save the parameter's values and electron count
           
    header=''  
    
    for i in range(len(guess)):
        header= header + minuit.parameters[i] + "\t"
        
    np.savetxt("/Users/yasaman/AMS02/plots/template_fit/values.txt", param_values, header = header )
    np.savetxt("/Users/yasaman/AMS02/plots/template_fit/errors.txt", param_errors, header = header )
    np.savetxt("/Users/yasaman/AMS02/plots/template_fit/el_count.txt", el_count )
    np.savetxt("/Users/yasaman/AMS02/plots/template_fit/el_count_error.txt", el_count_error )
    
    
    for i in range(len(guess)):   

        figure = plt.figure(figsize=(12, 10))
        plot = figure.subplots(1, 1)
        plot.scatter(y,param_values[:,i])
        plot.set_xlabel("Ecal Energy/ Gev",fontsize = 26,fontweight = 'bold')
        plot.set_ylabel(param_name[i], fontsize = 26,fontweight = 'bold')
        plot.set_xscale("log")
        
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
        
        figure.savefig("/Users/yasaman/AMS02/plots/template_fit/"+minuit.parameters[i]+".pdf" , dpi=250)
        plt.close(figure)


    
    figure = plt.figure(figsize=(12, 10))
    plot = figure.subplots(1, 1)
    plot.scatter(y,param_values[:,5]+param_values[:,3])
    plot.set_xlabel("Ecal Energy/ Gev",fontsize = 26,fontweight = 'bold')
    plot.set_ylabel(r'$\mu_{G,e}$', fontsize = 26,fontweight = 'bold')
    plot.set_xscale("log")
    
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
    
    figure.savefig("/Users/yasaman/AMS02/plots/template_fit/"+"m1.pdf" , dpi=250)
    plt.close(figure)


    
    figure = plt.figure(figsize=(12, 10))
    plot = figure.subplots(1, 1)
    plot.scatter(y,param_values[:,5]+param_values[:,8])
    plot.set_xlabel("Ecal Energy/ Gev",fontsize = 26,fontweight = 'bold')
    plot.set_ylabel(r'$\mu_{N,cpp}$', fontsize = 26,fontweight = 'bold')
    plot.set_xscale("log")
    
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
    
    figure.savefig("/Users/yasaman/AMS02/plots/template_fit/"+"mp.pdf" , dpi=250)
    plt.close(figure)
    
#-------------------------------------------------------------------------    
    figure = plt.figure(figsize=(12, 10))
    plot = figure.subplots(1, 1)
    plot.scatter(y,el_count, label="my data")
    plot.scatter(el_count_ref[:,0],el_count_ref[:,1], label="Nikolas data")
    plot.set_xlabel("Ecal Energy/ Gev",fontsize = 26,fontweight = 'bold')
    plot.set_ylabel('Numberof electrons', fontsize = 26,fontweight = 'bold')
    plot.set_xscale("log")
    plot.set_yscale("log")
    plot.set_ylim(1,None)
    
    plot.tick_params(axis='both', which="major",direction='in', length=14, width=1.5, labelsize=25)
    plot.tick_params(axis='both', which="minor",direction='in', length=8, width=1.5, labelsize=25)

    for axis in ['top','bottom','left','right']:
        plot.spines[axis].set_linewidth(2)
    
    plt.rcParams['xtick.top'] = True
    plt.rcParams['ytick.right'] = True
    plt.minorticks_on()
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    plt.legend()
    
    figure.savefig("/Users/yasaman/AMS02/plots/template_fit/"+"el_count.pdf" , dpi=250)
    plt.close(figure)
        
    
# ######### and we plot the chisqures per degree of freedom
    
    figure = plt.figure(figsize=(12, 8))
    plot = figure.subplots(1, 1)
    plot.tick_params(axis='both', which="major",direction='in', length=14, width=1.5, labelsize=25)
    plot.tick_params(axis='both', which="minor",direction='in', length=8, width=1.5, labelsize=25)
    # plot.xaxis.set_tick_params(labeltop='on')

    for axis in ['top','bottom','left','right']:
        plot.spines[axis].set_linewidth(2)
    plot.scatter(y,el_chi,s=50,marker='s',color='k')
    plot.set_xlabel("Ecal Energy/ Gev",fontsize = font_size)
    plot.set_ylabel(r'$\chi^{2} /dof$', fontsize = font_size)
    plot.set_xscale("log")    
    plt.rcParams['xtick.top'] = True
    plt.rcParams['ytick.right'] = True
    plt.minorticks_on()
    # plt.ylim(0,10)
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
            
    figure.savefig("/Users/yasaman/AMS02/plots/template_fit/"+"el_chisquares"+".pdf" , dpi=250)
    plt.close(figure)
    
    
        
    figure = plt.figure(figsize=(12, 8))
    plot = figure.subplots(1, 1)
    plot.tick_params(axis='both', which="major",direction='in', length=14, width=1.5, labelsize=25)
    plot.tick_params(axis='both', which="minor",direction='in', length=8, width=1.5, labelsize=25)
    # plot.xaxis.set_tick_params(labeltop='on')

    for axis in ['top','bottom','left','right']:
        plot.spines[axis].set_linewidth(2)
    plot.scatter(y,ccp_chi,s=50,marker='s',color='k')
    plot.set_xlabel("Ecal Energy/ Gev",fontsize = font_size)
    plot.set_ylabel(r'$\chi^{2} /dof$', fontsize = font_size)
    plot.set_xscale("log")    
    plt.rcParams['xtick.top'] = True
    plt.rcParams['ytick.right'] = True
    plt.minorticks_on()
    plt.ylim(0,10)
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
            
    figure.savefig("/Users/yasaman/AMS02/plots/template_fit/"+"ccp_chisquares"+".pdf" , dpi=250)
    plt.close(figure)
    
                      
if __name__ == "__main__":
    main()      

        
        
        





    
