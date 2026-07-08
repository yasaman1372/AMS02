#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 10:25:17 2023

@author: yasaman
"""

from iminuit import Minuit
import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit
from iminuit.cost import ExtendedBinnedNLL, LeastSquares
from scipy.integrate import quad
from iminuit.util import describe
import os


def chisquare(d,m):
    err=np.maximum(m,1)
    return ((d-m)**2)/err

def residuals(d,m):
    err=np.maximum(m,1)
    return (d-m)/np.sqrt(err)


def electron_model(x,alpha_el,s1_el,delta_m_el,s2_el,m2_el,t_el):
    m1_el=m2_el + delta_m_el
    l = 1 + t_el * (x-m2_el) *  np.sinh(t_el * np.sqrt(np.log(4))) / (s2_el*t_el* np.sqrt(np.log(4))) 
    l = np.maximum(l,1e-7) 
    G = (1/np.sqrt(2* np.pi* s1_el**2)) * np.exp (- ((x-m1_el)**2) /(2 * s1_el**2))
    N = np.exp(-0.5 * ((np.log(l)**2 / t_el**2) + t_el**2) )   
    return (1- alpha_el)*G + alpha_el*N

def ccproton_model(x,s_ccp,delta_m_ccp,t_ccp,m2_el):
    m_ccp=m2_el+delta_m_ccp
    l = 1 + t_ccp * (x-m_ccp) *  np.sinh(t_ccp * np.sqrt(np.log(4))) / (s_ccp*t_ccp* np.sqrt(np.log(4))) 
    l = np.maximum(l,1e-7)
    N = np.exp(-0.5 * ((np.log(l)**2 / t_ccp**2) + t_ccp**2) )    
    return N

def proton_model(x,alpha_p,s1_p,delta_m_p,s2_p,m2_p,t_p):
    m1_p = m2_p + delta_m_p
    l = 1 + t_p * (x-m2_p) *  np.sinh(t_p * np.sqrt(np.log(4))) / (s2_p*t_p* np.sqrt(np.log(4))) 
    l = np.maximum(l,1e-7) 
    G = (1/np.sqrt(2* np.pi* s1_p**2)) * np.exp (- ((x-m1_p)**2) /(2 * s1_p**2))
    N = np.exp(-0.5 * ((np.log(l)**2 / t_p**2) + t_p**2) )   
    return (1- alpha_p)*G + alpha_p*N

def electron_template(x,n_el,n_ccp,s1_el,delta_m_el,s2_el,m2_el,t_el,s_ccp,delta_m_ccp,t_ccp,alpha_el):
    return n_el*electron_model(x,alpha_el,s1_el,delta_m_el,s2_el,m2_el,t_el) + n_ccp*ccproton_model(x,s_ccp,delta_m_ccp,t_ccp,m2_el)
    
def ccproton_template(x,s1_el,delta_m_el,s2_el,m2_el,t_el,s_ccp,delta_m_ccp,t_ccp,alpha_el,ncc,npcc):
    return ncc*electron_model(x,alpha_el,s1_el,delta_m_el,s2_el,m2_el,t_el) + npcc*ccproton_model(x,s_ccp,delta_m_ccp,t_ccp,m2_el)

def proton_template(x,np_el,np_p,alpha_el,s1_el,delta_m_el,s2_el,m2_el,t_el,alpha_p,s1_p,delta_m_p,s2_p,m2_p,t_p):
    return np_el*electron_model(x,alpha_el,s1_el,delta_m_el,s2_el,m2_el,t_el) + np_p*proton_model(x,alpha_p,s1_p,delta_m_p,s2_p,m2_p,t_p)
        


def electron_cumulative_model(edges,n_el,n_ccp,s1_el,delta_m_el,s2_el,m2_el,t_el,s_ccp,delta_m_ccp,t_ccp,alpha_el):
    p = electron_template(edges,n_el,n_ccp,s1_el,delta_m_el,s2_el,m2_el,t_el,s_ccp,delta_m_ccp,t_ccp,alpha_el)
    cp = np.cumsum(p)
    return cp


def ccproton_cumulative_model(edges,s1_el,delta_m_el,s2_el,m2_el,t_el,s_ccp,delta_m_ccp,t_ccp,alpha_el,ncc,npcc):
    p = ccproton_template(edges,s1_el,delta_m_el,s2_el,m2_el,t_el,s_ccp,delta_m_ccp,t_ccp,alpha_el,ncc,npcc)
    cp = np.cumsum(p)
    return cp
    

def proton_cumulative_model(edges,np_el,np_p,alpha_el,s1_el,delta_m_el,s2_el,m2_el,t_el,alpha_p,s1_p,delta_m_p,s2_p,m2_p,t_p):
    p = proton_template(edges,np_el,np_p,alpha_el,s1_el,delta_m_el,s2_el,m2_el,t_el,alpha_p,s1_p,delta_m_p,s2_p,m2_p,t_p)
    cp = np.cumsum(p)
    return cp 

# def electron_cumulative_model(edges,n_el,n_ccp,s1_el,delta_m_el,s2_el,m2_el,t_el,s_ccp,delta_m_ccp,t_ccp,alpha_el):
#     x = (edges[1:] + edges[:-1])/2
#     p = electron_template(x,n_el,n_ccp,s1_el,delta_m_el,s2_el,m2_el,t_el,s_ccp,delta_m_ccp,t_ccp,alpha_el)
#     cp = np.cumsum(p)
#     return np.concatenate(([0],cp))


# def ccproton_cumulative_model(edges,s1_el,delta_m_el,s2_el,m2_el,t_el,s_ccp,delta_m_ccp,t_ccp,alpha_el,ncc,npcc):
#     x = (edges[1:] + edges[:-1])/2
#     p = ccproton_template(x,s1_el,delta_m_el,s2_el,m2_el,t_el,s_ccp,delta_m_ccp,t_ccp,alpha_el,ncc,npcc)
#     cp = np.cumsum(p)
#     return np.concatenate(([0],cp))
    

# def proton_cumulative_model(edges,np_el,np_p,alpha_el,s1_el,delta_m_el,s2_el,m2_el,t_el,alpha_p,s1_p,delta_m_p,s2_p,m2_p,t_p):
#     x = (edges[1:] + edges[:-1])/2
#     p = proton_template(x,np_el,np_p,alpha_el,s1_el,delta_m_el,s2_el,m2_el,t_el,alpha_p,s1_p,delta_m_p,s2_p,m2_p,t_p)
#     cp = np.cumsum(p)
#     return np.concatenate(([0],cp))    


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataversion", default="pass6", help="version of the data set (e.g. pass6)")
    parser.add_argument("--datatype", default="ISS", help="title of the input data (e.g. ISS)")
    parser.add_argument("--tracknumber", default="single", help="number of tracks (single or multiple)")
    parser.add_argument("--filepath", default= '/Users/yasaman/AMS02/data/TRD_templatefit_data.npz',help="Path to root file to read tree from. Can also be path to a file with a list of root files or a pattern of root files (e.g. results/ExampleAnalysisTree*.root)")
    #parser.add_argument("--filepath", default= "/Users/yasaman/AMS02/data/templatefit_sample/pass6_new/Edependent_results_identification_test.npz",help="Path to root file to read tree from. Can also be path to a file with a list of root files or a pattern of root files (e.g. results/ExampleAnalysisTree*.root)")
    args = parser.parse_args()
    
    with np.load(os.path.join(args.filepath)) as result_file:
        
        if args.tracknumber == "single":
            pEvents = result_file["pEvents_st"]
            eEvents = result_file["eEvents_st_passed"]
            ccpEvents = result_file["ccpEvents_st"]
            
        elif args.tracknumber == "multiple":
            pEvents = result_file["pEvents_mt"]
            eEvents = result_file["eEvents_mt"]
            ccpEvents = result_file["ccpEvents_mt"]
            
        TRD_binning = result_file["var1_binning"]
        Energy_binning = result_file["var2_binning"] 
        
    T= (TRD_binning[1:] + TRD_binning[:-1])/2 
    E= (Energy_binning[1:] + Energy_binning[:-1])/2 
    
    # lowlim=0
    # upplim=-1
    
    #TRD_binning = TRD_binning
        

    intitial_values=dict(n_el=20000 , n_ccp =1000, s1_el=0.06, delta_m_el=-0.1, s2_el =0.1, m2_el=0.35, t_el=0.2, s_ccp =0.15,
                   delta_m_ccp =0.45, t_ccp = 0.1 , alpha_el =0.99 , ncc =200 , npcc =100, np_el=100 , np_p =20000,
                   s1_p=0.06, delta_m_p=-0.1, s2_p =0.1, m2_p=0.35, t_p=0.2, alpha_p=0.99)  

    electron_guess = {variable: intitial_values[variable] for variable in intitial_values if variable in describe(electron_cumulative_model)}
    ccproton_guess = {variable: intitial_values[variable] for variable in intitial_values if variable in describe(ccproton_cumulative_model)} 
    proton_guess = {variable: intitial_values[variable] for variable in intitial_values if variable in describe(proton_cumulative_model)}  
    central_guess = None
    guess= {**electron_guess,**ccproton_guess}
    
   
    
    electron_parameters = {key: [None]*len(E) for key in electron_guess}
    ccproton_parameters = {key: [None]*len(E) for key in ccproton_guess}
    proton_parameters = {key: [None]*len(E) for key in proton_guess}
    
    el_chi = np.zeros((len(E))) 
    ccp_chi = np.zeros((len(E)))
    p_chi = np.zeros((len(E)))
    
    el_res = np.zeros((len(T), len(E)))
    ccp_res = np.zeros((len(T), len(E)))
    p_res = np.zeros((len(T), len(E)))

    
    
    
    for binn in list(range(len(E)//2, len(E))) + list(range(len(E)//2-1, -1, -1)): 
        counter = binn+1
        
        if binn == len(E) // 2 - 1:
            guess = central_guess
            
        
        events_electron= eEvents[: , binn]
        events_ccproton= ccpEvents[: , binn]
        events_proton= pEvents[: , binn]
        
        
        liklihood_electron = ExtendedBinnedNLL(events_electron,TRD_binning,electron_cumulative_model)
        liklihood_ccproton = ExtendedBinnedNLL(events_ccproton,TRD_binning,ccproton_cumulative_model)
        liklihood=  liklihood_electron + liklihood_ccproton 
        
        
        minuit = Minuit(liklihood, **guess)
        
            
        minuit.limits["t_ccp"]=(-0.25,0.25)
        minuit.limits["alpha_el"]=(0,1)
        minuit.limits["s1_el"]=(0.05,0.2)
        minuit.limits["s2_el"]=(0,0.2)
        minuit.limits["s_ccp"]=(0,0.3)
        minuit.limits["t_el"]=(0,1)
        minuit.limits["delta_m_ccp"]=(0,1)
        #minuit.limits["m1"]=(0.2,0.8)
        minuit.limits["delta_m_el"]=(-0.005,0)
        minuit.limits["m2_el"]=(0,1)
        minuit.limits["n_ccp"]=(0,10000)
        minuit.limits["n_el"]=(0,None)
        minuit.limits["ncc"]=(0,20000)
        minuit.limits["npcc"]=(0,None)
        
        if E[binn] > 10:
            minuit.values['alpha_el']=1
            minuit.fixed['alpha_el']=True
            #minuit.fixed["m1"]=True
            minuit.fixed["s1_el"]=True
            
        minuit.migrad()
        print(binn)
        print(minuit.valid)
        
       
        electron_ccp_fit_values = dict(zip(minuit.parameters, minuit.values)) 
        guess = electron_ccp_fit_values
        
        for parameter in electron_parameters:
            electron_parameters[parameter][binn]= electron_ccp_fit_values[parameter]
            
        for parameter in ccproton_parameters:
            ccproton_parameters[parameter][binn] = electron_ccp_fit_values[parameter]
            
        if central_guess is None:
            central_guess = electron_ccp_fit_values
              
        for parameter in set(electron_ccp_fit_values) & set(proton_guess):
            proton_guess[parameter] = electron_ccp_fit_values[parameter]
        
        electrontemplate = electron_template(T,electron_ccp_fit_values['n_el'],electron_ccp_fit_values['n_ccp'],electron_ccp_fit_values['s1_el'],electron_ccp_fit_values['delta_m_el'],electron_ccp_fit_values['s2_el'],electron_ccp_fit_values['m2_el'],electron_ccp_fit_values['t_el'],electron_ccp_fit_values['s_ccp'],electron_ccp_fit_values['delta_m_ccp'],electron_ccp_fit_values['t_ccp'],electron_ccp_fit_values['alpha_el'])  
        ccprotontemplate = ccproton_template(T,electron_ccp_fit_values["s1_el"],electron_ccp_fit_values["delta_m_el"],electron_ccp_fit_values["s2_el"],electron_ccp_fit_values["m2_el"],electron_ccp_fit_values["t_el"],electron_ccp_fit_values["s_ccp"],electron_ccp_fit_values["delta_m_ccp"],electron_ccp_fit_values["t_ccp"],electron_ccp_fit_values["alpha_el"],electron_ccp_fit_values["ncc"],electron_ccp_fit_values["npcc"])
       
        el_chi[binn]= np.sum(chisquare(eEvents[:,binn],electrontemplate))
        ccp_chi[binn]= np.sum(chisquare(ccpEvents[:,binn],ccprotontemplate))
        
        el_res[:,binn]= residuals(eEvents[:,binn], electrontemplate)
        ccp_res[:,binn]= residuals(ccpEvents[:,binn], ccprotontemplate)
        
        liklihood_proton = ExtendedBinnedNLL(events_proton,TRD_binning,proton_cumulative_model)
        minuit= Minuit(liklihood_proton, **proton_guess)
        
        
        
        minuit.fixed["s1_el"]=True
        minuit.fixed["delta_m_el"]=True
        minuit.fixed["s2_el"]=True
        minuit.fixed["m2_el"]=True
        minuit.fixed["t_el"]=True
        minuit.fixed["alpha_el"]=True
        
        
        minuit.migrad()
        print(minuit.valid)
        
        proton_fit_values = dict(zip(minuit.parameters, minuit.values)) 
        
        for parameter in proton_fit_values:
            proton_parameters[parameter][binn] = proton_fit_values[parameter]
        if binn ==10:
            print(proton_fit_values) 
            
            
        proton_guess = {variable: proton_fit_values[variable] for variable in proton_fit_values if variable in describe(proton_cumulative_model)}     
        protontemplate = proton_template(T,proton_fit_values['np_el'],proton_fit_values['np_p'],proton_fit_values['alpha_el'],proton_fit_values['s1_el'],proton_fit_values['delta_m_el'],proton_fit_values['s2_el'],proton_fit_values['m2_el'],proton_fit_values['t_el'],proton_fit_values['alpha_p'],proton_fit_values['s1_p'],proton_fit_values['delta_m_p'],proton_fit_values['s2_p'],proton_fit_values['m2_p'],proton_fit_values['t_p'])
        
        p_chi[binn]= np.sum(chisquare(pEvents[:,binn], protontemplate))
        p_res[:,binn] = residuals(pEvents[:,binn], protontemplate)
                
        
        figure = plt.figure(figsize=(12, 10))
        ax1 = figure.add_subplot(211)
        
        a = round(Energy_binning[binn],2)
        b = round(Energy_binning[binn +1],2)
        ax1.set_ylabel("Events", fontsize = 15, fontweight = 'bold')
        ax1.set_yscale("log")
        ax1.tick_params(axis='both', which="major",direction='in', length=10, width=1.5, labelsize=15)
        ax1.tick_params(axis='both', which="minor",direction='in', length=6, width=1.5, labelsize=15)  
        
        for axis in ['top','bottom','left','right']:
            ax1.spines[axis].set_linewidth(2)
            
            
        ax1.plot(T,electron_ccp_fit_values['n_el']*electron_model(T,electron_ccp_fit_values['alpha_el'],electron_ccp_fit_values['s1_el'],electron_ccp_fit_values['delta_m_el'],electron_ccp_fit_values['s2_el'],electron_ccp_fit_values['m2_el'],electron_ccp_fit_values['t_el']),label="electron",color='c')
        ax1.fill_between(T,electron_ccp_fit_values['n_el']*electron_model(T,electron_ccp_fit_values['alpha_el'],electron_ccp_fit_values['s1_el'],electron_ccp_fit_values['delta_m_el'],electron_ccp_fit_values['s2_el'],electron_ccp_fit_values['m2_el'],electron_ccp_fit_values['t_el']),facecolor='c',alpha=0.2)
        ax1.plot(T,electron_ccp_fit_values['n_ccp']*ccproton_model(T,electron_ccp_fit_values['s_ccp'],electron_ccp_fit_values['delta_m_ccp'],electron_ccp_fit_values['t_ccp'],electron_ccp_fit_values['m2_el']),label="ccproton",color='deeppink')
        ax1.fill_between(T,electron_ccp_fit_values['n_ccp']*ccproton_model(T,electron_ccp_fit_values['s_ccp'],electron_ccp_fit_values['delta_m_ccp'],electron_ccp_fit_values['t_ccp'],electron_ccp_fit_values['m2_el']),color='deeppink',facecolor="deeppink",alpha=0.2)
        
        ax1.plot(T,electron_template(T,electron_ccp_fit_values['n_el'],electron_ccp_fit_values['n_ccp'],electron_ccp_fit_values['s1_el'],electron_ccp_fit_values['delta_m_el'],electron_ccp_fit_values['s2_el'],electron_ccp_fit_values['m2_el'],electron_ccp_fit_values['t_el'],electron_ccp_fit_values['s_ccp'],electron_ccp_fit_values['delta_m_ccp'],electron_ccp_fit_values['t_ccp'],electron_ccp_fit_values['alpha_el']), label= 'fit result',linestyle='dashed',color='orange',linewidth=5)
        ax1.errorbar(T, eEvents[: , binn], np.sqrt(eEvents[: , binn]), fmt=".", markersize=16,label="data",color='k')
        #plot.errorbar([],[],fmt= ' ', label= "Energy = "+ "[" +str(a)+" , "+ str(b) +"]"+" "+r"$\rm GeV$")
        
        
        dof = len(eEvents[:,binn]) - len(electron_ccp_fit_values) 
        echi = el_chi[binn]/dof
        
        fit_info = [f"ISS data {args.dataversion}\n"
            f"Energy = [{a} , {b}]GeV \n"
            f"$\\chi^2$ / $n_\\mathrm{{dof}}$ = {echi:.1f}",]
            
        ax1.legend(title='ISS data'.join(fit_info),title_fontsize=15,fontsize=15,frameon=False,loc='upper right')
        
       
        ax1.set_xlim(0,1.8)
        ax1.set_ylim(0.6,10**5)
        
        ax2 = figure.add_subplot(212,sharex=ax1)
        
        ax2.set_ylabel("residuals", fontsize = 15, fontweight = 'bold')
        ax2.tick_params(axis='both', which="major",direction='in', length=10, width=1.5, labelsize=15)
        ax2.tick_params(axis='both', which="minor",direction='in', length=6, width=1.5, labelsize=15)  
        
        for axis in ['top','bottom','left','right']:
            ax2.spines[axis].set_linewidth(2)
        
        ax2.scatter(T,el_res[:,binn])
        ax2.hlines(0,min(T),max(T),colors='k', linestyle='--')
        ax2.set_xlabel(r'$\Lambda_{\rm TRD}$',fontsize = 20,fontweight = 'bold') 
        
        # Adjust the size of the upper subplot to be larger than the lower one
        ax1.set_position([0.1, 0.4, 0.8, 0.5])
        ax2.set_position([0.1, 0.1, 0.8, 0.3])
        
        plt.rcParams['xtick.top'] = True
        plt.rcParams['ytick.right'] = True
        plt.minorticks_on()
        plt.rcParams["font.weight"] = "bold"
        plt.rcParams["axes.labelweight"] = "bold"
        
        figure.savefig("/Users/yasaman/AMS02/plots/template_fit/TRD/electron/"+args.tracknumber+"/electron_"+ args.tracknumber +"_histogram_"+str(counter)+".pdf" , dpi=250)
        plt.close(figure) 
        
        
        
        figure = plt.figure(figsize=(12, 10))
        ax1 = figure.add_subplot(211)

        ax1.set_ylabel("Events", fontsize = 15, fontweight = 'bold')
        ax1.set_yscale("log")

        ax1.tick_params(axis='both', which="major",direction='in', length=10, width=1.5, labelsize=15)
        ax1.tick_params(axis='both', which="minor",direction='in', length=6, width=1.5, labelsize=15)
        

        for axis in ['top','bottom','left','right']:
            ax1.spines[axis].set_linewidth(2)

          
        
        ax1.plot(T,electron_ccp_fit_values['ncc']*electron_model(T,electron_ccp_fit_values['alpha_el'],electron_ccp_fit_values['s1_el'],electron_ccp_fit_values['delta_m_el'],electron_ccp_fit_values['s2_el'],electron_ccp_fit_values['m2_el'],electron_ccp_fit_values['t_el']),label="electron",color='c')
        ax1.fill_between(T,electron_ccp_fit_values['ncc']*electron_model(T,electron_ccp_fit_values['alpha_el'],electron_ccp_fit_values['s1_el'],electron_ccp_fit_values['delta_m_el'],electron_ccp_fit_values['s2_el'],electron_ccp_fit_values['m2_el'],electron_ccp_fit_values['t_el']),facecolor='c',alpha=0.2)
        ax1.plot(T,electron_ccp_fit_values['npcc']*ccproton_model(T,electron_ccp_fit_values['s_ccp'],electron_ccp_fit_values['delta_m_ccp'],electron_ccp_fit_values['t_ccp'],electron_ccp_fit_values['m2_el']),label="ccproton",color='deeppink')
        ax1.fill_between(T,electron_ccp_fit_values['npcc']*ccproton_model(T,electron_ccp_fit_values['s_ccp'],electron_ccp_fit_values['delta_m_ccp'],electron_ccp_fit_values['t_ccp'],electron_ccp_fit_values['m2_el']),color='deeppink',facecolor="deeppink",alpha=0.2)
        ax1.plot(T,ccproton_template(T,electron_ccp_fit_values["s1_el"],electron_ccp_fit_values["delta_m_el"],electron_ccp_fit_values["s2_el"],electron_ccp_fit_values["m2_el"],electron_ccp_fit_values["t_el"],electron_ccp_fit_values["s_ccp"],electron_ccp_fit_values["delta_m_ccp"],electron_ccp_fit_values["t_ccp"],electron_ccp_fit_values["alpha_el"],electron_ccp_fit_values["ncc"],electron_ccp_fit_values["npcc"]), label="fit result",linestyle='dashed',color='orange',linewidth=5)
        ax1.errorbar(T, ccpEvents[: , binn], np.sqrt(ccpEvents[: , binn]), fmt=".", markersize=16,label="data",color='k')
     

        dof = len(ccpEvents[:,binn]) - len(electron_ccp_fit_values) 
        ccpchi = ccp_chi[binn]/dof
                
        fit_info = [f"ISS data {args.dataversion}\n"
            f"Energy = [{a} , {b}]GeV \n"
            f"$\\chi^2$ / $n_\\mathrm{{dof}}$ = {ccpchi:.1f}",]
        ax1.legend(title='ISS data'.join(fit_info),title_fontsize=14,fontsize=14,frameon=False, loc='upper right')                
        # plot.set_xlim(0,1.8)
        ax1.set_ylim(0.6, 10**5)
        
        ax2 = figure.add_subplot(212,sharex=ax1) 
        ax2.set_ylabel("residuals", fontsize = 15, fontweight = 'bold')
        ax2.tick_params(axis='both', which="major",direction='in', length=10, width=1.5, labelsize=15)
        ax2.tick_params(axis='both', which="minor",direction='in', length=6, width=1.5, labelsize=15)         
        for axis in ['top','bottom','left','right']:
            ax2.spines[axis].set_linewidth(2)
        
        ax2.scatter(T,ccp_res[:,binn])
        ax2.hlines(0,min(T),max(T),colors='k', linestyle='--')
        ax2.set_xlabel(r'$\Lambda_{\rm TRD}$',fontsize = 20,fontweight = 'bold') 
        
        # Adjust the size of the upper subplot to be larger than the lower one
        ax1.set_position([0.1, 0.4, 0.8, 0.5])
        ax2.set_position([0.1, 0.1, 0.8, 0.3])
        
        plt.rcParams['xtick.top'] = True
        plt.rcParams['ytick.right'] = True
        plt.minorticks_on()
        plt.rcParams["font.weight"] = "bold"
        plt.rcParams["axes.labelweight"] = "bold"
        
        
        figure.savefig("/Users/yasaman/AMS02/plots/template_fit/TRD/ccproton/"+args.tracknumber+"/ccproton_" +args.tracknumber +"_histogram_"+str(counter)+".pdf" , dpi=250)
        plt.close(figure)
        
        
                
        figure = plt.figure(figsize=(12, 10))
        ax1 = figure.add_subplot(211)

        ax1.set_ylabel("Events", fontsize = 15, fontweight = 'bold')
        ax1.set_yscale("log")

        ax1.tick_params(axis='both', which="major",direction='in', length=10, width=1.5, labelsize=15)
        ax1.tick_params(axis='both', which="minor",direction='in', length=6, width=1.5, labelsize=15)
        

        for axis in ['top','bottom','left','right']:
            ax1.spines[axis].set_linewidth(2)
            
            
        ax1.plot(T,proton_fit_values['np_p']*proton_model(T,proton_fit_values["alpha_p"],proton_fit_values["s1_p"],proton_fit_values["delta_m_p"],proton_fit_values["s2_p"],proton_fit_values["m2_p"],proton_fit_values["t_p"]),label="proton",color='c')
        ax1.fill_between(T,proton_fit_values['np_p']*proton_model(T,proton_fit_values["alpha_p"],proton_fit_values["s1_p"],proton_fit_values["delta_m_p"],proton_fit_values["s2_p"],proton_fit_values["m2_p"],proton_fit_values["t_p"]),facecolor='c',alpha=0.2)
        ax1.plot(T,proton_fit_values['np_el']*electron_model(T,proton_fit_values['alpha_el'],proton_fit_values['s1_el'],proton_fit_values['delta_m_el'],proton_fit_values['s2_el'],proton_fit_values['m2_el'],proton_fit_values['t_el']),label="positron",color='deeppink')
        ax1.fill_between(T,proton_fit_values['np_el']*electron_model(T,proton_fit_values['alpha_el'],proton_fit_values['s1_el'],proton_fit_values['delta_m_el'],proton_fit_values['s2_el'],proton_fit_values['m2_el'],proton_fit_values['t_el']),color='deeppink',facecolor="deeppink",alpha=0.2)
        ax1.plot(T,proton_template(T,proton_fit_values['np_el'],proton_fit_values['np_p'],proton_fit_values['alpha_el'],proton_fit_values['s1_el'],proton_fit_values['delta_m_el'],proton_fit_values['s2_el'],proton_fit_values['m2_el'],proton_fit_values['t_el'],proton_fit_values['alpha_p'],proton_fit_values['s1_p'],proton_fit_values['delta_m_p'],proton_fit_values['s2_p'],proton_fit_values['m2_p'],proton_fit_values['t_p']), label="fit result",linestyle='dashed',color='orange',linewidth=5)
        ax1.errorbar(T, pEvents[: , binn], np.sqrt(pEvents[: , binn]), fmt=".", markersize=16,label="data",color='k')
       

        dof = len(pEvents[:,binn]) - len(proton_fit_values) 
        pchi = p_chi[binn]/dof
                        
        fit_info = [f"ISS data {args.dataversion}\n"
            f"Energy = [{a} , {b}]GeV \n"
            f"$\\chi^2$ / $n_\\mathrm{{dof}}$ = {pchi:.1f}",]
        ax1.legend(title='ISS data'.join(fit_info),title_fontsize=14,fontsize=14,frameon=False, loc='upper right')
        # plot.set_xlim(0,1.8)
        ax1.set_ylim(0.6, 10**8)
        
        ax2 = figure.add_subplot(212,sharex=ax1) 
        ax2.set_ylabel("residuals", fontsize = 15, fontweight = 'bold')
        ax2.tick_params(axis='both', which="major",direction='in', length=10, width=1.5, labelsize=15)
        ax2.tick_params(axis='both', which="minor",direction='in', length=6, width=1.5, labelsize=15)         
        for axis in ['top','bottom','left','right']:
            ax2.spines[axis].set_linewidth(2)
        
        ax2.scatter(T,p_res[:,binn])
        ax2.hlines(0,min(T),max(T),colors='k', linestyle='--')
        ax2.set_xlabel(r'$\Lambda_{\rm TRD}$',fontsize = 20,fontweight = 'bold') 
        
        # Adjust the size of the upper subplot to be larger than the lower one
        ax1.set_position([0.1, 0.4, 0.8, 0.5])
        ax2.set_position([0.1, 0.1, 0.8, 0.3])
        
        
        plt.rcParams['xtick.top'] = True
        plt.rcParams['ytick.right'] = True
        plt.minorticks_on()
        plt.rcParams["font.weight"] = "bold"
        plt.rcParams["axes.labelweight"] = "bold"
        
        figure.savefig("/Users/yasaman/AMS02/plots/template_fit/TRD/proton/"+args.tracknumber+"/proton_" +args.tracknumber +"_histogram_"+str(counter)+".pdf" , dpi=250)
        plt.close(figure)
        
            
   
    np.savez("/Users/yasaman/AMS02/plots/template_fit/TRD/proton/proton_parameters_"+args.tracknumber+".npz", **proton_parameters)
    np.savez("/Users/yasaman/AMS02/plots/template_fit/TRD/electron/electron_parameters_"+args.tracknumber+".npz", **electron_parameters)
    np.savez("/Users/yasaman/AMS02/plots/template_fit/TRD/ccproton/ccproton_parameters_"+args.tracknumber+".npz", **ccproton_parameters)
        
if __name__ == "__main__":
    main()         
        
        

        
        
        
        
