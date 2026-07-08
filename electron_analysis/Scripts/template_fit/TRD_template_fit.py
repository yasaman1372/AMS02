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
    #return (1- alpha_el)*G + alpha_el*N
    return N

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
    # return (1- alpha_p)*G + alpha_p*N
    return N

def electron_cumulative_model(edges,n_el,n_ccp,s1_el,delta_m_el,s2_el,m2_el,t_el,s_ccp,delta_m_ccp,t_ccp,alpha_el):
    p = n_el*electron_model(edges,alpha_el,s1_el,delta_m_el,s2_el,m2_el,t_el) + n_ccp*ccproton_model(edges,s_ccp,delta_m_ccp,t_ccp,m2_el)
    cp = np.cumsum(p)
    return cp


def ccproton_cumulative_model(edges,s1_el,delta_m_el,s2_el,m2_el,t_el,s_ccp,delta_m_ccp,t_ccp,alpha_el,ncc,npcc):
    p = ncc*electron_model(edges,alpha_el,s1_el,delta_m_el,s2_el,m2_el,t_el) + npcc*ccproton_model(edges,s_ccp,delta_m_ccp,t_ccp,m2_el)
    cp = np.cumsum(p)
    return cp
    

def proton_cumulative_model(edges,np_el,np_p,alpha_el,s1_el,delta_m_el,s2_el,m2_el,t_el,alpha_p,s1_p,delta_m_p,s2_p,m2_p,t_p):
    p = np_el*electron_model(edges,alpha_el,s1_el,delta_m_el,s2_el,m2_el,t_el) + np_p*proton_model(edges,alpha_p,s1_p,delta_m_p,s2_p,m2_p,t_p)
    cp = np.cumsum(p)
    return cp 

def passed_electron_cumulative_model(edges,alpha_el,s1_el,delta_m_el,s2_el,m2_el,t_el,s_ccp,delta_m_ccp,t_ccp):
    el = electron_model(edges,alpha_el,s1_el,delta_m_el,s2_el,m2_el,t_el)
    ccp = ccproton_model(edges,s_ccp,delta_m_ccp,t_ccp,m2_el)
    el_temp = el.cumsum()/el.sum()
    ccp_temp =ccp.cumsum()/ccp.sum()
    
    def passed_eff_model(edges,ne,nccp,e_sig,e_back):      
        return ne* e_sig * el_temp + nccp * e_back * ccp_temp
    return passed_eff_model

def failed_electron_cumulative_model(edges,alpha_el,s1_el,delta_m_el,s2_el,m2_el,t_el,s_ccp,delta_m_ccp,t_ccp):
    el = electron_model(edges,alpha_el,s1_el,delta_m_el,s2_el,m2_el,t_el) 
    ccp = ccproton_model(edges,s_ccp,delta_m_ccp,t_ccp,m2_el)
    el_temp = el.cumsum()/el.sum()
    ccp_temp =ccp.cumsum()/ccp.sum()
    
    def failed_eff_model(edges,ne,nccp,e_sig,e_back):      
        return ne * (1 - e_sig) * el_temp + nccp * (1 - e_back) * ccp_temp 
    return failed_eff_model

###for plotting
def passed_electron_cumulative_model_values(edges,alpha_el,s1_el,delta_m_el,s2_el,m2_el,t_el,s_ccp,delta_m_ccp,t_ccp,ne,nccp,e_sig,e_back):
    el = electron_model(edges,alpha_el,s1_el,delta_m_el,s2_el,m2_el,t_el)
    ccp = ccproton_model(edges,s_ccp,delta_m_ccp,t_ccp,m2_el)
    el_temp = el.cumsum()/el.sum()
    ccp_temp =ccp.cumsum()/ccp.sum()
    return ne* e_sig * el_temp + nccp * e_back * ccp_temp
  
#### for plotting
def failed_electron_cumulative_model_values(edges,alpha_el,s1_el,delta_m_el,s2_el,m2_el,t_el,s_ccp,delta_m_ccp,t_ccp,ne,nccp,e_sig,e_back):
    el = electron_model(edges,alpha_el,s1_el,delta_m_el,s2_el,m2_el,t_el) 
    ccp = ccproton_model(edges,s_ccp,delta_m_ccp,t_ccp,m2_el)
    el_temp = el.cumsum()/el.sum()
    ccp_temp =ccp.cumsum()/ccp.sum()     
    return ne * (1 - e_sig) * el_temp + nccp * (1 - e_back) * ccp_temp 


def electron_template(x,n_el,n_ccp,s1_el,delta_m_el,s2_el,m2_el,t_el,s_ccp,delta_m_ccp,t_ccp,alpha_el):
    return (electron_cumulative_model(x[1:],n_el,n_ccp,s1_el,delta_m_el,s2_el,m2_el,t_el,s_ccp,delta_m_ccp,t_ccp,alpha_el)
            - electron_cumulative_model(x[:-1],n_el,n_ccp,s1_el,delta_m_el,s2_el,m2_el,t_el,s_ccp,delta_m_ccp,t_ccp,alpha_el))

def ccproton_template(x,s1_el,delta_m_el,s2_el,m2_el,t_el,s_ccp,delta_m_ccp,t_ccp,alpha_el,ncc,npcc):
    return (ccproton_cumulative_model(x[1:],s1_el,delta_m_el,s2_el,m2_el,t_el,s_ccp,delta_m_ccp,t_ccp,alpha_el,ncc,npcc)
            - ccproton_cumulative_model(x[:-1],s1_el,delta_m_el,s2_el,m2_el,t_el,s_ccp,delta_m_ccp,t_ccp,alpha_el,ncc,npcc))

def proton_template(x,np_el,np_p,alpha_el,s1_el,delta_m_el,s2_el,m2_el,t_el,alpha_p,s1_p,delta_m_p,s2_p,m2_p,t_p):
    return (proton_cumulative_model(x[1:],np_el,np_p,alpha_el,s1_el,delta_m_el,s2_el,m2_el,t_el,alpha_p,s1_p,delta_m_p,s2_p,m2_p,t_p)
            - proton_cumulative_model(x[:-1],np_el,np_p,alpha_el,s1_el,delta_m_el,s2_el,m2_el,t_el,alpha_p,s1_p,delta_m_p,s2_p,m2_p,t_p))

def passed_electron_template(x,alpha_el,s1_el,delta_m_el,s2_el,m2_el,t_el,s_ccp,delta_m_ccp,t_ccp,ne,nccp,e_sig,e_back):
    return (passed_electron_cumulative_model_values(x[1:],alpha_el,s1_el,delta_m_el,s2_el,m2_el,t_el,s_ccp,delta_m_ccp,t_ccp,ne,nccp,e_sig,e_back)
            - passed_electron_cumulative_model_values(x[:-1],alpha_el,s1_el,delta_m_el,s2_el,m2_el,t_el,s_ccp,delta_m_ccp,t_ccp,ne,nccp,e_sig,e_back))

def failed_electron_template(x,alpha_el,s1_el,delta_m_el,s2_el,m2_el,t_el,s_ccp,delta_m_ccp,t_ccp,ne,nccp,e_sig,e_back):
    return (failed_electron_cumulative_model_values(x[1:],alpha_el,s1_el,delta_m_el,s2_el,m2_el,t_el,s_ccp,delta_m_ccp,t_ccp,ne,nccp,e_sig,e_back)
            - failed_electron_cumulative_model_values(x[:-1],alpha_el,s1_el,delta_m_el,s2_el,m2_el,t_el,s_ccp,delta_m_ccp,t_ccp,ne,nccp,e_sig,e_back))
    
####################################################################################################################################################################################################
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataversion", default="pass8", help="version of the data set (e.g. pass6)")
    parser.add_argument("--datatype", default="ISS", help="title of the input data (e.g. ISS)")
    parser.add_argument("--tracknumber", default="single", help="number of tracks (single or multiple)")
    parser.add_argument("--filepath", default="/Users/yasaman/AMS02/data/pass8/Ecal3DBDT/corrected/TRD_templatefit_data.npz",help="Path to root file to read tree from. Can also be path to a file with a list of root files or a pattern of root files (e.g. results/ExampleAnalysisTree*.root)")
    args = parser.parse_args()
    os.makedirs("results", exist_ok = True)
    os.makedirs("results/electron/"+args.tracknumber, exist_ok = True)
    os.makedirs("results/proton/"+args.tracknumber, exist_ok = True)
    os.makedirs("results/ccproton/"+args.tracknumber, exist_ok = True)
    os.makedirs("results/electron/efficiency/"+args.tracknumber, exist_ok = True)

    with np.load(os.path.join(args.filepath)) as result_file:
        
        if args.tracknumber == "single":
            pEvents = result_file["pEvents_st"]
            eEvents_passed = result_file["eEvents_st_passed"]
            eEvents_failed = result_file["eEvents_st_failed"]
            ccpEvents = result_file["ccpEvents_st"]
            
        elif args.tracknumber == "multiple":
            pEvents = result_file["pEvents_mt"]
            eEvents_passed = result_file["eEvents_mt_passed"]
            eEvents_failed = result_file["eEvents_mt_failed"]
            ccpEvents = result_file["ccpEvents_mt"]
            
        TRD_binning = result_file["var1_binning"]
        Energy_binning = result_file["var2_binning"] 
        
    T= (TRD_binning[1:] + TRD_binning[:-1])/2 
    E= (Energy_binning[1:] + Energy_binning[:-1])/2 
    
        

    intitial_values=dict(n_el=20000 , n_ccp =1000, s1_el=0.06, delta_m_el=0.2, s2_el =0.06, m2_el=0.5, t_el=0.2, s_ccp =0.07,
                   delta_m_ccp =0.45, t_ccp = 0.1 , alpha_el =0.5 , ncc =200 , npcc =100, np_el=100 , np_p =20000,
                   s1_p=0.06, delta_m_p=-0.1, s2_p =0.1, m2_p=0.35, t_p=0.2, alpha_p=0.99) 
    
    eff_guess = dict(ne = 20000, nccp = 1040, e_sig=0.9 ,e_back=0.1)
    
    electron_guess = {variable: intitial_values[variable] for variable in intitial_values if variable in describe(electron_cumulative_model)}
    ccproton_guess = {variable: intitial_values[variable] for variable in intitial_values if variable in describe(ccproton_cumulative_model)} 
    proton_guess = {variable: intitial_values[variable] for variable in intitial_values if variable in describe(proton_cumulative_model)}  
    central_guess_el = None
    central_guess_pr = None
    guess= {**electron_guess,**ccproton_guess}
    
   
    
    electron_parameters = {key: [None]*len(E) for key in electron_guess}
    ccproton_parameters = {key: [None]*len(E) for key in ccproton_guess}
    proton_parameters = {key: [None]*len(E) for key in proton_guess}
    ecal_eff_parameters = {key: [None]*len(E) for key in eff_guess}
    
    
    
    el_chi = np.zeros((len(E))) 
    ccp_chi = np.zeros((len(E)))
    p_chi = np.zeros((len(E)))
    
    el_passed_chi = np.zeros((len(E)))
    el_failed_chi = np.zeros((len(E)))    
    
    
    el_res = np.zeros((len(T), len(E)))
    ccp_res = np.zeros((len(T), len(E)))
    p_res = np.zeros((len(T), len(E)))
    
    el_passed_res = np.zeros((len(T), len(E)))
    el_failed_res = np.zeros((len(T), len(E)))

    
    
    
    for binn in list(range(len(E)//2, len(E))) + list(range(len(E)//2-1, -1, -1)): 
        # counter = binn+1
                
        if binn == len(E) // 2 - 1:
            guess = central_guess_el
        
        if binn == len(E) // 2 - 1:
            proton_guess = central_guess_pr    
            
        events_electron_passed = eEvents_passed[: , binn]
        events_electron_failed = eEvents_failed[: , binn]
        events_electron= eEvents_passed[: , binn]
        events_ccproton= ccpEvents[: , binn]
        events_proton= pEvents[: , binn]
        
        ##############################################################################################################
        ################# ELECTRON CCPROTON FIT
        
        liklihood_electron = ExtendedBinnedNLL(events_electron,TRD_binning,electron_cumulative_model)
        liklihood_ccproton = ExtendedBinnedNLL(events_ccproton,TRD_binning,ccproton_cumulative_model)
        liklihood=  liklihood_electron + liklihood_ccproton 
        
        
        minuit = Minuit(liklihood, **guess)
        
            
        minuit.limits["t_ccp"]=(-0.2,0.25)
        minuit.limits["alpha_el"]=(0,1)
        minuit.limits["s1_el"]=(0.05,0.2)
        minuit.limits["s2_el"]=(0,0.2)
        minuit.limits["s_ccp"]=(0.05,0.3)
        minuit.limits["t_el"]=(0,0.4)
        
        if args.tracknumber == "multiple":
            minuit.limits["delta_m_ccp"]=(0.3,1)
        elif args.tracknumber == "single":
            minuit.limits["delta_m_ccp"]=(0.2,1)            
        #minuit.limits["m1"]=(0.2,0.8)
        minuit.limits["delta_m_el"]=(-0.2,0.2)
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
        print("binn:",binn)
        print("electron:", minuit.valid)
        
       
        electron_ccp_fit_values = dict(zip(minuit.parameters, minuit.values)) 
        guess = electron_ccp_fit_values
        
        for parameter in electron_parameters:
            electron_parameters[parameter][binn]= electron_ccp_fit_values[parameter]
            
        for parameter in ccproton_parameters:
            ccproton_parameters[parameter][binn] = electron_ccp_fit_values[parameter]
            
        if central_guess_el is None:
            central_guess_el = electron_ccp_fit_values
              
        for parameter in set(electron_ccp_fit_values) & set(proton_guess):
            proton_guess[parameter] = electron_ccp_fit_values[parameter]
        
        electrontemplate = electron_template(TRD_binning,electron_ccp_fit_values['n_el'],electron_ccp_fit_values['n_ccp'],electron_ccp_fit_values['s1_el'],electron_ccp_fit_values['delta_m_el'],electron_ccp_fit_values['s2_el'],electron_ccp_fit_values['m2_el'],electron_ccp_fit_values['t_el'],electron_ccp_fit_values['s_ccp'],electron_ccp_fit_values['delta_m_ccp'],electron_ccp_fit_values['t_ccp'],electron_ccp_fit_values['alpha_el'])  
        ccprotontemplate = ccproton_template(TRD_binning,electron_ccp_fit_values["s1_el"],electron_ccp_fit_values["delta_m_el"],electron_ccp_fit_values["s2_el"],electron_ccp_fit_values["m2_el"],electron_ccp_fit_values["t_el"],electron_ccp_fit_values["s_ccp"],electron_ccp_fit_values["delta_m_ccp"],electron_ccp_fit_values["t_ccp"],electron_ccp_fit_values["alpha_el"],electron_ccp_fit_values["ncc"],electron_ccp_fit_values["npcc"])
        
        
        el_dof = len(eEvents_passed[:,binn]) - len(electron_ccp_fit_values) 
        el_chi[binn]= np.sum(chisquare(eEvents_passed[:,binn],electrontemplate))/el_dof
        
        ccp_dof = len(ccpEvents[:,binn]) - len(electron_ccp_fit_values) 
        ccp_chi[binn]= np.sum(chisquare(ccpEvents[:,binn],ccprotontemplate))/ccp_dof
        
        el_res[:,binn]= residuals(eEvents_passed[:,binn], electrontemplate)
        ccp_res[:,binn]= residuals(ccpEvents[:,binn], ccprotontemplate)
        
        
        ###################################################################################################
        ##### EFFICIENCY FIT
        passed_el_cumulative_model = passed_electron_cumulative_model(TRD_binning,electron_ccp_fit_values['alpha_el'],electron_ccp_fit_values['s1_el'],electron_ccp_fit_values['delta_m_el'],
                                                                      electron_ccp_fit_values['s2_el'],electron_ccp_fit_values['m2_el'],electron_ccp_fit_values['t_el'],
                                                                      electron_ccp_fit_values['s_ccp'],electron_ccp_fit_values['delta_m_ccp'],electron_ccp_fit_values['t_ccp'])
        
        failed_el_cumulative_model = failed_electron_cumulative_model(TRD_binning,electron_ccp_fit_values['alpha_el'],electron_ccp_fit_values['s1_el'],electron_ccp_fit_values['delta_m_el'],
                                                                      electron_ccp_fit_values['s2_el'],electron_ccp_fit_values['m2_el'],electron_ccp_fit_values['t_el'],
                                                                      electron_ccp_fit_values['s_ccp'],electron_ccp_fit_values['delta_m_ccp'],electron_ccp_fit_values['t_ccp'])
        
        liklihood_passed_electron = ExtendedBinnedNLL(events_electron_passed, TRD_binning, passed_el_cumulative_model)
        liklihood_failed_electron =  ExtendedBinnedNLL(events_electron_failed, TRD_binning, failed_el_cumulative_model)
        
        eff_liklihood = liklihood_passed_electron + liklihood_failed_electron    
        eff_minuit = Minuit(eff_liklihood, **eff_guess)
        
        eff_minuit.migrad()
        print("efficiency: ", eff_minuit.valid)   
        
        eff_fit_values = dict(zip(eff_minuit.parameters, eff_minuit.values)) 
        eff_guess = eff_fit_values
        
        for parameter in ecal_eff_parameters:
            ecal_eff_parameters[parameter][binn]= eff_fit_values[parameter]
            
        
        passed_electrontemplate = passed_electron_template(TRD_binning,electron_ccp_fit_values["alpha_el"],electron_ccp_fit_values["s1_el"],
                                                           electron_ccp_fit_values["delta_m_el"],electron_ccp_fit_values["s2_el"],
                                                           electron_ccp_fit_values["m2_el"],electron_ccp_fit_values["t_el"],
                                                           electron_ccp_fit_values["s_ccp"],electron_ccp_fit_values["delta_m_ccp"],
                                                           electron_ccp_fit_values["t_ccp"],eff_fit_values["ne"],
                                                           eff_fit_values["nccp"],eff_fit_values["e_sig"],
                                                           eff_fit_values["e_back"])
        
        failed_electrontemplate = failed_electron_template(TRD_binning,electron_ccp_fit_values["alpha_el"],electron_ccp_fit_values["s1_el"],
                                                           electron_ccp_fit_values["delta_m_el"],electron_ccp_fit_values["s2_el"],
                                                           electron_ccp_fit_values["m2_el"],electron_ccp_fit_values["t_el"],
                                                           electron_ccp_fit_values["s_ccp"],electron_ccp_fit_values["delta_m_ccp"],
                                                           electron_ccp_fit_values["t_ccp"],eff_fit_values["ne"],
                                                           eff_fit_values["nccp"],eff_fit_values["e_sig"],
                                                           eff_fit_values["e_back"])
        
        el_passed_dof = len(eEvents_passed[:,binn]) - len(eff_fit_values)
        el_failed_dof =len(eEvents_failed[:,binn]) - len(eff_fit_values)
        
        el_passed_chi[binn] = np.sum(chisquare(eEvents_passed[:,binn],passed_electrontemplate))/el_passed_dof
        el_failed_chi[binn] = np.sum(chisquare(eEvents_failed[:,binn],failed_electrontemplate))/el_failed_dof
        
        el_passed_res[:,binn]= residuals(eEvents_passed[:,binn], passed_electrontemplate)
        el_failed_res[:,binn]= residuals(eEvents_failed[:,binn], failed_electrontemplate)
        
        
        #######################################################################################################
        liklihood_proton = ExtendedBinnedNLL(events_proton,TRD_binning,proton_cumulative_model)
        minuit= Minuit(liklihood_proton, **proton_guess)
            
        
        minuit.fixed["s1_el"]=True
        minuit.fixed["delta_m_el"]=True
        minuit.fixed["s2_el"]=True
        minuit.fixed["m2_el"]=True
        minuit.fixed["t_el"]=True
        minuit.fixed["alpha_el"]=True
        
        minuit.limits["t_p"]=(-0.4,0.4)
        minuit.migrad()
        print("proton: ",minuit.valid)
        
        proton_fit_values = dict(zip(minuit.parameters, minuit.values)) 
        
        for parameter in proton_fit_values:
            proton_parameters[parameter][binn] = proton_fit_values[parameter]
        # if binn ==10:
        #     print(proton_fit_values) 
            
            
        proton_guess = {variable: proton_fit_values[variable] for variable in proton_fit_values if variable in describe(proton_cumulative_model)}     
        protontemplate = proton_template(TRD_binning,proton_fit_values['np_el'],proton_fit_values['np_p'],proton_fit_values['alpha_el'],proton_fit_values['s1_el'],proton_fit_values['delta_m_el'],proton_fit_values['s2_el'],proton_fit_values['m2_el'],proton_fit_values['t_el'],proton_fit_values['alpha_p'],proton_fit_values['s1_p'],proton_fit_values['delta_m_p'],proton_fit_values['s2_p'],proton_fit_values['m2_p'],proton_fit_values['t_p'])
        
        if central_guess_pr is None:
            central_guess_pr = proton_guess
        
        p_dof = len(pEvents[:,binn]) - len(proton_fit_values)
        p_chi[binn]= np.sum(chisquare(pEvents[:,binn], protontemplate))/p_dof
        p_res[:,binn] = residuals(pEvents[:,binn], protontemplate)
                
       ################################################################################################ 
        figure = plt.figure(figsize=(10, 10))
        ax1 = figure.add_subplot(211)
        
        a = round(Energy_binning[binn],2)
        b = round(Energy_binning[binn +1],2)
        ax1.set_ylabel("Events", fontsize = 15, fontweight = 'bold')
        ax1.set_yscale("log")
        ax1.tick_params(axis='both', which="major",direction='in', length=10, width=1.5, labelsize=15)
        ax1.tick_params(axis='both', which="minor",direction='in', length=6, width=1.5, labelsize=15) 
        ax1.set_xticklabels([])
        
        for axis in ['top','bottom','left','right']:
            ax1.spines[axis].set_linewidth(2)
            
  
        
        ax1.plot(T,electron_template(TRD_binning,electron_ccp_fit_values['n_el'],0,electron_ccp_fit_values['s1_el'],electron_ccp_fit_values['delta_m_el'],electron_ccp_fit_values['s2_el'],electron_ccp_fit_values['m2_el'],electron_ccp_fit_values['t_el'],electron_ccp_fit_values['s_ccp'],electron_ccp_fit_values['delta_m_ccp'],electron_ccp_fit_values['t_ccp'],electron_ccp_fit_values['alpha_el']) ,label="electron",color='c')
        # ax1.plot(T,guess['alpha_el']*electron_template(TRD_binning,guess['n_el'],0,guess['s1_el'],guess['delta_m_el'],guess['s2_el'],guess['m2_el'],guess['t_el'],guess['s_ccp'],guess['delta_m_ccp'],guess['t_ccp'],1) ,label="N model",color='k')
        # ax1.plot(T,(1-guess['alpha_el'])*electron_template(TRD_binning,guess['n_el'],0,guess['s1_el'],guess['delta_m_el'],guess['s2_el'],guess['m2_el'],guess['t_el'],guess['s_ccp'],guess['delta_m_ccp'],guess['t_ccp'],0) ,label="G model",color='m')
        ax1.fill_between(T,electron_template(TRD_binning,electron_ccp_fit_values['n_el'],0,electron_ccp_fit_values['s1_el'],electron_ccp_fit_values['delta_m_el'],electron_ccp_fit_values['s2_el'],electron_ccp_fit_values['m2_el'],electron_ccp_fit_values['t_el'],electron_ccp_fit_values['s_ccp'],electron_ccp_fit_values['delta_m_ccp'],electron_ccp_fit_values['t_ccp'],electron_ccp_fit_values['alpha_el']) ,facecolor='c',alpha=0.2)
        ax1.plot(T,electron_template(TRD_binning,0,electron_ccp_fit_values['n_ccp'],electron_ccp_fit_values['s1_el'],electron_ccp_fit_values['delta_m_el'],electron_ccp_fit_values['s2_el'],electron_ccp_fit_values['m2_el'],electron_ccp_fit_values['t_el'],electron_ccp_fit_values['s_ccp'],electron_ccp_fit_values['delta_m_ccp'],electron_ccp_fit_values['t_ccp'],electron_ccp_fit_values['alpha_el']) ,label="ccproton",color='deeppink')
        ax1.fill_between(T,electron_template(TRD_binning,0,electron_ccp_fit_values['n_ccp'],electron_ccp_fit_values['s1_el'],electron_ccp_fit_values['delta_m_el'],electron_ccp_fit_values['s2_el'],electron_ccp_fit_values['m2_el'],electron_ccp_fit_values['t_el'],electron_ccp_fit_values['s_ccp'],electron_ccp_fit_values['delta_m_ccp'],electron_ccp_fit_values['t_ccp'],electron_ccp_fit_values['alpha_el']) ,color='deeppink',facecolor="deeppink",alpha=0.2)
        ax1.plot(T,electron_template(TRD_binning,electron_ccp_fit_values['n_el'],electron_ccp_fit_values['n_ccp'],electron_ccp_fit_values['s1_el'],electron_ccp_fit_values['delta_m_el'],electron_ccp_fit_values['s2_el'],electron_ccp_fit_values['m2_el'],electron_ccp_fit_values['t_el'],electron_ccp_fit_values['s_ccp'],electron_ccp_fit_values['delta_m_ccp'],electron_ccp_fit_values['t_ccp'],electron_ccp_fit_values['alpha_el']) , label= 'fit result',linestyle='dashed',color='orange',linewidth=5)
        ax1.errorbar(T, eEvents_passed[: , binn], np.sqrt(eEvents_passed[: , binn]), fmt=".", markersize=16,label="data",color='k')

        
        # dof = len(eEvents_passed[:,binn]) - len(electron_ccp_fit_values) 
        # echi = el_chi[binn]/dof
        
        fit_info = [
            rf"$\mu_{{\rm el, novo}}$ = {electron_ccp_fit_values['m2_el']:.2f}",
            f"$\\tau_{{\\rm el, novo}}$ = {electron_ccp_fit_values['t_el']:.2f}",
            f"$\\sigma_{{\\rm el, novo}}$ = {electron_ccp_fit_values['s2_el']:.2f}",
            f"$\\epsilon_{{\\rm sig}}$ = {eff_fit_values['e_sig']:.2f}",
            f"$\\epsilon_{{\\rm back}}$ = {eff_fit_values['e_back']:.2f}",
            f"$\\chi^2 / n_\\mathrm{{dof}}$ = {el_chi[binn]:.2f}",]
            
        ax1.text(0.05,0.95,f"ISS data {args.dataversion}\nEnergy range = [{a} , {b}]GeV \n", transform=ax1.transAxes, va="top", fontsize=15)
        ax1.legend(title="\n".join(fit_info),title_fontsize=15,fontsize=15,frameon=False,loc='upper right')
        
       
        ax1.set_xlim(0,1.8)
        ax1.set_ylim(0.6,max(eEvents_passed[: , binn])*10)
        
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
        
        figure.savefig("results/electron/"+args.tracknumber+"/electron_"+ args.tracknumber +"_histogram_"+str(binn)+".pdf" , dpi=250)
        plt.close(figure) 
        
        
        ######################################################################################################
        figure = plt.figure(figsize=(10, 10))
        ax1 = figure.add_subplot(211)

        ax1.set_ylabel("Events", fontsize = 15, fontweight = 'bold')
        ax1.set_yscale("log")

        ax1.tick_params(axis='both', which="major",direction='in', length=10, width=1.5, labelsize=15)
        ax1.tick_params(axis='both', which="minor",direction='in', length=6, width=1.5, labelsize=15)
        

        for axis in ['top','bottom','left','right']:
            ax1.spines[axis].set_linewidth(2)

          
        
        ax1.plot(T,ccproton_template(TRD_binning,electron_ccp_fit_values["s1_el"],electron_ccp_fit_values["delta_m_el"],electron_ccp_fit_values["s2_el"],electron_ccp_fit_values["m2_el"],electron_ccp_fit_values["t_el"],electron_ccp_fit_values["s_ccp"],electron_ccp_fit_values["delta_m_ccp"],electron_ccp_fit_values["t_ccp"],electron_ccp_fit_values["alpha_el"],0,electron_ccp_fit_values["npcc"]),label="ccproton",color='c')
        ax1.fill_between(T,ccproton_template(TRD_binning,electron_ccp_fit_values["s1_el"],electron_ccp_fit_values["delta_m_el"],electron_ccp_fit_values["s2_el"],electron_ccp_fit_values["m2_el"],electron_ccp_fit_values["t_el"],electron_ccp_fit_values["s_ccp"],electron_ccp_fit_values["delta_m_ccp"],electron_ccp_fit_values["t_ccp"],electron_ccp_fit_values["alpha_el"],0,electron_ccp_fit_values["npcc"]),facecolor='c',alpha=0.2)
        ax1.plot(T,ccproton_template(TRD_binning,electron_ccp_fit_values["s1_el"],electron_ccp_fit_values["delta_m_el"],electron_ccp_fit_values["s2_el"],electron_ccp_fit_values["m2_el"],electron_ccp_fit_values["t_el"],electron_ccp_fit_values["s_ccp"],electron_ccp_fit_values["delta_m_ccp"],electron_ccp_fit_values["t_ccp"],electron_ccp_fit_values["alpha_el"],electron_ccp_fit_values["ncc"],0),label="electron",color='deeppink')
        ax1.fill_between(T,ccproton_template(TRD_binning,electron_ccp_fit_values["s1_el"],electron_ccp_fit_values["delta_m_el"],electron_ccp_fit_values["s2_el"],electron_ccp_fit_values["m2_el"],electron_ccp_fit_values["t_el"],electron_ccp_fit_values["s_ccp"],electron_ccp_fit_values["delta_m_ccp"],electron_ccp_fit_values["t_ccp"],electron_ccp_fit_values["alpha_el"],electron_ccp_fit_values["ncc"],0),color='deeppink',facecolor="deeppink",alpha=0.2)
        ax1.plot(T,ccproton_template(TRD_binning,electron_ccp_fit_values["s1_el"],electron_ccp_fit_values["delta_m_el"],electron_ccp_fit_values["s2_el"],electron_ccp_fit_values["m2_el"],electron_ccp_fit_values["t_el"],electron_ccp_fit_values["s_ccp"],electron_ccp_fit_values["delta_m_ccp"],electron_ccp_fit_values["t_ccp"],electron_ccp_fit_values["alpha_el"],electron_ccp_fit_values["ncc"],electron_ccp_fit_values["npcc"]), label="fit result",linestyle='dashed',color='orange',linewidth=5)
        ax1.errorbar(T, ccpEvents[: , binn], np.sqrt(ccpEvents[: , binn]), fmt=".", markersize=16,label="data",color='k')
     

        # dof = len(ccpEvents[:,binn]) - len(electron_ccp_fit_values) 
        # ccpchi = ccp_chi[binn]/dof
        
        m_ccp=electron_ccp_fit_values["m2_el"]+electron_ccp_fit_values["delta_m_ccp"]
        
        fit_info = [
            rf"$\mu_{{\rm ccp, novo}}$ = {m_ccp:.2f}",
            f"$\\tau_{{\\rm ccp, novo}}$ = {electron_ccp_fit_values['t_ccp']:.2f}",
            f"$\\sigma_{{\\rm ccp, novo}}$ = {electron_ccp_fit_values['s_ccp']:.2f}",
            f"$\\chi^2 / n_\\mathrm{{dof}}$ = {ccp_chi[binn]:.2f}",]
            
        ax1.text(0.05,0.95,f"ISS data {args.dataversion}\nEnergy range = [{a} , {b}]GeV \n", transform=ax1.transAxes, va="top", fontsize=15)
        ax1.legend(title="\n".join(fit_info),title_fontsize=15,fontsize=15,frameon=False,loc='upper right')
                                
        # plot.set_xlim(0,1.8)
        ax1.set_ylim(0.6, max(ccpEvents[: , binn])*10)
        
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
        
        
        figure.savefig("results/ccproton/"+args.tracknumber+"/ccproton_" +args.tracknumber +"_histogram_"+str(binn)+".pdf" , dpi=250)
        plt.close(figure)
        
        
        ####################################################################################################        
        figure = plt.figure(figsize=(10, 10))
        ax1 = figure.add_subplot(211)

        ax1.set_ylabel("Events", fontsize = 15, fontweight = 'bold')
        ax1.set_yscale("log")

        ax1.tick_params(axis='both', which="major",direction='in', length=10, width=1.5, labelsize=15)
        ax1.tick_params(axis='both', which="minor",direction='in', length=6, width=1.5, labelsize=15)
        

        for axis in ['top','bottom','left','right']:
            ax1.spines[axis].set_linewidth(2)
            
            
        ax1.plot(T,proton_template(TRD_binning,0,proton_fit_values['np_p'],proton_fit_values['alpha_el'],proton_fit_values['s1_el'],proton_fit_values['delta_m_el'],proton_fit_values['s2_el'],proton_fit_values['m2_el'],proton_fit_values['t_el'],proton_fit_values['alpha_p'],proton_fit_values['s1_p'],proton_fit_values['delta_m_p'],proton_fit_values['s2_p'],proton_fit_values['m2_p'],proton_fit_values['t_p']),label="proton",color='c')
        ax1.fill_between(T,proton_template(TRD_binning,0,proton_fit_values['np_p'],proton_fit_values['alpha_el'],proton_fit_values['s1_el'],proton_fit_values['delta_m_el'],proton_fit_values['s2_el'],proton_fit_values['m2_el'],proton_fit_values['t_el'],proton_fit_values['alpha_p'],proton_fit_values['s1_p'],proton_fit_values['delta_m_p'],proton_fit_values['s2_p'],proton_fit_values['m2_p'],proton_fit_values['t_p']),facecolor='c',alpha=0.2)
        ax1.plot(T,proton_template(TRD_binning,proton_fit_values['np_el'],0,proton_fit_values['alpha_el'],proton_fit_values['s1_el'],proton_fit_values['delta_m_el'],proton_fit_values['s2_el'],proton_fit_values['m2_el'],proton_fit_values['t_el'],proton_fit_values['alpha_p'],proton_fit_values['s1_p'],proton_fit_values['delta_m_p'],proton_fit_values['s2_p'],proton_fit_values['m2_p'],proton_fit_values['t_p']),label="positron",color='deeppink')
        ax1.fill_between(T,proton_template(TRD_binning,proton_fit_values['np_el'],0,proton_fit_values['alpha_el'],proton_fit_values['s1_el'],proton_fit_values['delta_m_el'],proton_fit_values['s2_el'],proton_fit_values['m2_el'],proton_fit_values['t_el'],proton_fit_values['alpha_p'],proton_fit_values['s1_p'],proton_fit_values['delta_m_p'],proton_fit_values['s2_p'],proton_fit_values['m2_p'],proton_fit_values['t_p']),color='deeppink',facecolor="deeppink",alpha=0.2)
        ax1.plot(T,proton_template(TRD_binning,proton_fit_values['np_el'],proton_fit_values['np_p'],proton_fit_values['alpha_el'],proton_fit_values['s1_el'],proton_fit_values['delta_m_el'],proton_fit_values['s2_el'],proton_fit_values['m2_el'],proton_fit_values['t_el'],proton_fit_values['alpha_p'],proton_fit_values['s1_p'],proton_fit_values['delta_m_p'],proton_fit_values['s2_p'],proton_fit_values['m2_p'],proton_fit_values['t_p']), label="fit result",linestyle='dashed',color='orange',linewidth=5)
        ax1.errorbar(T, pEvents[: , binn], np.sqrt(pEvents[: , binn]), fmt=".", markersize=16,label="data",color='k')
       

        #dof = len(pEvents[:,binn]) - len(proton_fit_values) 
        #pchi = p_chi[binn]/dof
        
        fit_info = [
            rf"$\mu_{{\rm p, novo}}$ = {proton_fit_values['m2_p']:.2f}",
            f"$\\tau_{{\\rm p, novo}}$ = {electron_ccp_fit_values['t_el']:.2f}",
            f"$\\sigma_{{\\rm p, novo}}$ = {proton_fit_values['s2_p']:.2f}",
            f"$\\chi^2 / n_\\mathrm{{dof}}$ = {p_chi[binn]:.2f}",]
            
        ax1.text(0.05,0.95,f"ISS data {args.dataversion}\nEnergy range = [{a} , {b}]GeV \n", transform=ax1.transAxes, va="top", fontsize=15)
        ax1.legend(title="\n".join(fit_info),title_fontsize=15,fontsize=15,frameon=False,loc='upper right')
                        
        # fit_info = [f"ISS data {args.dataversion}\n"
        #     f"Energy = [{a} , {b}]GeV \n"
        #     f"$\\chi^2$ / $n_\\mathrm{{dof}}$ = {p_chi[binn]:.1f}",]
        # ax1.legend(title='ISS data'.join(fit_info),title_fontsize=14,fontsize=14,frameon=False, loc='upper right')
        # plot.set_xlim(0,1.8)
        ax1.set_ylim(0.6, max(pEvents[: , binn])*10)
        
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
        
        figure.savefig("results/proton/"+args.tracknumber+"/proton_" +args.tracknumber +"_histogram_"+str(binn)+".pdf" , dpi=250)
        plt.close(figure)
        
        ########################################################################################################
        figure = plt.figure(figsize=(12, 10))
        ax1 = figure.add_subplot(212)

        ax1.set_ylabel("Events", fontsize = 15, fontweight = 'bold')
        ax1.set_yscale("log")
        
        ax1.set_xlabel(r'$\Lambda_{\rm TRD}$', fontsize = 15, fontweight = 'bold')

        ax1.tick_params(axis='both', which="major",direction='in', length=10, width=1.5, labelsize=15)
        ax1.tick_params(axis='both', which="minor",direction='in', length=6, width=1.5, labelsize=15)
        

        for axis in ['top','bottom','left','right']:
            ax1.spines[axis].set_linewidth(2)
            
            
        ax1.plot(T,passed_electron_template(TRD_binning,electron_ccp_fit_values["alpha_el"],electron_ccp_fit_values["s1_el"],
                                                           electron_ccp_fit_values["delta_m_el"],electron_ccp_fit_values["s2_el"],
                                                           electron_ccp_fit_values["m2_el"],electron_ccp_fit_values["t_el"],
                                                           electron_ccp_fit_values["s_ccp"],electron_ccp_fit_values["delta_m_ccp"],
                                                           electron_ccp_fit_values["t_ccp"],eff_fit_values["ne"],
                                                           0,eff_fit_values["e_sig"],
                                                           eff_fit_values["e_back"]),label="electron",color='c')
        
        ax1.fill_between(T,passed_electron_template(TRD_binning,electron_ccp_fit_values["alpha_el"],electron_ccp_fit_values["s1_el"],
                                                           electron_ccp_fit_values["delta_m_el"],electron_ccp_fit_values["s2_el"],
                                                           electron_ccp_fit_values["m2_el"],electron_ccp_fit_values["t_el"],
                                                           electron_ccp_fit_values["s_ccp"],electron_ccp_fit_values["delta_m_ccp"],
                                                           electron_ccp_fit_values["t_ccp"],eff_fit_values["ne"],
                                                           0,eff_fit_values["e_sig"],
                                                           eff_fit_values["e_back"]),facecolor='c',alpha=0.2)
        
        ax1.plot(T,passed_electron_template(TRD_binning,electron_ccp_fit_values["alpha_el"],electron_ccp_fit_values["s1_el"],
                                                           electron_ccp_fit_values["delta_m_el"],electron_ccp_fit_values["s2_el"],
                                                           electron_ccp_fit_values["m2_el"],electron_ccp_fit_values["t_el"],
                                                           electron_ccp_fit_values["s_ccp"],electron_ccp_fit_values["delta_m_ccp"],
                                                           electron_ccp_fit_values["t_ccp"],0,
                                                           eff_fit_values["nccp"],eff_fit_values["e_sig"],
                                                           eff_fit_values["e_back"]),label="p-",color='deeppink')
        
        ax1.fill_between(T,passed_electron_template(TRD_binning,electron_ccp_fit_values["alpha_el"],electron_ccp_fit_values["s1_el"],
                                                           electron_ccp_fit_values["delta_m_el"],electron_ccp_fit_values["s2_el"],
                                                           electron_ccp_fit_values["m2_el"],electron_ccp_fit_values["t_el"],
                                                           electron_ccp_fit_values["s_ccp"],electron_ccp_fit_values["delta_m_ccp"],
                                                           electron_ccp_fit_values["t_ccp"],0,
                                                           eff_fit_values["nccp"],eff_fit_values["e_sig"],
                                                           eff_fit_values["e_back"]),color='deeppink',facecolor="deeppink",alpha=0.2)
        
        ax1.plot(T, passed_electrontemplate, label="fit result",linestyle='dashed',color='orange',linewidth=5)
        ax1.errorbar(T, eEvents_passed[: , binn], np.sqrt(eEvents_passed[: , binn]), fmt=".", markersize=16,label="data",color='k')
        # ax1.legend()
       
        #dof = len(eEvents_passed[:,binn]) - len(eff_fit_values) 
        # eff_chi_passed = el_passed_chi[binn]/dof
                        
        fit_info = [f"ISS data {args.dataversion}\n"
            f"Energy = [{a} , {b}]GeV \n"
            f"$\\chi^2$ / $n_\\mathrm{{dof}}$ = {el_passed_chi[binn]:.1f}",]
        ax1.legend(title='ISS data'.join(fit_info),title_fontsize=14,fontsize=14,frameon=False, loc='upper right')
        # plot.set_xlim(0,1.8)
        ax1.set_ylim(0.6, max(eEvents_passed[: , binn])*10)
        
        ax2 = figure.add_subplot(212,sharex=ax1) 
        ax2.set_ylabel("residuals", fontsize = 15, fontweight = 'bold')
        ax2.tick_params(axis='both', which="major",direction='in', length=10, width=1.5, labelsize=15)
        ax2.tick_params(axis='both', which="minor",direction='in', length=6, width=1.5, labelsize=15)         
        for axis in ['top','bottom','left','right']:
            ax2.spines[axis].set_linewidth(2)
        
        ax2.scatter(T,el_passed_res[:,binn])
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
        
        figure.savefig("results/electron/efficiency/"+args.tracknumber+"/passed_data_" +args.tracknumber +"_histogram_"+str(binn)+".pdf" , dpi=250)
        plt.close(figure)
        
        #############################################################################################
        figure = plt.figure(figsize=(12, 10))
        ax1 = figure.add_subplot(211)

        ax1.set_ylabel("Events", fontsize = 15, fontweight = 'bold')
        ax1.set_yscale("log")
        
        ax1.set_xlabel(r'$\Lambda_{\rm TRD}$', fontsize = 15, fontweight = 'bold')

        ax1.tick_params(axis='both', which="major",direction='in', length=10, width=1.5, labelsize=15)
        ax1.tick_params(axis='both', which="minor",direction='in', length=6, width=1.5, labelsize=15)
        

        for axis in ['top','bottom','left','right']:
            ax1.spines[axis].set_linewidth(2)
            
            
        ax1.plot(T,failed_electron_template(TRD_binning,electron_ccp_fit_values["alpha_el"],electron_ccp_fit_values["s1_el"],
                                                                   electron_ccp_fit_values["delta_m_el"],electron_ccp_fit_values["s2_el"],
                                                                   electron_ccp_fit_values["m2_el"],electron_ccp_fit_values["t_el"],
                                                                   electron_ccp_fit_values["s_ccp"],electron_ccp_fit_values["delta_m_ccp"],
                                                                   electron_ccp_fit_values["t_ccp"],eff_fit_values["ne"],
                                                                   0,eff_fit_values["e_sig"],
                                                                   eff_fit_values["e_back"]),label="electron",color='c')
        
        ax1.fill_between(T,failed_electron_template(TRD_binning,electron_ccp_fit_values["alpha_el"],electron_ccp_fit_values["s1_el"],
                                                                   electron_ccp_fit_values["delta_m_el"],electron_ccp_fit_values["s2_el"],
                                                                   electron_ccp_fit_values["m2_el"],electron_ccp_fit_values["t_el"],
                                                                   electron_ccp_fit_values["s_ccp"],electron_ccp_fit_values["delta_m_ccp"],
                                                                   electron_ccp_fit_values["t_ccp"],eff_fit_values["ne"],
                                                                   0,eff_fit_values["e_sig"],
                                                                   eff_fit_values["e_back"]),facecolor='c',alpha=0.2)
        
        ax1.plot(T,failed_electron_template(TRD_binning,electron_ccp_fit_values["alpha_el"],electron_ccp_fit_values["s1_el"],
                                                                   electron_ccp_fit_values["delta_m_el"],electron_ccp_fit_values["s2_el"],
                                                                   electron_ccp_fit_values["m2_el"],electron_ccp_fit_values["t_el"],
                                                                   electron_ccp_fit_values["s_ccp"],electron_ccp_fit_values["delta_m_ccp"],
                                                                   electron_ccp_fit_values["t_ccp"],0,
                                                                   eff_fit_values["nccp"],eff_fit_values["e_sig"],
                                                                   eff_fit_values["e_back"]),label="ccproton",color='deeppink')
        
        ax1.fill_between(T,failed_electron_template(TRD_binning,electron_ccp_fit_values["alpha_el"],electron_ccp_fit_values["s1_el"],
                                                                   electron_ccp_fit_values["delta_m_el"],electron_ccp_fit_values["s2_el"],
                                                                   electron_ccp_fit_values["m2_el"],electron_ccp_fit_values["t_el"],
                                                                   electron_ccp_fit_values["s_ccp"],electron_ccp_fit_values["delta_m_ccp"],
                                                                   electron_ccp_fit_values["t_ccp"],0,
                                                                   eff_fit_values["nccp"],eff_fit_values["e_sig"],
                                                                   eff_fit_values["e_back"]),color='deeppink',facecolor="deeppink",alpha=0.2)
        
        ax1.plot(T, failed_electrontemplate,label="fit result",linestyle='dashed',color='orange',linewidth=5)
        ax1.errorbar(T, eEvents_failed[: , binn], np.sqrt(eEvents_failed[: , binn]), fmt=".", markersize=16,label="data",color='k')
        # ax1.legend()
       
        # dof = len(eEvents_failed[:,binn]) - len(eff_fit_values) 
        # eff_chi_failed = el_failed_chi[binn]/dof
                        
        fit_info = [
            f"$\\chi^2$ / $n_\\mathrm{{dof}}$ = {el_failed_chi[binn]:.1f}",]
        
        ax1.text(0.05,0.95,f"ISS data {args.dataversion}\nEnergy range = [{a} , {b}]GeV \n", transform=ax1.transAxes, va="top", fontsize=15)
        ax1.legend(title="\n".join(fit_info),title_fontsize=15,fontsize=15,frameon=False,loc='upper right')
        
        # ax1.legend(title='ISS data'.join(fit_info),title_fontsize=14,fontsize=14,frameon=False, loc='upper right')
        # plot.set_xlim(0,1.8)
        ax1.set_ylim(0.6, max(eEvents_failed[: , binn])*10)
        
        ax2 = figure.add_subplot(212,sharex=ax1) 
        ax2.set_ylabel("residuals", fontsize = 15, fontweight = 'bold')
        ax2.tick_params(axis='both', which="major",direction='in', length=10, width=1.5, labelsize=15)
        ax2.tick_params(axis='both', which="minor",direction='in', length=6, width=1.5, labelsize=15)         
        for axis in ['top','bottom','left','right']:
            ax2.spines[axis].set_linewidth(2)
        
        ax2.scatter(T,el_failed_res[:,binn])
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
        
        
        figure.savefig("results/electron/efficiency/"+args.tracknumber+"/failed_data_" +args.tracknumber +"_histogram_"+str(binn)+".pdf" , dpi=250)
        plt.close(figure)
               
  
  
  
  
  ################################################################################################################# 
  
    figure = plt.figure(figsize=(12, 10))
    ax1 = figure.add_subplot(111)
    ax1.set_ylabel(r"$\chi^{2}/n_{\rm dof}$", fontsize = 15, fontweight = 'bold')        
    ax1.set_xlabel("Energy (GeV)", fontsize = 15, fontweight = 'bold')

    ax1.tick_params(axis='both', which="major",direction='in', length=10, width=1.5, labelsize=15)
    ax1.tick_params(axis='both', which="minor",direction='in', length=6, width=1.5, labelsize=15)
            
    for axis in ['top','bottom','left','right']:
        ax1.spines[axis].set_linewidth(2)    
    ax1.scatter(E,el_chi)    
    
    plt.rcParams['xtick.top'] = True
    plt.rcParams['ytick.right'] = True
    plt.minorticks_on()
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    
    figure.savefig("results/electron/"+args.tracknumber+"/el_goodness_of_fits.pdf" , dpi=250)
    plt.close(figure)     
   
   #################################################################################################################  
   
    figure = plt.figure(figsize=(12, 10))
    ax1 = figure.add_subplot(111)
    ax1.set_ylabel(r"$\chi^{2}/n_{\rm dof}$", fontsize = 15, fontweight = 'bold')        
    ax1.set_xlabel("Energy (GeV)", fontsize = 15, fontweight = 'bold')

    ax1.tick_params(axis='both', which="major",direction='in', length=10, width=1.5, labelsize=15)
    ax1.tick_params(axis='both', which="minor",direction='in', length=6, width=1.5, labelsize=15)
            
    for axis in ['top','bottom','left','right']:
        ax1.spines[axis].set_linewidth(2)    
    ax1.scatter(E,ccp_chi)    
    
    plt.rcParams['xtick.top'] = True
    plt.rcParams['ytick.right'] = True
    plt.minorticks_on()
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    
    figure.savefig("results/ccproton/"+args.tracknumber+"/ccp_goodness_of_fits.pdf" , dpi=250)
    plt.close(figure)     
   
   #################################################################################################################
   
    figure = plt.figure(figsize=(12, 10))
    ax1 = figure.add_subplot(111)
    ax1.set_ylabel(r"$\chi^{2}/n_{\rm dof}$", fontsize = 15, fontweight = 'bold')        
    ax1.set_xlabel("Energy (GeV)", fontsize = 15, fontweight = 'bold')

    ax1.tick_params(axis='both', which="major",direction='in', length=10, width=1.5, labelsize=15)
    ax1.tick_params(axis='both', which="minor",direction='in', length=6, width=1.5, labelsize=15)
            
    for axis in ['top','bottom','left','right']:
        ax1.spines[axis].set_linewidth(2)    
    ax1.scatter(E,p_chi)    
    
    plt.rcParams['xtick.top'] = True
    plt.rcParams['ytick.right'] = True
    plt.minorticks_on()
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    
    figure.savefig("results/proton/"+args.tracknumber+"/el_goodness_of_fits.pdf" , dpi=250)
    plt.close(figure) 
    
   #################################################################################################################
   
    figure = plt.figure(figsize=(12, 10))
    ax1 = figure.add_subplot(111)
    ax1.set_ylabel(r"$\chi^{2}/n_{\rm dof}$", fontsize = 15, fontweight = 'bold')        
    ax1.set_xlabel("Energy (GeV)", fontsize = 15, fontweight = 'bold')

    ax1.tick_params(axis='both', which="major",direction='in', length=10, width=1.5, labelsize=15)
    ax1.tick_params(axis='both', which="minor",direction='in', length=6, width=1.5, labelsize=15)
            
    for axis in ['top','bottom','left','right']:
        ax1.spines[axis].set_linewidth(2)    
    ax1.scatter(E,el_passed_chi)    
    
    plt.rcParams['xtick.top'] = True
    plt.rcParams['ytick.right'] = True
    plt.minorticks_on()
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    
    figure.savefig("results/electron/efficiency/"+args.tracknumber+"/passed_el_goodness_of_fits.pdf" , dpi=250)
    plt.close(figure) 
    
  #####################################################################################################################    

    figure = plt.figure(figsize=(12, 10))
    ax1 = figure.add_subplot(111)
    ax1.set_ylabel(r"$\chi^{2}/n_{\rm dof}$", fontsize = 15, fontweight = 'bold')        
    ax1.set_xlabel("Energy (GeV)", fontsize = 15, fontweight = 'bold')

    ax1.tick_params(axis='both', which="major",direction='in', length=10, width=1.5, labelsize=15)
    ax1.tick_params(axis='both', which="minor",direction='in', length=6, width=1.5, labelsize=15)
            
    for axis in ['top','bottom','left','right']:
        ax1.spines[axis].set_linewidth(2)    
    ax1.scatter(E,el_failed_chi)    
    
    plt.rcParams['xtick.top'] = True
    plt.rcParams['ytick.right'] = True
    plt.minorticks_on()
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    
    figure.savefig("results/electron/efficiency/"+args.tracknumber+"/failed_el_goodness_of_fits.pdf" , dpi=250)
    plt.close(figure)    
   #######################################################################################################################
   
    np.savez("results/proton/proton_parameters_"+args.tracknumber+".npz", **proton_parameters)
    np.savez("results/electron/electron_parameters_"+args.tracknumber+".npz", **electron_parameters)
    np.savez("results/ccproton/ccproton_parameters_"+args.tracknumber+".npz", **ccproton_parameters)
    np.savez("results/electron/eff_parameters_"+args.tracknumber+".npz", **ecal_eff_parameters)
    
        
if __name__ == "__main__":
    main()         
        
        

        
        
        
        
