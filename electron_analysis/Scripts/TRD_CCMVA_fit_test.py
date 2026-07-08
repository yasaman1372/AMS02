#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  4 13:05:42 2023

@author: yasaman
"""

from iminuit import Minuit
import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit
from iminuit.cost import ExtendedBinnedNLL
from scipy.integrate import quad
from scipy.interpolate import interp1d
from scipy.stats import chisquare
from iminuit.util import describe
import matplotlib.colors as colors
import os

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


def make_combined_template(TRD_binning, CCMVA_binning, signed_trd_binning,
                           proton_trd_template, electron_trd_template, ccp_trd_template,
                           CCMVA_pTemplate, CCMVA_eTemplate, CCMVA_ccpTemplate, CCMVA_cceTemplate):
    
    positive_electron_trd_template = np.concatenate((np.zeros(len(TRD_binning) - 1), electron_trd_template))
    positive_electron_trd_template = positive_electron_trd_template.cumsum() / positive_electron_trd_template.sum()
    positive_electron_trd_template = interp1d(signed_trd_binning, positive_electron_trd_template, fill_value="extrapolate")
    
    negative_electron_trd_template = np.concatenate((electron_trd_template[::-1], np.zeros(len(TRD_binning) - 1)))
    negative_electron_trd_template = negative_electron_trd_template.cumsum() / negative_electron_trd_template.sum()
    negative_electron_trd_template = interp1d(signed_trd_binning, negative_electron_trd_template, fill_value="extrapolate")
    
    
    positive_proton_trd_template = np.concatenate((np.zeros(len(TRD_binning) - 1), proton_trd_template))
    positive_proton_trd_template = positive_proton_trd_template.cumsum() / positive_proton_trd_template.sum()  
    positive_proton_trd_template = interp1d(signed_trd_binning, positive_proton_trd_template, fill_value="extrapolate")
    
    
    negative_proton_trd_template = np.concatenate((ccp_trd_template[::-1], np.zeros(len(TRD_binning) - 1)))
    negative_proton_trd_template = negative_proton_trd_template.cumsum() / negative_proton_trd_template.sum()
    negative_proton_trd_template = interp1d(signed_trd_binning, negative_proton_trd_template, fill_value="extrapolate")
    
    
    CCMVA_pTemplate = interp1d(CCMVA_binning[1:], CCMVA_pTemplate.cumsum()/CCMVA_pTemplate.sum(), fill_value="extrapolate")
    CCMVA_eTemplate = interp1d(CCMVA_binning[1:], CCMVA_eTemplate.cumsum()/CCMVA_eTemplate.sum(), fill_value="extrapolate")
    CCMVA_ccpTemplate = interp1d(CCMVA_binning[1:], CCMVA_ccpTemplate.cumsum()/CCMVA_ccpTemplate.sum(), fill_value="extrapolate")
    CCMVA_cceTemplate = interp1d(CCMVA_binning[1:], CCMVA_cceTemplate.cumsum()/CCMVA_cceTemplate.sum(), fill_value="extrapolate")
    
    def combined_template(edges, nel, npos, npr, fecc, nccp):
        r_trd_edges = edges[0,:]
        ccmva_edges = edges[1,:]
        #print(r_trd_edges.shape, ccmva_edges.shape)
        
        return (nel * (1 - fecc) * negative_electron_trd_template(r_trd_edges) * CCMVA_eTemplate(ccmva_edges)
                + nel*fecc * positive_electron_trd_template(r_trd_edges) * CCMVA_cceTemplate(ccmva_edges)
                + npos*(1-fecc) * positive_electron_trd_template(r_trd_edges) * CCMVA_eTemplate(ccmva_edges)
                + npos*fecc * negative_electron_trd_template(r_trd_edges) * CCMVA_cceTemplate(ccmva_edges)
                + npr * positive_proton_trd_template(r_trd_edges) * CCMVA_pTemplate(ccmva_edges)
                + nccp * negative_proton_trd_template(r_trd_edges) * CCMVA_ccpTemplate(ccmva_edges))
    return combined_template
    
    

def combined_model(TRD_binning, proton_trd_template, electron_trd_template, ccp_trd_template,
                           CCMVA_pTemplate, CCMVA_eTemplate, CCMVA_ccpTemplate, CCMVA_cceTemplate, nel, npos, npr, fecc, nccp):
    
    positive_electron_trd_template = np.concatenate((np.zeros(len(TRD_binning) - 1), electron_trd_template))
    positive_electron_trd_template = positive_electron_trd_template / positive_electron_trd_template.sum()
    positive_electron_trd_template = positive_electron_trd_template[:,None]
    
    negative_electron_trd_template = np.concatenate((electron_trd_template[::-1], np.zeros(len(TRD_binning) - 1)))
    negative_electron_trd_template = negative_electron_trd_template / negative_electron_trd_template.sum()
    negative_electron_trd_template = negative_electron_trd_template[:,None]
    
    
    positive_proton_trd_template = np.concatenate((np.zeros(len(TRD_binning) - 1), proton_trd_template))
    positive_proton_trd_template = positive_proton_trd_template / positive_proton_trd_template.sum() 
    positive_proton_trd_template = positive_proton_trd_template[:,None]
      
    negative_proton_trd_template = np.concatenate((ccp_trd_template[::-1], np.zeros(len(TRD_binning) - 1)))
    negative_proton_trd_template = negative_proton_trd_template / negative_proton_trd_template.sum()
    negative_proton_trd_template =negative_proton_trd_template[:,None]

        
    CCMVA_pTemplate = (CCMVA_pTemplate/CCMVA_pTemplate.sum())[None,:]
    CCMVA_ccpTemplate = (CCMVA_ccpTemplate/ CCMVA_cceTemplate.sum())[None,:]
    CCMVA_cceTemplate = (CCMVA_cceTemplate/ CCMVA_cceTemplate.sum())[None,:]
    CCMVA_eTemplate = (CCMVA_eTemplate/ CCMVA_eTemplate.sum())[None,:]
        
        
    return (nel * (1 - fecc) * negative_electron_trd_template * CCMVA_eTemplate
               + nel*fecc * positive_electron_trd_template * CCMVA_cceTemplate
               + npos*(1-fecc) * positive_electron_trd_template * CCMVA_eTemplate
               + npos*fecc * negative_electron_trd_template * CCMVA_cceTemplate
               + npr * positive_proton_trd_template * CCMVA_pTemplate
               + nccp * negative_proton_trd_template* CCMVA_ccpTemplate)



def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataversion", default="pass6", help="version of the data set (e.g. pass6)")
    parser.add_argument("--datatype", default="ISS", help="title of the input data (e.g. ISS)")
    parser.add_argument("--tracknumber", default="multiple", help="number of tracks (single or multiple)")
    parser.add_argument("--TRDtemplate_filepath",default="/Users/yasaman/AMS02/plots/template_fit/TRD/" ,help="")
    parser.add_argument("--CCMVAtemplate_filepath_ISS",default="/Users/yasaman/AMS02/data/templatefit_sample/ElectronCCMVBDTTemplateFitISS6/CCMVABDT_template_ISS.npz" ,help="")
    parser.add_argument("--CCMVAtemplate_filepath_MC",default="/Users/yasaman/AMS02/data/templatefit_sample/ElectronCCMVBDTTemplateFitMC6/CCMVABDT_template_MC.npz" ,help="")
    parser.add_argument("--data_filepath",default="/Users/yasaman/AMS02/plots/template_fit/TRD-CCMVABDT/results.npz" ,help="")
    args = parser.parse_args()
    
    
    #load TRD template parameters from 1d fits
    TRD_pParameters = np.load(args.TRDtemplate_filepath + "/proton/proton_parameters_"+args.tracknumber+".npz")
    TRD_eParameters =np.load(args.TRDtemplate_filepath + "electron/electron_parameters_"+args.tracknumber+".npz")
    TRD_ccpParameters =np.load(args.TRDtemplate_filepath + "ccproton/ccproton_parameters_"+args.tracknumber+".npz")


     
    #load CCMVABDT 1d histograms      
    with np.load(os.path.join(args.CCMVAtemplate_filepath_ISS)) as result_file:
        
        if args.tracknumber == "single":   
            CCMVA_pTemplate = result_file["pEvents_st"]
            CCMVA_eTemplate = result_file["eEvents_st"]
            CCMVA_ccpTemplate = result_file["ccpEvents_st"]
                
        elif args.tracknumber == "multiple" :
            CCMVA_pTemplate = result_file["pEvents_mt"]
            CCMVA_eTemplate = result_file["eEvents_mt"]
            CCMVA_ccpTemplate = result_file["ccpEvents_mt"] 
            
    with np.load(os.path.join(args.CCMVAtemplate_filepath_MC)) as result_file:
            
        if args.tracknumber == "single":   
            CCMVA_cceTemplate = result_file["cc_eEvents_st"] 
            
        elif args.tracknumber == "multiple" :
            CCMVA_cceTemplate = result_file["cc_eEvents_mt"] 
       
                
     # load 2d histograms of data           
    with np.load(os.path.join(args.data_filepath)) as result_file:  
        TRD_binning = result_file["var1_binning"]
        CCMVA_binning = result_file["var2_binning"] 
        Energy_binning = result_file["var3_binning"]
        Rigidity_binning = result_file["var4_binning"]
                
        if args.tracknumber == "single": 
            Events = result_file["Events_st"]
        
        elif args.tracknumber == "multiple" :
            Events = result_file["Events_mt"]
            
    signed_trd_binning = np.concatenate((-TRD_binning[:0:-1], TRD_binning))
    
    guess = dict(nel=1000, npos=100, npr=100000, fecc=0.2, nccp =100)
    
    nel = 700
    npos= 40
    npr =200000
    fecc=0.05
    nccp =300
    
    parameters = {key: [] for key in guess}
    
    i=0
    for binn in range(len(Energy_binning) -1):
        
        i=i+1
        
        a = round(Energy_binning[binn],2)
        b = round(Energy_binning[binn +1],2)
        
        proton_trd_template =  proton_model(TRD_binning,TRD_pParameters['alpha_p'][binn],TRD_pParameters['s1_p'][binn],TRD_pParameters['delta_m_p'][binn],TRD_pParameters['s2_p'][binn],
                                            TRD_pParameters['m2_p'][binn],TRD_pParameters['t_p'][binn])
        
        electron_trd_template = electron_model(TRD_binning,TRD_eParameters['alpha_el'][binn],TRD_eParameters['s1_el'][binn],
                                               TRD_eParameters['delta_m_el'][binn],TRD_eParameters['s2_el'][binn],TRD_eParameters['m2_el'][binn],
                                               TRD_eParameters['t_el'][binn])
        
        ccp_trd_template = ccproton_model(TRD_binning,TRD_ccpParameters['s_ccp'][binn],TRD_ccpParameters["delta_m_ccp"][binn],
                                          TRD_ccpParameters["t_ccp"][binn],TRD_ccpParameters["m2_el"][binn])
    
        testdata = combined_model((TRD_binning[1:]+TRD_binning[:-1])/2, proton_trd_template, electron_trd_template, ccp_trd_template,
                                   CCMVA_pTemplate[:,binn], CCMVA_eTemplate[:,binn], CCMVA_ccpTemplate[:,binn], CCMVA_cceTemplate[:,binn], nel, npos, npr, fecc, nccp)
        
        template = make_combined_template(TRD_binning, CCMVA_binning, signed_trd_binning,
                                            proton_trd_template, electron_trd_template, ccp_trd_template,
                                            CCMVA_pTemplate[:,binn], CCMVA_eTemplate[:,binn], CCMVA_ccpTemplate[:,binn], CCMVA_cceTemplate[:,binn])
        
        # print(np.amax(testdata))   
        # print(template)
        # print(TRD_binning, CCMVA_binning, signed_trd_binning,
        #                                     proton_trd_template, electron_trd_template, ccp_trd_template,
        #                                     CCMVA_pTemplate[:,binn], CCMVA_eTemplate[:,binn], CCMVA_ccpTemplate[:,binn], CCMVA_cceTemplate[:,binn])
        liklihood = ExtendedBinnedNLL(testdata, [signed_trd_binning, CCMVA_binning], template)
        
        minuit = Minuit(liklihood, **guess)
        minuit.migrad()
        print(minuit.values)
        print(minuit.valid)



if __name__ == "__main__":
    main()                          
           


