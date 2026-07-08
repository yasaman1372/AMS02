#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 13:58:16 2023

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
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import SymLogNorm, LogNorm
from matplotlib.gridspec import GridSpec
import os

def chisquare(d,m):
    err=np.maximum(m,1)
    return ((d-m)**2)/err

def residuals(d,m):
    err=np.maximum(m,1)
    return (d-m)/np.sqrt(err)

# Analytical models for TRD estimator template fit

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
    #return (1- alpha_p)*G + alpha_p*N
    return N


#combined commulative model for the template fit

def make_combined_template(TRD_binning, CCMVA_binning, signed_trd_binning,
                           proton_trd_template, electron_trd_template, ccp_trd_template,
                           CCMVA_pTemplate, CCMVA_eTemplate, CCMVA_ccpTemplate, CCMVA_cceTemplate):
    ##TRD templates:
        
        #positron and charged confused electron
    positive_electron_trd_template = np.concatenate((np.zeros(len(TRD_binning) - 1), electron_trd_template))
    positive_electron_trd_template = positive_electron_trd_template.cumsum() / np.maximum(positive_electron_trd_template.sum(),1)
    positive_electron_trd_template = interp1d(signed_trd_binning, positive_electron_trd_template, fill_value="extrapolate")
    
        #electron and charge confused positron
    negative_electron_trd_template = np.concatenate((electron_trd_template[::-1], np.zeros(len(TRD_binning) - 1)))
    negative_electron_trd_template = negative_electron_trd_template.cumsum() / np.maximum(negative_electron_trd_template.sum(),1)
    negative_electron_trd_template = interp1d(signed_trd_binning, negative_electron_trd_template, fill_value="extrapolate")
    
        #proton
    positive_proton_trd_template = np.concatenate((np.zeros(len(TRD_binning) - 1), proton_trd_template))
    positive_proton_trd_template = positive_proton_trd_template.cumsum() / np.maximum(positive_proton_trd_template.sum(),1)  
    positive_proton_trd_template = interp1d(signed_trd_binning, positive_proton_trd_template, fill_value="extrapolate")
    
        #antiproton and charged confused proton
    negative_proton_trd_template = np.concatenate((ccp_trd_template[::-1], np.zeros(len(TRD_binning) - 1)))
    negative_proton_trd_template = negative_proton_trd_template.cumsum() / np.maximum(negative_proton_trd_template.sum(),1)
    negative_proton_trd_template = interp1d(signed_trd_binning, negative_proton_trd_template, fill_value="extrapolate")
    
    ##CCMVA templates:
    CCMVA_pTemplate = interp1d(CCMVA_binning[1:], CCMVA_pTemplate.cumsum()/np.maximum(CCMVA_pTemplate.sum(),1), fill_value="extrapolate")
    CCMVA_eTemplate = interp1d(CCMVA_binning[1:], CCMVA_eTemplate.cumsum()/np.maximum(CCMVA_eTemplate.sum(),1), fill_value="extrapolate")
    CCMVA_ccpTemplate = interp1d(CCMVA_binning[1:], CCMVA_ccpTemplate.cumsum()/np.maximum(CCMVA_ccpTemplate.sum(),1), fill_value="extrapolate")
    CCMVA_cceTemplate = interp1d(CCMVA_binning[1:], CCMVA_cceTemplate.cumsum()/np.maximum(CCMVA_cceTemplate.sum(),1), fill_value="extrapolate")
    
    def combined_template(edges, nel, npos, npr, fecc, nccp):
        r_trd_edges = edges[0,:]
        ccmva_edges = edges[1,:]
        #print(r_trd_edges.shape, ccmva_edges.shape)
        
        return (nel * (1 - fecc) * negative_electron_trd_template(r_trd_edges) * CCMVA_eTemplate(ccmva_edges) #electron
                + nel*fecc * positive_electron_trd_template(r_trd_edges) * CCMVA_cceTemplate(ccmva_edges)  #charged confused electron
                + npos*(1-fecc) * positive_electron_trd_template(r_trd_edges) * CCMVA_eTemplate(ccmva_edges) #positron
                + npos*fecc * negative_electron_trd_template(r_trd_edges) * CCMVA_cceTemplate(ccmva_edges) #charge confused positron
                + npr * positive_proton_trd_template(r_trd_edges) * CCMVA_pTemplate(ccmva_edges) #proton
                + nccp * negative_proton_trd_template(r_trd_edges) * CCMVA_ccpTemplate(ccmva_edges)) #charge confused proton
    
    return combined_template    
    

def make_combined_template_all_track(TRD_binning, CCMVA_binning, signed_trd_binning,
                           proton_trd_template_st, electron_trd_template_st, ccp_trd_template_st,
                           CCMVA_pTemplate_st, CCMVA_eTemplate_st, CCMVA_ccpTemplate_st, CCMVA_cceTemplate_st,
                           proton_trd_template_mt, electron_trd_template_mt, ccp_trd_template_mt,
                           CCMVA_pTemplate_mt, CCMVA_eTemplate_mt, CCMVA_ccpTemplate_mt, CCMVA_cceTemplate_mt):
    
    ##TRD templates:
        
        #positron and charged confused electron
    positive_electron_trd_template_st = np.concatenate((np.zeros(len(TRD_binning) - 1), electron_trd_template_st))
    positive_electron_trd_template_st = positive_electron_trd_template_st.cumsum() / np.maximum(positive_electron_trd_template_st.sum(),1)
    positive_electron_trd_template_st = interp1d(signed_trd_binning, positive_electron_trd_template_st, fill_value="extrapolate")
    
    positive_electron_trd_template_mt = np.concatenate((np.zeros(len(TRD_binning) - 1), electron_trd_template_mt))
    positive_electron_trd_template_mt = positive_electron_trd_template_mt.cumsum() / np.maximum(positive_electron_trd_template_mt.sum(),1)
    positive_electron_trd_template_mt = interp1d(signed_trd_binning, positive_electron_trd_template_mt, fill_value="extrapolate")
    
        #electron and charge confused positron
    negative_electron_trd_template_st = np.concatenate((electron_trd_template_st[::-1], np.zeros(len(TRD_binning) - 1)))
    negative_electron_trd_template_st = negative_electron_trd_template_st.cumsum() / np.maximum(negative_electron_trd_template_st.sum(),1)
    negative_electron_trd_template_st = interp1d(signed_trd_binning, negative_electron_trd_template_st, fill_value="extrapolate")
    
    negative_electron_trd_template_mt = np.concatenate((electron_trd_template_mt[::-1], np.zeros(len(TRD_binning) - 1)))
    negative_electron_trd_template_mt = negative_electron_trd_template_mt.cumsum() / np.maximum(negative_electron_trd_template_mt.sum(),1)
    negative_electron_trd_template_mt = interp1d(signed_trd_binning, negative_electron_trd_template_mt, fill_value="extrapolate")
    
        #proton
    positive_proton_trd_template_st = np.concatenate((np.zeros(len(TRD_binning) - 1), proton_trd_template_st))
    positive_proton_trd_template_st = positive_proton_trd_template_st.cumsum() / np.maximum(positive_proton_trd_template_st.sum(),1)  
    positive_proton_trd_template_st = interp1d(signed_trd_binning, positive_proton_trd_template_st, fill_value="extrapolate")
    
    positive_proton_trd_template_mt = np.concatenate((np.zeros(len(TRD_binning) - 1), proton_trd_template_mt))
    positive_proton_trd_template_mt = positive_proton_trd_template_mt.cumsum() / np.maximum(positive_proton_trd_template_mt.sum(),1)  
    positive_proton_trd_template_mt = interp1d(signed_trd_binning, positive_proton_trd_template_mt, fill_value="extrapolate")
    
        #antiproton and charged confused proton
    negative_proton_trd_template_st = np.concatenate((ccp_trd_template_st[::-1], np.zeros(len(TRD_binning) - 1)))
    negative_proton_trd_template_st = negative_proton_trd_template_st.cumsum() / np.maximum(negative_proton_trd_template_st.sum(),1)
    negative_proton_trd_template_st = interp1d(signed_trd_binning, negative_proton_trd_template_st, fill_value="extrapolate")
    
    negative_proton_trd_template_mt = np.concatenate((ccp_trd_template_mt[::-1], np.zeros(len(TRD_binning) - 1)))
    negative_proton_trd_template_mt = negative_proton_trd_template_mt.cumsum() / np.maximum(negative_proton_trd_template_mt.sum(),1)
    negative_proton_trd_template_mt = interp1d(signed_trd_binning, negative_proton_trd_template_mt, fill_value="extrapolate")
    
    ##CCMVA templates:
    # print(CCMVA_binning[1:].shape)
    # print(CCMVA_pTemplate_st.shape)
    # print((CCMVA_pTemplate_st.cumsum()/CCMVA_pTemplate_st.sum()).shape)
   

    
    CCMVA_pTemplate_st = interp1d(CCMVA_binning[1:], CCMVA_pTemplate_st.cumsum()/np.maximum(CCMVA_pTemplate_st.sum(),1), fill_value="extrapolate")
    CCMVA_eTemplate_st = interp1d(CCMVA_binning[1:], CCMVA_eTemplate_st.cumsum()/np.maximum(CCMVA_eTemplate_st.sum(),1), fill_value="extrapolate")
    CCMVA_ccpTemplate_st = interp1d(CCMVA_binning[1:], CCMVA_ccpTemplate_st.cumsum()/np.maximum(CCMVA_ccpTemplate_st.sum(),1), fill_value="extrapolate")
    CCMVA_cceTemplate_st = interp1d(CCMVA_binning[1:], CCMVA_cceTemplate_st.cumsum()/np.maximum(CCMVA_cceTemplate_st.sum(),1), fill_value="extrapolate")
    
    
    
    ##CCMVA templates:
    CCMVA_pTemplate_mt = interp1d(CCMVA_binning[1:], CCMVA_pTemplate_mt.cumsum()/np.maximum(CCMVA_pTemplate_mt.sum(),1), fill_value="extrapolate")
    CCMVA_eTemplate_mt = interp1d(CCMVA_binning[1:], CCMVA_eTemplate_mt.cumsum()/np.maximum(CCMVA_eTemplate_mt.sum(),1), fill_value="extrapolate")
    CCMVA_ccpTemplate_mt = interp1d(CCMVA_binning[1:], CCMVA_ccpTemplate_mt.cumsum()/np.maximum(CCMVA_ccpTemplate_mt.sum(),1), fill_value="extrapolate")
    CCMVA_cceTemplate_mt = interp1d(CCMVA_binning[1:], CCMVA_cceTemplate_mt.cumsum()/np.maximum(CCMVA_cceTemplate_mt.sum(),1), fill_value="extrapolate")
    
    def combined_template_all_track(edges, nel, npos, npr, nccp, fecc_st, fecc_mt ,
                                    alpha_el, alpha_pr, alpha_cce, alpha_ccp, alpha_ccpos):
        r_trd_edges = edges[0,:]
        ccmva_edges = edges[1,:]
        #print(r_trd_edges.shape, ccmva_edges.shape)
        
        return (alpha_el * (nel * (1 - fecc_st) * negative_electron_trd_template_st(r_trd_edges) * CCMVA_eTemplate_st(ccmva_edges)) #electron
         + alpha_cce * (nel*fecc_st * positive_electron_trd_template_st(r_trd_edges) * CCMVA_cceTemplate_st(ccmva_edges))  #charged confused electron
         + alpha_el * (npos*(1-fecc_st) * positive_electron_trd_template_st(r_trd_edges) * CCMVA_eTemplate_st(ccmva_edges)) #positron
         + alpha_ccpos * (npos * fecc_st * negative_electron_trd_template_st(r_trd_edges) * CCMVA_cceTemplate_st(ccmva_edges)) #charge confused positron
         + alpha_pr * (npr * positive_proton_trd_template_st(r_trd_edges) * CCMVA_pTemplate_st(ccmva_edges)) #proton
         + alpha_ccp * (nccp * negative_proton_trd_template_st(r_trd_edges) * CCMVA_ccpTemplate_st(ccmva_edges)) #charge confused proton
    
        +(1-alpha_el)*(nel * (1 - fecc_mt) * negative_electron_trd_template_mt(r_trd_edges) * CCMVA_eTemplate_mt(ccmva_edges)) #electron
        +(1 -alpha_cce)*(nel*fecc_mt * positive_electron_trd_template_mt(r_trd_edges) * CCMVA_cceTemplate_mt(ccmva_edges))  #charged confused electron
        +(1-alpha_el)*(npos*(1-fecc_mt) * positive_electron_trd_template_mt(r_trd_edges) * CCMVA_eTemplate_mt(ccmva_edges)) #positron
        +(1-alpha_ccpos)*(npos * fecc_mt * negative_electron_trd_template_mt(r_trd_edges) * CCMVA_cceTemplate_mt(ccmva_edges)) #charge confused positron
        +(1-alpha_pr)*(npr * positive_proton_trd_template_mt(r_trd_edges) * CCMVA_pTemplate_mt(ccmva_edges)) #proton
        +(1-alpha_ccp)*(nccp * negative_proton_trd_template_mt(r_trd_edges) * CCMVA_ccpTemplate_mt(ccmva_edges))) #charge confused proton    
    
    return combined_template_all_track  
    
    

       

# the combined (non-commulative) model

def combined_model(TRD_binning, proton_trd_template, electron_trd_template, ccp_trd_template,
                           CCMVA_pTemplate, CCMVA_eTemplate, CCMVA_ccpTemplate, CCMVA_cceTemplate, nel, npos, npr, fecc, nccp):
    ##TRD templates:
    
        #positron and charge confused electron
    positive_electron_trd_template = np.concatenate((np.zeros(len(TRD_binning) - 1), electron_trd_template))
    positive_electron_trd_template = positive_electron_trd_template / np.maximum(1,positive_electron_trd_template.sum())
    positive_electron_trd_template = positive_electron_trd_template[:,None]
    
    # print(positive_electron_trd_template)
    
        #electron and charge confused electron
    negative_electron_trd_template = np.concatenate((electron_trd_template[::-1], np.zeros(len(TRD_binning) - 1)))
    negative_electron_trd_template = negative_electron_trd_template / np.maximum(1,negative_electron_trd_template.sum())
    negative_electron_trd_template = negative_electron_trd_template[:,None]
    
    # print(negative_electron_trd_template)
    
        #proton
    positive_proton_trd_template = np.concatenate((np.zeros(len(TRD_binning) - 1), proton_trd_template))
    positive_proton_trd_template = positive_proton_trd_template / np.maximum(1,positive_proton_trd_template.sum()) 
    positive_proton_trd_template = positive_proton_trd_template[:,None]
    
    # print(positive_proton_trd_template)
      
        #antiproton and chrge confused proton
    negative_proton_trd_template = np.concatenate((ccp_trd_template[::-1], np.zeros(len(TRD_binning) - 1)))
    
    # print(" negative_proton_trd_template.sum()", negative_proton_trd_template.sum())
    # print("negative_proton_trd_template", negative_proton_trd_template)
    
    negative_proton_trd_template = negative_proton_trd_template / np.maximum(1,negative_proton_trd_template.sum())
    negative_proton_trd_template =negative_proton_trd_template[:,None]
    
    # print('negative_proton_trd_template',negative_proton_trd_template)

        
    CCMVA_pTemplate = (CCMVA_pTemplate/CCMVA_pTemplate.sum())[None,:]
    CCMVA_ccpTemplate = (CCMVA_ccpTemplate/ CCMVA_ccpTemplate.sum())[None,:]
    CCMVA_cceTemplate = (CCMVA_cceTemplate/ CCMVA_cceTemplate.sum())[None,:]
    CCMVA_eTemplate = (CCMVA_eTemplate/ CCMVA_eTemplate.sum())[None,:]
        
        
    return (nel * (1 - fecc) * negative_electron_trd_template * CCMVA_eTemplate
               + nel*fecc * positive_electron_trd_template * CCMVA_cceTemplate
               + npos*(1-fecc) * positive_electron_trd_template * CCMVA_eTemplate
               + npos*fecc * negative_electron_trd_template * CCMVA_cceTemplate
               + npr * positive_proton_trd_template * CCMVA_pTemplate
               + nccp * negative_proton_trd_template* CCMVA_ccpTemplate)
 




def combined_model_all_track(TRD_binning, 
                           proton_trd_template_st, electron_trd_template_st, ccp_trd_template_st,
                           CCMVA_pTemplate_st, CCMVA_eTemplate_st, CCMVA_ccpTemplate_st, CCMVA_cceTemplate_st,
                           proton_trd_template_mt, electron_trd_template_mt, ccp_trd_template_mt,
                           CCMVA_pTemplate_mt, CCMVA_eTemplate_mt, CCMVA_ccpTemplate_mt, CCMVA_cceTemplate_mt,nel, 
                           npos, npr, nccp, fecc_st, fecc_mt, 
                           alpha_el, alpha_pr, alpha_cce, alpha_ccp, alpha_ccpos):
    
    ##TRD templates:
        
        #positron and charged confused electron
    positive_electron_trd_template_st = np.concatenate((np.zeros(len(TRD_binning) - 1), electron_trd_template_st))
    print(len(TRD_binning), len(electron_trd_template_st))
    positive_electron_trd_template_st = positive_electron_trd_template_st/ np.maximum(positive_electron_trd_template_st.sum(),1)
    positive_electron_trd_template_st = positive_electron_trd_template_st[:,None]
    
    positive_electron_trd_template_mt = np.concatenate((np.zeros(len(TRD_binning) - 1), electron_trd_template_mt))
    positive_electron_trd_template_mt = positive_electron_trd_template_mt/ np.maximum(positive_electron_trd_template_mt.sum(),1)
    positive_electron_trd_template_mt = positive_electron_trd_template_mt[:,None]
    
        #electron and charge confused positron
    negative_electron_trd_template_st = np.concatenate((electron_trd_template_st[::-1], np.zeros(len(TRD_binning) - 1)))
    negative_electron_trd_template_st = negative_electron_trd_template_st/ np.maximum(negative_electron_trd_template_st.sum(),1)
    negative_electron_trd_template_st = negative_electron_trd_template_st[:,None]
    
    negative_electron_trd_template_mt = np.concatenate((electron_trd_template_mt[::-1], np.zeros(len(TRD_binning) - 1)))
    negative_electron_trd_template_mt = negative_electron_trd_template_mt/ np.maximum(negative_electron_trd_template_mt.sum(),1)
    negative_electron_trd_template_mt = negative_electron_trd_template_mt[:,None]
    
        #proton
    positive_proton_trd_template_st = np.concatenate((np.zeros(len(TRD_binning) - 1), proton_trd_template_st))
    positive_proton_trd_template_st = positive_proton_trd_template_st/ np.maximum(positive_proton_trd_template_st.sum(),1)  
    positive_proton_trd_template_st = positive_proton_trd_template_st[:,None]
    
    positive_proton_trd_template_mt = np.concatenate((np.zeros(len(TRD_binning) - 1), proton_trd_template_mt))
    positive_proton_trd_template_mt = positive_proton_trd_template_mt/ np.maximum(positive_proton_trd_template_mt.sum(),1)  
    positive_proton_trd_template_mt = positive_proton_trd_template_mt[:,None]
    
        #antiproton and charged confused proton
    negative_proton_trd_template_st = np.concatenate((ccp_trd_template_st[::-1], np.zeros(len(TRD_binning) - 1)))
    negative_proton_trd_template_st = negative_proton_trd_template_st/ np.maximum(negative_proton_trd_template_st.sum(),1)
    negative_proton_trd_template_st = negative_proton_trd_template_st[:,None]
    
    negative_proton_trd_template_mt = np.concatenate((ccp_trd_template_mt[::-1], np.zeros(len(TRD_binning) - 1)))
    negative_proton_trd_template_mt = negative_proton_trd_template_mt/ np.maximum(negative_proton_trd_template_mt.sum(),1)
    negative_proton_trd_template_mt = negative_proton_trd_template_mt[:,None]
    
    ##CCMVA templates:
    CCMVA_pTemplate_st = (CCMVA_pTemplate_st/np.maximum(CCMVA_pTemplate_st.sum(),1))[None,:]
    CCMVA_eTemplate_st = (CCMVA_eTemplate_st/np.maximum(CCMVA_eTemplate_st.sum(),1))[None,:]
    CCMVA_ccpTemplate_st = (CCMVA_ccpTemplate_st/np.maximum(CCMVA_ccpTemplate_st.sum(),1))[None,:]
    CCMVA_cceTemplate_st = (CCMVA_cceTemplate_st/np.maximum(CCMVA_cceTemplate_st.sum(),1))[None,:]
    
    ##CCMVA templates:
    CCMVA_pTemplate_mt = (CCMVA_pTemplate_mt/np.maximum(CCMVA_pTemplate_mt.sum(),1))[None,:]
    CCMVA_eTemplate_mt = (CCMVA_eTemplate_mt/np.maximum(CCMVA_eTemplate_mt.sum(),1))[None,:]
    CCMVA_ccpTemplate_mt = (CCMVA_ccpTemplate_mt/np.maximum(CCMVA_ccpTemplate_mt.sum(),1))[None,:]
    CCMVA_cceTemplate_mt = (CCMVA_cceTemplate_mt/np.maximum(CCMVA_cceTemplate_mt.sum(),1))[None,:]
    

        
    return (alpha_el * (nel * (1 - fecc_st) * negative_electron_trd_template_st * CCMVA_eTemplate_st) #electron
      +alpha_cce * (nel*fecc_st * positive_electron_trd_template_st * CCMVA_cceTemplate_st)  #charged confused electron
      +alpha_el * (npos*(1-fecc_st) * positive_electron_trd_template_st * CCMVA_eTemplate_st) #positron
      +alpha_ccpos * (npos * fecc_st * negative_electron_trd_template_st * CCMVA_cceTemplate_st) #charge confused positron
      +alpha_pr * (npr * positive_proton_trd_template_st * CCMVA_pTemplate_st) #proton
      +alpha_ccp * (nccp *negative_proton_trd_template_st * CCMVA_ccpTemplate_st) #charge confused proton
    
    +(1-alpha_el) * (nel * (1 - fecc_mt) * negative_electron_trd_template_mt * CCMVA_eTemplate_mt) #electron
    +(1-alpha_cce) * (nel*fecc_mt * positive_electron_trd_template_mt * CCMVA_cceTemplate_mt)  #charged confused electron
    +(1-alpha_el) * (npos*(1-fecc_mt) * positive_electron_trd_template_mt * CCMVA_eTemplate_mt) #positron
    +(1-alpha_ccpos)* (npos * fecc_mt * negative_electron_trd_template_mt * CCMVA_cceTemplate_mt) #charge confused positron
    +(1-alpha_pr)* (npr * positive_proton_trd_template_mt * CCMVA_pTemplate_mt) #proton
    +(1-alpha_ccp)*(nccp * negative_proton_trd_template_mt * CCMVA_ccpTemplate_mt)) #charge confused proton   


#only electron and positron model
def combined_model_all_el_pos(TRD_binning, 
                           electron_trd_template_st,
                           CCMVA_eTemplate_st,  electron_trd_template_mt,CCMVA_eTemplate_mt,nel,npos, 
                           fecc_st, fecc_mt, 
                           alpha_el):
    
        #positron and charged confused electron
    positive_electron_trd_template_st = np.concatenate((np.zeros(len(TRD_binning) - 1), electron_trd_template_st))
    positive_electron_trd_template_st = positive_electron_trd_template_st/ np.maximum(positive_electron_trd_template_st.sum(),1)
    positive_electron_trd_template_st = positive_electron_trd_template_st[:,None]
    
    positive_electron_trd_template_mt = np.concatenate((np.zeros(len(TRD_binning) - 1), electron_trd_template_mt))
    positive_electron_trd_template_mt = positive_electron_trd_template_mt/ np.maximum(positive_electron_trd_template_mt.sum(),1)
    positive_electron_trd_template_mt = positive_electron_trd_template_mt[:,None]
    
        #electron and charge confused positron
    negative_electron_trd_template_st = np.concatenate((electron_trd_template_st[::-1], np.zeros(len(TRD_binning) - 1)))
    negative_electron_trd_template_st = negative_electron_trd_template_st/ np.maximum(negative_electron_trd_template_st.sum(),1)
    negative_electron_trd_template_st = negative_electron_trd_template_st[:,None]
    
    negative_electron_trd_template_mt = np.concatenate((electron_trd_template_mt[::-1], np.zeros(len(TRD_binning) - 1)))
    negative_electron_trd_template_mt = negative_electron_trd_template_mt/ np.maximum(negative_electron_trd_template_mt.sum(),1)
    negative_electron_trd_template_mt = negative_electron_trd_template_mt[:,None]
    
    CCMVA_eTemplate_st = (CCMVA_eTemplate_st/np.maximum(CCMVA_eTemplate_st.sum(),1))[None,:]
    CCMVA_eTemplate_mt = (CCMVA_eTemplate_mt/np.maximum(CCMVA_eTemplate_mt.sum(),1))[None,:]
    
    
    
    
    return (alpha_el * (nel * (1 - fecc_st) * negative_electron_trd_template_st * CCMVA_eTemplate_st) #electron
            +alpha_el * (npos*(1-fecc_st) * positive_electron_trd_template_st * CCMVA_eTemplate_st) #positron
            +(1-alpha_el) * (nel * (1 - fecc_mt) * negative_electron_trd_template_mt * CCMVA_eTemplate_mt)
            +(1-alpha_el) * (npos*(1-fecc_mt) * positive_electron_trd_template_mt * CCMVA_eTemplate_mt)) #positron



#only cc electron and poitron
def combined_model_all_ccel_ccpos(TRD_binning, electron_trd_template_st,
                           CCMVA_cceTemplate_st,electron_trd_template_mt,CCMVA_cceTemplate_mt,nel, npos, fecc_st, fecc_mt, 
                           alpha_cce, alpha_ccpos):
    
        #positron and charged confused electron
    positive_electron_trd_template_st = np.concatenate((np.zeros(len(TRD_binning) - 1), electron_trd_template_st))
    positive_electron_trd_template_st = positive_electron_trd_template_st/ np.maximum(positive_electron_trd_template_st.sum(),1)
    positive_electron_trd_template_st = positive_electron_trd_template_st[:,None]
    
    positive_electron_trd_template_mt = np.concatenate((np.zeros(len(TRD_binning) - 1), electron_trd_template_mt))
    positive_electron_trd_template_mt = positive_electron_trd_template_mt/ np.maximum(positive_electron_trd_template_mt.sum(),1)
    positive_electron_trd_template_mt = positive_electron_trd_template_mt[:,None]
    
        #electron and charge confused positron
    negative_electron_trd_template_st = np.concatenate((electron_trd_template_st[::-1], np.zeros(len(TRD_binning) - 1)))
    negative_electron_trd_template_st = negative_electron_trd_template_st/ np.maximum(negative_electron_trd_template_st.sum(),1)
    negative_electron_trd_template_st = negative_electron_trd_template_st[:,None]
    
    negative_electron_trd_template_mt = np.concatenate((electron_trd_template_mt[::-1], np.zeros(len(TRD_binning) - 1)))
    negative_electron_trd_template_mt = negative_electron_trd_template_mt/ np.maximum(negative_electron_trd_template_mt.sum(),1)
    negative_electron_trd_template_mt = negative_electron_trd_template_mt[:,None]
    
    CCMVA_cceTemplate_st = (CCMVA_cceTemplate_st/np.maximum(CCMVA_cceTemplate_st.sum(),1))[None,:]
    CCMVA_cceTemplate_mt = (CCMVA_cceTemplate_mt/np.maximum(CCMVA_cceTemplate_mt.sum(),1))[None,:]
    
    
    return (alpha_cce * (nel*fecc_st * positive_electron_trd_template_st * CCMVA_cceTemplate_st)  #charged confused electron
    +alpha_ccpos * (npos * fecc_st * negative_electron_trd_template_st * CCMVA_cceTemplate_st) #charge confused positron 
    +(1-alpha_cce) * (nel*fecc_mt * positive_electron_trd_template_mt * CCMVA_cceTemplate_mt)  #charged confused electron
    +(1-alpha_ccpos)* (npos * fecc_mt * negative_electron_trd_template_mt * CCMVA_cceTemplate_mt)) #charge confused positron
    
            

######################################################################################################################    
# getting all the info we need to run the code
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataversion", default="pass8", help="version of the data set (e.g. pass6)")
    parser.add_argument("--datatype", default="ISS", help="title of the input data (e.g. ISS)")
    #parser.add_argument("--tracknumber", default="single", help="number of tracks (single or multiple or all)")
    parser.add_argument("--TRDtemplate_filepath",default="/Users/yasaman/AMS02/plots/newpass8/fitparametersplotsISS8/results/" ,help="")
    parser.add_argument("--CCMVAtemplate_filepath_ISS",default="/Users/yasaman/AMS02/plots/newpass8/CCMVABDT_template_ISS.npz" ,help="")
    parser.add_argument("--CCMVAtemplate_filepath_MC",default="/Users/yasaman/AMS02/plots/newpass8/CCMVABDT_template_MC.npz" ,help="")
    parser.add_argument("--data_filepath",default="/Users/yasaman/AMS02/plots/newpass8/results.npz" ,help="")
    args = parser.parse_args()
    
    os.makedirs("results/plots/single", exist_ok = True)
    os.makedirs("results/plots/multiple", exist_ok = True)
    os.makedirs("results/plots/All", exist_ok = True)

    #load TRD template parameters from 1d fits
    TRD_pParameters_st = np.load(args.TRDtemplate_filepath + "/proton_parameters_single_smooth.npz")
    TRD_eParameters_st =np.load(args.TRDtemplate_filepath + "/electron_parameters_single_smooth.npz")
    TRD_ccpParameters_st =np.load(args.TRDtemplate_filepath + "/ccproton_parameters_single_smooth.npz")

    TRD_pParameters_mt = np.load(args.TRDtemplate_filepath + "/proton_parameters_multiple_smooth.npz")
    TRD_eParameters_mt =np.load(args.TRDtemplate_filepath + "/electron_parameters_multiple_smooth.npz")
    TRD_ccpParameters_mt =np.load(args.TRDtemplate_filepath + "/ccproton_parameters_multiple_smooth.npz")

     
    #load CCMVABDT 1d histograms      
    with np.load(os.path.join(args.CCMVAtemplate_filepath_ISS)) as result_file:
        
        CCMVA_pTemplate_st = result_file["pEvents_st"]
        CCMVA_eTemplate_st = result_file["eEvents_st"]
        CCMVA_ccpTemplate_st = result_file["ccpEvents_st"]
                
    
        CCMVA_pTemplate_mt = result_file["pEvents_mt"]
        CCMVA_eTemplate_mt = result_file["eEvents_mt"]
        CCMVA_ccpTemplate_mt = result_file["ccpEvents_mt"] 
            
            
    with np.load(os.path.join(args.CCMVAtemplate_filepath_MC)) as result_file:
            
        CCMVA_cceTemplate_st = result_file["cc_eEvents_st"] 
        CCMVA_cceTemplate_mt = result_file["cc_eEvents_mt"] 
       
                
     # load 2d histograms of data           
    with np.load(os.path.join(args.data_filepath)) as result_file:  
        TRD_binning = result_file["var1_binning"]
        CCMVA_binning = result_file["var2_binning"] 
        Energy_binning = result_file["var3_binning"]
        Rigidity_binning = result_file["var4_binning"]
                
        
        Events_st = result_file["Events_st"]
        Events_mt = result_file["Events_mt"]
        Events_all = result_file["Events_all"]     
        
    


        
     
    #TRD_binning=TRD_binning[10:]   
    
    #rigidity sign times trd binning     
    signed_trd_binning = np.concatenate((-TRD_binning[:0:-1], TRD_binning))
       
    i=0
    chi_st = np.zeros((len(Energy_binning) -1)) 
    TRD_chi_st = np.zeros((len(Energy_binning) -1))
    CCMVA_chi_st = np.zeros((len(Energy_binning) -1))
    
    chi_mt = np.zeros((len(Energy_binning) -1)) 
    TRD_chi_mt = np.zeros((len(Energy_binning) -1))
    CCMVA_chi_mt = np.zeros((len(Energy_binning) -1))
    
    # TRD_pvalue = np.zeros((len(Energy_binning) -1))
    # CCMVA_pvalue = np.zeros((len(Energy_binning) -1))
    
    TRD_res_st = np.zeros((len(Energy_binning)-1 , len(signed_trd_binning) -1))
    CCMVA_res_st = np.zeros((len(Energy_binning)-1, len(CCMVA_binning) -1))
    
    TRD_res_mt = np.zeros((len(Energy_binning)-1 , len(signed_trd_binning) -1))
    CCMVA_res_mt = np.zeros((len(Energy_binning)-1, len(CCMVA_binning) -1))
    
    TRD_res_all = np.zeros((len(Energy_binning)-1 , len(signed_trd_binning) -1))
    CCMVA_res_all = np.zeros((len(Energy_binning)-1, len(CCMVA_binning) -1))
    
    guess_st = dict(nel=1000, npos=100, npr=100000, fecc=0.2, nccp =100) #guess values for template fit
    guess_mt = dict(nel=1000, npos=100, npr=100000, fecc=0.2, nccp =100)
    guess_all = dict(nel = 0, npos = 0, npr = 0, nccp = 0, fecc_st = 0, fecc_mt = 0,
                     alpha_el = 0, alpha_pr = 0, alpha_cce = 0, alpha_ccp = 0, alpha_ccpos = 0)
    
    parameters_st = {key: [] for key in guess_st}
    parameters_mt = {key: [] for key in guess_mt}
    parameters_all = {key: [] for key in guess_all}
    
    err_st = {key: [] for key in guess_st}
    err_mt = {key: [] for key in guess_mt}
    err_all = {key: [] for key in guess_all}
    
    for binn in range(len(Energy_binning) -1):
        
        i=i+1
        
        #upeer and lower edeges of enrgy bins
        a = round(Energy_binning[binn],2)
        b = round(Energy_binning[binn +1],2)
        
        #make trd template fits
        proton_trd_template_st =  proton_model(TRD_binning,TRD_pParameters_st['alpha_p'][binn],TRD_pParameters_st['s1_p'][binn],TRD_pParameters_st['delta_m_p'][binn],TRD_pParameters_st['s2_p'][binn],
                                            TRD_pParameters_st['m2_p'][binn],TRD_pParameters_st['t_p'][binn])
        
        proton_trd_template_mt =  proton_model(TRD_binning,TRD_pParameters_mt['alpha_p'][binn],TRD_pParameters_mt['s1_p'][binn],TRD_pParameters_mt['delta_m_p'][binn],TRD_pParameters_mt['s2_p'][binn],
                                            TRD_pParameters_mt['m2_p'][binn],TRD_pParameters_mt['t_p'][binn])
        
        
        electron_trd_template_st = electron_model(TRD_binning,TRD_eParameters_st['alpha_el'][binn],TRD_eParameters_st['s1_el'][binn],
                                               TRD_eParameters_st['delta_m_el'][binn],TRD_eParameters_st['s2_el'][binn],TRD_eParameters_st['m2_el'][binn],
                                               TRD_eParameters_st['t_el'][binn])
        
        electron_trd_template_mt = electron_model(TRD_binning,TRD_eParameters_mt['alpha_el'][binn],TRD_eParameters_mt['s1_el'][binn],
                                               TRD_eParameters_mt['delta_m_el'][binn],TRD_eParameters_mt['s2_el'][binn],TRD_eParameters_mt['m2_el'][binn],
                                               TRD_eParameters_mt['t_el'][binn])
        
        ccp_trd_template_st = ccproton_model(TRD_binning,TRD_ccpParameters_st['s_ccp'][binn],TRD_ccpParameters_st["delta_m_ccp"][binn],
                                          TRD_ccpParameters_st["t_ccp"][binn],TRD_ccpParameters_st["m2_el"][binn])
        
        ccp_trd_template_mt = ccproton_model(TRD_binning,TRD_ccpParameters_mt['s_ccp'][binn],TRD_ccpParameters_mt["delta_m_ccp"][binn],
                                          TRD_ccpParameters_mt["t_ccp"][binn],TRD_ccpParameters_mt["m2_el"][binn])
        
        
        #scaled_cdf = make_combined_template(TRD_binning, CCMVA_binning, signed_trd_binning,
                                            #proton_trd_template, electron_trd_template, ccp_trd_template,
                                           #CCMVA_pTemplate[:,binn], CCMVA_eTemplate[:,binn], CCMVA_ccpTemplate[:,binn], CCMVA_cceTemplate[:,binn])
        
        #separate negative and positive data    
        data_negative_st = Events_st[:,:,binn,0]
        data_positive_st = Events_st[:,:,binn,1]
        signed_data_st = np.concatenate((data_negative_st[::-1,:], data_positive_st), axis=0)
        
        data_negative_mt = Events_mt[:,:,binn,0]
        data_positive_mt = Events_mt[:,:,binn,1]
        signed_data_mt = np.concatenate((data_negative_mt[::-1,:], data_positive_mt), axis=0)
        
        data_negative_all = Events_all[:,:,binn,0]
        data_positive_all = Events_all[:,:,binn,1]
        signed_data_all = np.concatenate((data_negative_all[::-1,:], data_positive_all), axis=0)
        
        # print(np.max(Events_all[:,:,binn,0]))
        # print(np.max(Events_st[:,:,binn,0]))
        
        
        # first fit the single track and multi track samples
        
        
        
        template_st = make_combined_template(TRD_binning, CCMVA_binning, signed_trd_binning,
                                            proton_trd_template_st, electron_trd_template_st, ccp_trd_template_st,
                                            CCMVA_pTemplate_st[:,binn], CCMVA_eTemplate_st[:,binn], CCMVA_ccpTemplate_st[:,binn], CCMVA_cceTemplate_st[:,binn])
        
        template_mt = make_combined_template(TRD_binning, CCMVA_binning, signed_trd_binning,
                                            proton_trd_template_mt, electron_trd_template_mt, ccp_trd_template_mt,
                                            CCMVA_pTemplate_mt[:,binn], CCMVA_eTemplate_mt[:,binn], CCMVA_ccpTemplate_mt[:,binn], CCMVA_cceTemplate_mt[:,binn])
        
        #make liklihoodfunction
        liklihood_st = ExtendedBinnedNLL(signed_data_st, [signed_trd_binning, CCMVA_binning], template_st)
        liklihood_mt = ExtendedBinnedNLL(signed_data_mt, [signed_trd_binning, CCMVA_binning], template_mt)
        
        #call minuit for single track sample
        minuit_st = Minuit(liklihood_st, **guess_st) 
        #put limit on free parameters values
        minuit_st.limits["nccp"]=(0,None)
        minuit_st.limits["nel"]=(0,None)
        minuit_st.limits["npos"]=(0,None)
        minuit_st.limits["npr"]=(0,None)
        minuit_st.limits["fecc"]=(0,None)
        
        minuit_st.migrad()
        #print(minuit_st.values)
        
        minuit_mt = Minuit(liklihood_mt, **guess_mt)   
        #put limit on free parameters values
        minuit_mt.limits["nccp"]=(0,None)
        minuit_mt.limits["nel"]=(0,None)
        minuit_mt.limits["npos"]=(0,None)
        minuit_mt.limits["npr"]=(0,None)
        minuit_mt.limits["fecc"]=(0,None)
        
        minuit_mt.migrad()
        #print(minuit_mt.values)
    
        #update the new guess with previous fit results
        guess_st = dict(zip(minuit_st.parameters, minuit_st.values))
        error_st = dict(zip(minuit_st.parameters, minuit_st.errors))
        
        guess_mt = dict(zip(minuit_mt.parameters, minuit_mt.values))
        error_mt = dict(zip(minuit_mt.parameters, minuit_mt.errors))
        
        for parameter in parameters_st:
            parameters_st[parameter].append(guess_st[parameter])
            err_st[parameter].append(error_st[parameter])
        
        for parameter in parameters_mt:
            parameters_mt[parameter].append(guess_mt[parameter])
            err_mt[parameter].append(error_mt[parameter])    
        
        print(binn)
        print("multiple track fit: ",minuit_mt.valid)
        print("single track fit: ",minuit_st.valid)
        
        if i ==1:
            alpha_el = 0.8
            alpha_pr = 0.9
            alpha_cce = 0.9
            alpha_ccp = 0.2
            alpha_ccpos = 0.9
            
            guess_all = dict(nel = guess_st['nel'], npos = guess_st['npos'], 
                         npr = guess_st['npr'], nccp = guess_st['nccp'],
                         fecc_st = guess_st['fecc'],  
                         fecc_mt = guess_mt['fecc'],alpha_el = alpha_el, alpha_pr = alpha_pr, 
                         alpha_cce = alpha_cce, alpha_ccp = alpha_ccp, alpha_ccpos = alpha_ccpos)
        
        else: # keep the alphas from the previous fit but update the other variables from st and mt fit results
 
            guess_all['fecc_st'] = guess_st['fecc']
            guess_all['fecc_mt'] = guess_mt['fecc']

            

        template_all = make_combined_template_all_track(TRD_binning, CCMVA_binning, signed_trd_binning,
                                   proton_trd_template_st, electron_trd_template_st, ccp_trd_template_st,
                                   CCMVA_pTemplate_st[:,binn], CCMVA_eTemplate_st[:,binn], CCMVA_ccpTemplate_st[:,binn], CCMVA_cceTemplate_st[:,binn],
                                   proton_trd_template_mt, electron_trd_template_mt, ccp_trd_template_mt,
                                   CCMVA_pTemplate_mt[:,binn], CCMVA_eTemplate_mt[:,binn], CCMVA_ccpTemplate_mt[:,binn], CCMVA_cceTemplate_mt[:,binn])
        
        liklihood_all = ExtendedBinnedNLL(signed_data_all, [signed_trd_binning, CCMVA_binning], template_all)
        minuit_all = Minuit(liklihood_all, **guess_all) 
        
        minuit_all.limits["nccp"]=(0,None)
        minuit_all.limits["nel"]=(0,None)
        minuit_all.limits["npos"]=(0,None)
        minuit_all.limits["npr"]=(0,None)
        # minuit_all.limits["fecc_mt"]=(0,None)
        # minuit_all.limits["fecc_st"]=(0,None)
        
        minuit_all.limits['alpha_el']=(0.1,0.9)
        minuit_all.limits['alpha_pr']=(0.1,0.9)
        minuit_all.limits['alpha_cce']=(0.1,0.9)
        minuit_all.limits['alpha_ccp']=(0.1,0.9)
        minuit_all.limits['alpha_ccpos']=(0.1,0.9)

        minuit_all.fixed["fecc_st"]=True
        minuit_all.fixed["fecc_mt"]=True
        minuit_all.fixed["alpha_cce"]=True
        minuit_all.fixed["alpha_ccpos"]=True
        
        minuit_all.migrad()

        print("all track fit: ",minuit_st.valid)
        
        #update the new guess with previous fit results
        print(minuit_all.values)
        guess_all = dict(zip(minuit_all.parameters, minuit_all.values))
        error_all = dict(zip(minuit_all.parameters, minuit_all.errors))
        
        for parameter in parameters_all:
            parameters_all[parameter].append(guess_all[parameter])
            err_all[parameter].append(error_all[parameter])
        
        
        
        
        #resulted model from fit for plotting 
        model_st = combined_model((TRD_binning[1:]+TRD_binning[:-1])/2, proton_trd_template_st, electron_trd_template_st, ccp_trd_template_st,
                                   CCMVA_pTemplate_st[:,binn], CCMVA_eTemplate_st[:,binn], CCMVA_ccpTemplate_st[:,binn], CCMVA_cceTemplate_st[:,binn], guess_st["nel"], guess_st["npos"], guess_st["npr"], guess_st["fecc"], guess_st["nccp"])
       
        #print("model_st:",model_st)
        #print(proton_trd_template_st, electron_trd_template_st, ccp_trd_template_st,CCMVA_pTemplate_st[:,binn], CCMVA_eTemplate_st[:,binn], CCMVA_ccpTemplate_st[:,binn], CCMVA_cceTemplate_st[:,binn])
        
        #break
        
        model_el_st = combined_model((TRD_binning[1:]+TRD_binning[:-1])/2, proton_trd_template_st, electron_trd_template_st, ccp_trd_template_st, CCMVA_pTemplate_st[:,binn], CCMVA_eTemplate_st[:,binn]
                                  , CCMVA_ccpTemplate_st[:,binn], CCMVA_cceTemplate_st[:,binn], guess_st["nel"], 0, 0, guess_st["fecc"], 0)
        
        model_pos_st = combined_model((TRD_binning[1:]+TRD_binning[:-1])/2, proton_trd_template_st, electron_trd_template_st, ccp_trd_template_st,
                                   CCMVA_pTemplate_st[:,binn], CCMVA_eTemplate_st[:,binn], CCMVA_ccpTemplate_st[:,binn], CCMVA_cceTemplate_st[:,binn], 0, guess_st["npos"], 0, guess_st["fecc"], 0)
        
        model_proton_st = combined_model((TRD_binning[1:]+TRD_binning[:-1])/2, proton_trd_template_st, electron_trd_template_st, ccp_trd_template_st,
                                   CCMVA_pTemplate_st[:,binn], CCMVA_eTemplate_st[:,binn], CCMVA_ccpTemplate_st[:,binn], CCMVA_cceTemplate_st[:,binn], 0, 0, guess_st["npr"],0, 0)
        
        model_ccproton_st = combined_model((TRD_binning[1:]+TRD_binning[:-1])/2, proton_trd_template_st, electron_trd_template_st, ccp_trd_template_st,
                                   CCMVA_pTemplate_st[:,binn], CCMVA_eTemplate_st[:,binn], CCMVA_ccpTemplate_st[:,binn], CCMVA_cceTemplate_st[:,binn],0, 0, 0, 0, guess_st["nccp"])
        #difference of the model and the data
        diff_st = signed_data_st - model_st
        
        #TRD model
        TRD_model_st = model_st.sum(axis=1)
        
        #CCMVA_model
        CCMVA_model_st = model_st.sum(axis=0)
        
        
        #resulted model from fit for plotting 
        model_mt = combined_model((TRD_binning[1:]+TRD_binning[:-1])/2, proton_trd_template_mt, electron_trd_template_mt, ccp_trd_template_mt,
                                   CCMVA_pTemplate_mt[:,binn], CCMVA_eTemplate_mt[:,binn], CCMVA_ccpTemplate_mt[:,binn], CCMVA_cceTemplate_mt[:,binn], guess_mt["nel"], guess_mt["npos"], guess_mt["npr"], guess_mt["fecc"], guess_mt["nccp"])
       
        model_el_mt = combined_model((TRD_binning[1:]+TRD_binning[:-1])/2, proton_trd_template_mt, electron_trd_template_mt, ccp_trd_template_mt, CCMVA_pTemplate_mt[:,binn], CCMVA_eTemplate_mt[:,binn]
                                  , CCMVA_ccpTemplate_mt[:,binn], CCMVA_cceTemplate_mt[:,binn], guess_mt["nel"], 0, 0, guess_mt["fecc"], 0)
        
        model_pos_mt = combined_model((TRD_binning[1:]+TRD_binning[:-1])/2, proton_trd_template_mt, electron_trd_template_mt, ccp_trd_template_mt,
                                   CCMVA_pTemplate_mt[:,binn], CCMVA_eTemplate_mt[:,binn], CCMVA_ccpTemplate_mt[:,binn], CCMVA_cceTemplate_mt[:,binn], 0, guess_mt["npos"], 0, guess_mt["fecc"], 0)
        
        model_proton_mt = combined_model((TRD_binning[1:]+TRD_binning[:-1])/2, proton_trd_template_mt, electron_trd_template_mt, ccp_trd_template_mt,
                                   CCMVA_pTemplate_mt[:,binn], CCMVA_eTemplate_mt[:,binn], CCMVA_ccpTemplate_mt[:,binn], CCMVA_cceTemplate_mt[:,binn], 0, 0, guess_mt["npr"],0, 0)
        
        model_ccproton_mt = combined_model((TRD_binning[1:]+TRD_binning[:-1])/2, proton_trd_template_mt, electron_trd_template_mt, ccp_trd_template_mt,
                                   CCMVA_pTemplate_mt[:,binn], CCMVA_eTemplate_mt[:,binn], CCMVA_ccpTemplate_mt[:,binn], CCMVA_cceTemplate_mt[:,binn],0, 0, 0, 0, guess_mt["nccp"])
        #difference of the model and the data
        diff_mt = signed_data_mt - model_mt
        
        #TRD model
        TRD_model_mt = model_mt.sum(axis=1)
        
        #CCMVA_model
        CCMVA_model_mt = model_mt.sum(axis=0)
        
        
        #resulted model from fit for plotting for all track sample
        model_all = combined_model_all_track((TRD_binning[1:]+TRD_binning[:-1])/2, 
                                   proton_trd_template_st, electron_trd_template_st, ccp_trd_template_st,
                                   CCMVA_pTemplate_st[:,binn], CCMVA_eTemplate_st[:,binn], CCMVA_ccpTemplate_st[:,binn], CCMVA_cceTemplate_st[:,binn],
                                   proton_trd_template_mt, electron_trd_template_mt, ccp_trd_template_mt,
                                   CCMVA_pTemplate_mt[:,binn], CCMVA_eTemplate_mt[:,binn], CCMVA_ccpTemplate_mt[:,binn], CCMVA_cceTemplate_mt[:,binn],
                                   guess_all['nel'], guess_all['npos'], guess_all['npr'], 
                                   guess_all['nccp'], guess_all['fecc_st'], guess_all['fecc_mt'],
                                   guess_all['alpha_el'], guess_all['alpha_pr'],
                                   guess_all['alpha_cce'], guess_all['alpha_ccp'],guess_all['alpha_ccpos'])
        print(model_all.shape, TRD_binning.shape)
        
        
      
       
        model_el_all = combined_model_all_track((TRD_binning[1:]+TRD_binning[:-1])/2, 
                                   proton_trd_template_st, electron_trd_template_st, ccp_trd_template_st,
                                   CCMVA_pTemplate_st[:,binn], CCMVA_eTemplate_st[:,binn], CCMVA_ccpTemplate_st[:,binn], CCMVA_cceTemplate_st[:,binn],
                                   proton_trd_template_mt, electron_trd_template_mt, ccp_trd_template_mt,
                                   CCMVA_pTemplate_mt[:,binn], CCMVA_eTemplate_mt[:,binn], CCMVA_ccpTemplate_mt[:,binn], CCMVA_cceTemplate_mt[:,binn],
                                   guess_all['nel'],0,0, 
                                   0, guess_all['fecc_st'],guess_all['fecc_mt'],
                                   guess_all['alpha_el'], guess_all['alpha_pr'],
                                   guess_all['alpha_cce'], guess_all['alpha_ccp'],guess_all['alpha_ccpos'])
        
        model_el_pos_all = combined_model_all_el_pos((TRD_binning[1:]+TRD_binning[:-1])/2, 
                                                     electron_trd_template_st,CCMVA_eTemplate_st[:,binn],electron_trd_template_mt,
                                                     CCMVA_eTemplate_mt[:,binn],guess_all['nel'],guess_all['npos'],guess_all['fecc_st'],
                                                     guess_all['fecc_mt'], guess_all['alpha_el'])
        
        model_ccel_ccpos_all= combined_model_all_ccel_ccpos((TRD_binning[1:]+TRD_binning[:-1])/2,electron_trd_template_st,
                                   CCMVA_cceTemplate_st[:,binn],electron_trd_template_mt,CCMVA_cceTemplate_mt[:,binn],
                                   guess_all['nel'],guess_all['npos'], guess_all['fecc_st'], guess_all['fecc_mt'], 
                                   guess_all['alpha_cce'], guess_all['alpha_ccpos'])
        
        
        model_pos_all = combined_model_all_track((TRD_binning[1:]+TRD_binning[:-1])/2, 
                                   proton_trd_template_st, electron_trd_template_st, ccp_trd_template_st,
                                   CCMVA_pTemplate_st[:,binn], CCMVA_eTemplate_st[:,binn], CCMVA_ccpTemplate_st[:,binn], CCMVA_cceTemplate_st[:,binn],
                                   proton_trd_template_mt, electron_trd_template_mt, ccp_trd_template_mt,
                                   CCMVA_pTemplate_mt[:,binn], CCMVA_eTemplate_mt[:,binn], CCMVA_ccpTemplate_mt[:,binn], CCMVA_cceTemplate_mt[:,binn],
                                   0, guess_all['npos'],0, 
                                   0, guess_all['fecc_st'], guess_all['fecc_mt'],
                                   guess_all['alpha_el'], guess_all['alpha_pr'],
                                   guess_all['alpha_cce'], guess_all['alpha_ccp'],guess_all['alpha_ccpos'])
        
        
        model_proton_all = combined_model_all_track((TRD_binning[1:]+TRD_binning[:-1])/2, 
                                   proton_trd_template_st, electron_trd_template_st, ccp_trd_template_st,
                                   CCMVA_pTemplate_st[:,binn], CCMVA_eTemplate_st[:,binn], CCMVA_ccpTemplate_st[:,binn], CCMVA_cceTemplate_st[:,binn],
                                   proton_trd_template_mt, electron_trd_template_mt, ccp_trd_template_mt,
                                   CCMVA_pTemplate_mt[:,binn], CCMVA_eTemplate_mt[:,binn], CCMVA_ccpTemplate_mt[:,binn], CCMVA_cceTemplate_mt[:,binn], 
                                   0,0, guess_all['npr'], 
                                   0, 0, 0,
                                   guess_all['alpha_el'], guess_all['alpha_pr'],
                                   guess_all['alpha_cce'], guess_all['alpha_ccp'],guess_all['alpha_ccpos'])
        
        model_ccproton_all = combined_model_all_track((TRD_binning[1:]+TRD_binning[:-1])/2, 
                                   proton_trd_template_st, electron_trd_template_st, ccp_trd_template_st,
                                   CCMVA_pTemplate_st[:,binn], CCMVA_eTemplate_st[:,binn], CCMVA_ccpTemplate_st[:,binn], CCMVA_cceTemplate_st[:,binn],
                                   proton_trd_template_mt, electron_trd_template_mt, ccp_trd_template_mt,
                                   CCMVA_pTemplate_mt[:,binn], CCMVA_eTemplate_mt[:,binn], CCMVA_ccpTemplate_mt[:,binn], CCMVA_cceTemplate_mt[:,binn], 
                                   0, 0, 0, 
                                   guess_all['nccp'], 0, 0,
                                   guess_all['alpha_el'], guess_all['alpha_pr'],
                                   guess_all['alpha_cce'], guess_all['alpha_ccp'],guess_all['alpha_ccpos'])
        #difference of the model and the data
       
        diff_all = signed_data_all - model_all
        
        #TRD model
        TRD_model_all = model_all.sum(axis=1)
        
        #CCMVA_model
        CCMVA_model_all = model_all.sum(axis=0)
        
        
        #chi square
        
        dof= (len(TRD_binning)-1)*(len(CCMVA_binning)-1) - len(minuit_st.parameters)
        TRD_dof = (len(TRD_binning)-1) - len(minuit_st.parameters)
        CCMVA_dof = (len(CCMVA_binning)-1) - len(minuit_st.parameters)
        
        chi_st[binn]= np.sum(chisquare(signed_data_st,model_st))/dof
        chi_mt[binn]= np.sum(chisquare(signed_data_st,model_st))/dof
        
         
        # print(signed_data_st.shape)
        # print(model.shape)
        #chi, p = chisquare(signed_data,f_exp=model,ddof=dof,axis=None)
        #TRD_chi[binn]= chisquare(signed_data.sum(axis=0),model.sum(axis=0), TRD_dof)[0]/TRD_dof
        #CCMVA_chi[binn]= chisquare(signed_data.sum(axis=1),model.sum(axis=1),CCMVA_dof)[0]/CCMVA_dof
        
        # TRD_pvalue[binn]= chisquare(signed_data.sum(axis=0),model.sum(axis=0), TRD_dof)[1]
        # CCMVA_pvalue[binn] = chisquare(signed_data.sum(axis=1),model.sum(axis=1),CCMVA_dof)[1]
        
        TRD_res_st[binn, :] = residuals(signed_data_st.sum(axis=1),model_st.sum(axis=1))
        CCMVA_res_st[binn, :] = residuals(signed_data_st.sum(axis=0),model_st.sum(axis=0))
        
        TRD_res_mt[binn, :] = residuals(signed_data_mt.sum(axis=1),model_mt.sum(axis=1))
        CCMVA_res_mt[binn, :] = residuals(signed_data_mt.sum(axis=0),model_mt.sum(axis=0))
        
        TRD_res_all[binn, :] = residuals(signed_data_all.sum(axis=1),model_all.sum(axis=1))
        CCMVA_res_all[binn, :] = residuals(signed_data_all.sum(axis=0),model_all.sum(axis=0))
        
        
        ############################################plot 2D templates for single track sample
        figure = plt.figure(figsize=(10, 4))
        gs = GridSpec(1, 6, wspace=0.1, width_ratios=[0.06, 0.3, 1, 1, 1, 0.06])
        ax1 = figure.add_subplot(gs[2])
        ax2 = figure.add_subplot(gs[3], sharey=ax1)
        ax3 = figure.add_subplot(gs[4], sharey=ax1)
        cax1 = figure.add_subplot(gs[0])
        cax2 = figure.add_subplot(gs[5])
        #figure, (cax1, ax1, ax2, ax3, cax2) = plt.subplots(ncols=5, figsize=(10, 4), sharey=True,gridspec_kw=dict(wspace=0.1, width_ratios = [0.25,1,1,1,0.25]))
        norm = LogNorm (vmin = 1, vmax = np.maximum(np.max(signed_data_st), np.max(signed_data_st))+1)
        print(np.maximum(np.max(signed_data_st), np.max(signed_data_st)))
        #norm =  SymLogNorm(linthresh=0.01, linscale=1.0,vmin = np.minimum(np.min(diff),(np.minimum(np.min(signed_data),np.min(model)))) , vmax=np.maximum(np.max(diff),(np.maximum(np.max(signed_data),np.max(model)))))
        
        #figure.delaxes(ax_empty)
        ax1.set_title('Data')
        ax2.set_title('Model')
        ax3.set_title('data - model')
        
        #data plot
        ax1.set_xlabel(r'$Z \times \Lambda_{\rm TRD}$',fontsize = 12)
        ax1.set_ylabel(r'$\Lambda_{\rm CC}$', fontsize = 12)
        ax1.minorticks_on()

        
        signeddata_st = np.ma.masked_where(signed_data_st <=1, signed_data_st).transpose()  
        mesh1=ax1.pcolormesh(0.5*(signed_trd_binning[1:]+signed_trd_binning[:-1]), 0.5*(CCMVA_binning[1:]+CCMVA_binning[:-1]) , signeddata_st , norm=norm, cmap =plt.cm.get_cmap('viridis'))

        
        #model plot
        ax2.set_xlabel(r'$Z \times \Lambda_{\rm TRD}$',fontsize = 12)
        #ax2.set_ylabel(r'$\Lambda_{\rm CC}$', fontsize = 12)
        ax2.tick_params(labelleft=False)
        #ax2.get_yaxis().set_visible(False)
        ax2.minorticks_on()

        
        modeldata = np.ma.masked_where(model_st <= 1, model_st).transpose()   
        mesh2 = ax2.pcolormesh(0.5*(signed_trd_binning[1:]+signed_trd_binning[:-1]), 0.5*(CCMVA_binning[1:]+CCMVA_binning[:-1]) , modeldata , norm=norm, cmap =plt.cm.get_cmap('viridis'))
        
        #divider = make_axes_locatable(ax2)
        
        #Append a new axis for the colorbar
        #cax2 = divider.append_axes("right", size="6%", pad=0.1)

        #Create a colorbar on the new axis
        cbar2 = figure.colorbar(mesh2, cax=cax1)
        cbar2.set_label("Events",fontsize=10)
        cax1.get_yaxis().set_ticks_position("left")
        cax1.get_yaxis().set_label_position("left")
        
        
        ax3.set_xlabel(r'$Z \times \Lambda_{\rm TRD}$',fontsize = 12)
        #ax3.set_ylabel(r'$\Lambda_{\rm CC}$', fontsize = 12)
        ax3.tick_params(labelleft=False)
        ax3.minorticks_on()
        
        diff = np.ma.masked_where(np.abs(diff_st) <= 1, diff_st ).transpose()   
        mesh3 = ax3.pcolormesh(0.5*(signed_trd_binning[1:]+signed_trd_binning[:-1]), 0.5*(CCMVA_binning[1:]+CCMVA_binning[:-1]) , diff , cmap =plt.cm.get_cmap('jet'))
        
        #divider = make_axes_locatable(ax3)

        # Append a new axis for the colorbar
        #cax3 = divider.append_axes("right", size="6%", pad=0.1)

        # Create a colorbar on the new axis
        cbar3 = figure.colorbar(mesh3, cax=cax2)
        cbar3.set_label("Events",fontsize=10)
        
        figure.subplots_adjust(left=0.08, right=0.9, bottom=0.15, top=0.9)
        
        info = [f"ISS data {args.dataversion}\n"
            f"Energy = [{a} , {b}]GeV \n",
            f"$\\chi_st^2$ / $n_\\mathrm{{dof}}$ = {chi_st[binn]:.1f}"]
        
        #ax1.legend(title=''.join(info),title_fontsize=6,fontsize=6,frameon=False, loc="upper right")
        ax2.legend(title='Single Track sample'.join(info),title_fontsize=6,fontsize=6,frameon=False, loc="upper right")
        
        figure.savefig("results/plots/single/2D_histogram_CCMVABDT_TRD_Estimators_"+str(i)+".pdf" , dpi=250)
        plt.close(figure) 
        
        
        ###################################################################plot 2D templates for multi track sample
        figure = plt.figure(figsize=(10, 4))
        gs = GridSpec(1, 6, wspace=0.1, width_ratios=[0.06, 0.3, 1, 1, 1, 0.06])
        ax1 = figure.add_subplot(gs[2])
        ax2 = figure.add_subplot(gs[3], sharey=ax1)
        ax3 = figure.add_subplot(gs[4], sharey=ax1)
        cax1 = figure.add_subplot(gs[0])
        cax2 = figure.add_subplot(gs[5])
        #figure, (cax1, ax1, ax2, ax3, cax2) = plt.subplots(ncols=5, figsize=(10, 4), sharey=True,gridspec_kw=dict(wspace=0.1, width_ratios = [0.25,1,1,1,0.25]))
        norm = LogNorm (vmin = 1, vmax = np.maximum(np.max(signed_data_mt), np.max(signed_data_mt))+1)
        #norm =  SymLogNorm(linthresh=0.01, linscale=1.0,vmin = np.minimum(np.min(diff),(np.minimum(np.min(signed_data),np.min(model)))) , vmax=np.maximum(np.max(diff),(np.maximum(np.max(signed_data),np.max(model)))))
        
        #figure.delaxes(ax_empty)
        ax1.set_title('Data')
        ax2.set_title('Model')
        ax3.set_title('data - model')
        
        #data plot
        ax1.set_xlabel(r'$Z \times \Lambda_{\rm TRD}$',fontsize = 12)
        ax1.set_ylabel(r'$\Lambda_{\rm CC}$', fontsize = 12)
        ax1.minorticks_on()

        
        signeddata_mt = np.ma.masked_where(signed_data_mt <=1, signed_data_mt).transpose()  
        mesh1=ax1.pcolormesh(0.5*(signed_trd_binning[1:]+signed_trd_binning[:-1]), 0.5*(CCMVA_binning[1:]+CCMVA_binning[:-1]) , signeddata_mt , norm=norm, cmap =plt.cm.get_cmap('viridis'))

        
        #model plot
        ax2.set_xlabel(r'$Z \times \Lambda_{\rm TRD}$',fontsize = 12)
        #ax2.set_ylabel(r'$\Lambda_{\rm CC}$', fontsize = 12)
        ax2.tick_params(labelleft=False)
        #ax2.get_yaxis().set_visible(False)
        ax2.minorticks_on()

        
        modeldata = np.ma.masked_where(model_mt <= 1, model_mt).transpose()   
        mesh2 = ax2.pcolormesh(0.5*(signed_trd_binning[1:]+signed_trd_binning[:-1]), 0.5*(CCMVA_binning[1:]+CCMVA_binning[:-1]) , modeldata , norm=norm, cmap =plt.cm.get_cmap('viridis'))
        
        #divider = make_axes_locatable(ax2)
        
        #Append a new axis for the colorbar
        #cax2 = divider.append_axes("right", size="6%", pad=0.1)

        #Create a colorbar on the new axis
        cbar2 = figure.colorbar(mesh2, cax=cax1)
        cbar2.set_label("Events",fontsize=10)
        cax1.get_yaxis().set_ticks_position("left")
        cax1.get_yaxis().set_label_position("left")
        
        
        ax3.set_xlabel(r'$Z \times \Lambda_{\rm TRD}$',fontsize = 12)
        #ax3.set_ylabel(r'$\Lambda_{\rm CC}$', fontsize = 12)
        ax3.tick_params(labelleft=False)
        ax3.minorticks_on()
        
        diff = np.ma.masked_where(np.abs(diff_mt) <= 1, diff_mt ).transpose()   
        mesh3 = ax3.pcolormesh(0.5*(signed_trd_binning[1:]+signed_trd_binning[:-1]), 0.5*(CCMVA_binning[1:]+CCMVA_binning[:-1]) , diff , cmap =plt.cm.get_cmap('jet'))
        
        #divider = make_axes_locatable(ax3)

        # Append a new axis for the colorbar
        #cax3 = divider.append_axes("right", size="6%", pad=0.1)

        # Create a colorbar on the new axis
        cbar3 = figure.colorbar(mesh3, cax=cax2)
        cbar3.set_label("Events",fontsize=10)
        
        figure.subplots_adjust(left=0.08, right=0.9, bottom=0.15, top=0.9)
        
        info = [f"ISS data {args.dataversion}\n"
            f"Energy = [{a} , {b}]GeV \n",
            f"$\\chi_mt^2$ / $n_\\mathrm{{dof}}$ = {chi_mt[binn]:.1f}"]
        
        #ax1.legend(title=''.join(info),title_fontsize=6,fontsize=6,frameon=False, loc="upper right")
        ax2.legend(title=''.join(info),title_fontsize=6,fontsize=6,frameon=False, loc="upper right")
        
        figure.savefig("results/plots/multiple/2D_histogram_CCMVABDT_TRD_Estimators_"+str(i)+".pdf" , dpi=250)
        plt.close(figure) 
        
        ###############################################################plot 2D templates for all track sample
        figure = plt.figure(figsize=(10, 4))
        gs = GridSpec(1, 6, wspace=0.1, width_ratios=[0.06, 0.3, 1, 1, 1, 0.06])
        ax1 = figure.add_subplot(gs[2])
        ax2 = figure.add_subplot(gs[3], sharey=ax1)
        ax3 = figure.add_subplot(gs[4], sharey=ax1)
        cax1 = figure.add_subplot(gs[0])
        cax2 = figure.add_subplot(gs[5])
        #figure, (cax1, ax1, ax2, ax3, cax2) = plt.subplots(ncols=5, figsize=(10, 4), sharey=True,gridspec_kw=dict(wspace=0.1, width_ratios = [0.25,1,1,1,0.25]))
        norm = LogNorm (vmin = 1, vmax = np.maximum(np.max(signed_data_all), np.max(signed_data_all))+1)
        #norm =  SymLogNorm(linthresh=0.01, linscale=1.0,vmin = np.minimum(np.min(diff),(np.minimum(np.min(signed_data),np.min(model)))) , vmax=np.maximum(np.max(diff),(np.maximum(np.max(signed_data),np.max(model)))))
        
        #figure.delaxes(ax_empty)
        ax1.set_title('Data')
        ax2.set_title('Model')
        ax3.set_title('data - model')
        
        #data plot
        ax1.set_xlabel(r'$Z \times \Lambda_{\rm TRD}$',fontsize = 12)
        ax1.set_ylabel(r'$\Lambda_{\rm CC}$', fontsize = 12)
        ax1.minorticks_on()

        
        signeddata_all = np.ma.masked_where(signed_data_all <=1, signed_data_all).transpose()  
        mesh1=ax1.pcolormesh(0.5*(signed_trd_binning[1:]+signed_trd_binning[:-1]), 0.5*(CCMVA_binning[1:]+CCMVA_binning[:-1]) , signeddata_all , norm=norm, cmap =plt.cm.get_cmap('viridis'))

        
        #model plot
        ax2.set_xlabel(r'$Z \times \Lambda_{\rm TRD}$',fontsize = 12)
        #ax2.set_ylabel(r'$\Lambda_{\rm CC}$', fontsize = 12)
        ax2.tick_params(labelleft=False)
        #ax2.get_yaxis().set_visible(False)
        ax2.minorticks_on()

        
        modeldata_all = np.ma.masked_where(model_all <= 1, model_all).transpose()   
        mesh2 = ax2.pcolormesh(0.5*(signed_trd_binning[1:]+signed_trd_binning[:-1]), 0.5*(CCMVA_binning[1:]+CCMVA_binning[:-1]) , modeldata_all , norm=norm, cmap =plt.cm.get_cmap('viridis'))
        
        #divider = make_axes_locatable(ax2)
        
        #Append a new axis for the colorbar
        #cax2 = divider.append_axes("right", size="6%", pad=0.1)

        #Create a colorbar on the new axis
        cbar2 = figure.colorbar(mesh2, cax=cax1)
        cbar2.set_label("Events",fontsize=10)
        cax1.get_yaxis().set_ticks_position("left")
        cax1.get_yaxis().set_label_position("left")
        
        
        ax3.set_xlabel(r'$Z \times \Lambda_{\rm TRD}$',fontsize = 12)
        #ax3.set_ylabel(r'$\Lambda_{\rm CC}$', fontsize = 12)
        ax3.tick_params(labelleft=False)
        ax3.minorticks_on()
        
        diff = np.ma.masked_where(np.abs(diff_all) <= 1, diff_all ).transpose()   
        mesh3 = ax3.pcolormesh(0.5*(signed_trd_binning[1:]+signed_trd_binning[:-1]), 0.5*(CCMVA_binning[1:]+CCMVA_binning[:-1]) , diff , cmap =plt.cm.get_cmap('jet'))
        
        #divider = make_axes_locatable(ax3)

        # Append a new axis for the colorbar
        #cax3 = divider.append_axes("right", size="6%", pad=0.1)

        # Create a colorbar on the new axis
        cbar3 = figure.colorbar(mesh3, cax=cax2)
        cbar3.set_label("Events",fontsize=10)
        
        figure.subplots_adjust(left=0.08, right=0.9, bottom=0.15, top=0.9)
        
        info = [f"ISS data {args.dataversion}\n"
            f"Energy = [{a} , {b}]GeV \n",
            f"$\\chi_mt^2$ / $n_\\mathrm{{dof}}$ = {chi_mt[binn]:.1f}"]
        
        #ax1.legend(title=''.join(info),title_fontsize=6,fontsize=6,frameon=False, loc="upper right")
        ax2.legend(title=''.join(info),title_fontsize=6,fontsize=6,frameon=False, loc="upper right")
        
        figure.savefig("results/plots/All/2D_histogram_CCMVABDT_TRD_Estimators_"+str(i)+".pdf" , dpi=250)

        plt.close(figure) 
        
        
        ###############################################################plot TRD template for single track sample       
        figure = plt.figure(figsize=(12, 12))
        ax1 = figure.add_subplot(211)
        #plot.set_xlabel(r'$Z \times \Lambda_{\rm TRD}$',fontsize = 26)
        ax1.set_ylabel(r'$\rm Events$', fontsize = 26)
        ax1.tick_params(axis='both', which="major",direction='in', length=12, width=1.5, labelsize=15)
        ax1.tick_params(axis='both', which="minor",direction='in', length=7, width=1.5)
        # ax1.rcParams['xtick.top'] = True
        # ax1.rcParams['ytick.right'] = True
        ax1.minorticks_on()
        # ax1.rcParams["font.weight"] = "bold"
        # ax1.rcParams["axes.labelweight"] = "bold"
        for axis in ['top','bottom','left','right']:
            ax1.spines[axis].set_linewidth(2)
        
        ax1.plot((signed_trd_binning[1:]+ signed_trd_binning[:-1])/2, TRD_model_st, label = "model") 
        ax1.errorbar((signed_trd_binning[1:]+ signed_trd_binning[:-1])/2, signed_data_st.sum(axis=1), np.sqrt(signed_data_st.sum(axis=1)),fmt=".", color="k", label="data" )
        # print(TRD_binning.shape)
        # print(proton_trd_template_st.shape)
        ax1.plot((signed_trd_binning[1:]+ signed_trd_binning[:-1])/2, model_proton_st.sum(axis=1), label = "proton model")
        ax1.plot((signed_trd_binning[1:]+ signed_trd_binning[:-1])/2, model_pos_st.sum(axis=1), label = "positron model")
        ax1.plot((signed_trd_binning[1:]+ signed_trd_binning[:-1])/2, model_ccproton_st.sum(axis=1), label = "ccproton model")
        ax1.plot((signed_trd_binning[1:]+ signed_trd_binning[:-1])/2, model_el_st.sum(axis=1), label = "electron model")
        
        #ax1.plot((data_positive[1:]+data_positive[:1])/2, guess["npr"]*proton_trd_template, label = "proton model")
        
        
        info = [f"ISS data {args.dataversion}\n"
            f"Energy = [{a} , {b}]GeV \n"]
        
            # f"$\\chi^2$ / $n_\\mathrm{{dof}}$ = {TRD_chi[binn]:.1f}",
            # f"$\\p value$ = {TRD_pvalue[binn]:.1f}"]
        ax1.set_yscale("log")     
        ax1.set_ylim(bottom=1)
        ax1.legend(title=''.join(info),title_fontsize=10,fontsize=10,frameon=False, loc='upper right')
        
        ax2 = figure.add_subplot(212,sharex=ax1) 
        ax2.set_ylabel("residuals", fontsize = 15, fontweight = 'bold')
        ax2.set_xlabel(r'$Z \times \Lambda_{\rm TRD}$',fontsize = 26)
        ax2.tick_params(axis='both', which="major",direction='in', length=10, width=1.5, labelsize=15)
        ax2.tick_params(axis='both', which="minor",direction='in', length=6, width=1.5, labelsize=15)         
        for axis in ['top','bottom','left','right']:
            ax2.spines[axis].set_linewidth(2)
        
        T= (signed_trd_binning[1:]+ signed_trd_binning[:-1])/2
        ax2.scatter(T,TRD_res_st[binn,:])
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
        
        figure.savefig("results/plots/single/1D_histogram_TRD_"+str(i)+".pdf" , dpi=250)
        plt.close(figure) 
        
        
        #############################################################plot TRD template  for multi track sample      
        figure = plt.figure(figsize=(12, 12))
        ax1 = figure.add_subplot(211)
        #plot.set_xlabel(r'$Z \times \Lambda_{\rm TRD}$',fontsize = 26)
        ax1.set_ylabel(r'$\rm Events$', fontsize = 26)
        ax1.tick_params(axis='both', which="major",direction='in', length=12, width=1.5, labelsize=15)
        ax1.tick_params(axis='both', which="minor",direction='in', length=7, width=1.5)
        # ax1.rcParams['xtick.top'] = True
        # ax1.rcParams['ytick.right'] = True
        ax1.minorticks_on()
        # ax1.rcParams["font.weight"] = "bold"
        # ax1.rcParams["axes.labelweight"] = "bold"
        for axis in ['top','bottom','left','right']:
            ax1.spines[axis].set_linewidth(2)
        
        ax1.plot((signed_trd_binning[1:]+ signed_trd_binning[:-1])/2, TRD_model_mt, label = "model") 
        ax1.errorbar((signed_trd_binning[1:]+ signed_trd_binning[:-1])/2, signed_data_mt.sum(axis=1), np.sqrt(signed_data_mt.sum(axis=1)),fmt=".", color="k", label="data" )
        # print(TRD_binning.shape)
        # print(proton_trd_template.shape)
        ax1.plot((signed_trd_binning[1:]+ signed_trd_binning[:-1])/2, model_proton_mt.sum(axis=1), label = "proton model")
        ax1.plot((signed_trd_binning[1:]+ signed_trd_binning[:-1])/2, model_pos_mt.sum(axis=1), label = "positron model")
        ax1.plot((signed_trd_binning[1:]+ signed_trd_binning[:-1])/2, model_ccproton_mt.sum(axis=1), label = "ccproton model")
        ax1.plot((signed_trd_binning[1:]+ signed_trd_binning[:-1])/2, model_el_mt.sum(axis=1), label = "electron model")
        
        #ax1.plot((data_positive[1:]+data_positive[:1])/2, guess["npr"]*proton_trd_template, label = "proton model")
        
        
        info = [f"ISS data {args.dataversion}\n"
            f"Energy = [{a} , {b}]GeV \n"]
        
            # f"$\\chi^2$ / $n_\\mathrm{{dof}}$ = {TRD_chi[binn]:.1f}",
            # f"$\\p value$ = {TRD_pvalue[binn]:.1f}"]
        ax1.set_yscale("log")     
        ax1.set_ylim(bottom=1)
        ax1.legend(title=''.join(info),title_fontsize=10,fontsize=10,frameon=False, loc='upper right')
        
        ax2 = figure.add_subplot(212,sharex=ax1) 
        ax2.set_ylabel("residuals", fontsize = 15, fontweight = 'bold')
        ax2.set_xlabel(r'$Z \times \Lambda_{\rm TRD}$',fontsize = 26)
        ax2.tick_params(axis='both', which="major",direction='in', length=10, width=1.5, labelsize=15)
        ax2.tick_params(axis='both', which="minor",direction='in', length=6, width=1.5, labelsize=15)         
        for axis in ['top','bottom','left','right']:
            ax2.spines[axis].set_linewidth(2)
        
        T= (signed_trd_binning[1:]+ signed_trd_binning[:-1])/2
        ax2.scatter(T,TRD_res_mt[binn,:])
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
        
        figure.savefig("results/plots/multiple/1D_histogram_TRD_"+str(i)+".pdf" , dpi=250)
        plt.close(figure) 
        
        ############################################################plot TRD template  for all track sample      
        figure = plt.figure(figsize=(12, 12))
        ax1 = figure.add_subplot(211)
        #plot.set_xlabel(r'$Z \times \Lambda_{\rm TRD}$',fontsize = 26)
        ax1.set_ylabel(r'$\rm Events$', fontsize = 26)
        ax1.tick_params(axis='both', which="major",direction='in', length=12, width=1.5, labelsize=15)
        ax1.tick_params(axis='both', which="minor",direction='in', length=7, width=1.5)
        # ax1.rcParams['xtick.top'] = True
        # ax1.rcParams['ytick.right'] = True
        ax1.minorticks_on()
        # ax1.rcParams["font.weight"] = "bold"
        # ax1.rcParams["axes.labelweight"] = "bold"
        for axis in ['top','bottom','left','right']:
            ax1.spines[axis].set_linewidth(2)
        
        ax1.plot((signed_trd_binning[1:]+ signed_trd_binning[:-1])/2, TRD_model_all, label = "model") 
        ax1.errorbar((signed_trd_binning[1:]+ signed_trd_binning[:-1])/2, signed_data_all.sum(axis=1), np.sqrt(signed_data_all.sum(axis=1)),fmt=".", color="k", label="data" )
        # print(TRD_binning.shape)
        # print(proton_trd_template.shape)
        ax1.plot((signed_trd_binning[1:]+ signed_trd_binning[:-1])/2, model_proton_all.sum(axis=1), label = "proton model")
        #ax1.plot((signed_trd_binning[1:]+ signed_trd_binning[:-1])/2, model_pos_all.sum(axis=1), label = "positron model")
        ax1.plot((signed_trd_binning[1:]+ signed_trd_binning[:-1])/2, model_ccproton_all.sum(axis=1), label = "ccproton model")
        #ax1.plot((signed_trd_binning[1:]+ signed_trd_binning[:-1])/2, model_el_all.sum(axis=1), label = "electron model")
        ax1.plot((signed_trd_binning[1:]+ signed_trd_binning[:-1])/2, model_el_pos_all.sum(axis=1), label = "electron and positron model")
        ax1.plot((signed_trd_binning[1:]+ signed_trd_binning[:-1])/2, model_ccel_ccpos_all.sum(axis=1), label = "cc electron and positron model")
        
        #ax1.plot((data_positive[1:]+data_positive[:1])/2, guess["npr"]*proton_trd_template, label = "proton model")
        
        
        info = [f"ISS data {args.dataversion}\n"
            f"Energy = [{a} , {b}]GeV \n"]
        
            # f"$\\chi^2$ / $n_\\mathrm{{dof}}$ = {TRD_chi[binn]:.1f}",
            # f"$\\p value$ = {TRD_pvalue[binn]:.1f}"]
        ax1.set_yscale("log")     
        ax1.set_ylim(bottom=1)
        ax1.legend(title=''.join(info),title_fontsize=10,fontsize=10,frameon=False, loc='upper right')
        
        ax2 = figure.add_subplot(212,sharex=ax1) 
        ax2.set_ylabel("residuals", fontsize = 15, fontweight = 'bold')
        ax2.set_xlabel(r'$Z \times \Lambda_{\rm TRD}$',fontsize = 26)
        ax2.tick_params(axis='both', which="major",direction='in', length=10, width=1.5, labelsize=15)
        ax2.tick_params(axis='both', which="minor",direction='in', length=6, width=1.5, labelsize=15)         
        for axis in ['top','bottom','left','right']:
            ax2.spines[axis].set_linewidth(2)
        
        T= (signed_trd_binning[1:]+ signed_trd_binning[:-1])/2
        ax2.scatter(T,TRD_res_all[binn,:])
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
        
        figure.savefig("results/plots/All/1D_histogram_TRD_all_"+str(i)+".pdf" , dpi=250)
        plt.close(figure) 
        
        
        ###########################################################plot CCMVA templates for single track sample               
        figure = plt.figure(figsize=(12, 10))
        ax1 = figure.add_subplot(211)
        #ax1.set_xlabel(r'$\Lambda_{\rm CC}$',fontsize = 26)
        ax1.set_ylabel("Single track Events", fontsize = 15, fontweight = 'bold')
        ax1.tick_params(axis='both', which="major",direction='in', length=12, width=1.5, labelsize=20)
        ax1.tick_params(axis='both', which="minor",direction='in', length=7, width=1.5)
        # ax1.rcParams['xtick.top'] = True
        # ax1.rcParams['ytick.right'] = True
        ax1.minorticks_on()
        # ax1.rcParams["font.weight"] = "bold"
        # ax1.rcParams["axes.labelweight"] = "bold"
        for axis in ['top','bottom','left','right']:
            ax1.spines[axis].set_linewidth(2)
        
       # ax1.plot((CCMVA_binning[1:]+ CCMVA_binning[:-1])/2, CCMVA_model_st , label = "model") 
        ax1.step(np.concatenate(([CCMVA_binning[0]], CCMVA_binning)), np.concatenate(([0],CCMVA_model_st, [0])),color ='m', where="post",linewidth=2,label="model")
        ax1.errorbar((CCMVA_binning[1:]+ CCMVA_binning[:-1])/2, signed_data_st.sum(axis=0), np.sqrt(signed_data_st.sum(axis=0)),fmt=".", color="k", label="data" )
        ax1.step(np.concatenate(([CCMVA_binning[0]], CCMVA_binning)), np.concatenate(([0],model_proton_st.sum(axis=0), [0])), color='C1',where="post",linewidth=2,label="proton model")
        ax1.step(np.concatenate(([CCMVA_binning[0]], CCMVA_binning)), np.concatenate(([0],model_pos_st.sum(axis=0), [0])),color='y', where="post",linewidth=2,label="positron model")
        ax1.step(np.concatenate(([CCMVA_binning[0]], CCMVA_binning)), np.concatenate(([0],model_ccproton_st.sum(axis=0), [0])), color='C2',where="post",linewidth=2,label="ccproton model")
        ax1.step(np.concatenate(([CCMVA_binning[0]], CCMVA_binning)), np.concatenate(([0],model_el_st.sum(axis=0), [0])),color='C0', where="post",linewidth=2,label="electron model")
        #ax1.plot((CCMVA_binning[1:]+ CCMVA_binning[:-1])/2, model_proton_st.sum(axis=0), label = "proton model")
        #ax1.plot((CCMVA_binning[1:]+ CCMVA_binning[:-1])/2, model_pos_st.sum(axis=0), label = "positron and cc-positron model")
        #ax1.plot((CCMVA_binning[1:]+ CCMVA_binning[:-1])/2, model_ccproton_st.sum(axis=0), label = "ccproton model")
        #ax1.plot((CCMVA_binning[1:]+ CCMVA_binning[:-1])/2, model_el_st.sum(axis=0), label = "electron and cc-electron model")
        
        info = [f"ISS data {args.dataversion}\n"
            f"Energy = [{a} , {b}]GeV \n"]
            # f"$\\chi^2$ / $n_\\mathrm{{dof}}$ = {CCMVA_chi[binn]:.1f} \n",
            # f"$\\p value$ = {CCMVA_pvalue[binn]:.1f}"]
        
        ax1.legend(title=''.join(info),title_fontsize=9,fontsize=9,frameon=True, loc='upper left')
        ax1.set_yscale("log")
        #ax1.set_ylim(bottom=1)
        
        ax2 = figure.add_subplot(212,sharex=ax1) 
        ax2.set_ylabel("residuals", fontsize = 15, fontweight = 'bold')
        ax2.set_xlabel(r'$Z \times \Lambda_{\rm TRD}$',fontsize = 26)
        ax2.tick_params(axis='both', which="major",direction='in', length=10, width=1.5, labelsize=15)
        ax2.tick_params(axis='both', which="minor",direction='in', length=6, width=1.5, labelsize=15)         
        for axis in ['top','bottom','left','right']:
            ax2.spines[axis].set_linewidth(2)
        
        C= (CCMVA_binning[1:]+ CCMVA_binning[:-1])/2
        ax2.scatter(C,CCMVA_res_st[binn,:])
        ax2.hlines(0,min(C),max(C),colors='k', linestyle='--')
        ax2.set_xlabel(r'$\Lambda_{\rm CC}$',fontsize = 20,fontweight = 'bold') 
        
        # Adjust the size of the upper subplot to be larger than the lower one
        ax1.set_position([0.1, 0.4, 0.8, 0.5])
        ax2.set_position([0.1, 0.1, 0.8, 0.3])
        
        
        plt.rcParams['xtick.top'] = True
        plt.rcParams['ytick.right'] = True
        plt.minorticks_on()
        plt.rcParams["font.weight"] = "bold"
        
        figure.savefig("results/plots/single/1D_histogram_CCMVA_"+str(i)+".pdf" , dpi=250)
        plt.close(figure) 
        
        
        ########################################################plot CCMVA templates  for multiple track sample               
        figure = plt.figure(figsize=(12, 10))
        ax1 = figure.add_subplot(211)
        #ax1.set_xlabel(r'$\Lambda_{\rm CC}$',fontsize = 26)
        ax1.set_ylabel(r'$Events$', fontsize = 26)
        ax1.tick_params(axis='both', which="major",direction='in', length=12, width=1.5, labelsize=20)
        ax1.tick_params(axis='both', which="minor",direction='in', length=7, width=1.5)
        # ax1.rcParams['xtick.top'] = True
        # ax1.rcParams['ytick.right'] = True
        ax1.minorticks_on()
        # ax1.rcParams["font.weight"] = "bold"
        # ax1.rcParams["axes.labelweight"] = "bold"
        for axis in ['top','bottom','left','right']:
            ax1.spines[axis].set_linewidth(2)
        
        ax1.plot((CCMVA_binning[1:]+ CCMVA_binning[:-1])/2, CCMVA_model_mt , label = "model") 
        ax1.errorbar((CCMVA_binning[1:]+ CCMVA_binning[:-1])/2, signed_data_mt.sum(axis=0), np.sqrt(signed_data_mt.sum(axis=0)),fmt=".", color="k", label="data" )
        ax1.plot((CCMVA_binning[1:]+ CCMVA_binning[:-1])/2, model_proton_mt.sum(axis=0), label = "proton model")
        ax1.plot((CCMVA_binning[1:]+ CCMVA_binning[:-1])/2, model_pos_mt.sum(axis=0), label = "positron and cc-positron model")
        ax1.plot((CCMVA_binning[1:]+ CCMVA_binning[:-1])/2, model_ccproton_mt.sum(axis=0), label = "ccproton model")
        ax1.plot((CCMVA_binning[1:]+ CCMVA_binning[:-1])/2, model_el_mt.sum(axis=0), label = "electron and cc-electron model")
        
        info = [f"ISS data {args.dataversion}\n"
            f"Energy = [{a} , {b}]GeV \n"]
            # f"$\\chi^2$ / $n_\\mathrm{{dof}}$ = {CCMVA_chi[binn]:.1f} \n",
            # f"$\\p value$ = {CCMVA_pvalue[binn]:.1f}"]
        
        ax1.legend(title=''.join(info),title_fontsize=14,fontsize=14,frameon=False, loc='upper right')
        ax1.set_yscale("log")
        ax1.set_ylim(bottom=1)
        
        ax2 = figure.add_subplot(212,sharex=ax1) 
        ax2.set_ylabel("residuals", fontsize = 15, fontweight = 'bold')
        ax2.set_xlabel(r'$Z \times \Lambda_{\rm TRD}$',fontsize = 26)
        ax2.tick_params(axis='both', which="major",direction='in', length=10, width=1.5, labelsize=15)
        ax2.tick_params(axis='both', which="minor",direction='in', length=6, width=1.5, labelsize=15)         
        for axis in ['top','bottom','left','right']:
            ax2.spines[axis].set_linewidth(2)
        
        C= (CCMVA_binning[1:]+ CCMVA_binning[:-1])/2
        ax2.scatter(C,CCMVA_res_mt[binn,:])
        ax2.hlines(0,min(C),max(C),colors='k', linestyle='--')
        ax2.set_xlabel(r'$\Lambda_{\rm CC}$',fontsize = 20,fontweight = 'bold') 
        
        # Adjust the size of the upper subplot to be larger than the lower one
        ax1.set_position([0.1, 0.4, 0.8, 0.5])
        ax2.set_position([0.1, 0.1, 0.8, 0.3])
        
        
        plt.rcParams['xtick.top'] = True
        plt.rcParams['ytick.right'] = True
        plt.minorticks_on()
        plt.rcParams["font.weight"] = "bold"
        
        figure.savefig("results/plots/multiple/1D_histogram_CCMVA_"+str(i)+".pdf" , dpi=250)
        plt.close(figure) 
        
        ##################################################################plot CCMVA templates  for all track sample               
        figure = plt.figure(figsize=(12, 10))
        ax1 = figure.add_subplot(211)
        #ax1.set_xlabel(r'$\Lambda_{\rm CC}$',fontsize = 26)
        ax1.set_ylabel(r'$Events$', fontsize = 26)
        ax1.tick_params(axis='both', which="major",direction='in', length=12, width=1.5, labelsize=20)
        ax1.tick_params(axis='both', which="minor",direction='in', length=7, width=1.5)
        # ax1.rcParams['xtick.top'] = True
        # ax1.rcParams['ytick.right'] = True
        ax1.minorticks_on()
        # ax1.rcParams["font.weight"] = "bold"
        # ax1.rcParams["axes.labelweight"] = "bold"
        for axis in ['top','bottom','left','right']:
            ax1.spines[axis].set_linewidth(2)
        
        ax1.plot((CCMVA_binning[1:]+ CCMVA_binning[:-1])/2, CCMVA_model_all , label = "model") 
        ax1.errorbar((CCMVA_binning[1:]+ CCMVA_binning[:-1])/2, signed_data_all.sum(axis=0), np.sqrt(signed_data_all.sum(axis=0)),fmt=".", color="k", label="data" )
        ax1.plot((CCMVA_binning[1:]+ CCMVA_binning[:-1])/2, model_proton_all.sum(axis=0), label = "proton model")
        #ax1.plot((CCMVA_binning[1:]+ CCMVA_binning[:-1])/2, model_pos_all.sum(axis=0), label = "positron and cc-positron model")
        ax1.plot((CCMVA_binning[1:]+ CCMVA_binning[:-1])/2, model_ccproton_all.sum(axis=0), label = "ccproton model")
        #ax1.plot((CCMVA_binning[1:]+ CCMVA_binning[:-1])/2, model_el_all.sum(axis=0), label = "electron and cc-electron model")
        ax1.plot((CCMVA_binning[1:]+ CCMVA_binning[:-1])/2, model_el_pos_all.sum(axis=0), label = "electron and positron model")
        ax1.plot((CCMVA_binning[1:]+ CCMVA_binning[:-1])/2, model_ccel_ccpos_all.sum(axis=0), label = "cc-electron and cc-electron model")
        
        
        info = [f"ISS data {args.dataversion}\n"
            f"Energy = [{a} , {b}]GeV \n"]
            # f"$\\chi^2$ / $n_\\mathrm{{dof}}$ = {CCMVA_chi[binn]:.1f} \n",
            # f"$\\p value$ = {CCMVA_pvalue[binn]:.1f}"]
        
        ax1.legend(title=''.join(info),title_fontsize=8,fontsize=8,frameon=False, loc='upper right')
        ax1.set_yscale("log")
        ax1.set_ylim(bottom=1)
        
        ax2 = figure.add_subplot(212,sharex=ax1) 
        ax2.set_ylabel("residuals", fontsize = 15, fontweight = 'bold')
        ax2.set_xlabel(r'$Z \times \Lambda_{\rm TRD}$',fontsize = 26)
        ax2.tick_params(axis='both', which="major",direction='in', length=10, width=1.5, labelsize=15)
        ax2.tick_params(axis='both', which="minor",direction='in', length=6, width=1.5, labelsize=15)         
        for axis in ['top','bottom','left','right']:
            ax2.spines[axis].set_linewidth(2)
        
        C= (CCMVA_binning[1:]+ CCMVA_binning[:-1])/2
        ax2.scatter(C,CCMVA_res_all[binn,:])
        ax2.hlines(0,min(C),max(C),colors='k', linestyle='--')
        ax2.set_xlabel(r'$\Lambda_{\rm CC}$',fontsize = 20,fontweight = 'bold') 
        
        # Adjust the size of the upper subplot to be larger than the lower one
        ax1.set_position([0.1, 0.4, 0.8, 0.5])
        ax2.set_position([0.1, 0.1, 0.8, 0.3])
        
        
        plt.rcParams['xtick.top'] = True
        plt.rcParams['ytick.right'] = True
        plt.minorticks_on()
        plt.rcParams["font.weight"] = "bold"
        
        figure.savefig("results/plots/All/1D_histogram_CCMVA_"+str(i)+".pdf" , dpi=250)
        plt.close(figure) 
        
        
    ########################### plot fit parameters for all track sample #############################
    
    for parameter in parameters_all:
        #parameters_all[parameter].append(guess_all[parameter])
        #err_all[parameter].append(error_all[parameter])
    
        figure = plt.figure(figsize=(12, 10))
        ax = figure.add_subplot(111)
        ax.set_ylabel(str(parameter), fontsize = 26)
        ax.set_xlabel("Energy GeV", fontsize = 26)
        ax.tick_params(axis='both', which="major",direction='in', length=12, width=1.5, labelsize=20)
        ax.tick_params(axis='both', which="minor",direction='in', length=7, width=1.5)
        # ax1.rcParams['xtick.top'] = True
        # ax1.rcParams['ytick.right'] = True
        ax.minorticks_on()
        # ax1.rcParams["font.weight"] = "bold"
        # ax1.rcParams["axes.labelweight"] = "bold"
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(2)
         
        ax.set_xscale("log")
        
        plt.rcParams['xtick.top'] = True
        plt.rcParams['ytick.right'] = True
        plt.minorticks_on()
        plt.rcParams["font.weight"] = "bold"  
        
        ax.scatter((Energy_binning[1:]+Energy_binning[:-1])/2, parameters_all[parameter])
        figure.savefig("results/plots/All/parameters_"+str(parameter)+".pdf" , dpi=250)
        plt.close(figure)
            

         
    np.savez("results/parameters_single.npz", **parameters_st) 
    np.savez("results/errors_single.npz", **err_st) 
    np.savetxt("results/goodnessoffit_single.npz",chi_st)
        
        
    np.savez("results/parameters_multiple.npz", **parameters_mt) 
    np.savez("results/errors_multiple.npz", **err_mt) 
    np.savetxt("results/goodnessoffit_multiple.npz",chi_mt)
    
    np.savez("results/parameters_all.npz", **parameters_all) 
    np.savez("results/errors_all.npz", **err_all) 
    

    


if __name__ == "__main__":
    main()                          
                
                
                
                
            
            
            
