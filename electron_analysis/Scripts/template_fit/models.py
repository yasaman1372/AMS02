#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  2 10:55:46 2022

@author: yasaman


"""
from iminuit import Minuit
import numpy as np
from iminuit.cost import ExtendedBinnedNLL, LeastSquares
from scipy.integrate import quad



def chisquare(d,m):
    err=np.maximum(m,1)
    return ((d-m)**2)/err


def electron_model_Gaus(x,alpha,s1,delta_m,s2,m2,t):  
    m1=m2+delta_m    
    G = (1/np.sqrt(2* np.pi* s1**2)) * np.exp (- ((x-m1)**2) /(2 * s1**2))
    return (1- alpha)*G

def electron_model_Nov(x,alpha,s1,delta_m,s2,m2,t):  
    l = 1 + t * (x-m2) *  np.sinh(t * np.sqrt(np.log(4))) / (s2*t* np.sqrt(np.log(4))) 
    l = np.maximum(l,1e-7)
    N = np.exp(-0.5 * ((np.log(l)**2 / t**2) + t**2) )
    
    return alpha*N



def electron_model(x,alpha,s1,delta_m,s2,m2,t):
    m1=m2 + delta_m
    l = 1 + t * (x-m2) *  np.sinh(t * np.sqrt(np.log(4))) / (s2*t* np.sqrt(np.log(4))) 
    l = np.maximum(l,1e-7)
    
    G = (1/np.sqrt(2* np.pi* s1**2)) * np.exp (- ((x-m1)**2) /(2 * s1**2))
    N = np.exp(-0.5 * ((np.log(l)**2 / t**2) + t**2) )
    
    return (1- alpha)*G + alpha*N

def CCproton_model(x,sp,delta_mp,tp,m2):
    mp=m2+delta_mp
    l = 1 + tp * (x-mp) *  np.sinh(tp * np.sqrt(np.log(4))) / (sp*tp* np.sqrt(np.log(4))) 
    l = np.maximum(l,1e-7)
    N = np.exp(-0.5 * ((np.log(l)**2 / tp**2) + tp**2) )
    
    return N


def Com_model(x,n,npr,s1,delta_m,s2,m2,t,sp,delta_mp,tp,alpha):
    return n*electron_model(x,alpha,s1,delta_m,s2,m2,t) + npr*CCproton_model(x,sp,delta_mp,tp,m2)
    
def Com_model_CC(x,s1,delta_m,s2,m2,t,sp,delta_mp,tp,alpha,ncc,npcc):
    return ncc*electron_model(x,alpha,s1,delta_m,s2,m2,t) + npcc*CCproton_model(x,sp,delta_mp,tp,m2)
        

def cumulative_model(edges,n,npr,s1,delta_m,s2,m2,t,sp,delta_mp,tp,alpha):
    x = (edges[1:] + edges[:-1])/2
    p = Com_model(x,n,npr,s1,delta_m,s2,m2,t,sp,delta_mp,tp,alpha)
    cp = np.cumsum(p)
    return np.concatenate(([0],cp))


def cumulative_model_CC(edges,s1,delta_m,s2,m2,t,sp,delta_mp,tp,alpha,ncc,npcc):
    x = (edges[1:] + edges[:-1])/2
    p = Com_model_CC(x,s1,delta_m,s2,m2,t,sp,delta_mp,tp,alpha,ncc,npcc)
    cp = np.cumsum(p)
    return np.concatenate(([0],cp))

#########################################################

def El_template_fit_function(x,n,npr,s1,delta_m,s2,m2,t,sp,delta_mp,tp,alpha,TRD_binning,events_electron):
    
    #print(events_electron)
     
    dx=TRD_binning[1:]-TRD_binning[:-1]
    el_scale, el_scale_error = quad(electron_model,x[0],x[-1],args=(alpha,s1,delta_m,s2,m2,t))
    if el_scale==0:
        el_scale=1

    cpp_scale, cpp_scale_error= quad(CCproton_model,x[0],x[-1],args=(sp,delta_mp,tp,m2))
    if cpp_scale==0:
        cpp_scale=1
    
    #print(el_scale, cpp_scale)

    def normalized_electron_model(x):
                
        return (1/el_scale)* electron_model(x,alpha,s1,delta_m,s2,m2,t)
    def normalized_CCproton_model(x):
        return (1/cpp_scale)*CCproton_model(x,sp,delta_mp,tp,m2)
    
    # print(quad(normalized_electron_model,x[0],x[-1]))
    # print(quad(normalized_CCproton_model,x[0],x[-1]))
        
        
    def combined_model(x,ne,nc):
        return ne* normalized_electron_model(x) + nc* normalized_CCproton_model(x)
    
    def cumulativemodel(edges,ne,nc):
        #x = (edges[1:] + edges[:-1])/2
        p = combined_model(edges,ne,nc)* np.concatenate((dx,[dx[-1]]))
        #cp=np.array([quad(combined_model,x[0],edge,args=(ne,nc))[0] for edge in edges])
        cp = np.cumsum(p)
        return cp
        # print(cp[-1],ne,nc)
        # return np.concatenate(([0],cp))
        
    liklihood = ExtendedBinnedNLL(events_electron,TRD_binning,cumulativemodel)
   
    guess=dict(ne=5000,nc=5000)
    minuit = Minuit(liklihood, **guess)
    minuit.limits['ne']=(0,None)
    minuit.limits['nc']=(0,None)
    
    minuit.migrad()
    #print(minuit)
    
    
    return minuit.values, minuit.errors
    
    
    
    
    
    






