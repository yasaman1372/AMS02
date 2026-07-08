#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 17:27:45 2023

@author: yasaman
"""

import multiprocessing as mp
import os
import numpy as np
import awkward as ak
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.colors as colors

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataversion", default="pass6", help="version of the data set (e.g. pass6)")
    parser.add_argument("--datatype", default="ISS", help="title of the input data (e.g. ISS)")
    parser.add_argument("--version", default="pass8", help="version of the input data (e.g. pass8)")
    parser.add_argument("--tracknumber", default="multiple", help="number of tracks (single or multiple)")
    parser.add_argument("--filepath",default="/Users/yasaman/AMS02/data/pass8/CCMVABDT_template_ISS.npz",help="Path to root file to read tree from. Can also be path to a file with a list of root files or a pattern of root files (e.g. results/ExampleAnalysisTree*.root)")
    #parser.add_argument("--filepath",default="/Users/yasaman/AMS02/data/templatefit_sample/ElectronCCMVBDTTemplateFitISS6/CCMVABDT_template_ISS.npz" ,help="Path to root file to read tree from. Can also be path to a file with a list of root files or a pattern of root files (e.g. results/ExampleAnalysisTree*.root)")
    args = parser.parse_args()
    
    
    with np.load(os.path.join(args.filepath)) as result_file:
        
        CCBDT_binning = result_file["var1_binning"]
        Energy_binning = result_file["var2_binning"] 
        
        C= (CCBDT_binning[1:] + CCBDT_binning[:-1])/2 
        E= (Energy_binning[1:] + Energy_binning[:-1])/2 
        
        if args.datatype == "ISS":
            if args.tracknumber == "single":   
                pEvents = result_file["pEvents_st"]
                eEvents = result_file["eEvents_st"]
                ccpEvents = result_file["ccpEvents_st"]
                
            elif args.tracknumber == "multiple" :
                pEvents = result_file["pEvents_mt"]
                eEvents = result_file["eEvents_mt"]
                ccpEvents = result_file["ccpEvents_mt"]
                
                
            i=0
            for binn in range(len(Energy_binning) -1):
                i=i+1
                  
                e1 = round(Energy_binning[binn],2)
                e2= round(Energy_binning[binn +1],2)
                  
                figure = plt.figure(figsize=(12, 10))
                plot = figure.subplots(1, 1)
                plot.set_xlabel(r'$\Lambda_{\rm CC}$',fontsize = 26,fontweight = 'bold')
                plot.set_ylabel("Events", fontsize = 26,fontweight = 'bold')
                plot.set_yscale("log")
                plot.tick_params(axis='both', which="major",direction='in', length=14, width=1.5, labelsize=25)
                plot.tick_params(axis='both', which="minor",direction='in', length=8, width=1.5, labelsize=25)
                plt.rcParams['xtick.top'] = True
                plt.rcParams['ytick.right'] = True
                plt.minorticks_on()
                plt.rcParams["font.weight"] = "bold"
                plt.rcParams["axes.labelweight"] = "bold"

                for axis in ['top','bottom','left','right']:
                    plot.spines[axis].set_linewidth(2)
                  
                plot.step(np.concatenate(([CCBDT_binning[0]], CCBDT_binning)), np.concatenate(([0], eEvents[:,binn], [0])), where="post",linewidth=2)
                plt.fill_between(CCBDT_binning, np.concatenate(([0], eEvents[:,binn])),step="pre",alpha=0.4,color='none',hatch='//',edgecolor='b')
                
                fit_info = [f"ISS data {args.dataversion}\n"
                    f"Energy = [{e1} , {e2}]GeV "]
                plt.legend(title='ISS data'.join(fit_info),title_fontsize=14,fontsize=14,frameon=False, loc='upper right')
                
                  
                figure.savefig("/Users/yasaman/AMS02/plots/template_fit/CCMVA/"+args.version+"/electron/"+args.tracknumber+"/electron_" +args.tracknumber +"_histogram_"+str(binn)+".pdf" , dpi=250)
                plt.close(figure)   
                
                
                figure = plt.figure(figsize=(12, 10))
                plot = figure.subplots(1, 1)
                plot.set_xlabel(r'$\Lambda_{\rm CC}$',fontsize = 26,fontweight = 'bold')
                plot.set_ylabel("Events", fontsize = 26,fontweight = 'bold')
                plot.set_yscale("log")
                plot.tick_params(axis='both', which="major",direction='in', length=14, width=1.5, labelsize=25)
                plot.tick_params(axis='both', which="minor",direction='in', length=8, width=1.5, labelsize=25)
                plt.rcParams['xtick.top'] = True
                plt.rcParams['ytick.right'] = True
                plt.minorticks_on()
                plt.rcParams["font.weight"] = "bold"
                plt.rcParams["axes.labelweight"] = "bold"

                for axis in ['top','bottom','left','right']:
                    plot.spines[axis].set_linewidth(2)
                  
                plot.step(np.concatenate(([CCBDT_binning[0]], CCBDT_binning)), np.concatenate(([0], pEvents[:,binn], [0])), where="post",linewidth=2)
                plt.fill_between(CCBDT_binning, np.concatenate(([0], pEvents[:,binn])),step="pre",alpha=0.4,color='none',hatch='//',edgecolor='b')
                
                fit_info = [f"ISS data {args.dataversion}\n"
                    f"Energy = [{e1} , {e2}]GeV "]
                plt.legend(title='ISS data'.join(fit_info),title_fontsize=14,fontsize=14,frameon=False, loc='upper right')
                
                  
                figure.savefig("/Users/yasaman/AMS02/plots/template_fit/CCMVA/"+args.version +"/proton/"+args.tracknumber+"/proton_" +args.tracknumber +"_histogram_"+str(binn)+".pdf" , dpi=250)
                plt.close(figure) 
                
                figure = plt.figure(figsize=(12, 10))
                plot = figure.subplots(1, 1)
                plot.set_xlabel(r'$\Lambda_{\rm CC}$',fontsize = 26,fontweight = 'bold')
                plot.set_ylabel("Events", fontsize = 26,fontweight = 'bold')
                plot.set_yscale("log")
                plot.tick_params(axis='both', which="major",direction='in', length=14, width=1.5, labelsize=25)
                plot.tick_params(axis='both', which="minor",direction='in', length=8, width=1.5, labelsize=25)
                plt.rcParams['xtick.top'] = True
                plt.rcParams['ytick.right'] = True
                plt.minorticks_on()
                plt.rcParams["font.weight"] = "bold"
                plt.rcParams["axes.labelweight"] = "bold"

                for axis in ['top','bottom','left','right']:
                    plot.spines[axis].set_linewidth(2)
                  
                plot.step(np.concatenate(([CCBDT_binning[0]], CCBDT_binning)), np.concatenate(([0], ccpEvents[:,binn], [0])), where="post",linewidth=2)
                plt.fill_between(CCBDT_binning, np.concatenate(([0], ccpEvents[:,binn])),step="pre",alpha=0.4,color='none',hatch='//',edgecolor='b')
                
                fit_info = [f"ISS data {args.dataversion}\n"
                    f"Energy = [{e1} , {e2}]GeV "]
                plt.legend(title='ISS data'.join(fit_info),title_fontsize=14,fontsize=14,frameon=False, loc='upper right')
                
                  
                figure.savefig("/Users/yasaman/AMS02/plots/template_fit/CCMVA/"+args.version+"/ccproton/"+args.tracknumber+"/ccproton_" +args.tracknumber +"_histogram_"+str(binn)+".pdf" , dpi=250)
                plt.close(figure) 
                
                
                
        elif args.datatype == "MC":           
            if args.tracknumber == "single":
                cc_eEvents = result_file["cc_eEvents_st"]   
                
            elif args.tracknumber == "multiple" :    
                cc_eEvents = result_file["cc_eEvents_mt"]  
                
            i=0
            for binn in range(len(Energy_binning) -1):
                i=i+1
                      
                e1 = round(Energy_binning[binn],2)
                e2= round(Energy_binning[binn +1],2)
                      
                figure = plt.figure(figsize=(12, 10))
                plot = figure.subplots(1, 1)
                plot.set_xlabel(r'$\Lambda_{\rm CC}$',fontsize = 26,fontweight = 'bold')
                plot.set_ylabel("Events", fontsize = 26,fontweight = 'bold')
                plot.set_yscale("log")
                plot.tick_params(axis='both', which="major",direction='in', length=14, width=1.5, labelsize=25)
                plot.tick_params(axis='both', which="minor",direction='in', length=8, width=1.5, labelsize=25)
                plt.rcParams['xtick.top'] = True
                plt.rcParams['ytick.right'] = True
                plt.minorticks_on()
                plt.rcParams["font.weight"] = "bold"
                plt.rcParams["axes.labelweight"] = "bold"

                for axis in ['top','bottom','left','right']:
                    plot.spines[axis].set_linewidth(2)
                      
                plot.step(np.concatenate(([CCBDT_binning[0]], CCBDT_binning)), np.concatenate(([0], cc_eEvents[:,binn], [0])), where="post",linewidth=2)
                plt.fill_between(CCBDT_binning, np.concatenate(([0], cc_eEvents[:,binn])),step="pre",alpha=0.4,color='none',hatch='//',edgecolor='b')
                    
                fit_info = [f"ISS data {args.dataversion}\n"
                        f"Energy = [{e1} , {e2}]GeV "]
                plt.legend(title='ISS data'.join(fit_info),title_fontsize=14,fontsize=14,frameon=False, loc='upper right')
                
                figure.savefig("/Users/yasaman/AMS02/plots/template_fit/CCMVA/"+args.version+"/ccelectron/"+args.tracknumber+"/ccelectron_" +args.tracknumber +"_histogram_"+str(binn)+".pdf" , dpi=250)
                plt.close(figure) 
                
        
if __name__ == "__main__":
    main()     
  
            