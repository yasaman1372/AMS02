#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 10:50:52 2022

@author: yasaman
"""

from iminuit import Minuit
import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit
from iminuit.cost import ExtendedBinnedNLL, LeastSquares
import os


def m1_model(x,a,b):
    return a*x +b

def m2_model()

def cumulative_model(edges,a,b):
    x = (edges[1:] + edges[:-1])/2
    p = m1_model(x,a,b)
    cp = np.cumsum(p)
    return np.concatenate(([0],cp))


values = np.loadtxt("values.txt")
errors = np.loadtxt("errors.txt")
guess=dict(a=-0.1e-3,b=0.39)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--title", default="MC", help="title of the input data (e.g. ISS)")
    parser.add_argument("--filename", default= "/Users/yasaman/AMS02/data/",help="Path to root file to read tree from. Can also be path to a file with a list of root files or a pattern of root files (e.g. results/ExampleAnalysisTree*.root)")

    args = parser.parse_args()
    font_size='15'


    with np.load(os.path.join(args.filename, "results.npz")) as result_file:
        var2_binning = result_file["var2_binning"]
             
    y= (var2_binning[1:] + var2_binning[:-1])/2
    
    liklihood = ExtendedBinnedNLL(values[:,2],var2_binning,cumulative_model)
    minuit = Minuit(liklihood, **guess)
    minuit.migrad()
    print(minuit)
    
    figure = plt.figure(figsize=(12, 6.15))
    plot = figure.subplots(1, 1)
    plot.set_xlabel("Energy/Gev",fontsize = font_size)
    plot.set_ylabel("m1", fontsize = font_size)
    plot.set_xscale("log")
    plot.scatter(y,values[:,2], label="data")
    plot.plot(y,m1_model(y,**guess),label='guess')
    plot.plot(y,m1_model(y,*minuit.values), label= 'fit')
        
    plt.legend()
        
        
    figure.savefig("m1.pdf" , dpi=250)
    plt.close(figure)


if __name__ == "__main__":
    main() 