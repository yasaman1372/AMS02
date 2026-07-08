#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 11 13:53:08 2022

@author: yasaman
"""

import ROOT
inputfile = ROOT.TFile("LeptonAnalysis_EnergyDependantCuts.root")
tf = inputfile.Get('hTrackerEcalMatchingYUpperCut')
tg = ROOT.TGraph(tf)

outputfile = ROOT.TFile("TrackerEcalMatchingYUpperCut.root", "RECREATE")
outputfile.WriteObject(tg,"TrackerEcalMatchingYUpperCut")
