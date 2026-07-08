#!/usr/bin/env python3

import ROOT

inputfile = ROOT.TFile("LeptonAnalysis_EcalChiSquareLateralNormalizedCut.root")
tf = inputfile.Get("splineFitFunction")
tg = ROOT.TGraph(tf)

outputfile = ROOT.TFile("CutValue.root", "RECREATE")
outputfile.WriteObject(tg, "CutValues")
