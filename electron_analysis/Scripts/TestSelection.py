#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 09:26:38 2022

@author: yasaman
"""

import yaml
import ElectronAnalysisTools

file = "/Users/yasaman/AMS02/YasamanAnalysis/Configuration/LeptonAnalysisSelectionCuts.yaml"

with open(file) as config_file:
    config= yaml.safe_load(config_file)['IdentificationCuts']
    
ElectronAnalysisTools.Selection.load(config)    

    

    
