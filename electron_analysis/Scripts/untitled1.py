#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 12:58:23 2023

@author: yasaman
"""

import numpy as np

var1_name= 'TrdEstimator'
var1_min = 0
var1_max = 2
bin1_num = 100
TRD_binning = np.linspace(var1_min, var1_max, bin1_num +1)

signed_trd_binning = np.concatenate((-TRD_binning[:0:-1], TRD_binning))

# print(TRD_binning[:0:-1])
print(TRD_binning[:2:-1])
# print(signed_trd_binning)
