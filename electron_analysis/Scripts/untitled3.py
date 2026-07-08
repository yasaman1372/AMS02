#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  4 15:49:17 2023

@author: yasaman
"""

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
time = np.linspace(0, 10, 1000)
height = np.sin(time)
weight = time*0.3 + 2
score = time**2 + height
distribution = np.random.normal(0, 1, len(time))
fig = plt.figure(figsize=(10, 5))
gs = GridSpec(nrows=2, ncols=2)
ax0 = fig.add_subplot(gs[0, 0])
ax0.plot(time, height)
ax1 = fig.add_subplot(gs[1, 0])
ax1.plot(time, weight)
ax2 = fig.add_subplot(gs[:, 1])
ax2.plot(time, score)
ax3 = fig.add_axes([0.6, 0.6, 0.2, 0.2])
ax3.hist(distribution)
plt.show()

