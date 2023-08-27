# -*- coding: utf-8 -*-
"""
Created on Sat Aug 26 17:02:27 2023

@author: mchamaillard
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
u_L = cv2.imread('image_L.jpg',cv2.IMREAD_GRAYSCALE)

u=u_L[0]


u_test=u


correlation_croise=np.correlate(u*1., u_test)

plt.plot(correlation_croise)
