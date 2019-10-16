# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 16:59:52 2019

@author: Ruibzhan

return the threshold value.
"""

import numpy as np
import pandas as pd

def top_percentage_distribution(data, percentage, pick_highest = True, return_values = False):
    '''
    Given a dataset, find the top % of this distribution.
    @ Parameters:
        data: the data set
        percentage: this is in percentage e.g. 10 for 10%
        pick_highest: Ture for picking highest data, False for picking lowest data.
        return_values: True for returning the picked data as well, False for returning the threshold only. 
    '''
    data = np.asarray(data,dtype = float,order = 'C').flatten()
    total_number = len(data)
    if len(data.flatten())<10:
        raise ValueError("Too few data points.")
    pick_number = int(round(percentage * total_number / 100))
    if pick_highest == True:
        index = np.argsort(data)[::-1]# Highest to lowest
    else:
        index = np.argsort(data)# lowest to highest
    threshold = data[index[pick_number-1]]
    values = data[index[:pick_number]]
    if return_values:
        return threshold, values
    else:
        return threshold