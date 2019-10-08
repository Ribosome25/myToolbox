

import pandas as pd
import numpy as np

def top_percentage_distribution(data, percentage, pick_highest = True, return_values = False):
    '''
    Given a dataset, find the top % of this distribution.
    @ Parameters:
        data: the data set
        percentage: this is in percentage e.g. 10 for 10%
        pick_highest: Ture for picking highest data, False for picking lowest data.
        return_values: True for returning the chosen data as well, False for returning the threshold only. 
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
    
def normalize_int_between(data,low = 0,high = 255):
    if isinstance(data, pd.DataFrame):
        _is_df = True
        _df_index = data.index
        _df_col = data.columns
    else:
        _is_df = False
        
    data = np.asarray(data)
    if data.min() == data.max():
        raise ValueError ("Min == max. ")
    norm_data = (data-data.min()) / (data.max() - data.min())
    int_data = (norm_data * (high+1-low) + low).astype(int)
    int_data[int_data == high+1] = high
    
    if _is_df:
        int_data = pd.DataFrame(int_data,index = _df_index,columns = _df_col)
    
    return int_data

def symmetrize_matrix(K, mode='average'):
    """
    3 modes.
    or 取最大？ and 取最小？#TODO
    """
    if mode == 'average':
        return 0.5*(K + K.transpose())
    elif mode == 'or':
        Ktrans = K.transpose()
        dK = abs(K - Ktrans)
        K = K + Ktrans
        K = K + dK
        return 0.5*K
    elif mode == 'and':
        Ktrans = K.transpose()
        dK = abs(K - Ktrans)
        K = K + Ktrans
        K = K - dK
        return 0.5*K
    else:
        raise ValueError('Did not understand symmetrization method')
        

def nearestPSD(A,epsilon=0):
    """
    https://stackoverflow.com/questions/10939213/how-can-i-calculate-the-nearest-positive-semi-definite-matrix
    """
    n = A.shape[0]
    eigval, eigvec = np.linalg.eig(A)
    val = np.matrix(np.maximum(eigval,epsilon))
    vec = np.matrix(eigvec)
    T = 1/(np.multiply(vec,vec) * val.T)
    T = np.matrix(np.sqrt(np.diag(np.array(T).reshape((n)) )))
    B = T * vec * np.diag(np.array(np.sqrt(val)).reshape((n)))
    out = B*B.T
    return(out)