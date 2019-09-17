# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 15:32:47 2019

@author: Ruibzhan

The ToolBox
"""


import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
#%% String operations
class Str:
    @staticmethod
    def truncate_cell_line_names(data,index = True,separator = '_',preserve_str_location = 0):
        '''
        This is for truncating cell line names such as 'CAL120_breast' to 'CAL120'
        Split the input Str by separator, and preserve the No. preserve_str_location th part. 
        Input is DaraFrame, List, or Index. (Todo)
        If input data is in DataFrame, a option is given as: trucate the index(0) or the columns (1).
        Return the same type as input. 
        The cell line names has to be in the index or column. otw will try truc the index.
        @ Parameters:
            index,
            separator,
            preserve_str_location
        @ Returns:
            Same data type.
        '''
        if isinstance(data,pd.DataFrame) or isinstance(data, pd.Series):
            if index or isinstance(data, pd.Series):
                data.index = [x[preserve_str_location] for x in data.index.str.split(separator)]
                return data
            else:
                data.columns = [x[preserve_str_location] for x in data.columns.str.split(separator)]
                return data
        
        elif isinstance(data,pd.Index):
            ''' Todo: check if this is passed as Ref or Copy. 
            '''
            return [x[preserve_str_location] for x in data.str.split(separator)]
        
        elif isinstance(data,list):
            return [x.split(separator)[preserve_str_location] for x in data]
        
        else:
            raise TypeError

    def transform_underscore(data, underscore_to_minus = True, target = 'columns'):
        """
        This is for unify the symbols, change all the - or _ in the index or columns.
        @ Parameter:
            data: DataFrame;
            underscore_to_minusminus: Change _ to - , if False, viseversa.
            target: Do it on index, or columns, or both.
        @ Returns:
            Same type as given.
        """
        if isinstance(data,pd.DataFrame):
            if target in ['columns','column','col']:
                if underscore_to_minus:
                    data.columns = data.columns.str.replace('_','-')
                else:
                    data.columns = data.columns.str.replace('-','_')
            elif target in ['index','idx']:
                if underscore_to_minus:
                    data.columns = data.columns.str.replace('_','-')
                else:
                    data.columns = data.columns.str.replace('-','_')
            return data
        
        elif isinstance(data,list):
            print('To do')
        
        


                
#%% Stats, Math

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

    
#%%  Preprocessing
def Impute(X,k = 5,metric = 'correlation',axis = 0,weighted = False):
    '''
    numeric knn imputation.
    (Need to check if the array ref is passed or the array value is passes. Better to use array.copy() when calling this func())
    @ Input:
        np.array or pd.Dataframe
    @ Parameters:
        k: k nearest neighbors are selected to impute the missing values
        metric: distance metric. See scipy.cdist.
        axis: 0 means finding the nearest k rows, 1 means finding the nearest k cols.
        weighted: When nearest k neighbors are picked out. Should we use the weighted avg of them. 
    @ Returns:
        DF or array after imputation.
    '''

    # If imputate col-wise: transpose it here, and transpose back in the end
    if axis == 1:
        X = X.T
    if X.shape[0]<k+1:
        raise Warning ("Samples are not enough for imputation. Return as the same.")
        return X
    if isinstance(X, pd.DataFrame):
        _is_df = True
        _df_cols = X.columns
        _df_idxs = X.index
    else:
        _is_df = False
    # Forcing convert to numpy.array
    X = np.asarray(X,dtype = float,order = 'C')

    # Get NaN map, where is NaN, and which row needs to be impu    
    _shape = X.shape
    NaN_map = np.isnan(X)
    sample_contains_NaN = NaN_map.sum(axis = 1)

    dist_matr, global_mean = _distance_matrix(X.copy(),metric = metric) # global mean is for the case when k-nn are all mssing at that point.
    for row_id in range(_shape[0]):
        if sample_contains_NaN[row_id]:
            sort_idx = dist_matr[row_id,:].argsort() # Returns [where is 1st smallest, where is 2nd smallest, ... etc].
            min_dist_sample_idxs = sort_idx[1:k+1]# For sure itself is the smallest distance. 1:k+1, Discard it.
            min_dist_samples = X[min_dist_sample_idxs,:]
            min_dists = dist_matr[row_id,:][min_dist_sample_idxs]
            if weighted:
                # Need to be tested.
                weighted_min_dist_samples = np.dot(min_dists.reshape(-1,1), min_dist_samples)
                knn_col_mean = np.nanmean(weighted_min_dist_samples,axis = 0)
            else:
                knn_col_mean = np.nanmean(min_dist_samples,axis = 0)
            NaN_col_idx = np.where(np.isnan(X[row_id,:]))[0]# Return a tuple. For only having 1 dim, so write a [0]
            for each_idx in NaN_col_idx:
                if sum(np.isnan(min_dist_samples[:,each_idx]))==k:
                    # all this col is NaN,take universal col mean, otw take knn_mean
                    X[row_id,each_idx] = np.take(global_mean,each_idx)
                else:
                    X[row_id,each_idx] = np.take(knn_col_mean,each_idx)
            
        else:
            continue

    # if do imputation col-wise: transpose back here
    if axis == 1:
        X = X.T
        
    if _is_df:
        X = pd.DataFrame(X,index = _df_idxs,columns = _df_cols)
    
    return X

def _distance_matrix(X,metric):
    # When calculating the distance, Replace NaN with col mean first.
    
    #Obtain mean of columns as you need, nanmean is just convenient.
    global_mean = np.nanmean(X, axis=0)
    #Find indicies that you need to replace
    inds = np.where(np.isnan(X)) # return tuple with 2 arrays, one-on-one matched, idx in [0], col in [1] 
    #Place column means in the indices. Align the arrays using take
    X[inds] = np.take(global_mean, inds[1])
    dist_matr = cdist(X,X,metric = metric)
    return dist_matr,global_mean

#%% Machine learning, datasets
def grid_search_para_comb(Paras,Para_names):
    """
    Generate the parameter combinations from given parameter lists.
    i.e. from n-dim to 1-dim. Finding out all the combinations
    
    Example:
    Paras = {'kern': ['lin','rbf'], 'sigma':np.logspace(-1,1,10), 'B': np.linspace(1,5,5), 'lmbd' : [0,1]}
    Para_names = list(Paras.keys())
    Para_table = pd.DataFrame(GridSearchTL.get_comb(Paras,Para_names),columns = Para_names)
    
    """
    if len(Para_names) == 1:
        return np.vstack(Paras[Para_names[0]])
    
    this_para = Para_names[-1]
    temp_array = []
    for each_value in Paras[this_para]:
        get_array = grid_search_para_comb(Paras,Para_names[:-1])
        length = len(get_array)
        attach_array = np.repeat(np.array(each_value).reshape(-1,1),length,axis = 0)
        return_array = np.hstack((get_array,attach_array))
        temp_array.append(return_array)
    return np.vstack(temp_array)
    
def grid_search_dict_to_df(Paras_dict):
    assert isinstance(Paras_dict, dict)
    Para_names = list(Paras_dict.keys())
    Para_table = pd.DataFrame(grid_search_para_comb(Paras_dict,Para_names),columns = Para_names)
    return Para_table

def expand_col_to_bool_dataset(input_df,sele_cols):
    '''
    for some data sets, targets are given as str, str, in some columns
    this is to expand this kind of dataset into sparsed boolean dataframe (exsit or not)
    paras: input dataframe, with index=instances, selected columns to expand.
    returns: a dataframe, with these two cols expanded. 
    features are not having sequnces, all sele_cols are just mixed together. 
    '''
    if not isinstance(input_df,pd.DataFrame):
        input_df = pd.DataFrame(input_df)
        
    if isinstance(sele_cols,str):
        sele_cols = [].append(sele_cols)
    
    all_columns = set()
    for each in input_df[sele_cols].values.flatten():
        readed = [x.strip() for x in each.strip().split(',')]
        all_columns = all_columns|set(readed)
    all_columns = list(all_columns)
    all_columns.sort()
    
    table = pd.DataFrame(np.zeros((input_df.shape[0],len(all_columns)),dtype = bool),
                         index = input_df.index,columns=all_columns)
    for idx,each_row in input_df[sele_cols].iterrows():
        for each in each_row:
            readed = [x.strip() for x in each.strip().split(',')]
            for each_ft in readed:
                table.loc[idx,each_ft] = True
    return table

#%% I/O
import datetime
def write_to_log(Str):
    theTime = datetime.datetime.now().strftime('%H:%M:%S')
    print(theTime,Str)
    try:
        Str = str(Str)
        with open('log.txt','a') as file:
            file.write(theTime+'\t'+Str+'\n')
    except:
        raise Warning ("Failed to save log. ")