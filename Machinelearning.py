
import pandas as pd
import numpy as np

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
    """
    For simpler usage. 
    Example:
    paras = {'kern': ['lin','rbf'], 'sigma':np.logspace(-1,1,10), 'B': np.linspace(1,5,5), 'lmbd' : [0,1]}
    Params = grid_search_dict_to_df(paras)
    for idx, each_para in Params.iterrows():
        pass
    """
    assert isinstance(Paras_dict, dict)
    Para_names = list(Paras_dict.keys())
    Para_table = pd.DataFrame(grid_search_para_comb(Paras_dict,Para_names),columns = Para_names)
    return Para_table

def expand_col_to_onehot(input_df,sele_cols):
    raise ValueError("I moved this to ./Preprocessing. ")
