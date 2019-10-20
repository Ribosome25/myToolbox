
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
    assert isinstance(Paras_dict, dict)
    Para_names = list(Paras_dict.keys())
    Para_table = pd.DataFrame(grid_search_para_comb(Paras_dict,Para_names),columns = Para_names)
    return Para_table

def expand_col_to_onehot(input_df,sele_cols):
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

