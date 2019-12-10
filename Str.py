import pandas as pd


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

def transform_invalid_char_in_df(df, which_col = None, to_char = '_'):
    """Strip the strs, and replace the invalid chars into underscore"""
    inv_chars = [' ','/','\\']
    if not isinstance(df,pd.DataFrame):
        df = pd.DataFrame(df)
    if which_col is None:
        which_col = df.columns
    if isinstance(which_col,str):
        which_col = [which_col]
    for each_col in which_col:
        df[each_col] = df[each_col].str.strip()
        for each_inv in inv_chars:
            df[each_col] = df[each_col].str.replace(each_inv,to_char)
    return df

def _check_match(A,B,print_right_here=1):
    """list like A B
    等长吗，等集合吗，等顺序吗
    """
    rst = [False,False,False]
    if len(A)==len(B):
        rst[0]=True
    if set(A)==set(B):
        rst[1]=True
    if rst[0] and rst[1]:
        A = list(A)
        B = list(B)
        if A==B:
            rst[2]=True
            
    if print_right_here:
        if rst[2]:
            print("Excat match.")
        elif rst[1] and rst[0]:
            print("Not same sequence.")
        elif (not rst[0]) and rst[1]:
            print("Duplicated elements.")
        elif not rst[1]:
            print("Different elements.")
            
    return tuple(rst)

def check_dataset_matched(A,B):
    """"""
    print('\n>>> Check data sets index and cols matched:')
    if isinstance(A,pd.DataFrame) and isinstance(B,pd.DataFrame):
        print(">> Check matching: two DataFrames. ")
        print("> Indexs:")
        idxs = _check_match(A.index,B.index)
        print("> Columns:")
        cols = _check_match(A.columns,B.columns)
    elif isinstance(A,pd.DataFrame) and isinstance(B,pd.Series):
        print('>> Check matching: DataFrame -> Series. ')
        idxs = _check_match(A.index,B.index)
        cols = (1,1,1)
    elif isinstance(A,pd.Series) and isinstance(B,pd.DataFrame):
        print('>> Check matching: Series -> DataFrame. ')
        idxs = _check_match(A.index,B.index)
        cols = (1,1,1)
    elif isinstance(A,pd.Series) and isinstance(B,pd.Series):
        print('>> Check matching: Series -> Series. ')
        idxs = _check_match(A.index,B.index)
        cols = (1,1,1)
    else:
        raise TypeError("now only support pandas things.")
        
    return all(idxs) and all(cols) # a py thing. all(iterable), any(iterable), logic compu of iter boolean.