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