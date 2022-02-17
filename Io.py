import datetime
import json
from typing import List, Optional
import pandas as pd
import os
import warnings
import math
import pickle

#%%

def check_path_exists(output_path: str):
    """
    if output_path is a dir, check the dir exist, or mkdir. 
    if output_path is a file, check the folder exists, or mkdir.
    """
    if output_path is None:
        return None
    if os.path.isdir(output_path):
        # if this is a folder that already exsits:
        return output_path
    else:
        # if there is no extension in file name, consider it as a dir.
        # check if it exsits, and if not, create the dir.
        path, ext = os.path.splitext(output_path)

        if ext == '':
            # This is a dir or a file without ext.
            if not os.path.exists(path) or os.path.isfile(path):
                os.makedirs(path)
        else:
            # This is a file name.
            if os.path.exists(output_path):
                warnings.warn("{} already exists. Will be over-written.".format(output_path))
            dir, f_name = os.path.split(output_path)
            if not os.path.exists(dir):
                # if the dir not exsit.
                os.mkdir(dir)
            elif os.path.isfile(path):
                # if it is a file with the same name.
                os.mkdir(dir)

        return output_path


def check_is_file(path: str or List) -> bool:
    """
    Check if the path(s) is a file or a dir.
    If a list is passed, it's not a file.
    Check if the path has a extention. If not it'll be considered a dir.
    """
    if not isinstance(path, str):
        # if path is a list, or something, return a False. It's not a file.
        return False
    file, ext = os.path.splitext(path)
    return ext != ''


def parent_dir(path: str) -> str:
    """If given a file path, strip it to only keep the folder"""
    if check_is_file:
        return os.path.split(path)[0]
    else:
        return path
#%%
def read_df(path: str, convert_index=True) -> pd.DataFrame:
    """
    Read single df. 
    If convert_index is True, this function will search for that if the index or columns is RangeIndex. 
    If yes, it will be converted to I{} or F{} format. 
    """
    if path.endswith(".parquet"):
        df = pd.read_parquet(path, engine='fastparquet')
    elif path.endswith(".csv"):
        df = pd.read_csv(path, index_col=0)
    elif path.endswith(".tsv"):
        df = pd.read_table(path, index_col=0)
    elif path.endswith(".pickle") or path.endswith('.pkl'):
        with open(path, 'rb') as f:
            df = pickle.load(f)
    elif path.endswith(".json"):
        dict = json.load(path)
        df = pd.DataFrame(dict)
    elif path.endswith(".txt"):
        df = pd.read_table(path, index_col=0)
    else:
        raise ValueError("Not supported format: " + path)

    if convert_index:
        if isinstance(df.index, pd.RangeIndex):
            df.index = ["I{}".format(ii) for ii in range(len(df))]
        if isinstance(df.columns, pd.RangeIndex):
            df.columns = ["I{}".format(ii) for ii in range(df.shape[1])]

    return df


def read_df_list(paths: List) -> pd.DataFrame:
    dfs = []
    for each_path in paths:
        dfs.append(read_df(each_path))
    cdf = pd.concat(dfs)
    if sum(cdf.index.duplicated()) > 0:
        print(cdf.index[cdf.index.duplicated()])
        # raise ValueError("Duplicated index are found in the DFs.")
        print("Duplicated index are found in the DFs.")
    return cdf


def save_df(obj: pd.DataFrame, save_path:str, tag=None):
    check_path_exists(save_path)
    if tag is not None:
        tag = "_" + tag.strip("_")
    else:
        tag = ""
    if save_path.endswith(".parquet"):
        if isinstance(obj.index, pd.RangeIndex):
            obj.index = ["F{}".format(ii) for ii in range(obj.shape[1])]
        save_path = save_path.replace(".parquet", tag + ".parquet")
        obj.to_parquet(save_path, compression='gzip')
    elif save_path.endswith(".csv"):
        save_path = save_path.replace(".csv", tag + ".csv")
        obj.to_csv(save_path)
    else:
        raise ValueError("Unkonwn save type.")


#%%
def log_to_csv(message, path: str) -> None:
    """Write (append) a list to csv. """
    if isinstance(message, str):
        message = [message]
    check_path_exists(path)
    if not check_is_file(path):
        path = os.path.join(path, "log.csv")
    string = "\n{}" + ",{}"*(len(message)-1)
    with open(path, 'a') as file:
        file.write(string.format(*message))


#%%
def float_to_int(x):
    # int becomes float when loaded by json
    if not isinstance(x, float):
        return x
    elif math.isnan(x) or math.isinf(x):
        return x
    else:
        if int(x) == x:
            return int(x)
        else:
            return x


def load_int_dict_from_json(path: str) -> dict:
    # automatic convert floated int back
    with open(path, 'r') as f:
        origin = json.load(f)
    processed = {key: float_to_int(origin[key]) for key in origin}
    return processed

#%%  Below are the init ones from early ver ToolBox
file_cwd = os.path.dirname(__file__)

def write_to_log(*arg):
    Str = ''
    for each in arg:
        Str += ' '
        Str += str(each)
    theTime = datetime.datetime.now().strftime('%H:%M:%S')
    print(theTime, Str)
    try:
        Str = str(Str)
        with open('log.txt','a') as file:
            file.write(theTime+'\t'+Str+'\n')
    except:
        raise Warning ("Failed to save log. ")

def mkdir(dir_name):
    try:
        os.mkdir(dir_name)
    except:
        print (dir_name+" already exsits.")
    finally:
        os.chdir(dir_name)
        
def write_csv(array,name = '_save.csv'):
    import pandas as pd
    assert isinstance(name,str)
    if not name.endswith('.csv'):
        name += '.csv'
    
    pd.DataFrame(array).to_csv(name)
    return True
    
def add_to_csv(obj, name = '_save.csv'):
    from collections.abc import Iterable
    if isinstance(obj, dict):
        if not os.path.exists(name):
            list_to_write = list(args)
            with open(name,'a') as file:
                file.write('\n')
                file.writelines(["%s," % item  for item in list_to_write])
        else:
            list_to_write = list(args.values())
            with open(name,'a') as file:
                file.write('\n')
                file.writelines(["%s," % item  for item in list_to_write])
    elif isinstance(obj, Iterable):
        with open(name,'a') as file:
            file.write('\n')
            file.writelines(["%s," % item  for item in obj])
            
"""
havent think of a way to write this
if not os.path.exists("CNN_BF_grid.csv"):
    list_to_write = list(para_dict)
    list_to_write.extend(["NRMSE",'MSE','Correlation'])
    with open("CNN_BF_grid.csv",'a') as file:
        file.writelines(["%s," % item  for item in list_to_write])
        file.write('\n')
else:
    list_to_write = list(para_dict.values())
    list_to_write.extend(cnn_results)
    with open("CNN_BF_grid.csv",'a') as file:
        file.writelines(["%s," % item  for item in list_to_write])
        file.write('\n')
"""