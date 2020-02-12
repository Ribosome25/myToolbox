import datetime
import os

file_cwd = os.path.dirname(__file__)

def write_to_log(*arg):
    Str = ''
    for each in arg:
        Str += ' '
        Str += str(each)
    theTime = datetime.datetime.now().strftime('%H:%M:%S')
    print(theTime,Str)
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