import datetime
import os
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