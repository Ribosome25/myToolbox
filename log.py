
def write_to_log(Str):
    theTime = datetime.datetime.now().strftime('%H:%M:%S')
    try:
        with open('log.txt','a') as file:
            file.write('\n'+theTime+'\t'+Str)
    except:
        raise Warning ("Filed to save log. ")
            
