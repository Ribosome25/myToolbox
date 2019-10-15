# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 14:54:19 2019

@author: Ruibzhan
"""
import numpy as np
from scipy.stats import pearsonr

def NRMSE(Y_Target, Y_Predict, multi_dimension = False):
    Y_Target = np.array(Y_Target); Y_Predict = np.array(Y_Predict);
    if multi_dimension:
        Y_Target = Y_Target.flatten()
        Y_Predict = Y_Predict.flatten()
    else:
        Y_Target = Y_Target.reshape(len(Y_Target),1)
        Y_Predict = Y_Predict.reshape(len(Y_Predict),1)
    Y_Bar = np.mean(Y_Target)
    Nom = np.sum((Y_Predict - Y_Target)**2)
    Denom = np.sum((Y_Bar - Y_Target)**2)
    MSE = np.mean((Y_Predict - Y_Target)**2)
    NRMSE_Val = np.sqrt(Nom/Denom)
    return NRMSE_Val, MSE

def triplet(Y_Target, Y_Predict, multi_dimension = False, output_format = list):
    Y_Target = np.asarray(Y_Target,order = 'C')
    Y_Predict = np.asarray(Y_Predict,order = 'C')
    if multi_dimension:
        Y_Target = Y_Target.flatten()
        Y_Predict = Y_Predict.flatten()
    else:
        Y_Target = Y_Target.reshape(len(Y_Target),1)
        Y_Predict = Y_Predict.reshape(len(Y_Predict),1)
        
    if len(Y_Target)==len(Y_Predict):
        mse = np.mean((Y_Predict - Y_Target)**2)
        var = np.var(Y_Target)
        nrmse = np.sqrt( mse/var )
        pcorrs, pvalue = pearsonr(Y_Predict,Y_Target)
        if multi_dimension:
            pcorr = pcorrs
        else:
            pcorr = pcorrs[0]
    else:
        raise ValueError ('Target & Predicted are not of same length.')
        return np.nan
    
    if output_format==list:
        return([nrmse, mse, pcorr])
    elif output_format==dict:
        return({'NRMSE':nrmse,
                'MSE':mse,
                'P-CorrCoef':pcorr
                })
    else:
        print("Unknown format, to be done. return list.")
        return([nrmse,mse,pcorr])
    
def test_triplet(n_dim=3):
    if n_dim==1:
        print("TBD")
        return True
    
    y1 = np.random.randint(0,19,(17,n_dim))
    y2 = np.random.randint(0,9,(17,n_dim))
    tri_nrmse, tri_mse, tri_pcorr = triplet(y1,y2,True,list)
    old_nrmse, old_mse = NRMSE(y1,y2,True)
    assert(tri_nrmse == old_nrmse)
    assert(tri_mse == old_mse)
    
    rst_dict = triplet(y1,y2,True,dict)
    assert(rst_dict['NRMSE'] == old_nrmse)
    return None
    
def Accuracy(Y_Target,Y_Predict,error_rate = False,multi_dimension = False):
    Y_Target = np.asarray(Y_Target,order = 'C')
    Y_Predict = np.asarray(Y_Predict,order = 'C')
    if multi_dimension:
        Y_Target = Y_Target.flatten()
        Y_Predict = Y_Predict.flatten()
    else:
        Y_Target = Y_Target.reshape(len(Y_Target),1)
        Y_Predict = Y_Predict.reshape(len(Y_Predict),1)
    if len(Y_Target)==len(Y_Predict):
        correct = np.sum(Y_Target==Y_Predict)
        if error_rate:
            Value = 1-(correct/len(Y_Predict))
        else:
            Value = correct/len(Y_Predict)
    else:
        raise ValueError ('Target & Predicted are not of same length.')
        return np.nan
    
    return Value

def F1_score(Y_Target, Y_Predict, multi_dimension = False):
    # Y target and Y predict are forced to convert to bool. 
    # Only applicatble for 2-class problems.
    Y_Target = np.asarray(Y_Target,order = 'C').astype(bool)
    Y_Predict = np.asarray(Y_Predict,order = 'C').astype(bool)
    if multi_dimension:
        Y_Target = Y_Target.flatten()
        Y_Predict = Y_Predict.flatten()
    else:
        Y_Target = Y_Target.reshape(len(Y_Target),1)
        Y_Predict = Y_Predict.reshape(len(Y_Predict),1)
        
    if len(Y_Target)==len(Y_Predict):
        ...
        _true_posi = sum(Y_Target & Y_Predict)
        _false_posi = sum(~Y_Target & Y_Predict)
        _false_neg = sum(Y_Target & ~Y_Predict)
        _precision = _true_posi/(_true_posi + _false_posi)
        _recall = _true_posi/(_true_posi + _false_neg)
        Value = 2*_precision / (_precision + _recall)
        
    else:
        raise ValueError ('Target & Predicted are not of same length.')
        return np.nan
    
    return Value
        
        
from sklearn.model_selection import KFold

def kFold_NRMSE(Mdl,X,y,k = 5):
    kf = KFold(n_splits=k)
    X = np.asarray(X,order = 'C')
    y = np.asarray(y,order = 'C')
    
    errors = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        Mdl.fit(X_train,y_train.ravel())
        y_prd = Mdl.predict(X_test)
        errors.append(NRMSE(y_test,y_prd)[0])
#        errors.append(np.corrcoef(y_test.ravel(),y_prd.ravel()))# tempera
#        return errors # tempra 
    return np.mean(errors)

        
    