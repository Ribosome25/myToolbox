# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 14:54:19 2019

@author: Ruibzhan
"""
from sklearn.model_selection import KFold
import numpy as np
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import copy
# %%


def get_performance(model, metric_func, X_train, Y_train, X_test, Y_test):
    mdl = copy.deepcopy(model)
    mdl.fit(X_train, Y_train)
    pred = mdl.predict(X_test)
    perfm = metric_func(Y_test, pred)
    print(perfm)
    return perfm
# %%


def NRMSE(Y_Target, Y_Predict, multi_dimension=False):
    Y_Target = np.array(Y_Target)
    Y_Predict = np.array(Y_Predict)
    if multi_dimension:
        Y_Target = Y_Target.flatten()
        Y_Predict = Y_Predict.flatten()
    else:
        Y_Target = Y_Target.reshape(len(Y_Target), 1)
        Y_Predict = Y_Predict.reshape(len(Y_Predict), 1)
    Y_Bar = np.mean(Y_Target)
    Nom = np.sum((Y_Predict - Y_Target)**2)
    Denom = np.sum((Y_Bar - Y_Target)**2)
    MSE = np.mean((Y_Predict - Y_Target)**2)
    NRMSE_Val = np.sqrt(Nom/Denom)
    return NRMSE_Val, MSE


def two_correlations(Y_Target, Y_Predict, multi_dimension=True, output_format=list):
    """
    Returns the Spearman correlation and Pearson Correlation
    """
    Y_Target, Y_Predict = _check_ys(Y_Target, Y_Predict, multi_dimension)
    scorrs, pvalue = spearmanr(Y_Target, Y_Predict)
    pcorrs, pvalue = pearsonr(Y_Predict, Y_Target)
    if multi_dimension:
        scorr = scorrs
        pcorr = pcorrs
    else:
        scorr = scorrs[0]
        pcorr = pcorrs[0]
    if output_format == list:
        return([scorr, pcorr])
    elif output_format == dict:
        return({'s-corr': scorr,
                'p-corr': pcorr})
    else:
        print("Unknown format, to be done. return list.")
        return([scorr, pcorr])


def avg_correlation(Y_Target, Y_Predict):
    # Return a line of avg of correlations. This is for the DREAM project
    Y_Target = np.asarray(Y_Target)
    Y_Predict = np.asarray(Y_Predict)
    if Y_Target.ndim == 1:
        Y_Target = Y_Target.reshape(-1, 1)
    if Y_Predict.ndim == 1:
        Y_Predict = Y_Predict.reshape(-1, 1)
    cols = Y_Target.shape[1]
    rt = np.zeros(cols)

    for ii in range(cols):
        tgt = Y_Target[:, ii]
        prd = Y_Predict[:, ii]
        nan_mask = np.isnan(tgt)
        tgt = tgt[~nan_mask]
        prd = prd[~nan_mask]
        rt[ii] = two_correlations(tgt, prd)[0]
    return rt.mean()


def corr_and_error(Y_Target, Y_Predict, multi_dimension=True, output_format=list):
    """
    Returns the Spearman correlation and Pearson Correlation,
        and MSE, MAE.
    """
    Y_Target, Y_Predict = _check_y_same_dim(
        Y_Target, Y_Predict, multi_dimension)
    if multi_dimension:
        scorrs = []
        pcorrs = []
        mses = []
        maes = []
        for ii in range(Y_Target.shape[1]):
            scorr, pcorr, mse, mae = _single_corr_and_error(
                Y_Target[:, ii], Y_Predict[:, ii])
            scorrs.append(scorr)
            pcorrs.append(pcorr)
            mses.append(mse)
            maes.append(mae)
        scorr = np.mean(scorrs)
        pcorr = np.mean(pcorrs)
        mse = np.mean(mses)
        mae = np.mean(maes)
    else:
        scorr, pcorr, mse, mae = _single_corr_and_error(Y_Target, Y_Predict)
    if output_format == list:
        return([scorr, pcorr, mse, mae])
    elif output_format == dict:
        return({'s-corr': scorr,
                'p-corr': pcorr,
                'MSE': mse,
                'MAE': mae})
    else:
        print("Unknown format, to be done. return list.")
        return([scorr, pcorr, mse, mae])


def _single_corr_and_error(Y_Target, Y_Predict):
    nan_maps = np.isnan(Y_Target) | np.isnan(Y_Predict)
    Y_Target = Y_Target[~nan_maps]
    Y_Predict = Y_Predict[~nan_maps]
    scorrs, pvalue = spearmanr(Y_Target, Y_Predict)
    pcorrs, pvalue = pearsonr(Y_Predict, Y_Target)
    scorr = scorrs
    pcorr = pcorrs
    # scorr = scorrs[0]
    # pcorr = pcorrs[0]
    mse = mean_squared_error(Y_Target, Y_Predict)
    mae = mean_absolute_error(Y_Target, Y_Predict)
    return scorr, pcorr, mse, mae


def _check_y_same_dim(Y_Target, Y_Predict, multi_dimension=True):
    """ Confused by _check_ys """
    Y_Target = np.asarray(Y_Target, order='C', dtype=float)
    Y_Predict = np.asarray(Y_Predict, order='C', dtype=float)
    if multi_dimension:
        assert len(Y_Target.shape) > 1
    else:
        assert len(Y_Target.shape) == 1
    assert Y_Target.shape == Y_Predict.shape
    return Y_Target, Y_Predict


def _check_ys(Y_Target, Y_Predict, multi_dimension=True):
    """Check and transform to np.array"""
    Y_Target = np.asarray(Y_Target, order='C', dtype=float)
    Y_Predict = np.asarray(Y_Predict, order='C', dtype=float)
    if multi_dimension:
        Y_Target = Y_Target.flatten()
        Y_Predict = Y_Predict.flatten()
    else:
        Y_Target = Y_Target.reshape(len(Y_Target), 1)
        Y_Predict = Y_Predict.reshape(len(Y_Predict), 1)
    assert(len(Y_Target) == len(Y_Predict))
    return Y_Target, Y_Predict


def triplet(Y_Target, Y_Predict, multi_dimension=True, output_format=list):
    """
    Accepts appended results in each iterations.
    Put multi_dimension = Ture
    """
    # Y_Target = np.asarray(Y_Target,order = 'C',dtype=float)
    # Y_Predict = np.asarray(Y_Predict,order = 'C',dtype=float)
    # if multi_dimension:
    #     Y_Target = Y_Target.flatten()
    #     Y_Predict = Y_Predict.flatten()
    # else:
    #     Y_Target = Y_Target.reshape(len(Y_Target),1)
    #     Y_Predict = Y_Predict.reshape(len(Y_Predict),1)
    Y_Target, Y_Predict = _check_ys(Y_Target, Y_Predict, multi_dimension)

    if len(Y_Target) == len(Y_Predict):
        mse = np.mean((Y_Predict - Y_Target)**2)
        var = np.var(Y_Target)
        nrmse = np.sqrt(mse/var)
        pcorrs, pvalue = pearsonr(Y_Predict, Y_Target)
        if multi_dimension:
            pcorr = pcorrs
        else:
            pcorr = pcorrs[0]
    else:
        raise ValueError('Target & Predicted are not of same length.')
        return np.nan

    if output_format == list:
        return([nrmse, mse, pcorr])
    elif output_format == dict:
        return({'NRMSE': nrmse,
                'MSE': mse,
                'P-CorrCoef': pcorr
                })
    else:
        print("Unknown format, to be done. return list.")
        return([nrmse, mse, pcorr])


def test_triplet(n_dim=3):
    if n_dim == 1:
        print("TBD")
        return True

    y1 = np.random.randint(0, 19, (17, n_dim))
    y2 = np.random.randint(0, 9, (17, n_dim))
    tri_nrmse, tri_mse, tri_pcorr = triplet(y1, y2, True, list)
    old_nrmse, old_mse = NRMSE(y1, y2, True)
    assert(tri_nrmse == old_nrmse)
    assert(tri_mse == old_mse)

    rst_dict = triplet(y1, y2, True, dict)
    assert(rst_dict['NRMSE'] == old_nrmse)
    return None


def triplet_transfer(Y_Target, Y_Predict, Y_Source, multi_dimension=True, output_format=list):
    """
    Change da thang subtrac in var to the mean(Source Mean),
    Because we don't know the mean of DomainTarget.
    Thus, if we predict all values as the mean of Sourse, NRMSE will be 1.
    """
    Y_Target, Y_Predict = _check_ys(Y_Target, Y_Predict, multi_dimension)

    if len(Y_Target) == len(Y_Predict):
        mse = np.mean((Y_Predict - Y_Target)**2)
        #TODO: confusion
#        var = np.mean( (Y_Predict - np.mean(Y_Source.ravel()))**2 )
        var = np.mean((Y_Target - np.mean(Y_Source.ravel()))**2)
        nrmse = np.sqrt(mse/var)
        pcorrs, pvalue = pearsonr(Y_Predict, Y_Target)
        if multi_dimension:
            pcorr = pcorrs
        else:
            pcorr = pcorrs[0]
    else:
        raise ValueError('Target & Predicted are not of same length.')
        return np.nan

    if output_format == list:
        return([nrmse, mse, pcorr])
    elif output_format == dict:
        return({'NRMSE': nrmse,
                'MSE': mse,
                'P-CorrCoef': pcorr
                })
    else:
        print("Unknown format, to be done. return list.")
        return([nrmse, mse, pcorr])


def test_triplet_transfer(n_dim=3):
    if n_dim == 1:
        ys = np.random.random(size=5)
        yt = np.array([1, 2, 1, 2, 1])
        yprd = np.array([4, 3, 4, 3, 4])
        ymeans = np.array([ys.mean()]*5)
        rslt1 = triplet_transfer(yt, yprd, ys)
        rslt2 = triplet_transfer(yt, ymeans, ys)
        print(rslt1)
        print(rslt2)
        assert(rslt2[0]) == 1
        return True
    else:
        print("没想好")
    return None
# %%


def Accuracy(Y_Target, Y_Predict, error_rate=False, multi_dimension=False):
    Y_Target = np.asarray(Y_Target, order='C')
    Y_Predict = np.asarray(Y_Predict, order='C')
    if multi_dimension:
        Y_Target = Y_Target.flatten()
        Y_Predict = Y_Predict.flatten()
    else:
        Y_Target = Y_Target.reshape(len(Y_Target), 1)
        Y_Predict = Y_Predict.reshape(len(Y_Predict), 1)
    if len(Y_Target) == len(Y_Predict):
        correct = np.sum(Y_Target == Y_Predict)
        if error_rate:
            Value = 1-(correct/len(Y_Predict))
        else:
            Value = correct/len(Y_Predict)
    else:
        raise ValueError('Target & Predicted are not of same length.')
        return np.nan

    return Value


def F1_score(Y_Target, Y_Predict, multi_dimension=False):
    # Y target and Y predict are forced to convert to bool.
    # Only applicatble for 2-class problems.
    Y_Target = np.asarray(Y_Target, order='C').astype(bool)
    Y_Predict = np.asarray(Y_Predict, order='C').astype(bool)
    if multi_dimension:
        Y_Target = Y_Target.flatten()
        Y_Predict = Y_Predict.flatten()
    else:
        Y_Target = Y_Target.reshape(len(Y_Target), 1)
        Y_Predict = Y_Predict.reshape(len(Y_Predict), 1)

    if len(Y_Target) == len(Y_Predict):
        ...
        _true_posi = sum(Y_Target & Y_Predict)
        _false_posi = sum(~Y_Target & Y_Predict)
        _false_neg = sum(Y_Target & ~Y_Predict)
        _precision = _true_posi/(_true_posi + _false_posi)
        _recall = _true_posi/(_true_posi + _false_neg)
        Value = 2*_precision / (_precision + _recall)

    else:
        raise ValueError('Target & Predicted are not of same length.')
        return np.nan

    return Value


def kFold_NRMSE(Mdl, X, y, k=5):
    kf = KFold(n_splits=k)
    X = np.asarray(X, order='C')
    y = np.asarray(y, order='C')

    errors = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        Mdl.fit(X_train, y_train.ravel())
        y_prd = Mdl.predict(X_test)
        errors.append(NRMSE(y_test, y_prd)[0])
#        errors.append(np.corrcoef(y_test.ravel(),y_prd.ravel()))# tempera
#        return errors # tempra
    return np.mean(errors)


# %%
if __name__ == '__main__':
    t1 = np.random.randn(10, 2)
    t2 = -t1
    sg = avg_correlation(t1, t2)
    test_triplet()
    test_triplet_transfer()
