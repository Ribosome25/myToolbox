# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 14:54:19 2019

@author: Ruibzhan
"""
import math
from sklearn.model_selection import KFold
import numpy as np
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import copy
import warnings
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


def avg_correlation(Y_Target, Y_Predict, two_corrs=False):
    # Return a line of avg of correlations. This is for the DREAM project
    Y_Target = np.asarray(Y_Target)
    Y_Predict = np.asarray(Y_Predict)
    if Y_Target.ndim == 1:
        Y_Target = Y_Target.reshape(-1, 1)
    if Y_Predict.ndim == 1:
        Y_Predict = Y_Predict.reshape(-1, 1)
    cols = Y_Target.shape[1]
    rt = np.zeros(cols)
    rt2 = np.zeros(cols)

    for ii in range(cols):
        tgt = Y_Target[:, ii]
        prd = Y_Predict[:, ii]
        nan_mask = np.isnan(tgt)
        tgt = tgt[~nan_mask]
        prd = prd[~nan_mask]
        rt[ii], rt2[ii] = two_correlations(tgt, prd)

    if two_corrs:
        return rt.mean(), rt2.mean()
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

def sextuple(Y_Target, Y_Predict, multi_dimension=True, output_format=list):
    """
    Returns the Spearman correlation and Pearson Correlation, MSE, MAE.
    Plus NRMSE, NMAE
    """
    Y_Target, Y_Predict = _check_y_same_dim(
        Y_Target, Y_Predict, multi_dimension)
    if multi_dimension:
        scorrs = []
        pcorrs = []
        mses = []
        maes = []
        nrmses = []
        nmaes = []

        t_mean = np.abs(Y_Target.mean(axis=0))
        t_std = Y_Target.std(axis=0)

        for ii in range(Y_Target.shape[1]):
            scorr, pcorr, mse, mae = _single_corr_and_error(
                Y_Target[:, ii], Y_Predict[:, ii])
            scorrs.append(scorr)
            pcorrs.append(pcorr)
            mses.append(mse)
            maes.append(mae)
            nrmses.append(np.sqrt(mse) / t_std[ii])
            nmaes.append(mae / t_mean[ii])

        scorr = np.mean(scorrs)
        pcorr = np.mean(pcorrs)
        mse = np.mean(mses)
        mae = np.mean(maes)
        nrmse = np.mean(nrmses)
        nmae = np.mean(nmaes)
    else:
        scorr, pcorr, mse, mae = _single_corr_and_error(Y_Target, Y_Predict)
        nrmse = np.sqrt(mse) / np.std(Y_Target)
        nmae = mae / np.abs(np.mean(Y_Target))

    if output_format == list:
        return([scorr, pcorr, mse, mae, nrmse, nmae])
    elif output_format == dict:
        return({'s-corr': scorr,
                'p-corr': pcorr,
                'MSE': mse,
                'MAE': mae,
                "NRMSE": nrmse,
                "NMAE": nmae})
    else:
        print("Unknown format, to be done. return list.")
        return([scorr, pcorr, mse, mae, nrmse, nmae])

from sklearn.metrics import r2_score
def octuple(Y_Target, Y_Predict, multi_dimension=True, output_format=list):
    sext = sextuple(Y_Target, Y_Predict, multi_dimension)
    rmse = math.sqrt(sext[2])
    r2 = r2_score(Y_Target, Y_Predict)
    return [*sext, rmse, r2]


def test_sextuple():
    x = np.random.randn(20, 5)
    y = np.random.randn(20, 5)
    kx = x * 10
    ky = y * 10
    s, p, mse, mae, nrmse, nmae = sextuple(x, y, True)
    ks, kp, kmse, kmae, knrmse, knmae = sextuple(kx, ky, True)
    assert(np.allclose(s, ks))
    assert(np.allclose(p, kp))
    assert(np.allclose(100 * mse, kmse))
    assert(np.allclose(10 * mae, kmae))
    assert(np.allclose(nrmse, knrmse))
    assert(np.allclose(nmae, knmae))

    x = np.random.randn(20)
    y = np.random.randn(20)
    kx = x * 10
    ky = y * 10
    s, p, mse, mae, nrmse, nmae = sextuple(x, y, False)
    ks, kp, kmse, kmae, knrmse, knmae = sextuple(kx, ky, False)
    assert(np.allclose(s, ks))
    assert(np.allclose(p, kp))
    assert(np.allclose(100 * mse, kmse))
    assert(np.allclose(10 * mae, kmae))
    assert(np.allclose(nrmse, knrmse))
    assert(np.allclose(nmae, knmae))
    assert(np.allclose(nrmse, np.sqrt(mse)/x.std()))

def _single_corr_and_error(Y_Target, Y_Predict):
    nan_maps = np.isnan(Y_Target) | np.isnan(Y_Predict)
    Y_Target = Y_Target[~nan_maps]
    Y_Predict = Y_Predict[~nan_maps]
    scorrs, pvalue = spearmanr(Y_Target, Y_Predict)
    pcorrs, pvalue = pearsonr(Y_Predict, Y_Target)
    scorr = scorrs
    pcorr = pcorrs
    if scorr == np.nan or pcorr == np.nan:
        warnings.warn("{}: S-corr or P-corr contains NaN. Replaced with 0.".format(__name__))
        scorr = 0
        pcorr = 0
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

#%%
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import auc, roc_auc_score, precision_recall_curve
def twocat_sextuple_1d(y_true, y_pred_prob):
    """
    Input: Predicted probability for each class. (the raw output from model.predict_proba())
    Accuracy, Precision, Recall, F1-score, ROC AUC, PRC AUC

    Input must be 1D. Multi-target version will be done later.
    Input must be in int. A more general interface TBD.
    Target must be 2 categories.
    The second cat 1 is the positive cat.
    """
    y_true = np.asarray_chkfinite(y_true, dtype=int)
    y_pred_prob = np.asarray_chkfinite(y_pred_prob, dtype=float)
    assert y_true.ndim == 1 or len(y_true) == y_true.size, "cls_1d only works for 1d targets."
    assert len(np.unique(y_true)) == 2, "this metric function only works for 2 cat problems."

    yhat = 1 * (y_pred_prob[:, 1] > 0.5)
    acc = accuracy_score(y_true, yhat)
    precision = precision_score(y_true, yhat)
    recall = recall_score(y_true, yhat)
    f1 = f1_score(y_true, yhat)

    precision_list, recall_list, _ = precision_recall_curve(y_true, y_pred_prob[:, 1])
    prc_auc = auc(recall_list, precision_list)
    roc_auc = roc_auc_score(y_true, y_pred_prob[:, 1])

    return acc, precision, recall, f1, roc_auc, prc_auc

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
def test_twocat_1d():
    X, y = make_classification(n_samples=1000, n_classes=2, random_state=1)
    # split into train/test sets
    trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.5, random_state=2)
    # generate a no skill prediction (majority class)
    ns_probs = [0 for _ in range(len(testy))]
    # fit a model
    model = LogisticRegression(solver='lbfgs')
    model.fit(trainX, trainy)
    # predict probabilities
    lr_probs = model.predict_proba(testX)
    result = twocat_sextuple_1d(testy, lr_probs)
    print(result)
    assert np.allclose(result[4], 0.903, atol=1e-3)
    assert np.allclose(result[5], 0.898, atol=1e-3)
    assert np.allclose(result[3], 0.841, atol=1e-3)



# %%
if __name__ == '__main__':
    test_twocat_1d()
    raise
    test_sextuple()
    t1 = np.random.randn(10, 2)
    t2 = -t1
    sg = avg_correlation(t1, t2)
    test_triplet()
    test_triplet_transfer()
