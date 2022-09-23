# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 15:12:16 2022

@author: ruibzhan
"""

import numpy as np
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
import pandas as pd

from myToolbox.Metrics import octuple, twocat_sextuple_1d, twocat_sextuple_2d

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
    # Y_Target, Y_Predict = _check_ys(Y_Target, Y_Predict, multi_dimension)
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


def test_new_octuple():

    prediction = pd.read_parquet("test_data/prediction.parquet")
    target = pd.read_parquet("test_data/y_target.parquet")

    skl_r2 = r2_score(target.fillna(target.mean()), prediction,
                      multioutput="raw_values")  # Returns a numpy array.
    
    new_scores = octuple(target, prediction)
    old_corrs = corr_and_error(target, prediction)
    assert(np.allclose(new_scores[:4], np.array(old_corrs)))

    new_scores = octuple(target.iloc[:, 0], prediction.iloc[:, 0])
    old_corrs = corr_and_error(target.iloc[:, 0], prediction.iloc[:, 0], multi_dimension=False)
    assert(np.allclose(new_scores[:4], np.array(old_corrs)))

    impu_target = target.fillna(target.mean())
    new_scores = octuple(target, prediction, omit_nan=False)
    new_scores_impu = octuple(impu_target, prediction)
    assert(np.allclose(new_scores, new_scores_impu))

    new_scores_impu = octuple(impu_target.iloc[:, 0], prediction.iloc[:, 0])
    old_nrmse = NRMSE(impu_target.iloc[:, 0], prediction.iloc[:, 0])
    assert(new_scores_impu[4] == old_nrmse[0])

    new_sex = octuple(impu_target, prediction)
    old_sex = sextuple(impu_target, prediction)
    assert(np.allclose(new_sex[:6], np.array(old_sex)))



def test_octuple_multiply():
    x = np.random.randn(20, 5)
    y = np.random.randn(20, 5)
    kx = x * 10
    ky = y * 10
    s, p, mse, mae, nrmse, nmae, rmse, r2 = octuple(x, y)
    ks, kp, kmse, kmae, knrmse, knmae, krmse, kr2 = octuple(kx, ky)
    assert(np.allclose(s, ks))
    assert(np.allclose(p, kp))
    assert(np.allclose(100 * mse, kmse))
    assert(np.allclose(10 * mae, kmae))
    assert(np.allclose(nrmse, knrmse))
    assert(np.allclose(nmae, knmae))
    assert(np.allclose(10 * rmse, krmse))
    assert(np.allclose(r2, kr2))

    x = np.random.randn(20)
    y = np.random.randn(20)
    kx = x * 10
    ky = y * 10
    s, p, mse, mae, nrmse, nmae, rmse, r2 = octuple(x, y)
    ks, kp, kmse, kmae, knrmse, knmae, krmse, kr2 = octuple(kx, ky)
    assert(np.allclose(s, ks))
    assert(np.allclose(p, kp))
    assert(np.allclose(100 * mse, kmse))
    assert(np.allclose(10 * mae, kmae))
    assert(np.allclose(nrmse, knrmse))
    assert(np.allclose(nmae, knmae))
    assert(np.allclose(nrmse, np.sqrt(mse)/x.std()))
    assert(np.allclose(10 * rmse, krmse))
    assert(np.allclose(r2, kr2))


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

from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier as RFC
def test_twocat_2d():
    """
    Test code. 500 test samples, 3 targets, 2 classes, predict prob.
    the output of sklearn model predict proba is a list with len of targets:
    [ ndarray for target 1 (N x nClasses),  ndarray for target 2 (N x nClasses), ...]
    """
    X1, y1 = make_classification(n_samples=1000, n_classes=2, random_state=1)
    X2, y2 = make_classification(n_samples=1000, n_classes=2, random_state=2)
    X3, y3 = make_classification(n_samples=1000, n_classes=2, random_state=3)
    X = np.hstack((X1, X2, X3))
    y = np.hstack([each.reshape(-1, 1) for each in (y1, y2, y3)])
    # split into train/test sets
    trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.5, random_state=2)
    # generate a no skill prediction (majority class)
    ns_probs = [0 for _ in range(len(testy))]
    # fit a model
    model_1 = MultiOutputClassifier(LogisticRegression(solver='lbfgs'))
    model_1.fit(trainX, trainy)
    # predict probabilities
    lr_probs_1 = model_1.predict_proba(testX)  # a list of predictions cols = 2 for 2 classes.

    result_1 = twocat_sextuple_2d(testy, lr_probs_1)
    print(result_1)

    # another model, comparing the format of output
    model_2 = RFC()
    model_2.fit(trainX, trainy)
    lr_probs_2 = model_2.predict_proba(testX)  # a list of predictions cols = 2 for 2 classes.
    result_2 = twocat_sextuple_2d(testy, lr_probs_2)
    print(result_2)
