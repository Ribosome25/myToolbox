# -*- coding: utf-8 -*-
"""
Created on Sept 23 2022

Since the sklearn Metrics has been extensively updated,
it is necessary to rewrite the regression section
for the sake of clarity and convenience.


@author: Ruibzhan


"""
import math
from sklearn.model_selection import KFold
import numpy as np

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

# %% Regression metrics
reg_metric_octuple = ["Spearman", "Pearson", "MSE", "MAE", "NRMSE", "NMAE", "RMSE", "R2"]

from sklearn.utils.validation import check_consistent_length, _num_samples
from sklearn.exceptions import UndefinedMetricWarning, DataConversionWarning
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def _check_nan_reg_targets(y_true, y_pred, multioutput="uniform_average", dtype="numeric"):
    """
    Copied from sklearn.
    https://github.com/scikit-learn/scikit-learn/blob/0bf24792d69ebc2024821adafd63ee37d0110cd3/sklearn/metrics/_regression.py#L66
    """
    check_consistent_length(y_true, y_pred)

    if y_true.ndim == 1:
        y_true = y_true.reshape((-1, 1))

    if y_pred.ndim == 1:
        y_pred = y_pred.reshape((-1, 1))

    if y_true.shape[1] != y_pred.shape[1]:
        raise ValueError(
            "y_true and y_pred have different number of output ({0}!={1})".format(
                y_true.shape[1], y_pred.shape[1]
            )
        )

    n_outputs = y_true.shape[1]
    allowed_multioutput_str = ("raw_values", "uniform_average", "variance_weighted")
    if isinstance(multioutput, str):
        if multioutput not in allowed_multioutput_str:
            raise ValueError(
                "Allowed 'multioutput' string values are {}. "
                "You provided multioutput={!r}".format(
                    allowed_multioutput_str, multioutput
                )
            )
    elif multioutput is not None:  # Custom weights
        if n_outputs == 1:
            raise ValueError("Custom weights are useful only in multi-output cases.")
        elif n_outputs != len(multioutput):
            raise ValueError(
                "There must be equally many custom weights (%d) as outputs (%d)."
                % (len(multioutput), n_outputs)
            )
    y_type = "continuous" if n_outputs == 1 else "continuous-multioutput"

    return y_type, y_true, y_pred, multioutput


def octuple(y_true, y_pred, omit_nan=True, multioutput="uniform_average"):
    """
    NRMSE is RMSE divided by std var of y_true
    NMAE is MAE divided by mean of y_true
    omit_nan: True for dropping nan for each target then find its score,
            False for impute with target mean then find the score. 
    multioutput: str, "raw_values" or "uniform_average". 
                raw values means returning values for every target (col of y)
                uniform average means returning 
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_type, y_true, y_pred, multioutput = _check_nan_reg_targets(y_true, y_pred, multioutput)
    n_targets = y_true.shape[1]

    if _num_samples(y_pred) < 2:
        msg = "Correlations and R^2 score are not well-defined with less than two samples."
        warnings.warn(msg, UndefinedMetricWarning)
        return np.full(shape=8, fill_value=np.nan)
    if np.isnan(y_true).sum() > 0 and omit_nan is False:
        msg = "NaNs found in the y_target and will be imputed with target mean."
        warnings.warn(msg, DataConversionWarning)

    output_scores = np.full(shape=(n_targets, 8), fill_value=np.nan)  # rows for samples, cols for metrics
    for ii in range(n_targets):
        
        y_true_i = y_true[:, ii]
        y_pred_i = y_pred[:, ii]
        if omit_nan:
            nan_maps = np.isnan(y_true_i) | np.isnan(y_pred_i)
            y_true_i = y_true_i[~ nan_maps]
            y_pred_i = y_pred_i[~ nan_maps]
        else:
            y_true_i = np.nan_to_num(y_true_i, nan=np.nanmean(y_true_i))

        scorr, pvalue = spearmanr(y_true_i, y_pred_i)
        pcorr, pvalue = pearsonr(y_true_i, y_pred_i)
        output_scores[ii, 0] = scorr
        output_scores[ii, 1] = pcorr
        
        mse = mean_squared_error(y_true_i, y_pred_i)
        mae = mean_absolute_error(y_true_i, y_pred_i)
        output_scores[ii, 2] = mse
        output_scores[ii, 3] = mae

        rmse = mean_squared_error(y_true_i, y_pred_i, squared=False)
        r2 = r2_score(y_true_i, y_pred_i)
        output_scores[ii, 6] = rmse
        output_scores[ii, 7] = r2
        
        std_i = np.std(y_true_i)
        mean_i = np.mean(y_true_i)
        output_scores[ii, 4] = rmse / std_i
        output_scores[ii, 5] = mae / mean_i
        
    if multioutput == "raw_values":
        return output_scores
    elif multioutput == "uniform_average":
        return output_scores.mean(axis=0)
    else:
        raise ValueError("Unknown output aggregation method. ")
        


#%%
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import auc, roc_auc_score, precision_recall_curve

clf_metrics_sextuple = ["Accuracy", "Precision", "Recall", "F1-score", "ROC AUC", "PRC AUC"]

def twocat_sextuple_1d(y_true, y_pred_prob):
    """
    Input:
        y_true is a 1-D array, with int 0, 1 indicating the label.
        y_pred_prob is predicted probability for each class. (the raw output from model.predict_proba())
        A 2-D array. First column is the prob of being 0, 2nd is the prob of 1.
    Output:
        Accuracy, Precision, Recall, F1-score, ROC AUC, PRC AUC


    Input must be one-target. y_true must be 1-D. Multi-target version will be done later.
    Labels must be int. A more general interface to be done.
    Target must have 2 categories.
    The second cat 1 is the positive class.
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


def twocat_sextuple_2d(y_true, y_pred_prob):
    """
    Input: Predicted probability for each class. (the raw output from model.predict_proba())
    Accuracy, Precision, Recall, F1-score, ROC AUC, PRC AUC

    Input is Multi-targets. Each column stands for a target.
    The average value of every col is the final output.

    Target must be 2 categories.
    The second cat 1 is the positive cat.
    """
    y_true = np.asarray_chkfinite(y_true, dtype=int)
    y_pred_prob = np.asarray_chkfinite(y_pred_prob, dtype=float)
    assert len(np.unique(y_true)) == 2, "this metric function only works for 2 cat problems."
    if y_true.ndim == 1:
        return twocat_sextuple_1d(y_true, y_pred_prob)
    assert y_true.ndim == 2 and y_pred_prob.ndim == 3,\
        "Suppose y_true is a 2d matrix, and probs is a 3D matrix. While y_true has {} dims and y_pred has {} dims".format(y_true.ndim, y_pred_prob.ndim)

    results = np.empty((6, y_true.shape[1]))  # Create a array to save single result. 6 metrics, n_cols = n targets
    for ii in range(y_true.shape[1]):
        results[:, ii] = twocat_sextuple_1d(y_true[:, ii], y_pred_prob[ii, :, :])
    mean = np.nanmean(results, axis=1)
    # print(results)
    return mean.tolist()


# %%
if __name__ == '__main__':
    # test_twocat_1d()
    test_twocat_2d()
    raise

