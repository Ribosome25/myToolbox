import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist


def Impute(X, k=5, metric='correlation', axis=0, weighted=False):
    '''
    numeric knn imputation. Not in-place. 
    Returns the result in the end.
    KNNimpute uses pairwise information between the target gene with missing values
        and the K nearest reference genes to impute the missing values. 
        (by default, kn genes are find for one gene)

    @ Input:
        np.array or pd.Dataframe
    @ Parameters:
        k: k nearest neighbors are selected to impute the missing values.
        metric: distance metric. See scipy.cdist.
        axis: 0 means finding the nearest k rows, 1 means finding the nearest k cols.
        weighted: When nearest k neighbors are picked out. Should we use the weighted avg of them. 
    @ Returns:
        DF or array after imputation.

    ref: Troyanskaya O,  Cantor M,  Sherlock G, et al. Missing value estimation methods for DNA microarrays, Bioinformatics, 2001, vol. 17 6(pg. 520-5)
    '''

    # If imputate col-wise: transpose it here, and transpose back in the end
    if axis == 1:
        X = X.T
    if X.shape[0] < k+1:
        raise Warning(
            "Samples are not enough for imputation. Return as the same.")
        return X
    if isinstance(X, pd.DataFrame):
        _is_df = True
        _df_cols = X.columns
        _df_idxs = X.index
    else:
        _is_df = False
    # Forcing convert to numpy.array
    X = np.asarray(X, dtype=float, order='C')

    # Get NaN map, where is NaN, and which row needs to be impu
    _shape = X.shape
    NaN_map = np.isnan(X)
    sample_contains_NaN = NaN_map.sum(axis=1)

    # global mean is for the case when k-nn are all mssing at that point.
    dist_matr, global_mean = _distance_matrix(X.copy(), metric=metric)
    for row_id in range(_shape[0]):
        if sample_contains_NaN[row_id]:
            # Returns [where is 1st smallest, where is 2nd smallest, ... etc].
            sort_idx = dist_matr[row_id, :].argsort()
            # For sure itself is the smallest distance. 1:k+1, Discard it.
            min_dist_sample_idxs = sort_idx[1:k+1]
            min_dist_samples = X[min_dist_sample_idxs, :]
            min_dists = dist_matr[row_id, :][min_dist_sample_idxs]
            if weighted:
                # Need to be tested.
                weighted_min_dist_samples = np.dot(
                    min_dists.reshape(-1, 1), min_dist_samples)
                knn_col_mean = np.nanmean(weighted_min_dist_samples, axis=0)
            else:
                knn_col_mean = np.nanmean(min_dist_samples, axis=0)
            # Return a tuple. For only having 1 dim, so write a [0]
            NaN_col_idx = np.where(np.isnan(X[row_id, :]))[0]
            for each_idx in NaN_col_idx:
                if sum(np.isnan(min_dist_samples[:, each_idx])) == k:
                    # all this col is NaN,take universal col mean, otw take knn_mean
                    X[row_id, each_idx] = np.take(global_mean, each_idx)
                else:
                    X[row_id, each_idx] = np.take(knn_col_mean, each_idx)

        else:
            continue

    if _is_df:
        X = pd.DataFrame(X, index=_df_idxs, columns=_df_cols)
    # if do imputation col-wise: transpose back here
    if axis == 1:
        X = X.T

    return X


def _distance_matrix(X, metric):
    # When calculating the distance, Replace NaN with col mean first.

    # Obtain mean of columns as you need, nanmean is just convenient.
    global_mean = np.nanmean(X, axis=0)
    # Find indicies that you need to replace
    # return tuple with 2 arrays, one-on-one matched, idx in [0], col in [1]
    inds = np.where(np.isnan(X))
    # Place column means in the indices. Align the arrays using take
    X[inds] = np.take(global_mean, inds[1])
    dist_matr = cdist(X, X, metric=metric)
    return dist_matr, global_mean


def how_many_nans(obj):
    obj = np.asarray(obj)
    nans = np.isnan(obj)
    how_many = np.sum(nans)
    percent = how_many/nans.size
    return percent


def drop_too_many_nans(obj, drop_more_than=0.9, drop_rows=True):
    """
    Drop the instance if its NaN exceed the ratio.
    Ratio is more consistant than define a number thresh.
    Drop rows = True, otherwise drop cols.
    Only support DataFrame, if required, update for sth else later.

    """
    if drop_more_than > 1:
        drop_more_than /= 100
    assert isinstance(obj, pd.DataFrame), (
        "Preprossing.drop_too_`many_nans, so far onlt pd.DF is supported.")
    if not drop_rows:
        obj = obj.T
    nans = np.isnan(obj)
    counts = nans.sum(axis=1)
    keeps = counts[counts < drop_more_than*nans.shape[1]]
    keeps_idx = keeps.index[~keeps.index.duplicated(keep='first')]
    print("drop NaNs: {} instances are droped.".format(
        counts.shape[0] - keeps_idx.shape[0]))
    obj = obj.reindex(keeps_idx)

    if not drop_rows:
        obj = obj.T

    return obj


def drop_duplicated(df, axis='rows', keep='first'):
    """
    Remove rows (columns) with duplicated index (column names)
    keep{‘first’, ‘last’, False}, default ‘first’,
        False : remv all duplicates.
    """
    _is_flipped = False
    if isinstance(axis, str):
        if 'row' in axis:
            axis = 0
        elif 'column' in axis:
            axis = 1
    if not keep in ['first', 'last', False]:
        raise Warning('kept method not understood. Too be done.')
        return None
    if axis == 1:
        _is_flipped = True
        df = df.T
    df = df.iloc[~df.index.duplicated(keep=keep)]
    if _is_flipped:
        df = df.T
    return df

def mean_duplicated(df, axis='rows'):
    """
    Remove duplicated columns or rows, replace with the mean of them.
    """
    # process axis
    _is_flipped = False
    if isinstance(axis, str):
        if 'row' in axis:
            axis = 0
        elif 'column' in axis:
            axis = 1
    if axis == 1:
        _is_flipped = True
        df = df.T
    # Mean and replace
    dup_idx = df.index[df.index.duplicated()].unique()
    for each in dup_idx:
        nan_mean = np.nanmean(df.loc[each], axis=0)
        mean = pd.Series(nan_mean, index=df.columns, name=each)
        df.drop(each, axis=0, inplace=True)
        df.loc[each] = mean
    # Transpose back if needed.
    if _is_flipped:
        df = df.T
    return df

        
def expand_col_to_onehot(input_df, sele_cols):
    '''
    For some data sets, targets are given as (str, str), in one cell.
    This func is to expand this kind of data into sparsed boolean dataframe (exsit or not)
    paras: input dataframe whose index=instances; selected column(s) to expand.
    returns: a dataframe, with these cols expanded to booleans. 
    features will not having sequncials, all sele_cols are just mixed together. 
    '''
    if not isinstance(input_df, pd.DataFrame):
        input_df = pd.DataFrame(input_df)

    if isinstance(sele_cols, str):
        sele_cols = [].append(sele_cols)

    all_columns = set()
    for each in input_df[sele_cols].values.flatten():
        readed = [x.strip() for x in each.strip().split(',')]
        all_columns = all_columns | set(readed)
    all_columns = list(all_columns)
    all_columns.sort()

    table = pd.DataFrame(np.zeros((input_df.shape[0], len(all_columns)), dtype=bool),
                         index=input_df.index, columns=all_columns)
    for idx, each_row in input_df[sele_cols].iterrows():
        for each in each_row:
            readed = [x.strip() for x in each.strip().split(',')]
            for each_ft in readed:
                table.loc[idx, each_ft] = True
    return table


def expand_multiclass_to_onehot(input_df, sele_cols=None):
    """
    Common use.
    Some catagorical sets, classes are given as numbers 1 2 3 4.. or strs. 
    This func is for converting multi class coding to one-hot coding.
    Set the sele_cols to None to automatic detect multiclasses.
    """
    counts = input_df.nunique(axis=0)
    # It returns the count of unique elements along different axis.
    if sele_cols is None:
        sele_cols = counts[counts > 2].index
    expanded_df = []
    for each_col in sele_cols:  # to be done: int cols
        all_possible = input_df.loc[:, each_col].unique()
        each_col_df = []
        for each_possible in all_possible:
            each_possible_series = (
                input_df.loc[:, each_col] == each_possible).astype(int, copy=False)
            each_possible_series.name = str(each_col)+str(each_possible)
            each_col_df.append(each_possible_series)
        each_col_df = pd.concat(each_col_df, axis=1)
        expanded_df.append(each_col_df)
    expanded_df.append(input_df.drop(sele_cols, axis=1))
    expanded_df = pd.concat(expanded_df, axis=1, join='outer', sort='False')
    return expanded_df
