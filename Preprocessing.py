def Impute(X,k = 5,metric = 'correlation',axis = 0,weighted = False):
    '''
    numeric knn imputation.
    (Need to check if the array ref is passed or the array value is passes. Better to use array.copy() when calling this func())
    @ Input:
        np.array or pd.Dataframe
    @ Parameters:
        k: k nearest neighbors are selected to impute the missing values
        metric: distance metric. See scipy.cdist.
        axis: 0 means finding the nearest k rows, 1 means finding the nearest k cols.
        weighted: When nearest k neighbors are picked out. Should we use the weighted avg of them. 
    @ Returns:
        DF or array after imputation.
    '''

    # If imputate col-wise: transpose it here, and transpose back in the end
    if axis == 1:
        X = X.T
    if X.shape[0]<k+1:
        raise Warning ("Samples are not enough for imputation. Return as the same.")
        return X
    if isinstance(X, pd.DataFrame):
        _is_df = True
        _df_cols = X.columns
        _df_idxs = X.index
    else:
        _is_df = False
    # Forcing convert to numpy.array
    X = np.asarray(X,dtype = float,order = 'C')

    # Get NaN map, where is NaN, and which row needs to be impu    
    _shape = X.shape
    NaN_map = np.isnan(X)
    sample_contains_NaN = NaN_map.sum(axis = 1)

    dist_matr, global_mean = _distance_matrix(X.copy(),metric = metric) # global mean is for the case when k-nn are all mssing at that point.
    for row_id in range(_shape[0]):
        if sample_contains_NaN[row_id]:
            sort_idx = dist_matr[row_id,:].argsort() # Returns [where is 1st smallest, where is 2nd smallest, ... etc].
            min_dist_sample_idxs = sort_idx[1:k+1]# For sure itself is the smallest distance. 1:k+1, Discard it.
            min_dist_samples = X[min_dist_sample_idxs,:]
            min_dists = dist_matr[row_id,:][min_dist_sample_idxs]
            if weighted:
                # Need to be tested.
                weighted_min_dist_samples = np.dot(min_dists.reshape(-1,1), min_dist_samples)
                knn_col_mean = np.nanmean(weighted_min_dist_samples,axis = 0)
            else:
                knn_col_mean = np.nanmean(min_dist_samples,axis = 0)
            NaN_col_idx = np.where(np.isnan(X[row_id,:]))[0]# Return a tuple. For only having 1 dim, so write a [0]
            for each_idx in NaN_col_idx:
                if sum(np.isnan(min_dist_samples[:,each_idx]))==k:
                    # all this col is NaN,take universal col mean, otw take knn_mean
                    X[row_id,each_idx] = np.take(global_mean,each_idx)
                else:
                    X[row_id,each_idx] = np.take(knn_col_mean,each_idx)
            
        else:
            continue

    # if do imputation col-wise: transpose back here
    if axis == 1:
        X = X.T
        
    if _is_df:
        X = pd.DataFrame(X,index = _df_idxs,columns = _df_cols)
    
    return X

def _distance_matrix(X,metric):
    # When calculating the distance, Replace NaN with col mean first.
    
    #Obtain mean of columns as you need, nanmean is just convenient.
    global_mean = np.nanmean(X, axis=0)
    #Find indicies that you need to replace
    inds = np.where(np.isnan(X)) # return tuple with 2 arrays, one-on-one matched, idx in [0], col in [1] 
    #Place column means in the indices. Align the arrays using take
    X[inds] = np.take(global_mean, inds[1])
    dist_matr = cdist(X,X,metric = metric)
    return dist_matr,global_mean