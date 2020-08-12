1) write a common class for data reading. To coorperate, all sets should be included in one same object.  
  
    
2) There is a problem with Metrics: in 'corr and err', what should we do if the predictions are constant, and the corrs are NaN? So far it's neglected and non-nan values are taken the mean. I don't think that's fair. How about replace it with 0s? 
