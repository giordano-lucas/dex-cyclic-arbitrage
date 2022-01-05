from sklearn.base import BaseEstimator, TransformerMixin, _OneToOneFeatureMixin
import pandas as pd

class TokenStandardScaler(_OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
    
    
    def __init__(self, *, copy=True):
        self.moments   = None
        self.copy = copy
        
    def transform(self, X, copy=None):
        """Perform standardization by centering and scaling.
        Parameters
        ----------
        X : {array-like, sparse matrix of shape (n_samples, n_features)
            The data used to scale along the features axis.
        copy : bool, default=None
            Copy the input X or not.
        Returns
        -------
        X_tr : {ndarray, sparse matrix} of shape (n_samples, n_features)
            Transformed array.
        """
        if self.copy:
            X=X.copy()
            
        tX = X.reset_index()
                              
        t_moments = self.moments.reindex(tX.groupby(["token1","token2"]).count().index)
        t_moments.loc[pd.isna(t_moments).any(axis=1),"mean"] = self.default_mean.values 
        t_moments.loc[pd.isna(t_moments).any(axis=1),"std"] = self.default_std.values
        
        X_tr = (tX.set_index(["token1","token2"])-t_moments["mean"])/t_moments["std"]
    
        
        X_tr["cycle_id"] = X_tr["cycle_id"].astype("int32")
        X_tr = X_tr.reset_index().set_index(["cycle_id","token1","token2"])
        return X_tr
    
    def inverse_transform(self, X, copy=None):
        """Scale back the data to the original representation.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The data used to scale along the features axis.
        copy : bool, default=None
            Copy the input X or not.
        Returns
        -------
        X_tr : {ndarray, sparse matrix} of shape (n_samples, n_features)
            Transformed array.
        """

    
    def fit(self, X, y=None, sample_weight=None):
        """Compute the mean and std to be used for later scaling.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The data used to compute the mean and standard deviation
            used for later scaling along the features axis.
        y : None
            Ignored.
        sample_weight : array-like of shape (n_samples,), default=None
            Individual weights for each sample.
            .. versionadded:: 0.24
               parameter *sample_weight* support to StandardScaler.
        Returns
        -------
        self : object
            Fitted scaler.
        """

        self.moments = (X.reset_index()
                        .groupby(["token1","token2"])
                        .agg(["mean","std"])
                        .swaplevel(axis=1))
        
        self.moments["mean","cycle_id"] = 0
        self.moments["std","cycle_id"]  = 1 
        self.default_mean = self.moments["mean"].mean()
        self.default_std = self.moments["std"].mean()
        return self