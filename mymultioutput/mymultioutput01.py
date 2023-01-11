from sklearn.utils.validation import _check_fit_params
from sklearn.utils.validation import check_is_fitted
from sklearn.base import is_classifier
from sklearn.utils.fixes import delayed
from joblib import Parallel
from sklearn.multioutput import _fit_estimator
from sklearn.multioutput import MultiOutputRegressor


class MyMultiOutputRegressor(MultiOutputRegressor):
    
     def predict(self, X):
        """Predict multi-output variable using model for each target variable.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input data.
        Returns
        -------
        y : {array-like, sparse matrix} of shape (n_samples, n_outputs)
            Multi-output targets predicted across multiple predictors.
            Note: Separate models are generated for each predictor.
        """
        check_is_fitted(self)
        if not hasattr(self.estimators_[0], "predict"):
            raise ValueError("The base estimator should implement a predict method")

        y = Parallel(n_jobs=self.n_jobs)(
            delayed(e.predict)(X) for e in self.estimators_
        )

        return np.asarray(y).T
