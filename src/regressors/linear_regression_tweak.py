import numpy as np
from sklearn.base import BaseEstimator, _fit_context
from sklearn.utils.validation import check_is_fitted
import cvxpy as cp
from typing import Literal

class LinearRegressorTweak(BaseEstimator):
    """A template estimator to be used as a reference implementation.

    For more information regarding how to build your own estimator, read more
    in the :ref:`User Guide <user_guide>`.

    Parameters
    ----------
    demo_param : str, default='demo_param'
        A parameter used for demonstration of how to pass and store parameters.

    Attributes
    ----------
    is_fitted_ : bool
        A boolean indicating whether the estimator has been fitted.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

    Examples
    --------
    >>> from skltemplate import TemplateEstimator
    >>> import numpy as np
    >>> X = np.arange(100).reshape(100, 1)
    >>> y = np.zeros((100, ))
    >>> estimator = TemplateEstimator()
    >>> estimator.fit(X, y)
    TemplateEstimator()
    """

    # This is a dictionary allowing to define the type of parameters.
    # It used to validate parameter within the `_fit_context` decorator.
    _parameter_constraints = {
        "penalty": [int, float],
        "bound": [str, None],
        "positive": [tuple],
        "negative": [tuple]

    }

    def __init__(
            self, 
            penalty: float= 1, 
            bound: Literal['lower', 'upper', None]= None, 
            positive: tuple[int] = (), 
            negative: tuple[int] = ()
    ) -> None:
        self.penalty = penalty
        self.bound = bound
        self.positive = positive
        self.negative = negative

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y):
        """A reference implementation of a fitting function.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.

        y : array-like, shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels in classification, real numbers in
            regression).

        Returns
        -------
        self : object
            Returns self.
        """

        # Validate parameters passed in the __init__ method that depend on data
        parameter_validation = self._validate_parameters(X)
            
        # `_validate_data` is defined in the `BaseEstimator` class.
        # It allows to:
        # - run different checks on the input data;
        # - define some attributes associated to the input data: `n_features_in_` and
        #   `feature_names_in_`.
        
        X, y = self._validate_data(X, y, accept_sparse=True)
        
        # Build the model
        n_samples, n_features_in = X.shape

        # Decision Variables
        coef = cp.Variable(n_features_in)
        intercept = cp.Variable()

        # Objective
        rmse = cp.sum_squares(X @ coef + intercept*np.ones(n_samples) - y)
        penalty_term = (
            cp.sum(cp.pos(X @ coef + intercept*np.ones(n_samples) - y)) if self.bound == 'lower' else (
                cp.sum(cp.pos(y - X @ coef - intercept*np.ones(n_samples))) if self.bound == 'upper' else 0
            )
        )

        objective = cp.Minimize(rmse + self.penalty*penalty_term) 

        # Monotonicity Constraints
        positive_constraints = [coef[list(self.positive)] >= 0] if self.positive else []
        negative_constraints = [coef[list(self.negative)] <= 0] if self.negative else []

        monotonicity_constraints = positive_constraints + negative_constraints

        # Instantiate Convex Model
        cvxpy_model = cp.Problem(objective, monotonicity_constraints)
        
        # Solve the convex model
        cvxpy_model.solve()

        # Set attributes if status was not infeasible or unbounded
        if not cvxpy_model.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            raise Warning(f'Training ended with status {cvxpy_model.status}')
        
        print(f'RMSE Value: {rmse.value}')
        print(f'Violation_value: {penalty_term.value if self.bound else 0}')
        
        self.coef_ = coef.value
        self.intercept_ = intercept.value

        self.is_fitted_ = True

        # `fit` should always return `self`
        return self

    def predict(self, X):
        """A reference implementation of a predicting function.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.

        Returns
        -------
        y : ndarray, shape (n_samples,)
            Returns an array of ones.
        """
        # Check if fit had been called
        check_is_fitted(self)
        # We need to set reset=False because we don't want to overwrite `n_features_in_`
    
        # `feature_names_in_` but only check that the shape is consistent.
        X = self._validate_data(X, accept_sparse=True, reset=False)

        if len(X.shape) > 1:
            y_pred = np.matmul(X, self.coef_) + self.intercept_
        else:
            print('HOO')
            y_pred = np.dot(X[0], self.coef_) + self.intercept_
        
        return y_pred

    def _validate_parameters(self, X) -> None:
        
        # Get n_features_in
        n_features_in = X.shape[1] if len(X.shape)  >= 2 else 1

        # Check that bound correspond to one of the posssible values 'lower' and 'upper'
        bound_check = self.bound in ['lower', 'upper', None]
        if not bound_check:
            raise ValueError('Bound parameter must be one of "lower", "upper", or None')

        # Check that elements are integers in self.positive
        positive_integer_check = True
        try:
            test = [int(index) for index in self.positive]
        except:
            positive_integer_check = False
        
        if not positive_integer_check:
            raise ValueError('Parameter positive contains non-integer values')

        # Check that elements are integers in self.negative
        negative_integer_check = True
        try:
            test = [int(index) for index in self.negative]
        except:
            negative_integer_check = False
        
        if not negative_integer_check:
            raise ValueError('Parameter negative contains non-integer values')

        # Check that indexes in positive are in the range [0, n_features_in - 1]
        positive_range_check = (max(self.positive) <= n_features_in and min(self.positive) >= 0) if self.positive else True
        if not positive_range_check:
            raise ValueError('Parameter positive contains indexes out of range for this training data')

        # Check that indexes in positive are in the range [0, n_features_in - 1]
        negative_range_check = (max(self.negative) <= n_features_in and min(self.negative) >= 0) if self.negative else True
        if not negative_range_check:
            raise ValueError('Parameter negative contains indexes out of range for this training data')
        
    
