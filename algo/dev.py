from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

from collections import Counter
from collections import ChainMap

import warnings
import numpy as np
# from smote import ADASYN
from oversample import Oversample
from sklearn.base import is_regressor
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble.forest import BaseForest
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from sklearn.tree.tree import BaseDecisionTree
from sklearn.utils import check_random_state
from sklearn.utils import check_X_y, check_array
from sklearn.neighbors import NearestNeighbors

__author__ = 'Wenhao Zhang'
MAJORITY = 0
MINORITY = 1


class DEVALGO(AdaBoostClassifier):
    """Implementation of DEVALGO.
    DEVALGO introduces data resampling into the AdaBoost algorithm by
    oversampling the minority class using ADASYN and undersampling the majority at borderline on each boosting iteration [1].
    This implementation inherits methods from the scikit-learn 
    AdaBoostClassifier class, only overriding the `fit` method.
    Parameters
    ----------
    n_samples : int, optional (default=100)
        Number of new synthetic samples per boosting step.
    k_neighbors : int, optional (default=5)
        Number of nearest neighbors.
    base_estimator : object, optional (default=DecisionTreeClassifier)
        The base estimator from which the boosted ensemble is built.
        Support for sample weighting is required, as well as proper `classes_`
        and `n_classes_` attributes.
    n_estimators : int, optional (default=50)
        The maximum number of estimators at which boosting is terminated.
        In case of perfect fit, the learning procedure is stopped early.
    learning_rate : float, optional (default=1.)
        Learning rate shrinks the contribution of each classifier by
        ``learning_rate``. There is a trade-off between ``learning_rate`` and
        ``n_estimators``.
    algorithm : {'SAMME', 'SAMME.R'}, optional (default='SAMME.R')
        If 'SAMME.R' then use the SAMME.R real boosting algorithm.
        ``base_estimator`` must support calculation of class probabilities.
        If 'SAMME' then use the SAMME discrete boosting algorithm.
        The SAMME.R algorithm typically converges faster than SAMME,
        achieving a lower test error with fewer boosting iterations.
    random_state : int or None, optional (default=None)
        If int, random_state is the seed used by the random number generator.
        If None, the random number generator is the RandomState instance used
        by np.random.
    References
    ----------
    .. [1] https://3.basecamp.com/3929907/buckets/10670971/uploads/1565590241
    """

    def __init__(self,
                 n_samples=100,
                 k_neighbors=5,
                 base_estimator=None,
                 n_estimators=50,
                 learning_rate=1.,
                 algorithm='SAMME.R',
                 random_state=None,
                 D=None):

        self.n_samples = n_samples
        self.algorithm = algorithm
        self.k_neighbors = k_neighbors

        super(DEVALGO, self).__init__(
            base_estimator=base_estimator,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=random_state)

    def label_minority(self, X, y):
        """Iterate over all minority examples, and compute the K (default, k=5) nearest neighbors for each example. Amongst these K nearest neighbors, we rank the minority examples in terms of learning difficulty using this ratio = (# of majority examples in K)/(# of minority examples in K). For example, a minority example is regarded as "Safe" instance if ratio = 0 or 1/4; ratio of "Borderline" minority is 2/3 or 3/2. More on scoring the minority class is in [2].

        Type                Ratio (maj/min)
        ----                ---------------
        Safe                0/5 or 1/4
        Borderline          2/3 or 3/2
        Outlier             4/1 or 5/0

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape = [n_samples, n_features]
            The training input samples. Sparse matrix can be CSC, CSR, COO,
            DOK, or LIL. COO, DOK, and LIL are converted to CSR. The dtype is
            forced to DTYPE from tree._tree if the base classifier of this
            ensemble weighted boosting classifier is a tree or forest.
        y : array-like of shape = [n_samples]
            The target values (class labels in classification, real numbers in
            regression).
        Returns
        -------
            : dictionary
            A dictionary contains the indices of minority examples, which are classified into 3 groups: "Safe", "Borderline", and "Outlier".  
        References
        ----------
        [2] K. Napierala and J. Stefanowski, “Types of minority class examples and their influence on learning classifiers from imbalanced data,” Journal of Intelligent Information Systems, vol. 46, no. 3, pp. 563–597, Jun. 2016.
        """
        neigh = NearestNeighbors(n_neighbors=5)
        neigh.fit(X)
        nns = neigh.kneighbors(n_neighbors=5, return_distance=False)

        labels = {'Safe': set(), 'Borderline': set(), 'Outlier': set()}
        # Iterate k neighbors for all minority examples, calculate the ratio, and categorize them
        for i, neighbors in enumerate(nns):
            if y[i] == 1:
                # only check minority examples
                min_counts = sum(y[neighbors]) # regard 1 as minority class
                key = None
                if min_counts >= 4:
                    key = 'Safe'
                elif min_counts >=2 and min_counts <= 3:
                    key = 'Borderline'
                else:
                    key = 'Outlier'
                labels[key].add(i)

        return dict(labels)



    def undersample(self, X, y, borderline):
        """ Undersample the dataset by removing the majority examples of Tomek-link pairs at borderline[3].
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape = [n_samples, n_features]
            The training input samples. Sparse matrix can be CSC, CSR, COO,
            DOK, or LIL. COO, DOK, and LIL are converted to CSR. The dtype is
            forced to DTYPE from tree._tree if the base classifier of this
            ensemble weighted boosting classifier is a tree or forest.
        y : array-like of shape = [n_samples]
            The target values (class labels in classification, real numbers in
            regression).
        borderline: list
            The list contains the indices of borderline minority examples in X.
        Returns
        -------
            : list
            A list contains the indices of "Borderline" minority examples.  
        References
        ----------
        [3] Elhassan T and Aljurf M, “Classification of Imbalance Data using Tomek Link (T-link) Combined with Random Under-sampling (RUS) as a Data Reduction Method,” Jun. 2016.
        """
        neigh = NearestNeighbors(n_neighbors=1)
        neigh.fit(X)

        ret = []
        for i in borderline:
            # function kneighbors will return itself, hence setting n_neighbors=2
            tl = neigh.kneighbors([X[i]], n_neighbors=2, return_distance=False)
            if y[tl.item(0,1)] == MAJORITY:
                ret.append(tl.item(0,1))
        return list(ret)
        # return list()





    def fit(self, X, y, minority_target=None, sample_weight=None,):
        """Build a boosted classifier/regressor from the training set (X, y),
        performing ADASYN during each boosting step.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape = [n_samples, n_features]
            The training input samples. Sparse matrix can be CSC, CSR, COO,
            DOK, or LIL. COO, DOK, and LIL are converted to CSR. The dtype is
            forced to DTYPE from tree._tree if the base classifier of this
            ensemble weighted boosting classifier is a tree or forest.
        y : array-like of shape = [n_samples]
            The target values (class labels in classification, real numbers in
            regression).
        minority_target : int
            Minority class label.
        Returns
        -------
        self : object
            Returns self.
        Notes
        -----
        Based on the scikit-learn v0.20 AdaBoostClassifier and
        BaseWeightBoosting `fit` methods.
        """
        # Check that algorithm is supported.
        if self.algorithm not in ('SAMME', 'SAMME.R'):
            raise ValueError("algorithm %s is not supported" % self.algorithm)

        # Check parameters.
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be greater than zero")

        if (self.base_estimator is None or
                isinstance(self.base_estimator, (BaseDecisionTree,
                                                 BaseForest))):
            DTYPE = np.float64  # from fast_dict.pxd
            dtype = DTYPE
            accept_sparse = 'csc'
        else:
            dtype = None
            accept_sparse = ['csr', 'csc']

        X, y = check_X_y(X, y, accept_sparse=accept_sparse, dtype=dtype,
                         y_numeric=is_regressor(self))

        # Classify the minority examples
        labels = self.label_minority(np.array(X), np.array(y))
        
        # Undersampling 
        ret = self.undersample(np.array(X), np.array(y),labels['Borderline'])
        X_select = np.delete(X, ret, axis=0)
        y_select = np.delete(y, ret, axis=0)

        if sample_weight is None:
            # Initialize weights to 1 / n_samples.
            sample_weight = np.empty(X_select.shape[0], dtype=np.float64)
            sample_weight[:] = 1. / X_select.shape[0]
        else:
            sample_weight = check_array(sample_weight, ensure_2d=False)
            # Normalize existing weights.
            sample_weight = sample_weight / sample_weight.sum(dtype=np.float64)

            # Check that the sample weights sum is positive.
            if sample_weight.sum() <= 0:
                raise ValueError(
                    "Attempting to fit with a non-positive "
                    "weighted number of samples.")

        if minority_target is None:
            # Determine the minority class label.
            stats_c_ = Counter(y_select)
            maj_c_ = max(stats_c_, key=stats_c_.get)
            min_c_ = min(stats_c_, key=stats_c_.get)
            self.minority_target = min_c_
        else:
            self.minority_target = minority_target

        # Check parameters.
        self._validate_estimator()

        # Clear any previous fit results.
        self.estimators_ = []
        self.estimator_weights_ = np.zeros(self.n_estimators, dtype=np.float64)
        self.estimator_errors_ = np.ones(self.n_estimators, dtype=np.float64)

        random_state = check_random_state(self.random_state)

        # self.smote = SMOTE(k_neighbors=self.k_neighbors,
        #                     random_state=self.random_state)
        self.os = Oversample(verbose=False, 
                             N=self.n_samples,
                             random_state=self.random_state,)

        for iboost in range(self.n_estimators):
            # Oversample step using weights
            X_syn, y_syn = self.os.fit_transform(X_select, y_select, sample_weight)

            # Combine the original and synthetic samples.
            X_mrg = np.vstack((X_select, X_syn))
            y_mrg = np.append(y_select, y_syn)

            # Normalize synthetic sample weights based on current training set.
            sample_weight_syn = np.empty(X_syn.shape[0], dtype=np.float64)
            sample_weight_syn[:] = 1. / X.shape[0]

            # Combine the weights.
            sample_weight_mrg = \
                np.append(sample_weight, sample_weight_syn).reshape(-1, 1)
            sample_weight_new = \
                np.squeeze(normalize(sample_weight_mrg, axis=0, norm='l1'))

            # Boosting step.
            sample_weight, estimator_weight, estimator_error = self._boost(
                iboost,
                X_mrg, y_mrg,
                sample_weight_new,
                random_state)

            # Early termination.
            if sample_weight is None:
                break

            # Only take the weights of the original data
            sample_weight = sample_weight[:len(X_select)]


            self.estimator_weights_[iboost] = estimator_weight
            self.estimator_errors_[iboost] = estimator_error

            # Stop if error is zero.
            if estimator_error == 0:
                break

            sample_weight_sum = np.sum(sample_weight)

            # Stop if the sum of sample weights has become non-positive.
            if sample_weight_sum <= 0:
                break

            if iboost < self.n_estimators - 1:
                # Normalize.
                sample_weight /= sample_weight_sum

        return self