from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

from collections import Counter
from collections import ChainMap

import warnings
import numpy as np
# from smote import ADASYN
from smote import SMOTE
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
                 random_state=None):

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
        Based on the scikit-learn v0.18 AdaBoostClassifier and
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
        # X_select = X
        # y_select = y

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
        self.adasyn = ADASYN(k=self.k_neighbors,
                             random_state=self.random_state,
                             num_samples = self.n_samples)

        for iboost in range(self.n_estimators):
            # # SMOTE step.
            # X_min = X_select[np.where(y_select == self.minority_target)]
            # self.smote.fit(X_min)
            # X_syn = self.smote.sample(self.n_samples)
            # y_syn = np.full(X_syn.shape[0], fill_value=self.minority_target,
            #                 dtype=np.int64)
            X_syn, y_syn = self.adasyn.fit_transform(X_select, y_select)

            # Normalize synthetic sample weights based on current training set.
            sample_weight_syn = np.empty(X_syn.shape[0], dtype=np.float64)
            sample_weight_syn[:] = 1. / X_select.shape[0]

            # Combine the original and synthetic samples.
            X_select = np.vstack((X_select, X_syn))
            y_select = np.append(y_select, y_syn)

            # Combine the weights.
            sample_weight = \
                np.append(sample_weight, sample_weight_syn).reshape(-1, 1)
            sample_weight = \
                np.squeeze(normalize(sample_weight, axis=0, norm='l1'))

            # X, y, sample_weight = shuffle(X, y, sample_weight,
            #                              random_state=random_state)

            # Boosting step.
            sample_weight, estimator_weight, estimator_error = self._boost(
                iboost,
                X_select, y_select,
                sample_weight,
                random_state)

            # Early termination.
            if sample_weight is None:
                break

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

class ADASYN(object):
    """
    Oversampling parent class with the main methods required by scikit-learn:
    fit, transform and fit_transform. This class can only be used in binary-class classification
    """

    def __init__(self,
                 num_samples = 100,
                 imb_threshold=0.5,
                 k=5,
                 random_state=None,
                 maj_class_ = 0,
                 min_class_ = 1,
                 verbose=False):
        """
        :num_samples:
            Number of new synthetic samples.
        :imb_threshold:
            The imbalance ratio threshold to allow/deny oversampling.
            For example if imb_threshold=0.5 then minority class needs
            to be at most half the size of the majority in order for
            resampling to apply
        :k:
            Number of K-nearest-neighbors
        :random_state:
            seed for random number generation
        :verbose:
            Determines if messages will be printed to terminal or not
        Extra Instance variables:
        :self.X:
            Feature matrix to be oversampled
        :self.y:
            Class labels for data
        :self.clstats:
            Class populations to determine minority/majority
        :self.unique_classes_:
            Number of unique classes
        :self.min_class_:
            Label of majority class
        :self.random_state_:
            Seed
        """

        self.num_samples = num_samples
        self.imb_threshold = imb_threshold
        self.k = k
        self.random_state = random_state
        self.verbose = verbose
        self.clstats = {}
        self.num_new = 0
        self.index_new = []
        self.min_class_ = min_class_
        self.maj_class_ = maj_class_


    def fit(self, X, y):
        """
        Class method to define class populations and store them as instance
        variables. Also stores majority class label
        """
        self.X = check_array(X)
        self.y = np.array(y).astype(np.int64)
        self.random_state_ = check_random_state(self.random_state)
        self.unique_classes_ = set(self.y)

        # Initialize all class populations with zero
        for element in self.unique_classes_:
            self.clstats[element] = 0

        # Count occurences of each class
        for element in self.y:
            self.clstats[element] += 1

        # # Find majority class
        # v = list(self.clstats.values())
        # k = list(self.clstats.keys())
        # self.maj_class_ = k[v.index(max(v))]
        # self.min_class_ = k[v.index(min(v))]

        if self.verbose:
            print(
                'Majority class is %s and total number of classes is %s'
                % (self.maj_class_, len(self.unique_classes_)))
            print(
                'Minority class is %s and total number of classes is %s'
                % (self.min_class_, len(self.unique_classes_)))

    def transform(self, X, y):
        """
        Applies oversampling transformation to data as proposed by
        the ADASYN algorithm. Returns oversampled X,y
        """
        self.new_X, self.new_y = self.oversample()

    def fit_transform(self, X, y):
        """
        Fits the data and then returns the transformed version
        """
        self.fit(X, y)
        self.new_X, self.new_y = self.oversample()

        # self.new_X = np.concatenate((self.new_X, self.X), axis=0)
        # self.new_y = np.concatenate((self.new_y, self.y), axis=0)

        return self.new_X, self.new_y

    def generate_samples(self, x, knns, knnLabels, cl):

        # List to store synthetically generated samples and their labels
        new_data = []
        new_labels = []
        for ind, elem in enumerate(x):
            # calculating k-neighbors that belong to minority (their indexes in x)
            # Unfortunately knn returns the example itself as a neighbor. So it needs
            # to be ignored thats why it is iterated [1:-1] and
            # knnLabelsp[ind][+1].
            min_knns = [ele for index,ele in enumerate(knns[ind][1:-1])
                         if knnLabels[ind][index +1] == cl]

            if not min_knns:
                continue

            # generate gi synthetic examples for every minority example
            for i in range(0, int(self.gi[ind])):
                # randi holds an integer to choose a random minority kNNs
                randi = self.random_state_.randint(
                    0, len(min_knns))
                # l is a random number in [0,1)
                l = self.random_state_.random_sample()
                # X[min_knns[randi]] is the Xzi on equation [5]
                si = self.X[elem] + \
                    (self.X[min_knns[randi]] - self.X[elem]) * l
                    
                new_data.append(si)
                new_labels.append(self.y[elem])
                self.num_new += 1

        return(np.asarray(new_data), np.asarray(new_labels))

    def oversample(self):
        """
        Preliminary calculations before generation of
        synthetic samples. Calculates and stores as instance
        variables: img_degree(d),G,ri,gi as defined by equations
        [1],[2],[3],[4] in the original paper
        """

        try:
            # Checking if variable exists, i.e. if fit() was called
            self.unique_classes_ = self.unique_classes_
        except:
            raise RuntimeError("You need to fit() before applying tranform(),"
                               "or simply fit_transform()")
        int_X = np.zeros([1, self.X.shape[1]])
        int_y = np.zeros([1])

        # ADASYN is built upon eucliden distance so p=2 default
        self.nearest_neighbors_ = NearestNeighbors(n_neighbors=self.k + 1)
        self.nearest_neighbors_.fit(self.X)

        # keeping indexes of minority examples
        minx = [ind for ind, exam in enumerate(self.X) if self.y[ind] == self.min_class_]

        # Computing kNearestNeighbors for every minority example
        knn = self.nearest_neighbors_.kneighbors(
            self.X[minx], return_distance=False)

        # Getting labels of k-neighbors of each example to determine how many of them
        # are of different class than the one being oversampled
        knnLabels = self.y[knn.ravel()].reshape(knn.shape)

        tempdi = [Counter(i) for i in knnLabels]

        # Calculating ri as defined in ADASYN paper:
        # No. of k-neighbors belonging to different class than the minority divided by K
        # which is ratio of friendly/non-friendly neighbors
        self.ri = np.array(
            [(sum(i.values())- i[self.min_class_]) / float(self.k) for i in tempdi])

        # Normalizing so that ri is a density distribution (i.e.
        # sum(ri)=1)
        if np.sum(self.ri):
            self.ri = self.ri / np.sum(self.ri)

        # Calculating #synthetic_examples that need to be generated for
        # each minority instance and rounding to nearest integer because
        # it can't produce e.g 2.35 new examples.
        self.gi = np.rint(self.ri * self.num_samples)

        # Generation of synthetic samples
        inter_X, inter_y = self.generate_samples(
                             minx, knn, knnLabels, self.min_class_)
        # in case no samples where generated at all concatenation
        # won't be attempted
        if len(inter_X):
            int_X = np.concatenate((int_X, inter_X), axis=0)
        if len(inter_y):
            int_y = np.concatenate((int_y, inter_y), axis=0)
        # New samples are concatenated in the beggining of the X,y arrays
        # index_new contains the indiced of artificial examples
        self.index_new = [i for i in range(0,self.num_new)]
        return(int_X[1:-1], int_y[1:-1])