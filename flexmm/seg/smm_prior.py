"""
t-Student Mixture Models.

This class has reused code and comments from sklearn.mixture.gmm.

This class implements the following paper:
=================================================

'Robust mixture modelling using the t distribution', D. Peel and G. J.
McLachlan.
Published at: Statistics and Computing (2000) 10, 339-348.

Jonathan Vacher (jonathan.vacher@einstein.yu.edu)
March 2019.

Modified and augmented from original code by
Luis Carlos Garcia-Peraza Herrera (luis.herrera.14@ucl.ac.uk).
24 Nov 2015.
"""

import numpy as np
import scipy as sp
import sklearn
import sklearn.cluster
import sklearn.utils
import sklearn.decomposition
import scipy.linalg
import scipy.special
import scipy.optimize
import scipy.signal
import scipy.ndimage
from scipy.ndimage import gaussian_filter
from flexmm import toolbox as tb

import warnings
import time


class dofMaximizationError(ValueError):
    def __init__(self, message):
        super(dofMaximizationError, self).__init__(message)


class SMM(sklearn.base.BaseEstimator):
    """t-Student Mixture Model.

    Representation of a t-Student mixture model probability
    distribution. This class allows for easy evaluation of, sampling
    from, and maximum-likelihood estimation of the parameters of an
    SMM distribution.

    Initializes parameters such that every mixture component has
    zero mean and identity covariance.

    Parameters
    ----------
    n_components : int, optional.
                   Number of mixture components.
                   Defaults to 1.

    covariance_type : string, optional.
                      String describing the type of covariance
                      parameters to use. Must be one of 'spherical',
                      'tied', 'diag', 'full'.
                      Defaults to 'full'.

    random_state: RandomState or an int seed.
                  A random number generator instance.
                  None by default.

    tol : float, optional.
          Convergence threshold. EM iterations will stop when
          average gain in log-likelihood is below this threshold.
          Defaults to 1e-6.

    min_covar : float, optional.
                Floor on the diagonal of the covariance matrix to
                prevent overfitting.
                Defaults to 1e-6.

    n_iter : int, optional.
             Number of EM iterations to perform.
             Defaults to 1000.

    n_init : int, optional.
             Number of initializations to perform. The best result
             is kept.
             Defaults to 1.

    params : string, optional.
             Controls which parameters are updated in the training
             process. Can contain any combination of 'w' for
             weights, 'm' for means, 'c' for covars, and 'd' for the
             degrees of freedom.
             Defaults to 'wmcd'.

    init_params : string, optional.
                  Controls which parameters are updated in the
                  initialization process.  Can contain any
                  combination of 'w' for weights, 'm' for means,
                  'c' for covars, and 'd' for the degrees of
                  freedom.
                  Defaults to 'wmcd'.

    Attributes
    ----------
    weights_ : array, shape (`n_components`,).
               This attribute stores the mixing weights for each
               mixture component.

    means_ : array_like, shape (`n_components`, `n_features`).
             Mean parameters for each mixture component.

    covars_ : array_like.
              Covariance parameters for each mixture component. The
              shape depends on `covariance_type`:

                  (n_components, n_features)             if 'spherical',
                  (n_features, n_features)               if 'tied',
                  (n_components, n_features)             if 'diag',
                  (n_components, n_features, n_features) if 'full'

    converged_ : bool.
                 True when convergence was reached in fit(), False
                 otherwise.
    """

    def __init__(
        self,
        n_components=1,
        covariance_type="full",
        prior_weights=None,
        prior_means=None,
        prior_var=None,
        prior_init=False,
        im_shape=(256, 256),
        neigh_size=3,
        random_state=None,
        tol=1e-6,
        min_covar=1e-6,
        n_iter=1000,
        n_init=1,
        light=True,
        params="wqmcd",
        init_params="wqmcd",
        ppca=False,
        n_pca=0.95,
    ):

        # Store the parameters as class attributes
        # sets number of components in mixture
        self.n_components = n_components
        # sets covariance type to use?
        self.covariance_type = covariance_type
        # FIXME: prior_weights = "ext3" working but not prior_weights=None
        self.prior_weights = prior_weights

        # spatial smoothing case
        if self.prior_weights != None:
            self.prior_means = prior_means
            self.prior_var = prior_var
            self.prior_norm = 1.0
            self.prior_init = prior_init
            self.neigh_size = neigh_size
            Y, X = np.mgrid[
                -(neigh_size - 1) // 2 : (neigh_size - 1) // 2 + 1,
                -(neigh_size - 1) // 2 : (neigh_size - 1) // 2 + 1,
            ]
            ##NOTE: sets up a 2d Gaussian based on neigh_size
            ##DEBUG: failure to initialize self.neighbors leads to an error in convolution
            # self.neighbors = tb.gauss2d(X, Y, neigh_size / 4.0)
            ##NOTE: normalizes 2d Gaussian kernel
            self.neighbors = tb.gauss2d(X, Y, neigh_size / 4.0)
            self.neighbors /= self.neighbors.sum()
            self.neighbors = self.neighbors[..., np.newaxis]
            self.im_shape = im_shape
        else:
            # CHANGED: #6 added initialization for self.prior_init
            self.prior_init = prior_init
        # not the spatial smoothing case?
        # DEBUG: if prior_weights is set to None, none of the above parameters are defined
        self.random_state = random_state
        self.tol = tol
        self.min_covar = min_covar
        self.n_iter = n_iter
        self.n_init = n_init
        self.light = light
        self.params = params
        self.init_params = init_params
        self.ppca = ppca
        self.n_pca = n_pca
        self.converged_ = False

    def _expectation_step(self, X):
        """Performs the expectation step of the EM algorithm.

        This method uses the means, class-related weights,
        covariances and degrees of freedom stored in the attributes
        of this class:
        self.means_, self.weights_, self.covars_, and self.degrees_.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features).
            Matrix with all the data points, each row represents a
            new data point and the columns represent each one of the
            features.
        """

        # Sanity checks:
        #    - Check that the fit() method has been called before this
        #      one.
        #    - Convert input to 2d array, raise error on sparse
        #      matrices. Calls assert_all_finite by default.
        #    - Check that the the X array is not empty of samples.
        #    - Check that the no. of features is equivalent to the no.
        #      of means that we have in self.
        sklearn.utils.validation.check_is_fitted(self, "means_")
        X = sklearn.utils.validation.check_array(X, dtype=np.float64)
        if X.ndim == 1:
            X = X[:, np.newaxis]
        if X.size == 0:
            return np.array([]), np.empty((0, self.n_components))
        if X.shape[1] != self.means_.shape[1]:
            raise ValueError(
                "[SMM._expectation_step] Error, the "
                + "shape of X is not compatible with self."
            )

        # NOTE: Initialisation of reponsibilities and weight of each point for
        # the Gamma distribution
        n_samples, n_dim = X.shape
        responsibilities = np.ndarray(
            shape=(X.shape[0], self.n_components), dtype=np.float64
        )

        gammaweights_ = np.ndarray(
            shape=(X.shape[0], self.n_components), dtype=np.float64
        )

        # NOTE: Calculate the probability of each point belonging to each
        # t-Student distribution of the mixture
        # NOTE: the formula for the PDF is in McLachlan & Peel
        # NOTE: the q parameter optimizes the calculation of the covariance matrix?
        pr_before_weighting = self._multivariate_t_student_density(
            X,
            self.means_,
            self.covars_,
            self.degrees_,
            self.q_,
            self.covariance_type,
            self.min_covar,
        )

        pr = pr_before_weighting * self.weights_

        # Calculate the likelihood of each point
        likelihoods = pr.sum(axis=1)

        # Update responsibilities
        # NOTE: the likelihood of an observation being from component "i"
        # is the likelihood at one weight over the likelihood over the mixture
        responsibilities = pr / (likelihoods.reshape(likelihoods.shape[0], 1))  # \
        # + 10 * SMM._EPS)
        responsibilities = np.clip(responsibilities, 0, 1)

        # Update the Gamma weight for each observation
        # NOTE: the Gamma weight is E(mixer | observation)
        mahalanobis_distance_mix_func = SMM._mahalanobis_funcs[self.covariance_type]
        vp = self.degrees_ + n_dim
        maha_dist = mahalanobis_distance_mix_func(
            X, self.means_, self.covars_, self.q_, self.min_covar
        )

        gammaweights_ = np.zeros_like(responsibilities)
        # NOTE: by default, self.q_ is an array of ones?
        gammaweights_[:, self.q_] = vp[self.q_] / (
            self.degrees_[self.q_] + maha_dist[:, self.q_]
        )

        return likelihoods, responsibilities, gammaweights_

    def _maximisation_step(self, X, responsibilities, gammaweights_):
        """Perform the maximisation step of the EM algorithm.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features).
            Each row corresponds to a single data point.

        responsibilities : array_like, shape (n_samples, n_components).

        gammaweights_ : array_like, shape (n_samples, n_components).
        """

        n_samples, n_dim = X.shape
        z_sum = responsibilities.sum(axis=0)
        zu = responsibilities * gammaweights_
        zu_sum = zu.sum(axis=0)
        # Update weights
        if "w" in self.params:
            if self.prior_weights == None:
                # DEBUG: this is the correct update rule for no spatial smoothing
                # CHANGED: #5 change this case to match the "ext3" case
                # otherwise the prior_mean and prior_var attributes are not used
                # self.weights_ = z_sum / n_samples
                self.weights_ = self.prior_means / self.prior_norm
            elif self.prior_weights == "ext3":
                # outside this algo
                self.weights_ = self.prior_means / self.prior_norm
            elif self.prior_weights == "ext2":
                # prior_means, prior_var and prior_norm
                # are set outside this alg
                # and have the following form
                # prior_means = s1*m2+s2*m1
                # prior_var = s1*s2
                # prior_norm = s1+s2+s1*s2
                means = scipy.ndimage.convolve(
                    responsibilities.reshape(
                        (self.im_shape[0], self.im_shape[1], self.n_components)
                    ),
                    self.neighbors,
                    mode="nearest",
                ).reshape(n_samples, self.n_components)
                var = scipy.ndimage.convolve(
                    (responsibilities**2).reshape(
                        (self.im_shape[0], self.im_shape[1], self.n_components)
                    ),
                    self.neighbors,
                    mode="nearest",
                ).reshape(n_samples, self.n_components)

                var -= means**2
                var = var.mean()
                # transfer to prior weights
                self.weights_ = var * self.prior_means + self.prior_var * means
                self.weights_ /= var * self.prior_norm + self.prior_var
            elif self.prior_weights == "ext1":
                # prior_means and prior var is set outside this alg
                # transfer to prior weights
                self.weights_ = self.prior_means + self.prior_var * responsibilities
                self.weights_ /= 1 + self.prior_var
            elif self.prior_weights == "loc":
                # prior mean
                self.prior_means = scipy.ndimage.convolve(
                    responsibilities.reshape(
                        (self.im_shape[0], self.im_shape[1], self.n_components)
                    ),
                    self.neighbors,
                    mode="nearest",
                ).reshape(n_samples, self.n_components)
                # prior var
                self.prior_var = scipy.ndimage.convolve(
                    (responsibilities**2).reshape(
                        (self.im_shape[0], self.im_shape[1], self.n_components)
                    ),
                    self.neighbors,
                    mode="nearest",
                ).reshape(n_samples, self.n_components)

                self.prior_var -= self.prior_means**2
                self.prior_var = self.prior_var.mean()  # axis=1, keepdims=True)

                # transfer to prior weights
                self.weights_ = self.prior_means + self.prior_var * responsibilities
                self.weights_ /= 1 + self.prior_var

        if "q" in self.params:
            if self.prior_weights in {None, "loc"}:
                # self.cov_dist = sp.spatial.distance.pdist(
                #    self.covars_.reshape(self.n_components,n_dim**2),\
                #                metric='cosine')
                # self.be = (sp.spatial.distance.squareform(\
                #              self.cov_dist)<0.2).sum(0)-1
                # print(self.cov_dist)
                # print(self.covars_.reshape(self.n_components,n_dim**2))
                # if np.sum(self.be>self.n_components//2)>0:
                #    self.be = 0*self.be
                # self.q_weights_ = z_sum / (z_sum + n_samples / self.n_components)
                # self.be = self.be/3.0
                # print(self.be)

                # self.q_weights_ = z_sum/(z_sum+
                #                    self.be*n_samples/(self.q_.sum()))
                # print(self.q_weights_)
                # CHANGED: #5 match between None and "ext3" case
                self.q_weights_ = self.q_weights_
                self.q_ = self.q_weights_ > 0.5
                # self.q_ = self.be<4.0
                self.weights_ *= self.q_
            elif self.prior_weights == "ext3":
                # set q_weights outside
                self.q_weights_ = self.q_weights_
                self.q_ = self.q_weights_ > 0.5
                self.weights_ *= self.q_
        else:
            self.q_weights_ = np.ones(self.n_components)
            self.q_ = np.ones(self.n_components, dtype=bool)

        # Update means
        if "m" in self.params:
            if self.ppca:
                for k in range(self.n_components):
                    if self.q_[k]:
                        # print(self.pcs[k].shape,self.Y[k].shape)
                        self.means_[k] = np.sum(
                            zu[:, k][:, np.newaxis]
                            * (X - (self.pcs[k] @ self.Y[k].T).T),
                            axis=0,
                        )
                        self.means_[k] /= zu_sum[k] + 10 * SMM._EPS
            else:
                dot_zu_x = np.dot(zu.T, X)
                zu_sum_ndarray = zu_sum.reshape(zu_sum.shape[0], 1)
                self.means_ = dot_zu_x / (zu_sum_ndarray + 10 * SMM._EPS)

        # Update covariances
        if "c" in self.params:
            covar_mstep_func = SMM._covar_mstep_funcs[self.covariance_type]
            self.covars_ = covar_mstep_func(
                X, zu, z_sum, self.means_, self.q_, self.min_covar
            )

        # Update degrees of freedom
        if "d" in self.params:
            try:
                self.degrees_ = SMM._solve_dof_equation(
                    self.degrees_,
                    responsibilities,
                    z_sum,
                    gammaweights_,
                    self.q_,
                    n_dim,
                    self.tol,
                    self.n_iter,
                )
                # print(self.degrees_)
            except FloatingPointError as fpe:
                message = str(fpe)
                raise dofMaximizationError(message)
            except RuntimeError as re:
                message = str(re)
                if message.startswith("Failed to converge after"):
                    warnings.warn(message, RuntimeWarning)
                    pass

    def _initialization_step(
        self,
        X,
        gt=None,
        gt_eps=None,
        gt_partition=1,
        n_components_best=None,
        use_kmeans=False,
    ):
        """Performs the initialization step of the EM algorithm.

        This method initializes the means, class-related weights,
        covariances and degrees of freedom stored in the attributes
        of this class:
        self.means_, self.weights_, self.covars_, and self.degrees_.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features).
        """

        if "m" in self.init_params or not hasattr(self, "means_"):
            if use_kmeans:
                kmeans = sklearn.cluster.KMeans(
                    n_clusters=self.n_components,
                    init="k-means++",
                    random_state=self.random_state,
                )
                # NOTE: assign mean of t- components as mean of KMeans cluster centers
                self.means_ = kmeans.fit(X).cluster_centers_
            elif gt is not None:
                if gt_partition:
                    cluster_centers, _ = tb.gt_pca_cluster_centers(X, gt)
                    self.means_ = cluster_centers
                    assert self.means_.shape[0] == self.n_components
                else:
                    # TODO: fix this
                    self.means_ = np.mean(X.T)
                    assert self.means_.shape[0] == self.n_components
            else:
                # NOTE: if no k-means or groundtruth initialization, initialize means to 0
                self.means_ = np.zeros((self.n_components, X.shape[1]))

        if "w" in self.init_params or not hasattr(self, "weights_"):
            if self.prior_weights == None:
                if gt is not None:
                    epsilon = 0.0001
                    if gt_eps is not None:
                        epsilon = gt_eps
                    ny = gt.shape[0]
                    nx = gt.shape[1]
                    vec = np.reshape(gt, ny * nx)
                    vals, counts = np.asarray(np.unique(vec, return_counts=True))
                    # vals[counts<tb.THRESHOLD] = 0
                    adj_vals = vals[vals == vals]
                    indicators = []
                    for val in adj_vals:
                        indicator = np.zeros(vec.shape)
                        indicator += epsilon
                        indicator[vec == val] = 1 - (len(adj_vals) - 1) * epsilon
                        indicators.append(indicator)
                    indicators = np.asarray(indicators)
                    self.weights_ = indicators.T
                    assert self.weights_.shape[1] == self.n_components
                    self.weights_ /= self.weights_.sum(axis=1, keepdims=True)
                    # print(self.prior_means/self.prior_means.sum(axis=1, keepdims=True).shape)
                else:
                    self.weights_ = np.tile(1.0 / self.n_components, self.n_components)
            elif self.prior_weights in {"ext", "ext2", "ext3"}:
                if gt is not None:
                    epsilon = 0.0001
                    if gt_eps is not None:
                        epsilon = gt_eps
                    ny = gt.shape[0]
                    nx = gt.shape[1]
                    vec = np.reshape(gt, ny * nx)
                    vals, counts = np.asarray(np.unique(vec, return_counts=True))
                    # vals[counts<tb.THRESHOLD] = 0
                    adj_vals = vals[vals == vals]
                    indicators = []
                    for val in adj_vals:
                        indicator = np.zeros(vec.shape)
                        indicator += epsilon
                        indicator[vec == val] = 1 - (len(adj_vals) - 1) * epsilon
                        indicators.append(indicator)
                    indicators = np.asarray(indicators)
                    self.weights_ = indicators.T
                    assert self.weights_.shape[1] == self.n_components
                    self.weights_ /= self.weights_.sum(axis=1, keepdims=True)
                    # print(self.prior_means/self.prior_means.sum(axis=1, keepdims=True).shape)
                else:
                    # prior_means is initially set to np.ones
                    self.weights_ = self.prior_means
                    self.weights_ /= self.weights_.sum(axis=1, keepdims=True)
            elif self.prior_weights == "loc":
                self.weights_ = self.prior_means
                self.weights_ /= self.weights_.sum(axis=1, keepdims=True)

        if "q" in self.init_params or not hasattr(self, "q_weights_"):
            self.q_weights_ = np.ones(self.n_components)
            self.q_ = self.q_weights_ > 0.5

        if "c" in self.init_params or not hasattr(self, "covars_"):
            cv = np.cov(X.T) + self.min_covar * np.eye(X.shape[1])
            if not cv.shape:
                cv.shape = (1, 1)
            self.covars_ = SMM.dist_covar_to_match_cov_type(
                cv, self.covariance_type, self.n_components
            )
            # if self.ppca:
            #    self.pcs = np.zeros((self.n_components,X.shape[1],self.n_pca))
            #    for k in range(self.n_components):
            #        self.pcs[k] = np.random.normal(size=(X.shape[1],self.n_pca))
            #        self.covars_[k] = self.pcs[k]@self.pcs[k].T
            # print(self.covars_[k])
            # print(mat)

        if "d" in self.init_params or not hasattr(self, "degrees_"):
            # initial guess is 2 degrees of freedom
            self.degrees_ = np.tile(2.0, self.n_components)
            # 0.5+3*np.random.rand(self.n_components)
            #

        # prior_init is False for the first layer, true for following layers
        # CHANGED: #5 create a None case here
        if self.prior_weights is None and self.prior_init:
            self._maximisation_step(
                X, self.prior_means, np.ones((X.shape[0], self.n_components))
            )
        if self.prior_weights in {"ext", "loc"} and self.prior_init:
            self._maximisation_step(
                X, self.prior_means, np.ones((X.shape[0], self.n_components))
            )

    def fit(self, X, y=None):
        """Estimate model parameters with the EM algorithm.

        A initialization step is performed before entering the em
        algorithm. If you want to avoid this step, set the keyword
        argument init_params to the empty string '' when creating
        the SMM object. Likewise, if you would like just to do an
        initialization, set n_iter=0.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features).

        y : not used, just for compatibility with sklearn API.
        """

        # Sanity check: assert that the input matrix is not 1D
        if len(X.shape) == 1:
            raise ValueError(
                "[SMM.fit] Error, the input matrix must have a "
                + "shape of (n_samples, n_features)."
            )

        # Sanity checks:
        #    - Convert input to 2d array, raise error on sparse
        #      matrices. Calls assert_all_finite by default.
        #    - No. of samples is higher or equal to the no. of
        #      components in the mixture.
        X = sklearn.utils.validation.check_array(X, dtype=np.float64)
        if X.shape[0] < self.n_components:
            raise ValueError(
                "[SMM.fit] Error, SMM estimation with "
                + "%s components, but got only %s samples"
                % (self.n_components, X.shape[0])
            )

        # For all the initialisations we get the one with the best
        # parameters
        n_samples, n_dim = X.shape
        max_prob = -np.infty
        for _ in range(self.n_init):

            # EM initialisation
            self._initialization_step(X, use_kmeans=True)

            best_params = {
                "weights": self.weights_,
                "means": self.means_,
                "covars": self.covars_,
                "degrees": self.degrees_,
            }

            self.converged_ = False
            current_log_likelihood = None

            # EM algorithm
            for i in range(self.n_iter):
                prev_log_likelihood = current_log_likelihood
                # Expectation step
                likelihoods, responsibilities, gammaweights_ = self._expectation_step(X)

                # Sanity check: assert that the likelihoods,
                # responsibilities and gammaweights have the correct
                # dimensions
                assert len(likelihoods.shape) == 1
                assert likelihoods.shape[0] == n_samples
                assert len(responsibilities.shape) == 2
                assert responsibilities.shape[0] == n_samples
                assert responsibilities.shape[1] == self.n_components
                assert len(gammaweights_.shape) == 2
                assert gammaweights_.shape[0] == n_samples
                assert gammaweights_.shape[1] == self.n_components

                # Calculate loss function
                current_log_likelihood = np.log(likelihoods).mean()

                # Check for convergence
                if prev_log_likelihood is not None:
                    change = np.abs(current_log_likelihood - prev_log_likelihood)
                    if change < self.tol:
                        self.converged_ = True
                        break

                # Maximisation step
                # self._maximisation_step(X, responsibilities,
                #                               gammaweights_
                #     )

                try:
                    self._maximisation_step(X, responsibilities, gammaweights_)
                except dofMaximizationError as e:
                    print(
                        "[self._maximisation_step] Error in the "
                        + "maximization step of the degrees of "
                        + "freedom: "
                        + e.message
                    )
                    break

            # If the results are better, keep it
            # if self.n_iter and self.converged_:
            if current_log_likelihood > max_prob:
                max_prob = current_log_likelihood
                best_params = {
                    "weights": self.weights_,
                    "means": self.means_,
                    "covars": self.covars_,
                    "degrees": self.degrees_,
                }

        # Check the existence of an init param that was not subject to
        # likelihood computation issue
        if np.isneginf(max_prob) and self.n_iter:
            msg = (
                "EM algorithm was never able to compute a valid "
                + "likelihood given initial parameters. Try "
                + "different init parameters (or increasing "
                + "n_init) or check for degenerate data."
            )
            warnings.warn(msg, RuntimeWarning)

        # Choosing the best result of all the iterations as the actual
        # result
        if self.n_iter:
            self.weights_ = best_params["weights"]
            self.means_ = best_params["means"]
            self.covars_ = best_params["covars"]
            self.degrees_ = best_params["degrees"]

        if self.light:
            del (self.Y, self.B_, self.S)
            self.weights_ = np.float16(self.weights_)
            self.means_ = np.float32(self.means_)
            self.covars_ = np.float32(self.covars_)
            self.degrees_ = np.float32(self.degrees_)
            self.pcs = np.float32(self.pcs)
            self.taus = np.float32(self.taus)

        return self

    def _posterior_proba(self, X):
        """Compute the posterior probability of X.

        (i.e., the probability of X coming from the inferred mixture)

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features).
            Each row corresponds to a single data point.

        responsibilities : array_like, shape (n_samples, n_components).
        """

        # Sanity checks:
        #    - Check that the fit() method has been called before this
        #      one.
        #    - Convert input to 2d array, raise error on sparse
        #      matrices. Calls assert_all_finite by default.
        #    - Check that the the X array is not empty of samples.
        #    - Check that the no. of features is equivalent to the no.
        #      of means that we have in self.
        sklearn.utils.validation.check_is_fitted(self, "means_")
        X = sklearn.utils.validation.check_array(X, dtype=np.float64)
        if X.ndim == 1:
            X = X[:, np.newaxis]
        if X.size == 0:
            return np.array([]), np.empty((0, self.n_components))
        if X.shape[1] != self.means_.shape[1]:
            raise ValueError(
                "[SMM._expectation_step] Error, the "
                + "shape of X is not compatible with self."
            )

        # Initialisation of posterior probabilities (responsibilities)
        n_samples, n_dim = X.shape
        responsibilities = np.ndarray(
            shape=(X.shape[0], self.n_components), dtype=np.float64
        )

        # Calculate the probability of each point belonging to each
        # t-Student distribution of the mixture
        # DEBUG: should this add up to 1? it doesn't in the prior_weights == None case, check the prior_weighst=="ext3" case
        pr_before_weighting = self._multivariate_t_student_density(
            X,
            self.means_,
            self.covars_,
            self.degrees_,
            self.q_,
            self.covariance_type,
            self.min_covar,
        )
        pr = pr_before_weighting * self.weights_

        # Calculate the likelihood of each point
        likelihoods = pr.sum(axis=1)

        # Compute posterior probability
        responsibilities = pr / (likelihoods.reshape(likelihoods.shape[0], 1))

        responsibilities = np.clip(responsibilities, 0, 1)

        return responsibilities

    def predict(self, X):
        """Predict label for data.

        This function will tell you which component of the mixture
        most likely generated the sample.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features).

        Returns
        -------
        r_argmax : array_like, shape (n_samples,).
        """

        _, responsibilities, _ = self._expectation_step(X)
        r_argmax = responsibilities.argmax(axis=1)

        return r_argmax

    def predict_proba(self, X):
        """Give label probabilities for data.

        This function will tell the probability of each component
        generating each sample.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features).

        Returns
        -------
        responsibilities : array_like, shape (n_samples, n_components).
        """

        _, responsibilities, _ = self._expectation_step(X)

        return responsibilities

    def score(self, X, y=None):
        """Compute the log probability under the model.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features).

        Returns
        -------
        prob : array_like, shape (n_samples,).
               Probabilities of each data point in X.
        """

        prob, _ = self.score_samples(X)

        return prob

    def score_samples(self, X):
        """Per-sample likelihood of the data under the model.

        Compute the probability of X under the model and return the
        posterior distribution (responsibilities) of each mixture
        component for each element of X.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features).

        Returns
        -------
        prob : array_like, shape (n_samples,).
               Unnormalised probability of each data point in X,
               i.e. likelihoods.

        responsibilities : array_like, shape (n_samples,
                           n_components).
                           Posterior probabilities of each mixture
                           component for each observation.
        """

        sklearn.utils.validation.check_is_fitted(self, "means_")
        X = sklearn.utils.validation.check_array(X, dtype=np.float64)

        if X.ndim == 1:
            X = X[:, np.newaxis]
        if X.size == 0:
            return np.array([]), np.empty((0, self.n_components))
        if X.shape[1] != self.means_.shape[1]:
            raise ValueError(
                "[score_samples] ValueError, the shape of X is not "
                + "compatible with self."
            )

        # Initialisation of reponsibilities and weight of each point for
        # the Gamma distribution
        n_samples, n_dim = X.shape
        responsibilities = np.ndarray(
            shape=(X.shape[0], self.n_components), dtype=np.float64
        )
        gammaweights_ = np.ndarray(
            shape=(X.shape[0], self.n_components), dtype=np.float64
        )

        # Calculate the probability of each point belonging to each
        # t-Student distribution of the mixture
        pr = (
            self._multivariate_t_student_density(
                X,
                self.means_,
                self.covars_,
                self.degrees_,
                self.q_,
                self.covariance_type,
                self.min_covar,
            )
            * self.weights_
        )

        # Calculate the likelihood of each point
        likelihoods = pr.sum(axis=1)

        # Update responsibilities
        like_ndarray = likelihoods.reshape(likelihoods.shape[0], 1)
        responsibilities = pr / (like_ndarray + 10 * SMM._EPS)

        return likelihoods, responsibilities

    def _n_parameters(self):
        """Returns the number of free parameters in the model."""

        _, n_features = self.means_.shape

        if self.covariance_type == "full":
            cov_params = self.n_components * n_features * (n_features + 1) / 2.0
        elif self.covariance_type == "diag":
            cov_params = self.n_components * n_features
        elif self.covariance_type == "tied":
            cov_params = n_features * (n_features + 1) / 2.0
        elif self.covariance_type == "spherical":
            cov_params = self.n_components

        mean_params = n_features * self.n_components
        df_params = self.n_components
        total_param = int(cov_params + mean_params + df_params + self.n_components - 1)

        return total_param

    def bic(self, X):
        """Bayesian information criterion for the current model fit
        and the proposed data.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features).

        Returns
        -------
        A float (the lower the better).
        """

        retval = -2 * self.score(X).sum() + self._n_parameters() * np.log(X.shape[0])

        return retval

    def aic(self, X):
        """Akaike information criterion for the current model fit
        and the proposed data.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features).

        Returns
        -------
        A float (the lower the better).
        """

        retval = -2 * self.score(X).sum() + 2 * self._n_parameters()

        return retval

    @staticmethod
    def _solve_dof_equation(v_vector, z, z_sum, u, active, n_dim, tol, n_iter):
        """Solves the equation to calculate the next value of v
        (degrees of freedom).

        This method is part of the maximisation step. It is used to
        calculate the next value for the degrees of freedom of each
        t-Student component. This method uses the information from
        the E-step as well as the number of dimensions (features) of
        a point.

        Parameters
        ----------
        v_vector : array_like, shape (n_components,).
                   Degrees of freedoom of ALL the components of the
                   mixture.

        z : array_like, shape (n_samples, n_components).
            Matrix of responsibilities, each row represents a point
            and each column represents a component of the mixture.

        z_sum : array_like, shape (n_samples,).
                Sum of all the rows of the matrix of
                responsibilities.

        u : array_like, shape (n_samples, n_components).
            Matrix of gamma weights, each row represents a point and
            each column represents a component of the mixture.

        n_dim : integer.
                Number of features of each data point.

        Returns
        -------
        new_v_vector : array_like (n_components,).
                       Vector with the updated degrees of freedom for
                       each component in the mixture.
        """

        # Sanity check: check that the dimensionality of the vector of
        # degrees of freedom is correct
        assert len(v_vector.shape) == 1
        n_components = v_vector.shape[0]

        # Sanity check: the matrix of responsibilities should be
        # (n_samples x n_components)
        assert len(z.shape) == 2
        assert z.shape[1] == n_components

        # Sanity check: the top-to-bottom sum of the responsibilities
        # must have a shape (n_components, )
        assert len(z_sum.shape) == 1
        assert z_sum.shape[0] == n_components

        # Sanity check: gamma weights matrix must have the same
        # dimensionality as the responsibilities
        assert u.shape == z.shape

        # Initialisation
        new_v_vector = np.empty_like(v_vector)

        # Calculate the constant part of the equation to calculate the
        # new degrees of freedom
        zlogu_sum = np.zeros_like(v_vector)
        constant_part = np.zeros_like(v_vector)

        vdim = (v_vector + n_dim) / 2.0
        zlogu_sum[active] = np.sum(
            z[:, active] * (np.log(u[:, active]) - u[:, active]), axis=0
        )
        constant_part[active] = (
            1.0
            + zlogu_sum[active] / z_sum[active]
            + scipy.special.digamma(vdim[active])
            - np.log(vdim[active])
        )
        # Solve the equation numerically using Newton-Raphson for each
        # component of the mixture

        for c in range(n_components):
            if active[c] == 1:

                def func(x):
                    return (
                        np.log(x / 2.0)
                        - scipy.special.digamma(x / 2.0)
                        + constant_part[c]
                    )

                def fprime(x):
                    return 1.0 / x - scipy.special.polygamma(1, x / 2.0) / 2.0

                def fprime2(x):
                    return -1.0 / (x * x) - scipy.special.polygamma(2, x / 2.0) / 4.0

                #             if func(1e-9)*func(1e12)<0:
                #                 new_v_vector[c] = scipy.optimize.newton(
                #                     func, v_vector[c], fprime, args=(), tol=tol,
                #                     maxiter=1000, fprime2=fprime2)
                #             else:
                #                 print('pwet')
                if func(1e-9) * func(1e12) < 0:
                    new_v_vector[c] = scipy.optimize.bisect(func, 1e-9, 1e12)
                else:
                    new_v_vector[c] = v_vector[c]

                if new_v_vector[c] < 0.0:
                    raise ValueError(
                        "[_solve_dof_equation] Error, "
                        + "degree of freedom smaller than zero. \n"
                        + "n_components[c] = "
                        + str(n_components)
                        + ". \n"
                        + "v_vector[c] = "
                        + str(v_vector[c])
                        + ". \n"
                        + "new_v_vector[c] = "
                        + str(new_v_vector[c])
                        + ". \n"
                        + "constant_part[c] = "
                        + str(constant_part[c])
                        + ". \n"
                        + "zlogu_sum[c] = "
                        + str(zlogu_sum[c])
                        + ". \n"
                        + "z_sum[c] = "
                        + str(z_sum[c])
                        + ". \n"
                        + "z = "
                        + str(z)
                        + ". \n"
                    )

        return new_v_vector

    @staticmethod
    def _covar_mstep_diag(X, zu, z_sum, means, active, min_covar):
        """Performing the covariance m-step for diagonal
        covariances.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features).
            List of k_features-dimensional data points.
            Each row corresponds to a single data point.

        zu : array, shape (n_samples, n_components).
             Contains the element-wise multiplication of the
             responsibilities by the gamma weights.

        z_sum : array_like, shape (n_components,).
                Sum of all the responsibilities for each mixture.

        means : array_like, shape (n_components, n_features).
                List of n_features-dimensional mean vectors for
                n_components t-Students. Each row corresponds to a
                single mean vector.

        min_covar : float value.
                    Minimum amount that will be added to the
                    covariance matrix in case of trouble, usually
                    1.e-6.

        Returns
        -------
        retval : array_like, shape (n_components, n_features).
        """

        # Eq. 15 from K. Murphy, "Fitting a Conditional Linear Gaussian
        # Distribution" adapted to the mixture of t-students (i.e.
        # responsibilities matrix is multiplied element-wise by the
        # gamma weights matrix. See that zu.T is used in the calculation
        # of weighted_X_sum)
        norm = 1.0 / (z_sum[:, np.newaxis])  # + 10 * SMM._EPS)
        diff2 = (X.T[np.newaxis, ...] - means[..., np.newaxis]) ** 2
        retval = np.sum(zu[:, np.newaxis, :].T * diff2, axis=2) * norm

        return retval

    @staticmethod
    def _covar_mstep_spherical(*args):
        cv = SMM._covar_mstep_diag(*args)
        return np.tile(cv.mean(axis=1)[:, np.newaxis], (1, cv.shape[1]))

    @staticmethod
    def _covar_mstep_full(X, zu, z_sum, means, active, min_covar):
        """Performing the covariance m-step for full covariances.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features).
            List of k_features-dimensional data points. Each row
            corresponds to a single
            data point.

        zu : array_like, shape (n_samples, n_components).
             Contains the element-wise multiplication of the
             responsibilities by the gamma weights.

        z_sum : array_like, shape (n_components,)
                Sum of all the responsibilities for each mixture.

        means : array_like, shape (n_components, n_features).
                List of n_features-dimensional mean vectors for
                n_components t-Students.
                Each row corresponds to a single mean vector.

        min_covar : float value.
                    Minimum amount that will be added to the
                    covariance matrix in case of trouble, usually 1.e-7.

        Returns
        -------
        cv : array_like, shape (n_components, n_features,
             n_features).
             New array of updated covariance matrices.
        """

        # Sanity checks for dimensionality
        n_samples, n_features = X.shape
        n_components = means.shape[0]
        assert zu.shape[0] == n_samples
        assert zu.shape[1] == n_components
        assert z_sum.shape[0] == n_components

        # Eq. 31 from D. Peel and G. J. McLachlan, "Robust mixture
        # modelling using the t distribution"
        cv = np.empty((n_components, n_features, n_features))
        zu_sum = zu.sum(axis=0)
        for c in range(n_components):
            if active[c] == 1:
                post = zu[:, c]
                mu = means[c]
                diff = X - mu
                with np.errstate(under="ignore"):
                    # Underflow Errors in doing post * X.T are not important
                    if n_components == 1:
                        avg_cv = np.dot(post * diff.T, diff) / (
                            zu_sum[c] + 10 * SMM._EPS
                        )
                    else:
                        avg_cv = np.dot(post * diff.T, diff) / (
                            z_sum[c] + 10 * SMM._EPS
                        )

                cv[c] = avg_cv + min_covar * np.eye(n_features)

        return cv

    @staticmethod
    def _covar_mstep_tied(X, zu, z_sum, means, min_covar):
        """Performing the for tied a covariance.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features).
            List of k_features-dimensional data points. Each row
            corresponds to a single data point.

        zu : array_like, shape (n_samples, n_components).
             Contains the element-wise multiplication of the
             responsibilities by the gamma weights.

        z_sum : array_like, shape (n_components, )
                Sum of all the responsibilities for each mixture.

        means : array_like, shape (n_components, n_features).
                List of n_features-dimensional mean vectors for
                n_components t-Students. Each row corresponds to a
                single mean vector.

        min_covar : float value.
                    Minimum amount that will be added to the covariance
                    matrix in case of trouble, usually 1.e-6.

        Returns
        -------
        out : array_like, shape (n_features, n_features).
        """

        avg_X2 = np.dot(zu.T, X * X)
        avg_means2 = np.dot(z_sum * means.T, means)
        out = avg_X2 - avg_means2
        out /= z_sum.sum()
        out.flat[:: len(out) + 1] += min_covar

        return out

    @staticmethod
    def _multivariate_t_student_density_diag(X, means, covars, dfs, min_covar):
        """Multivariate t-Student PDF for a matrix of data points and
        diagonal covariance matrices.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features).
            Each row corresponds to a single data point.

        means : array_like, shape (n_components, n_features).
                Each row corresponds to a single mean vector.

        covars : array_like, shape (n_components, n_features).
                 List of n_components covariance parameters for each
                 t-Student.

        dfs : array_like, shape (n_components,).
              Degrees of freedom.

        min_covar : float value.
                    Minimum amount that will be added to the covariance
                    matrix in case of trouble, usually 1.e-6.

        Returns
        -------
        retval : array_like, shape (n_samples, n_components).
                 Evaluation of the multivariate probability density
                 function for a t-Student distribution.
        """

        # Sanity check: make sure that the shape of means and
        # covariances is as expected for diagonal matrices
        n_samples, n_dim = X.shape
        assert covars.shape[0] == means.shape[0]
        assert covars.shape[1] == means.shape[1]
        assert covars.shape[1] == n_dim

        # NOTE: Calculate inverse and determinant of the covariances
        #         inv_covars = 1.0 / covars
        log_det_covars = np.sum(np.log(covars), axis=1)

        # Calculate the value of the numerator
        log_num = scipy.special.gammaln((dfs + n_dim) / 2.0)

        # Calculate Mahalanobis distance from all the points to the
        # mean of each component in the mix
        maha = SMM._mahalanobis_distance_mix_diag(X, means, covars, min_covar)

        # Calculate the value of the denominator
        braces = 1.0 + maha / dfs
        log_denom = (
            np.log(np.pi * dfs) * n_dim / 2.0
            + scipy.special.gammaln(dfs / 2.0)
            + 0.5 * log_det_covars
            + np.log(braces) * (dfs + n_dim) / 2.0
        )

        retval = np.exp(log_num - log_denom)

        return retval

    @staticmethod
    def _multivariate_t_student_density_spherical(X, means, covars, dfs, min_covar):
        """Multivariate t-Student PDF for a matrix of data points.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features).
            Each row corresponds to a single data point.

        means : array_like, shape (n_components, n_features).
                Mean vectors for n_components t-Students.
                Each row corresponds to a single mean vector.

        covars : array_like, shape (n_components, n_features).
                 Covariance parameters for each t-Student.

        dfs : array_like, shape (n_components,).
              Degrees of freedom.

        min_covar : float value.
                    Minimum amount that will be added to the covariance
                    matrix in case of trouble, usually 1.e-6.

        Returns
        -------
        retval : array_like, shape (n_samples, n_components).
                 Evaluation of the multivariate probability density
                 function for a t-Student distribution.
        """

        cv = covars.copy()
        if covars.ndim == 1:
            cv = cv[:, np.newaxis]
        if covars.shape[1] == 1:
            cv = np.tile(cv, (1, X.shape[-1]))

        # Sanity check: make sure that the covariance is spherical,
        # i.e. all the elements of the diagonal must be equal
        for k in range(cv.shape[0]):
            assert np.unique(cv[k]).shape[0] == 1
        retval = SMM._multivariate_t_student_density_diag(X, means, cv, dfs, min_covar)

        return retval

    @staticmethod
    def _multivariate_t_student_density_tied(X, means, covars, dfs, min_covar):
        """Multivariate t-Student PDF for a matrix of data points.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features).
            Each row corresponds to a single data point.

        means : array_like, shape (n_components, n_features).
                Mean vectors for n_components t-Students.
                Each row corresponds to a single mean vector.

        covars : array_like, shape (n_features, n_features).
                 Covariance parameters for each t-Student.

        dfs : array_like, shape (n_components,).
              Degrees of freedom.

        min_covar : float value.
                    Minimum amount that will be added to the covariance
                    matrix in case of trouble, usually 1.e-6.

        Returns
        -------
        retval : array_like, shape (n_samples, n_components).
                 Evaluation of the multivariate probability density
                 function for a t-Student distribution.
        """

        # Sanity check: make sure that the shape is (n_features,
        # n_features) and that it matches the shape of the vector of
        # means
        assert len(covars.shape) == 2
        assert covars.shape[0] == covars.shape[1]
        assert means.shape[1] == covars.shape[0]

        cv = np.tile(covars, (means.shape[0], 1, 1))
        retval = SMM._multivariate_t_student_density_full(X, means, cv, dfs, min_covar)

        return retval

    @staticmethod
    def _multivariate_t_student_density_full(X, means, covars, dfs, active, min_covar):
        """Multivariate t-Student PDF for a matrix of data points.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features).
            Each row corresponds to a single data point.

        means : array_like, shape (n_components, n_features).
                Mean vectors for n_components t-Students.
                Each row corresponds to a single mean vector.

        covars : array_like, shape (n_components, n_features,
                 n_features).
                 Covariance parameters for each t-Student.

        dfs : array_like, shape (n_components,).
              Degrees of freedom.

        min_covar : float value.
                    Minimum amount that will be added to the covariance
                    matrix in case of trouble, usually 1.e-6.

        Returns
        -------
        prob : array_like, shape (n_samples, n_components).
               Evaluation of the multivariate probability density
               function for a t-Student distribution.
        """

        n_samples, n_dim = X.shape
        n_components = len(means)
        prob = np.zeros((n_samples, n_components))

        # Sanity check: assert that the received means and covars have
        # the right shape
        assert means.shape[0] == n_components
        assert covars.shape[0] == n_components
        assert dfs.shape[0] == n_components

        # We evaluate all the saples for each component 'c' in the
        # mixture
        for c, (mu, cv, df) in enumerate(zip(means, covars, dfs)):
            if active[c] == 1:
                # Calculate the Cholesky decomposition of the covariance
                # matrix
                cov_chol = SMM._cholesky(cv, min_covar)

                # Calculate the determinant of the covariance matrix
                log_cov_det = np.log(np.diagonal(cov_chol)).sum()
                # cov_det = np.power(np.prod(np.diagonal(cov_chol)), 2)
                # print(cov_det)

                # Calculate the Mahalanobis distance between each vector and
                # the mean
                maha = SMM._mahalanobis_distance_chol(X, mu, cov_chol)

                # Calculate the coefficient of the gamma functions
                r = np.asarray(df, dtype=np.float64)
                log_gamma_coef = scipy.special.gammaln(
                    (r + n_dim) / 2.0
                ) - scipy.special.gammaln(r / 2.0)

                # Calculate the denominator of the multivariate t-Student
                log_denom = log_cov_det  # np.log(np.sqrt(cov_det))
                log_denom += n_dim / 2.0 * np.log(np.pi * df)
                log_denom += (df + n_dim) / 2 * np.log(1 + maha / df)

                # Finally calculate the PDF of the class 'c' for all the X
                # samples
                log_diff = log_gamma_coef - log_denom

                # clip to max and min before taking exp
                log_diff[log_diff > 709.0] = 705.0
                log_diff[log_diff < -709.0] = -709.0

                prob[:, c] = np.exp(log_diff)

        return prob

    @staticmethod
    def _multivariate_t_student_density(
        X, means, covars, dfs, active, cov_type, min_covar
    ):
        """Calculates the PDF of the multivariate t-student for a group
        of samples.

        This method has a dictionary with the different types of
        covariance matrices and calls the appropriate PDF function
        depending on the type of covariance matrix that is passed as
        parameter.
        This method assumes that the covariance matrices are full if the
        parameter cov_type is not specified when calling.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features).
            Each row corresponds to a single data point.

        means : array_like, shape (n_components, n_features).
                Mean vectors for n_components t-Students.
                Each row corresponds to a single mean vector.

        covars : array_like, covariance parameters for each t-Student.
                 The shape depends on `covariance_type`:

                 (n_components, n_features)             if 'spherical',
                 (n_features, n_features)               if 'tied',
                 (n_components, n_features)             if 'diag',
                 (n_components, n_features, n_features) if 'full'

        cov_type : string.
                   Indicates the type of the covariance parameters.
                   It must be one of 'spherical', 'tied', 'diag',
                   'full'.
                   Defaults to 'full'.

        Returns
        -------
        retval : array_like, shape (n_samples, n_components).
                 Array containing the probabilities of each data point
                 in X under each of the n_components multivariate
                 t-Student distributions.
        """

        _multivariate_normal_density_dict = {
            "diag": SMM._multivariate_t_student_density_diag,
            "spherical": SMM._multivariate_t_student_density_spherical,
            "tied": SMM._multivariate_t_student_density_tied,
            "full": SMM._multivariate_t_student_density_full,
        }
        retval = _multivariate_normal_density_dict[cov_type](
            X, means, covars, dfs, active, min_covar
        )

        return retval

    @staticmethod
    def _cholesky(cv, min_covar):
        """Calculates the lower triangular Cholesky decomposition of a
        covariance matrix.

        Parameters
        ----------
        covar : array_like, shape (n_features, n_features).
                Covariance matrix whose Cholesky decomposition wants to
                be calculated.

        min_covar : float value.
                    Minimum amount that will be added to the covariance
                    matrix in case of trouble, usually 1.e-6.

        Returns
        -------
        cov_chol : array_like, shape (n_features, n_features).
                   Lower Cholesky decomposition of a covariance matrix.
        """

        # Sanity check: assert that the covariance matrix is squared
        assert cv.shape[0] == cv.shape[1]

        # Sanity check: assert that the covariance matrix is symmetric
        if (cv.transpose() - cv).sum() > min_covar:
            print(
                "[SMM._cholesky] Error, covariance matrix not "
                + "symmetric: "
                + str(cv)
            )

        n_dim = cv.shape[0]
        try:
            cov_chol = scipy.linalg.cholesky(cv, lower=True)
        except scipy.linalg.LinAlgError:
            # The model is most probably stuck in a component with too
            # few observations, we need to reinitialize this components
            try:
                cov_chol = scipy.linalg.cholesky(
                    cv + min_covar * np.eye(n_dim), lower=True
                )
            except scipy.linalg.LinAlgError:
                cov_chol = np.eye(n_dim)
        #                 scipy.linalg.cholesky(
        #                     cv + np.eye(n_dim), lower=True
        #                 )

        return cov_chol

    @staticmethod
    def _mahalanobis_distance_chol(X, mu, cov_chol):
        """Calculates the Mahalanobis distance between a matrix (set) of
        vectors (X) and another vector (mu).

        The vectors must be organised by row in X, that is, the features
        are the columns.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features).
            Sample in each row.

        mu : array_like (n_features).
             Mean vector of a single distribution (no mixture).

        cov_chol : array_like, shape (n_features, n_features).
                   Cholesky decomposition (L, i.e. lower triangular) of
                   the covariance (normalising) matrix in case that is
                   a full matrix.

        Returns
        -------
        retval : array_like, shape (n_samples,).
                 Array of distances, each row represents the distance
                 from the vector in the same row of X and mu.
        """

        z = scipy.linalg.solve_triangular(cov_chol, (X - mu).T, lower=True)
        retval = np.einsum("ij, ij->j", z, z)

        return retval

    @staticmethod
    def _mahalanobis_distance_mix_diag(X, means, covars, min_covar):
        """Calculates the mahalanobis distance between a matrix of
        points and a mixture of distributions when the covariance
        matrices are diagonal.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features).
            Matrix with a sample in each row.

        means : array_like, shape (n_components, n_features).
                Mean vectors for n_components t-Students.
                Each row corresponds to a single mean vector.

        covars : array_like, shape (n_components, n_features).
                 Covariance parameters for each t-Student.

        Returns
        -------
        result : array_like, shape (n_samples, n_components).
                 Mahalanobis distance from all the samples to all the i
                 component means.
        """

        n_samples, n_dim = X.shape
        n_components = len(means)
        result = np.empty((n_samples, n_components))
        for c, (mu, cv) in enumerate(zip(means, covars)):
            centred_X = X - mu
            inv_cov = np.float64(1) / cv
            result[:, c] = (centred_X * inv_cov * centred_X).sum(axis=1)

        return result

    @staticmethod
    def _mahalanobis_distance_mix_spherical(*args):
        return SMM._mahalanobis_distance_mix_diag(*args)

    @staticmethod
    def _mahalanobis_distance_mix_full(X, means, covars, active, min_covar):
        """Calculates the mahalanobis distance between a matrix of
        points and a mixture of distributions.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features).
            Matrix with a sample vector in each row.

        means : array_like, shape (n_components, n_features).
                Each row corresponds to a single mean vector.

        covars : array_like, shape (n_components, n_features,
                 n_features).
                 Covariance parameters for each t-Student.

        Returns
        -------
        result : array_like, shape (n_samples, n_components).
                 Mahalanobis distance from all the samples to all the
                 component means.
        """

        n_samples, n_dim = X.shape
        n_components = len(means)
        result = np.empty((n_samples, n_components))
        for c, (mu, cv) in enumerate(zip(means, covars)):
            if active[c]:
                cov_chol = SMM._cholesky(cv, min_covar)
                result[:, c] = SMM._mahalanobis_distance_chol(X, mu, cov_chol)

        return result

    @staticmethod
    def _mahalanobis_distance_mix_tied(X, means, covars, min_covar):
        """Calculates the mahalanobis distance between a matrix of
        points and a mixture of distributions.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features).
            Matrix with a sample vector in each row.

        means : array_like, shape (n_components, n_features).
                Each row corresponds to a single mean vector.

        covars : array_like, shape (n_features, n_features).
                 Covariance parameters for each t-Student.

        Returns
        -------
        retval : array_like, shape (n_samples, n_components).
                 Mahalanobis distance from all the samples to all the
                 component means.
        """

        cv = np.tile(covars, (means.shape[0], 1, 1))
        retval = SMM._mahalanobis_distance_mix_full(X, means, cv, min_covar)

        return retval

    @staticmethod
    def _validate_covariances(covars, covariance_type, n_components):
        """Do basic checks on matrix covariance sizes and values."""

        if covariance_type == "full":
            if len(covars.shape) != 3:
                raise ValueError(
                    "'full' covars must have shape (n_components, " + "n_dim, n_dim)"
                )
            elif covars.shape[1] != covars.shape[2]:
                raise ValueError(
                    "'full' covars must have shape (n_components, " + "n_dim, n_dim)"
                )
            for n, cv in enumerate(covars):
                if not np.allclose(cv, cv.T) or np.any(linalg.eigvalsh(cv) <= 0):
                    raise ValueError(
                        "component %d of 'full' covars must be "
                        + "symmetric, positive-definite" % n
                    )
                else:
                    raise ValueError(
                        "covariance_type must be one of "
                        + "'spherical', 'tied', 'diag', 'full'"
                    )
        elif covariance_type == "diag":
            if len(covars.shape) != 2:
                raise ValueError(
                    "'diag' covars must have shape (n_components, " + "n_dim)"
                )
            elif np.any(covars <= 0):
                raise ValueError("'diag' covars must be non-negative")
        elif covariance_type == "spherical":
            if len(covars) != n_components:
                raise ValueError("'spherical' covars have length n_components")
            elif np.any(covars <= 0):
                raise ValueError("'spherical' covars must be non-negative")
        elif covariance_type == "tied":
            if covars.shape[0] != covars.shape[1]:
                raise ValueError("'tied' covars must have shape (n_dim, n_dim)")
            elif not np.allclose(covars, covars.T) or np.any(
                np.linalg.eigvalsh(covars) <= 0
            ):
                raise ValueError(
                    "'tied' covars must be symmetric, " + "positive-definite"
                )

    @staticmethod
    def dist_covar_to_match_cov_type(tied_cv, covariance_type, n_components):
        """Create all the covariance matrices from a given template.

        Parameters
        ----------
        tied_cv : array_like, shape (n_features, n_features).
                  Tied covariance that is going to be converted to other
                  type.

        covariance_type : string.
                          Type of the covariance parameters.
                          Must be one of 'spherical', 'tied', 'diag',
                          'full'.

        n_components : integer value.
                       Number of components in the mixture.

        Returns
        -------
        cv : array_like, shape (depends on the covariance_type
             parameter).
             Tied covariance in the format specified by the user.
        """

        if covariance_type == "spherical":
            cv = np.tile(tied_cv.mean() * np.ones(tied_cv.shape[1]), (n_components, 1))
        elif covariance_type == "tied":
            cv = tied_cv
        elif covariance_type == "diag":
            cv = np.tile(np.diag(tied_cv), (n_components, 1))
        elif covariance_type == "full":
            cv = np.tile(tied_cv, (n_components, 1, 1))
        else:
            raise ValueError(
                "covariance_type must be one of "
                + "'spherical', 'tied', 'diag', 'full'"
            )
        return cv

    @staticmethod
    def multivariate_t_rvs(m, S, df=np.inf, n=1):
        """Generate multivariate random variable sample from a t-Student
        distribution.

        Author
        ------
        Original code by Enzo Michelangeli.
        Modified by Luis C. Garcia-Peraza Herrera.
        This static method is exclusively used by 'tests/smm_test.py'.

        Parameters
        ----------
        m : array_like, shape (n_features,).
            Mean vector, its length determines the dimension of the
            random variable.

        S : array_like, shape (n_features, n_features).
            Covariance matrix.

        df : int or float.
             Degrees of freedom.

        n : int.
            Number of observations.

        Returns
        -------
        rvs : array_like, shape (n, len(m)).
              Each row is an independent draw of a multivariate t
              distributed random variable.
        """

        # Sanity check: dimension of mean and covariance compatible
        assert len(m.shape) == 1
        assert len(S.shape) == 2
        assert m.shape[0] == S.shape[0]
        assert m.shape[0] == S.shape[1]

        # m = np.asarray(m)
        d = m.shape[0]
        # d = len(m)
        if df == np.inf:
            x = 1.0
        else:
            x = np.random.chisquare(df, n) / df
        z = np.random.multivariate_normal(np.zeros(d), S, (n,))
        retval = m + z / np.sqrt(x)[:, None]

        return retval

    @property
    def weights(self):
        """Returns the weights of each component in the mixture."""
        return self.weights_

    @property
    def means(self):
        """Returns the means of each component in the mixture."""
        return self.means_

    @property
    def degrees(self):
        """Returns the degrees of freedom of each component in the
        mixture."""
        return self.degrees_

    @property
    def covariances(self):
        """Covariance parameters for each mixture component.

        Returns
        -------
        The covariance matrices for all the classes.
        The shape depends on the type of covariance matrix:

            (n_classes,  n_features)               if 'diag',
            (n_classes,  n_features, n_features)   if 'full'
            (n_classes,  n_features)               if 'spherical',
            (n_features, n_features)               if 'tied',
        """

        if self.covariance_type == "full":
            return self.covars_
        elif self.covariance_type == "diag":
            return [np.diag(cov) for cov in self.covars_]
        elif self.covariance_type == "tied":
            return [self.covars_] * self.n_components
        elif self.covariance_type == "spherical":
            return [np.diag(cov) for cov in self.covars_]

    # Class constants
    _covar_mstep_funcs = {
        "spherical": _covar_mstep_spherical.__func__,
        "diag": _covar_mstep_diag.__func__,
        "tied": _covar_mstep_tied.__func__,
        "full": _covar_mstep_full.__func__,
    }

    _mahalanobis_funcs = {
        "spherical": _mahalanobis_distance_mix_spherical.__func__,
        "diag": _mahalanobis_distance_mix_diag.__func__,
        "tied": _mahalanobis_distance_mix_tied.__func__,
        "full": _mahalanobis_distance_mix_full.__func__,
    }

    _EPS = np.finfo(np.float64).eps


# to be moved


def pooling(mat, ksize, method="mean", pad=False):
    """Non-overlapping pooling on 2D or 3D data.

    <mat>: ndarray, input array to pool.
    <ksize>: tuple of 2, kernel size in (ky, kx).
    <method>: str, 'max for max-pooling,
                   'mean' for mean-pooling.
    <pad>: bool, pad <mat> or not. If no pad, output has size
           n//f, n being <mat> size, f being kernel size.
           if pad, output has size ceil(n/f).

    Return <result>: pooled matrix.
    """

    m, n = mat.shape[:2]
    ky, kx = ksize

    _ceil = lambda x, y: int(np.ceil(x / float(y)))

    if pad:
        ny = _ceil(m, ky)
        nx = _ceil(n, kx)
        size = (ny * ky, nx * kx) + mat.shape[2:]
        mat_pad = np.full(size, np.nan)
        mat_pad[:m, :n, ...] = mat
    else:
        ny = m // ky
        nx = n // kx
        mat_pad = mat[: ny * ky, : nx * kx, ...]

    new_shape = (ny, ky, nx, kx) + mat.shape[2:]

    if method == "max":
        result = np.nanmax(mat_pad.reshape(new_shape), axis=(1, 3))
    else:
        result = np.nanmean(mat_pad.reshape(new_shape), axis=(1, 3))

    return result


def unpooling(mat, ksize):
    return np.kron(mat, np.ones(ksize))
