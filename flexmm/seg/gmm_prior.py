"""
Gaussian Mixture Models.

This class has reused code and comments from sklearn.mixture.gmm.

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
import scipy.linalg
import scipy.special
import scipy.optimize
import scipy.signal
from scipy.ndimage import gaussian_filter
from flexmm import toolbox as tb

import warnings

class GMM(sklearn.base.BaseEstimator):
    """Gaussian Mixture Model.

    Representation of a Gaussian mixture model probability
    distribution. This class allows for easy evaluation of, sampling
    from, and maximum-likelihood estimation of the parameters of an
    GMM distribution.

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

    def __init__(self, n_components=1, covariance_type='full',
                 prior_weights=None, prior_means=None, prior_var=None,
                 prior_init=False, im_shape=(256,256), neigh_size=3,
                 random_state=None, tol=1e-6, min_covar=1e-6,
                 n_iter=1000, n_init=1, light=True, params='wqmc',
                 init_params='wqmc', ppca=False, n_pca=10):

        # Store the parameters as class attributes
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.prior_weights = prior_weights
        if self.prior_weights!=None:
            self.prior_means = prior_means
            self.prior_var = prior_var
            self.prior_norm = 1.0
            self.prior_init = prior_init
            self.neigh_size = neigh_size
            Y, X = np.mgrid[-(neigh_size-1)//2:(neigh_size-1)//2+1,
                            -(neigh_size-1)//2:(neigh_size-1)//2+1]
            self.neighbors = tb.gauss2d(X,Y,neigh_size/4.)
            self.neighbors /= self.neighbors.sum()
            self.neighbors = self.neighbors[...,np.newaxis]
            self.im_shape = im_shape

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
        self.log_lkls = np.array([])
        self.log_priors = np.array([])

    def _expectation_step(self, X):
        """Performs the expectation step of the EM algorithm.

        This method uses the means, class-related weights,
        covariances and degrees of freedom stored in the attributes
        of this class:
        self.means_, self.weights_, self.covars_.

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
        sklearn.utils.validation.check_is_fitted(self, 'means_')
        X = sklearn.utils.validation.check_array(X, dtype=np.float64)
        if X.ndim == 1:
            X = X[:, np.newaxis]
        if X.size == 0:
            return np.array([]), np.empty((0, self.n_components))
        if X.shape[1] != self.means_.shape[1]:
            raise ValueError(
                '[GMM._expectation_step] Error, the ' \
                + 'shape of X is not compatible with self.'
            )

        # Initialisation of reponsibilities
        n_samples, n_dim = X.shape
        responsibilities = np.ndarray(
            shape=(X.shape[0], self.n_components),
            dtype=np.float64)

        # Calculate the probability of each point belonging to each
        # Gaussian distribution of the mixture
        pr_before_weighting = self._multivariate_gaussian_density(
            X, self.means_, self.covars_, self.covariance_type,
            self.q_, self.min_covar)

        pr = pr_before_weighting * self.weights_
        # Calculate the likelihood of each point
        likelihoods = pr.sum(axis=1) #+ 1e-8

        # Update responsibilities
        responsibilities = pr / pr.sum(1, keepdims=True)

        if self.ppca:
            for k in range(self.n_components):
                if self.q_[k]:
                    self.B_[k] = self.taus[k]*self.pcs[k].T@self.pcs[k]\
                                + np.eye(self.n_pca)
                    try:
                        B_inv = scipy.linalg.pinvh(self.B_[k]\
                                            +self.min_covar*np.eye(self.n_pca))
                    except:
                        B_inv = np.eye(self.n_pca)

                    self.Y[k] = (self.taus[k]*B_inv\
                            @(self.pcs[k].T@(X-self.means_[k]).T)).T
                    self.S[k] = responsibilities[:,k].sum()*B_inv+\
                                   np.dot(responsibilities[:,k]*self.Y[k].T,\
                                                             self.Y[k])


        return likelihoods, responsibilities

    def _maximisation_step(self, X, responsibilities):
        """Perform the maximisation step of the EM algorithm.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features).
            Each row corresponds to a single data point.

        responsibilities : array_like, shape (n_samples, n_components).
        """

        n_samples, n_dim = X.shape
        z_sum = responsibilities.sum(axis=0)
        zu = np.copy(responsibilities)
        zu_sum = zu.sum(axis=0)

        # Update weights
        if 'w' in self.params:
            if self.prior_weights==None:
                self.weights_ = z_sum / n_samples
            elif self.prior_weights=='ext3':
                # outside this algo
                self.weights_ = self.prior_means/self.prior_norm
            elif self.prior_weights=='ext2':
                # prior_means, prior_var and prior_norm
                # are set outside this alg
                # and have the following form
                # prior_means = s1*m2+s2*m1
                # prior_var = s1*s2
                # prior_norm = s1+s2+s1*s2
                means = scipy.ndimage.convolve(
                            responsibilities.reshape((self.im_shape[0],
                            self.im_shape[1],self.n_components)),
                        self.neighbors, mode='nearest').reshape(n_samples, self.n_components)
                var = scipy.ndimage.convolve(
                            (responsibilities**2).reshape((self.im_shape[0],
                            self.im_shape[1],self.n_components)),
                        self.neighbors, mode='nearest').reshape(n_samples, self.n_components)

                var -= means**2
                var = var.mean()
                # transfer to prior weights
                self.weights_ = var*self.prior_means+self.prior_var*means
                self.weights_ /= var*self.prior_norm+self.prior_var
            elif self.prior_weights=='ext':
                # prior_means and prior_var are set outside this alg
                # transfer to prior weights
                self.weights_ = self.prior_means+self.prior_var*responsibilities
                self.weights_ /= 1+self.prior_var
            elif self.prior_weights=='loc':
                # prior mean
                self.prior_means = scipy.ndimage.convolve(
                    responsibilities.reshape((self.im_shape[0],
                            self.im_shape[1],self.n_components)),
                    self.neighbors, mode='nearest').reshape(n_samples, self.n_components)
                # prior var
                #self.prior_var = scipy.ndimage.convolve(
                #    (responsibilities**2).reshape((self.im_shape[0],
                #            self.im_shape[1],self.n_components)),
                #    self.neighbors, mode='nearest').reshape(n_samples, self.n_components)

                #self.prior_var -= self.prior_means**2
                #self.prior_var = self.prior_var.mean()#axis=1, keepdims=True)

                # transfer to prior weights
                self.weights_ = self.prior_means#+self.prior_var*responsibilities
                #self.weights_ /= 1+self.prior_var

        if 'q' in self.params:
            if self.prior_weights in {None,'loc'}:
                #self.q_weights_ = z_sum/(z_sum+n_samples/self.q_.sum())
                self.q_weights_ = z_sum/(z_sum+n_samples/self.n_components)
                self.q_ = self.q_weights_>0.5
                self.weights_ *= self.q_
            elif self.prior_weights=='ext3':
                # set q_weights outside
                self.q_weights_ = self.q_weights_
                self.q_ = self.q_weights_>0.5
                self.weights_ *= self.q_

        else:
            self.q_weights_ = np.ones(self.n_components)
            self.q_ = np.ones(self.n_components, dtype=bool)

        # Update means
        if 'm' in self.params:
            if self.ppca:
                for k in range(self.n_components):
                    if self.q_[k]:
                        #print(self.pcs[k].shape,self.Y[k].shape)
                        self.means_[k] = np.sum(zu[:,k][:,np.newaxis]*(X-
                                                (self.pcs[k]@self.Y[k].T).T),
                                                axis=0)
                        self.means_[k] /= (zu_sum[k] + 10 * GMM._EPS)
            else:
                dot_zu_x = np.dot(zu.T, X)
                zu_sum_ndarray = zu_sum.reshape(zu_sum.shape[0], 1)
                self.means_ = dot_zu_x / (zu_sum_ndarray + 10 * GMM._EPS)

        # Update covariances
        if 'c' in self.params:
            if self.ppca:
                for k in range(self.n_components):
                    if self.q_[k]:
                        try:
                            S_inv = scipy.linalg.pinvh(self.S[k]\
                                            +self.min_covar*np.eye(self.n_pca))
                        except:
                            S_inv = np.eye(self.n_pca)

                        post = zu[:,k]
                        mu = self.means[k]
                        diff = X - mu

                        with np.errstate(under='ignore'):
                            # Underflow Errors in doing post * X.T
                            # are not important
                            pcs_ = np.dot(post * diff.T, self.Y[k])@S_inv
                            #print(post.shape,diff.shape, pcs_.shape,self.Y[k].shape)
                            tau_inv=np.einsum('ij,ji->i',\
                                        post[:,np.newaxis] * diff,\
                                        (diff-2*(pcs_@self.Y[k].T).T).T).sum()
                            tau_inv+=np.trace(pcs_@self.S[k]@pcs_.T)
                            #print(tau_inv.shape)
                            tau_inv/=n_samples*n_dim*self.weights_.T[k].mean()
                            self.taus[k] = 1/(tau_inv + 10 * GMM._EPS)

                        self.pcs[k] = pcs_
                        self.covars_[k] = pcs_@pcs_.T\
                                          + np.eye(n_dim)/self.taus[k]
                #print(self.taus)

            else:
                covar_mstep_func = GMM._covar_mstep_funcs[
                    self.covariance_type
                ]
                self.covars_ = covar_mstep_func(
                    X, zu, z_sum, self.means_, self.q_, self.min_covar
                )

    def _initialization_step(self, X, gt=None, n_components_best=None, use_kmeans=False):
        """Performs the initialization step of the EM algorithm.

        This method initializes the means, class-related weights
        and covariances stored in the attributes
        of this class:
        self.means_, self.weights_ and self.covars_

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features).
        """

        if 'm' in self.init_params or not hasattr(self, 'means_'):
            if use_kmeans:
                kmeans = sklearn.cluster.KMeans(
                        n_clusters=self.n_components,
                        init='k-means++',
                        random_state=self.random_state
                    )
                self.means_ = kmeans.fit(X).cluster_centers_
            elif gt is not None:
                cluster_centers, n_components = tb.gt_pca_cluster_centers(X,gt)
                print("HERE")
                self.means_ = cluster_centers
                if n_components_best is not None:
                    self.n_components = n_components_best
            else:
                self.means_ = np.zeros((self.n_components, X.shape[1]))

        if 'w' in self.init_params or not hasattr(self, 'weights_'):
            if self.prior_weights==None:
                self.weights_ = np.tile(
                        1.0 / self.n_components, self.n_components
                        )
            elif self.prior_weights in {'ext','ext2','ext3'}:
                self.weights_ = self.prior_means
                self.weights_ /= self.weights_.sum(axis=1, keepdims=True)
            elif self.prior_weights=='loc':
                self.weights_ = self.prior_means
                self.weights_ /= self.weights_.sum(axis=1, keepdims=True)

        if 'q' in self.init_params or not hasattr(self, 'q_weights_'):
                self.q_weights_ = np.ones(self.n_components)
                self.q_ = self.q_weights_>0.5

        if 'c' in self.init_params or not hasattr(self, 'covars_'):
            cv = np.cov(X.T) + self.min_covar * np.eye(X.shape[1])
            if not cv.shape:
                cv.shape = (1, 1)
            self.covars_ = GMM.dist_covar_to_match_cov_type(
                    cv, self.covariance_type, self.n_components
                )

        if self.ppca:
            pca = sklearn.decomposition.PCA(n_components=self.n_pca)
            #np.random.randn(X.shape[0],X.shape[1])
            pcs_ = pca.fit(X).components_.T
            self.pcs = np.tile(pcs_,(self.n_components, 1, 1))
            self.taus = np.tile(1/(pca.noise_variance_+1e-8), (self.n_components,))
            self.B_ = np.tile(self.taus[0]*pcs_.T@pcs_ + np.eye(self.n_pca),
                              (self.n_components, 1, 1))
            self.Y = np.zeros((self.n_components, X.shape[0], self.n_pca))
            self.S = np.zeros((self.n_components, self.n_pca, self.n_pca))
            for k in range(self.n_components):
                B_inv = scipy.linalg.pinvh(self.B_[k]\
                                           +self.min_covar*np.eye(self.n_pca))
                self.Y[k] = (self.taus[k]*B_inv\
                            @(self.pcs[k].T@(X-self.means_[k]).T)).T
                self.S[k] = X.shape[0]*(B_inv+ np.dot(self.Y[k].T,self.Y[k]))\
                                        /self.n_components


        if self.prior_weights in {'ext','loc'} and self.prior_init:
                self._maximisation_step(X, self.prior_means)

    def fit(self, X, y=None):
        """Estimate model parameters with the EM algorithm.

        A initialization step is performed before entering the em
        algorithm. If you want to avoid this step, set the keyword
        argument init_params to the empty string '' when creating
        the GMM object. Likewise, if you would like just to do an
        initialization, set n_iter=0.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features).

        y : not used, just for compatibility with sklearn API.
        """

        # Sanity check: assert that the input matrix is not 1D
        if (len(X.shape) == 1):
            raise ValueError(
                '[GMM.fit] Error, the input matrix must have a ' \
                + 'shape of (n_samples, n_features).'
            )

        # Sanity checks:
        #    - Convert input to 2d array, raise error on sparse
        #      matrices. Calls assert_all_finite by default.
        #    - No. of samples is higher or equal to the no. of
        #      components in the mixture.
        X = sklearn.utils.validation.check_array(X, dtype=np.float64)
        if X.shape[0] < self.n_components:
            raise ValueError(
                '[GMM.fit] Error, GMM estimation with ' \
                + '%s components, but got only %s samples' % (
                    self.n_components, X.shape[0]
                )
            )

        # For all the initialisations we get the one with the best
        # parameters
        n_samples, n_dim = X.shape
        max_prob = -np.infty
        for _ in range(self.n_init):

            # EM initialisation
            self._initialization_step(X, use_kmeans=True)


            best_params = {
                'weights': self.weights_,
                'means': self.means_,
                'covars': self.covars_,
            }

            self.converged_ = False
            current_log_likelihood = None

            # EM algorithm
            for i in range(self.n_iter):
                prev_log_likelihood = current_log_likelihood
                # Expectation step
                likelihoods, responsibilities = \
                    self._expectation_step(X)
                #self.log_lkls = np.append(self.log_lkls, np.log(likelihoods).sum())

                idx_r = np.argmax(responsibilities, axis=-1)
                r = np.zeros(self.prior_means.shape)
                r[np.arange(r.shape[0]), idx_r] = 1.0
                print(responsibilities.shape, r.shape)
                self.log_lkls = np.append(self.log_lkls, np.nansum(r*np.log(responsibilities)))

                w = scipy.ndimage.convolve(r.reshape((self.im_shape[0],
                                        self.im_shape[1],self.n_components)),
                                        self.neighbors, mode='nearest')\
                                        .reshape(n_samples, self.n_components)

                log_priors = (w - r + 1)*np.log(self.weights)
                self.log_priors = np.append(self.log_priors, log_priors.sum())

                # Sanity check: assert that the likelihoods,
                # responsibilities and gammaweights have the correct
                # dimensions
                assert(len(likelihoods.shape) == 1)
                assert(likelihoods.shape[0] == n_samples)
                assert(len(responsibilities.shape) == 2)
                assert(responsibilities.shape[0] == n_samples)
                assert(responsibilities.shape[1] == self.n_components)

                # Calculate loss function
                current_log_likelihood = np.log(likelihoods).mean()

                # Check for convergence
                if prev_log_likelihood is not None:
                    change = np.abs(current_log_likelihood -
                        prev_log_likelihood
                    )
                    if change < self.tol:
                        self.converged_ = True
                        break

                # Maximisation step
                self._maximisation_step(X, responsibilities)

            # If the results are better, keep it
            #if self.n_iter and self.converged_:
            if current_log_likelihood > max_prob:
                max_prob = current_log_likelihood
                best_params = {
                    'weights': self.weights_,
                    'means': self.means_,
                    'covars': self.covars_,
                }

        # Check the existence of an init param that was not subject to
        # likelihood computation issue
        if np.isneginf(max_prob) and self.n_iter:
            msg = 'EM algorithm was never able to compute a valid ' \
                + 'likelihood given initial parameters. Try '       \
                + 'different init parameters (or increasing '       \
                + 'n_init) or check for degenerate data.'
            warnings.warn(msg, RuntimeWarning)

        # Choosing the best result of all the iterations as the actual
          # result
        if self.n_iter:
            self.weights_ = best_params['weights']
            self.means_ = best_params['means']
            self.covars_ = best_params['covars']

        if self.light:
            del(self.Y,self.B_,self.S)
            self.weights_ = np.float16(self.weights_)
            self.means_ = np.float32(self.means_)
            self.covars_ = np.float32(self.covars_)
            self.degrees_ = np.float32(self.degrees_)
            self.pcs = np.float32(self.pcs)
            self.taus = np.float32(self.taus)

        return self

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

        _, responsibilities = self._expectation_step(X)
        r_argmax = responsibilities.argmax(axis=1)

        return r_argmax

    def predict_proba(self, X):
        """Predict label for data.

        This function will tell the probability of each component
        generating each sample.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features).

        Returns
        -------
        responsibilities : array_like, shape (n_samples, n_components).
        """

        _, responsibilities = self._expectation_step(X)

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

    def _posterior_proba(self, X):
        """Compute the log probability under the model.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features).

        Returns
        -------
        prob : array_like, shape (n_samples,).
               Probabilities of each data point in X.
        """

        _, responsibilities = self.score_samples(X)

        return responsibilities


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

        sklearn.utils.validation.check_is_fitted(self, 'means_')
        X = sklearn.utils.validation.check_array(X, dtype=np.float64)

        if X.ndim == 1:
            X = X[:, np.newaxis]
        if X.size == 0:
            return np.array([]), np.empty((0, self.n_components))
        if X.shape[1] != self.means_.shape[1]:
            raise ValueError(
                '[score_samples] ValueError, the shape of X is not ' \
                + 'compatible with self.'
            )

        # Initialisation of reponsibilities
        n_samples, n_dim = X.shape
        responsibilities = np.ndarray(
            shape=(X.shape[0], self.n_components), dtype=np.float64
        )

        # Calculate the probability of each point belonging to each
        # Gaussian distribution of the mixture
        pr = self._multivariate_gaussian_density(
            X, self.means_, self.covars_, self.covariance_type,
            self.q_, self.min_covar) * self.weights_

        # Calculate the likelihood of each point
        likelihoods = pr.sum(axis=1)

        # Update responsibilities
        like_ndarray = likelihoods.reshape(likelihoods.shape[0], 1)
        responsibilities = pr / (like_ndarray + 10 * GMM._EPS)

        return likelihoods, responsibilities

    def _n_parameters(self):
        """Returns the number of free parameters in the model."""

        _, n_features = self.means_.shape

        if self.covariance_type == 'full':
            cov_params = self.n_components \
            * n_features                   \
            * (n_features + 1) / 2.0
        elif self.covariance_type == 'diag':
            cov_params = self.n_components * n_features
        elif self.covariance_type == 'tied':
            cov_params = n_features * (n_features + 1) / 2.0
        elif self.covariance_type == 'spherical':
            cov_params = self.n_components

        mean_params = n_features * self.n_components
        total_param = int(
            cov_params           \
            + mean_params        \
            + self.n_components  \
            - 1
        )

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

        retval = - 2 * self.score(X).sum() \
              + self._n_parameters() * np.log(X.shape[0])

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

        retval = - 2 * self.score(X).sum() + 2 * self._n_parameters()

        return retval

    @staticmethod
    def _covar_mstep_diag(X, zu, z_sum, means, min_covar):
        """Performing the covariance m-step for diagonal
        covariances.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features).
            List of k_features-dimensional data points.
            Each row corresponds to a single data point.

        zu : array, shape (n_samples, n_components).
             Contains responsibilities.

        z_sum : array_like, shape (n_components,).
                Sum of all the responsibilities for each mixture.

        means : array_like, shape (n_components, n_features).
                List of n_features-dimensional mean vectors for
                n_components Gaussian. Each row corresponds to a
                single mean vector.

        min_covar : float value.
                    Minimum amount that will be added to the
                    covariance matrix in case of trouble, usually
                    1.e-6.

        Returns
        -------
        retval : array_like, shape (n_components, n_features).
        """

        norm = 1.0 / (z_sum[:, np.newaxis]) #+ 10 * GMM._EPS)
        diff2 = (X.T[np.newaxis,...] - means[...,np.newaxis])**2
        retval = np.sum(zu[:, np.newaxis,:].T*diff2, axis=2)*norm

        return retval

    @staticmethod
    def _covar_mstep_spherical(*args):
        cv = GMM._covar_mstep_diag(*args)
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
             Contains responsibilities.

        z_sum : array_like, shape (n_components,)
                Sum of all the responsibilities for each mixture.

        means : array_like, shape (n_components, n_features).
                List of n_features-dimensional mean vectors for
                n_components Gaussian.
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
        assert(zu.shape[0] == n_samples)
        assert(zu.shape[1] == n_components)
        assert(z_sum.shape[0] == n_components)

        # Eq. 31 from D. Peel and G. J. McLachlan, "Robust mixture
        # modelling using the t distribution"
        cv = np.empty((n_components, n_features, n_features))
        zu_sum = zu.sum(axis=0)
        for c in range(n_components):
            if active[c]:
                post = zu[:, c]
                mu = means[c]
                diff = X - mu
                with np.errstate(under='ignore'):
                    # Underflow Errors in doing post * X.T are not important
                    if n_components == 1:
                        avg_cv = np.dot(post * diff.T, diff) \
                            / (zu_sum[c] + 10 * GMM._EPS)
                    else:
                        avg_cv = np.dot(post * diff.T, diff) \
                            / (z_sum[c] + 10 * GMM._EPS)

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
             Contains the responsibilities.

        z_sum : array_like, shape (n_components, )
                Sum of all the responsibilities for each mixture.

        means : array_like, shape (n_components, n_features).
                List of n_features-dimensional mean vectors for
                n_components Gaussian. Each row corresponds to a
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
        out.flat[::len(out) + 1] += min_covar

        return out

    @staticmethod
    def _multivariate_gaussian_density_diag(X, means, covars, min_covar):
        """Multivariate Gaussian PDF for a matrix of data points and
        diagonal covariance matrices.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features).
            Each row corresponds to a single data point.

        means : array_like, shape (n_components, n_features).
                Each row corresponds to a single mean vector.

        covars : array_like, shape (n_components, n_features).
                 List of n_components covariance parameters for each
                 Gaussian.

        min_covar : float value.
                    Minimum amount that will be added to the covariance
                    matrix in case of trouble, usually 1.e-6.

        Returns
        -------
        retval : array_like, shape (n_samples, n_components).
                 Evaluation of the multivariate probability density
                 function for a Gaussian distribution.
        """

        # Sanity check: make sure that the shape of means and
        # covariances is as expected for diagonal matrices
        n_samples, n_dim = X.shape
        assert(covars.shape[0] == means.shape[0])
        assert(covars.shape[1] == means.shape[1])
        assert(covars.shape[1] == n_dim)

        # Calculate inverse and determinant of the covariances
        log_det_covars = np.sum(np.log(covars), axis=1)

        # Calculate squared Mahalanobis distance from all the points
        # to the mean of each component in the mix
        maha = GMM._mahalanobis_distance_mix_diag(X, means, covars,
                                                  min_covar)

        # Calculate the log value of the numerator
        log_num = -0.5*maha

        # Calculate the value of the denominator
        log_denom = 0.5*log_det_covars + 0.5*n_dim*np.log(2*np.pi)

        retval = np.exp(log_num - log_denom)

        return retval

    @staticmethod
    def _multivariate_gaussian_density_spherical(X, means, covars, min_covar):
        """Multivariate Gaussian PDF for a matrix of data points.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features).
            Each row corresponds to a single data point.

        means : array_like, shape (n_components, n_features).
                Mean vectors for n_components Gaussian.
                Each row corresponds to a single mean vector.

        covars : array_like, shape (n_components, n_features).
                 Covariance parameters for each Gaussian.

        min_covar : float value.
                    Minimum amount that will be added to the covariance
                    matrix in case of trouble, usually 1.e-6.

        Returns
        -------
        retval : array_like, shape (n_samples, n_components).
                 Evaluation of the multivariate probability density
                 function for a Gaussian distribution.
        """

        cv = covars.copy()
        if covars.ndim == 1:
            cv = cv[:, np.newaxis]
        if covars.shape[1] == 1:
            cv = np.tile(cv, (1, X.shape[-1]))

        # Sanity check: make sure that the covariance is spherical,
        # i.e. all the elements of the diagonal must be equal
        for k in range(cv.shape[0]):
            assert(np.unique(cv[k]).shape[0] == 1)
        retval = GMM._multivariate_gaussian_density_diag(
            X, means, cv, min_covar
        )

        return retval

    @staticmethod
    def _multivariate_gaussian_density_tied(X, means, covars, min_covar):
        """Multivariate Gaussian PDF for a matrix of data points.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features).
            Each row corresponds to a single data point.

        means : array_like, shape (n_components, n_features).
                Mean vectors for n_components Gaussian.
                Each row corresponds to a single mean vector.

        covars : array_like, shape (n_features, n_features).
                 Covariance parameters for each Gaussian.

        min_covar : float value.
                    Minimum amount that will be added to the covariance
                    matrix in case of trouble, usually 1.e-6.

        Returns
        -------
        retval : array_like, shape (n_samples, n_components).
                 Evaluation of the multivariate probability density
                 function for a Gaussian distribution.
        """

        # Sanity check: make sure that the shape is (n_features,
        # n_features) and that it matches the shape of the vector of
        # means
        assert(len(covars.shape) == 2)
        assert(covars.shape[0] == covars.shape[1])
        assert(means.shape[1] == covars.shape[0])

        cv = np.tile(covars, (means.shape[0], 1, 1))
        retval = GMM._multivariate_gaussian_density_full(
            X, means, cv, min_covar
        )

        return retval

    @staticmethod
    def _multivariate_gaussian_density_full(X, means, covars, active, min_covar):
        """Multivariate Gaussian PDF for a matrix of data points.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features).
            Each row corresponds to a single data point.

        means : array_like, shape (n_components, n_features).
                Mean vectors for n_components Gaussian.
                Each row corresponds to a single mean vector.

        covars : array_like, shape (n_components, n_features,
                 n_features).
                 Covariance parameters for each Gaussian.

        min_covar : float value.
                    Minimum amount that will be added to the covariance
                    matrix in case of trouble, usually 1.e-6.

        Returns
        -------
        prob : array_like, shape (n_samples, n_components).
               Evaluation of the multivariate probability density
               function for a Gaussian distribution.
        """

        n_samples, n_dim = X.shape
        n_components = len(means)
        prob = np.empty((n_samples, n_components))

        # Sanity check: assert that the received means and covars have
        # the right shape
        assert(means.shape[0] == n_components)
        assert(covars.shape[0] == n_components)

        # We evaluate all the samples for each component 'c' in the
        # mixture
        for c, (mu, cv) in enumerate(zip(means, covars)):
            if active[c]:
                # Calculate the Cholesky decomposition of the covariance
                # matrix
                cov_chol = GMM._cholesky(cv, min_covar)

                # Calculate the log determinant of the covariance matrix
                log_cov_det = np.log(np.diagonal(cov_chol)).sum()

                # Calculate the squared Mahalanobis distance between
                # each vector and the mean
                try:
                    maha = GMM._mahalanobis_distance_chol(X, mu, cov_chol)
                except:
                    maha = GMM._mahalanobis_distance_chol(X,
                                                    np.zeros((1,X.shape[1])),
                                                    np.eye(X.shape[1]))
                # Calculate the log numerator
                log_num = -0.5*maha

                # Calculate the denominator of the multivariate Gaussian
                log_denom = 0.5*log_cov_det + 0.5*n_dim*np.log(2*np.pi)

                # Finally calculate the PDF of the class 'c' for all the X
                # samples
                log_diff = log_num - log_denom

                # clip to max and min before taking exp
                log_diff[log_diff>709.0] = 705.0
                log_diff[log_diff<-709.0] = -709.0

                prob[:, c] = np.exp(log_diff)

        return prob

    @staticmethod
    def _multivariate_gaussian_density(X, means, covars, cov_type,
                                       active, min_covar):
        """Calculates the PDF of the multivariate Gaussian for a group
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
                Mean vectors for n_components Gaussian.
                Each row corresponds to a single mean vector.

        covars : array_like, covariance parameters for each Gaussian.
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
                 Gaussian distributions.
        """

        _multivariate_normal_density_dict = {
            'diag': GMM._multivariate_gaussian_density_diag,
            'spherical': GMM._multivariate_gaussian_density_spherical,
            'tied': GMM._multivariate_gaussian_density_tied,
            'full': GMM._multivariate_gaussian_density_full
        }
        retval = _multivariate_normal_density_dict[cov_type](
            X, means, covars, active, min_covar
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
        assert(cv.shape[0] == cv.shape[1])

        # Sanity check: assert that the covariance matrix is symmetric
        if (cv.transpose() - cv).sum() > min_covar:
            print('[GMM._cholesky] Error, covariance matrix not ' \
                + 'symmetric: '
                + str(cv)
            )

        n_dim = cv.shape[0]
        try:
            cov_chol = scipy.linalg.cholesky(cv, lower=True)
        except:
            # The model is most probably stuck in a component with too
            # few observations, we need to reinitialize this components
            cov_chol = np.eye(n_dim)

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
        try:
            z = scipy.linalg.solve_triangular(
                cov_chol, (X - mu).T, lower=True
            )
        except:
            z = scipy.linalg.solve_triangular(
                np.eye(X.shape[1]), (X - np.zeros((1,X.shape[1]))).T,
                lower=True
            )

        retval = np.einsum('ij, ij->j', z, z)

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
                Mean vectors for n_components Gaussian.
                Each row corresponds to a single mean vector.

        covars : array_like, shape (n_components, n_features).
                 Covariance parameters for each Gaussian.

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
        return GMM._mahalanobis_distance_mix_diag(*args)

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
                 Covariance parameters for each Gaussian.

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
                cov_chol = GMM._cholesky(cv, min_covar)
                result[:, c] = GMM._mahalanobis_distance_chol(
                    X, mu, cov_chol)

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
                 Covariance parameters for each Gaussian.

        Returns
        -------
        retval : array_like, shape (n_samples, n_components).
                 Mahalanobis distance from all the samples to all the
                 component means.
        """

        cv = np.tile(covars, (means.shape[0], 1, 1))
        retval = GMM._mahalanobis_distance_mix_full(
            X, means, cv, min_covar
        )

        return retval

    @staticmethod
    def _validate_covariances(covars, covariance_type, n_components):
        """Do basic checks on matrix covariance sizes and values."""

        if covariance_type == 'full':
            if len(covars.shape) != 3:
                raise ValueError(
                    "'full' covars must have shape (n_components, " \
                    + "n_dim, n_dim)"
                )
            elif covars.shape[1] != covars.shape[2]:
                raise ValueError(
                    "'full' covars must have shape (n_components, " \
                    + "n_dim, n_dim)"
                )
            for n, cv in enumerate(covars):
                if (not np.allclose(cv, cv.T)
                        or np.any(linalg.eigvalsh(cv) <= 0)):
                    raise ValueError(
                        "component %d of 'full' covars must be "    \
                        + "symmetric, positive-definite" % n
                    )
                else:
                    raise ValueError(
                        "covariance_type must be one of " \
                        + "'spherical', 'tied', 'diag', 'full'"
                    )
        elif covariance_type == 'diag':
            if len(covars.shape) != 2:
                raise ValueError(
                    "'diag' covars must have shape (n_components, " \
                    + "n_dim)")
            elif np.any(covars <= 0):
                raise ValueError("'diag' covars must be non-negative")
        elif covariance_type == 'spherical':
            if len(covars) != n_components:
                raise ValueError(
                    "'spherical' covars have length n_components"
                )
            elif np.any(covars <= 0):
                raise ValueError(
                    "'spherical' covars must be non-negative"
                )
        elif covariance_type == 'tied':
            if covars.shape[0] != covars.shape[1]:
                raise ValueError(
                    "'tied' covars must have shape (n_dim, n_dim)")
            elif (not np.allclose(covars, covars.T)
                    or np.any(np.linalg.eigvalsh(covars) <= 0)):
                raise ValueError(
                    "'tied' covars must be symmetric, " \
                    + "positive-definite"
                )

    @staticmethod
    def dist_covar_to_match_cov_type(tied_cv, covariance_type,
             n_components):
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

        if covariance_type == 'spherical':
            cv = np.tile(
                tied_cv.mean() * np.ones(tied_cv.shape[1]),
                (n_components, 1)
            )
        elif covariance_type == 'tied':
            cv = tied_cv
        elif covariance_type == 'diag':
            cv = np.tile(np.diag(tied_cv), (n_components, 1))
        elif covariance_type == 'full':
            cv = np.tile(tied_cv, (n_components, 1, 1))
        else:
            raise ValueError(
                "covariance_type must be one of "
                + "'spherical', 'tied', 'diag', 'full'"
            )
        return cv

    @staticmethod
    def multivariate_t_rvs(m, S, df=np.inf, n=1):
        """Generate multivariate random variable sample from a Gaussian
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
        assert(len(m.shape) == 1)
        assert(len(S.shape) == 2)
        assert(m.shape[0] == S.shape[0])
        assert(m.shape[0] == S.shape[1])

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

        if self.covariance_type == 'full':
            return self.covars_
        elif self.covariance_type == 'diag':
            return [np.diag(cov) for cov in self.covars_]
        elif self.covariance_type == 'tied':
            return [self.covars_] * self.n_components
        elif self.covariance_type == 'spherical':
            return [np.diag(cov) for cov in self.covars_]

    # Class constants
    _covar_mstep_funcs = {
        'spherical': _covar_mstep_spherical.__func__,
        'diag': _covar_mstep_diag.__func__,
        'tied': _covar_mstep_tied.__func__,
        'full': _covar_mstep_full.__func__,
    }

    _mahalanobis_funcs = {
        'spherical': _mahalanobis_distance_mix_spherical.__func__,
        'diag': _mahalanobis_distance_mix_diag.__func__,
        'tied': _mahalanobis_distance_mix_tied.__func__,
        'full': _mahalanobis_distance_mix_full.__func__,
    }

    _EPS = np.finfo(np.float64).eps



def pooling(mat,ksize,method='mean',pad=False):
    '''Non-overlapping pooling on 2D or 3D data.

    <mat>: ndarray, input array to pool.
    <ksize>: tuple of 2, kernel size in (ky, kx).
    <method>: str, 'max for max-pooling,
                   'mean' for mean-pooling.
    <pad>: bool, pad <mat> or not. If no pad, output has size
           n//f, n being <mat> size, f being kernel size.
           if pad, output has size ceil(n/f).

    Return <result>: pooled matrix.
    '''

    m, n = mat.shape[:2]
    ky,kx=ksize

    _ceil=lambda x,y: int(numpy.ceil(x/float(y)))

    if pad:
        ny=_ceil(m,ky)
        nx=_ceil(n,kx)
        size=(ny*ky, nx*kx)+mat.shape[2:]
        mat_pad=np.full(size,numpy.nan)
        mat_pad[:m,:n,...]=mat
    else:
        ny=m//ky
        nx=n//kx
        mat_pad=mat[:ny*ky, :nx*kx, ...]

    new_shape=(ny,ky,nx,kx)+mat.shape[2:]

    if method=='max':
        result=np.nanmax(mat_pad.reshape(new_shape),axis=(1,3))
    else:
        result=np.nanmean(mat_pad.reshape(new_shape),axis=(1,3))

    return result


def unpooling(mat,ksize):
    return np.kron(mat, np.ones(ksize))
