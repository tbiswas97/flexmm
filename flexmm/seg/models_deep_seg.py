"""Multilayer segmentation with mixture models

License GNU GPLv3 : https://www.gnu.org/licenses/gpl-3.0.en.html
May 2019, Jonathan Vacher (jonathanvacher.github.io)

Helper: get deep features
Model ref: indeprendent layers / no smoothing
Model a: independent layers
Model b: single prior probability map
Model c: smoothed prior probability maps
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
import copy

import sys

from flexmm.seg.smm_prior import *
from flexmm.seg.gmm_prior import *

"""
Helper: get features from each layer and send them back as numpy arrays
"""


def _reshape_deep_layers(result, layer, N_list, d_list):
    if result.ndim == 1:
        result = result[np.newaxis, ...]

    Ny, Nx = N_list[0]
    ny, nx = N_list[layer]
    d = d_list[layer]
    multiplier = Ny // ny
    m = multiplier

    if result.ndim == 2:
        k = result.shape[-1]
        to_mult = result.reshape((ny, nx, -1))
        reshaped = to_mult.repeat(m, 0).repeat(m, 1)

        out = reshaped.reshape((-1, k))
    elif result.ndim == 3:
        reshaped = result.repeat(m, 0).repeat(m, 1)
        out = reshaped

    return out


# Get deep features from <model> as applied to <im_torch>
def get_deep_features(model, im_torch):
    if hasattr(model, "get_intermediate_layers"):
        from flexmm.deepnets.dinov3 import extract_features

        return extract_features(model, im_torch)
    else:
        return get_conv2d_features(model, im_torch)


def get_conv2d_features(model, im_torch):
    deep_features = []
    for i in range(1, len(model) + 1):
        if isinstance(model[:i][-1], nn.Conv2d):
            deep_features.append(model[:i](im_torch).detach().numpy()[0])

    return np.array(deep_features, dtype=object)


"""
Model ref: independent layers / no spatial smoothing
"""


def model_ref(
    model,
    im,
    K_list=np.array([3, 4, 5]),
    L=16,
    d_list=np.array([64, 128]),
    N_list=np.array([256, 128]),
    n_iter=50,
    params="q",
    ppca=False,
    gmm=True,
    light=True,
    verbose=None,
):
    model = copy.deepcopy(model)
    ny, nx = im.shape[:2]
    K = len(K_list)
    im_torch = (
        torch.from_numpy(np.moveaxis(im, [0, 1, 2], [1, 2, 0])).float().unsqueeze(0)
    )
    # prior_weights = 'loc'
    # prior_init = False #False

    res = np.zeros((L, K, 3, 1), dtype=object)

    # compute deep features
    deep_features = get_deep_features(model, im_torch)
    # deep_features = get_conv2d_features(model, im_torch)

    # run fit
    """
    example N_list: N_list = np.array([(ny,nx),(ny,nx),(ny//2,nx//2),(ny//2,nx//2),
                       (ny//4,nx//4),(ny//4,nx//4),(ny//4,nx//4),(ny//4,nx//4),
                       (ny//8,nx//8),(ny//8,nx//8),(ny//8,nx//8),(ny//8,nx//8),
                       (ny//16,nx//16),(ny//16,nx//16),(ny//16,nx//16),(ny//16,nx//16)])
    """
    for l in range(L):
        if N_list[l][0] != ny and l > 0:
            Xpca0 = pooling(Xpca0.reshape((ny, nx, Xpca0.shape[-1])), (2, 2)).reshape(
                ny // 2 * nx // 2, Xpca0.shape[-1]
            )

        res[l, 0, 0, 0] = PCA(n_components=0.95)
        d = d_list[l]
        ny, nx = N_list[l]
        X = deep_features[l].reshape(d, ny * nx).T
        if ppca == True:
            Xpca = X
            n_pca = res[l, 0, 0, 0].fit(X).n_components_ // 2
            # print(n_pca)
        else:
            Xpca = res[l, 0, 0, 0].fit_transform(X)

        if l == 0:
            Xpca0 = np.copy(Xpca)
        else:
            Xpca = np.concatenate((Xpca, Xpca0), 1)

        k = 0
        for kk in K_list:
            if gmm:
                res[l, k, 1, 0] = GMM(
                    n_components=kk,
                    n_init=1,
                    tol=1e-2,
                    n_iter=n_iter,
                    params="w" + params + "mc",
                    light=light,
                )
            res[l, k, 2, 0] = SMM(
                n_components=kk,
                n_init=1,
                tol=1e-2,
                n_iter=n_iter,
                params="w" + params + "mcd",
                light=light,
            )
            if gmm:
                try:
                    res[l, k, 1, 0].fit(Xpca)
                except:
                    pass

            try:
                res[l, k, 2, 0].fit(Xpca)
            except:
                pass

            k += 1

    return res


"""
Model a: independent layers
"""


def model_a(
    model,
    im,
    K_list=np.array([3, 4, 5]),
    L=16,
    d_list=np.array([64, 128]),
    N_list=np.array([256, 128]),
    neigh_size_list=np.tile(5, 16),
    n_iter=None,
    params="q",
    ppca=True,
    n_pca=12,
    gmm=True,
    light=True,
    verbose=None,
    keep=False,
):
    model = copy.deepcopy(model)
    ny, nx = im.shape[:2]
    K = K_list.shape[0]
    im_torch = (
        torch.from_numpy(np.moveaxis(im, [0, 1, 2], [1, 2, 0])).float().unsqueeze(0)
    )
    prior_weights = "loc"
    prior_init = False  # False

    res = np.zeros((L, K, 3, 1), dtype=object)
    proba_maps = np.zeros((n_iter, L, K, 2), dtype=object)

    # compute deep features
    deep_features = get_conv2d_features(model, im_torch)

    var = 1
    # run fit
    for l in range(L):
        if N_list[l][0] != ny and l > 0:
            Xpca0 = pooling(Xpca0.reshape((ny, nx, Xpca0.shape[-1])), (2, 2)).reshape(
                ny // 2 * nx // 2, Xpca0.shape[-1]
            )

        res[l, 0, 0, 0] = PCA(n_components=0.95)
        d = d_list[l]
        ny, nx = N_list[l]
        X = deep_features[l].reshape(d, ny * nx).T
        if ppca == True:
            Xpca = X
            n_pca = res[l, 0, 0, 0].fit(X).n_components_ // 2
            # print(n_pca)
        else:
            Xpca = res[l, 0, 0, 0].fit_transform(X)

        if l == 0:
            Xpca0 = np.copy(Xpca)
        else:
            Xpca = np.concatenate((Xpca, Xpca0), 1)

        k = 0
        for kk in K_list:
            prior_means_init = np.ones((ny * nx, kk)) / kk

            if gmm:
                res[l, k, 1, 0] = GMM(
                    n_components=kk,
                    prior_weights=prior_weights,
                    n_init=1,
                    prior_means=prior_means_init,
                    prior_var=var,
                    prior_init=prior_init,
                    im_shape=(ny, nx),
                    neigh_size=neigh_size_list[l],
                    tol=1e-2,
                    n_iter=30,
                    params="w" + params + "mc",
                    light=light,
                    ppca=ppca,
                    n_pca=n_pca,
                )
            res[l, k, 2, 0] = SMM(
                n_components=kk,
                prior_weights=prior_weights,
                n_init=1,
                prior_means=prior_means_init,
                prior_var=var,
                prior_init=prior_init,
                im_shape=(ny, nx),
                neigh_size=neigh_size_list[l],
                tol=1e-2,
                n_iter=30,
                params="w" + params + "mcd",
                light=light,
                ppca=ppca,
                n_pca=n_pca,
            )
            if gmm:
                try:
                    res[l, k, 1, 0].fit(Xpca)
                except:
                    pass

            try:
                res[l, k, 2, 0].fit(Xpca)
            except:
                pass

            k += 1

    if keep:
        return res  # proba_maps, lkl_smm, lkl_gmm
    else:
        return res


"""
Model b: single prior map
"""


def model_b(
    model,
    im,
    K_list=np.array([3, 4, 5]),
    L=16,
    d_list=np.array([64, 128]),
    N_list=np.array([256, 128]),
    neigh_size_list=np.tile(5, 16),
    n_iter=50,
    params="q",
    ppca=True,
    n_pca=12,
    gmm=True,
    light=True,
    verbose=True,
):
    model = copy.deepcopy(model)
    ny, nx = im.shape[:2]
    K = K_list.shape[0]
    im_torch = (
        torch.from_numpy(np.moveaxis(im, [0, 1, 2], [1, 2, 0])).float().unsqueeze(0)
    )
    prior_weights = "ext3"

    Xpca = np.zeros(L, dtype=object)
    res = np.zeros((L, K, 3, 1), dtype=object)
    proba_maps = np.zeros((n_iter, L, K, 2), dtype=object)

    deep_features = get_conv2d_features(model, im_torch)

    for l in range(L):
        if N_list[l][0] != ny and l > 0:
            Xpca0 = pooling(Xpca0.reshape((ny, nx, Xpca0.shape[-1])), (2, 2)).reshape(
                ny // 2 * nx // 2, Xpca0.shape[-1]
            )

        if l == 0:
            prior_init = False
        else:
            prior_init = True

        res[l, 0, 0, 0] = PCA(n_components=0.95)
        d = d_list[l]
        ny, nx = N_list[l]
        X = deep_features[l].reshape(d, ny * nx).T

        if ppca == True:
            Xpca[l] = X
            n_pca = np.maximum(res[l, 0, 0, 0].fit(X).n_components_ // 2, 2)
            # print(n_pca)
        else:
            Xpca[l] = res[l, 0, 0, 0].fit_transform(X)

        if l == 0:
            Xpca0 = np.copy(Xpca[0])
        else:
            Xpca[l] = np.concatenate((Xpca[l], Xpca0), 1)

        k = 0
        for kk in K_list:
            prior_means_init = np.ones((ny * nx, kk)) / kk
            prior_var = 1.0

            if gmm:
                res[l, k, 1, 0] = GMM(
                    n_components=kk,
                    prior_weights=prior_weights,
                    n_init=1,
                    prior_means=prior_means_init,
                    prior_var=prior_var,
                    prior_init=prior_init,
                    im_shape=(ny, nx),
                    neigh_size=neigh_size_list[l],
                    tol=1e-3,
                    n_iter=200,
                    params="w" + params + "mc",
                    ppca=ppca,
                    n_pca=n_pca,
                )
            res[l, k, 2, 0] = SMM(
                n_components=kk,
                prior_weights=prior_weights,
                n_init=1,
                prior_means=prior_means_init,
                prior_var=prior_var,
                prior_init=prior_init,
                im_shape=(ny, nx),
                neigh_size=neigh_size_list[l],
                tol=1e-3,
                n_iter=200,
                params="w" + params + "mcd",
                ppca=ppca,
                n_pca=n_pca,
            )

            k += 1
    # init
    if verbose:
        print("Initialization ...")
    for k in range(K):
        kk = K_list[k]

        for l in range(L):
            if N_list[l][0] != ny:
                pool = True
            else:
                pool = False

            ny, nx = N_list[l]
            if l == 0:
                # SMM
                res[l, k, 2, 0]._initialization_step(Xpca[l], use_kmeans=True)
                prior_param_smm = res[l, k, 2, 0]._posterior_proba(Xpca[l])
                # GMM
                if gmm:
                    res[l, k, 1, 0]._initialization_step(Xpca[l], use_kmeans=True)
                    _, prior_param_gmm = res[l, k, 1, 0]._expectation_step(Xpca[l])
            else:
                if pool:
                    # SMM
                    res[l, k, 2, 0].prior_means = pooling(
                        prior_param_smm.reshape(2 * ny, 2 * nx, kk), (2, 2)
                    ).reshape(ny * nx, kk)
                    # GMM
                    if gmm:
                        res[l, k, 1, 0].prior_means = pooling(
                            prior_param_gmm.reshape(2 * ny, 2 * nx, kk), (2, 2)
                        ).reshape(ny * nx, kk)
                else:
                    # SMM
                    res[l, k, 2, 0].prior_means = prior_param_smm
                    # GMM
                    if gmm:
                        res[l, k, 1, 0].prior_means = prior_param_gmm

                # SMM
                res[l, k, 2, 0].prior_var = 1.0
                res[l, k, 2, 0].prior_norm = 2.0
                res[l, k, 2, 0]._initialization_step(Xpca[l])
                prior_param_smm = res[l, k, 2, 0]._posterior_proba(Xpca[l])

                # GMM
                if gmm:
                    res[l, k, 1, 0].prior_var = 1.0
                    res[l, k, 1, 0].prior_norm = 2.0
                    res[l, k, 1, 0]._initialization_step(Xpca[l])
                    _, prior_param_gmm = res[l, k, 1, 0]._expectation_step(Xpca[l])

    lkl_smm = np.zeros((K, L, n_iter))
    prior_means_smm = np.zeros((K, L), dtype=object)
    prior_var_smm = np.zeros((K, L))
    tau_smm = np.zeros(L, dtype=object)
    nu = np.zeros(L, dtype=object)

    if gmm:
        lkl_gmm = np.zeros((K, L, n_iter))
        prior_means_gmm = np.zeros((K, L), dtype=object)
        prior_var_gmm = np.zeros((K, L))
        tau_gmm = np.zeros(L, dtype=object)

    # EM
    for k in range(K):
        kk = K_list[k]

        # SMM
        prior_param_smm = res[0, k, 2, 0]._posterior_proba(Xpca[0])
        # GMM
        if gmm:
            _, prior_param_gmm = res[0, k, 1, 0]._expectation_step(Xpca[0])

        i = 0
        while i < n_iter:
            for l in range(L):
                ny, nx = N_list[l]
                # SMM
                lkls_smm, tau_smm[l], nu[l] = res[l, k, 2, 0]._expectation_step(Xpca[l])
                lkl_smm[k, l, i] = np.log(lkls_smm).mean()
                prior_means_smm[k, l] = sp.ndimage.convolve(
                    tau_smm[l].reshape(ny, nx, kk),
                    res[l, k, 2, 0].neighbors,
                    mode="nearest",
                ).reshape(ny * nx, kk)
                prior_var = sp.ndimage.convolve(
                    (tau_smm[l] ** 2).reshape(ny, nx, kk),
                    res[l, k, 2, 0].neighbors,
                    mode="nearest",
                ).reshape(ny * nx, kk)
                prior_var -= prior_means_smm[k, l] ** 2
                prior_var_smm[k, l] = prior_var.mean()

                # GMM
                if gmm:
                    lkls_gmm, tau_gmm[l] = res[l, k, 1, 0]._expectation_step(Xpca[l])
                    lkl_gmm[k, l, i] = np.log(lkls_gmm).mean()
                    prior_means_gmm[k, l] = sp.ndimage.convolve(
                        tau_gmm[l].reshape(ny, nx, kk),
                        res[l, k, 1, 0].neighbors,
                        mode="nearest",
                    ).reshape(ny * nx, kk)
                    prior_var = sp.ndimage.convolve(
                        (tau_gmm[l] ** 2).reshape(ny, nx, kk),
                        res[l, k, 1, 0].neighbors,
                        mode="nearest",
                    ).reshape(ny * nx, kk)
                    prior_var -= prior_means_gmm[k, l] ** 2
                    prior_var_gmm[k, l] = prior_var.mean()

            prior_w_smm = np.prod(
                prior_var_smm[k][np.newaxis] * (1 - np.eye(L)) + np.eye(L), axis=1
            )
            if gmm:
                prior_w_gmm = np.prod(
                    prior_var_gmm[k][np.newaxis] * (1 - np.eye(L)) + np.eye(L), axis=1
                )

            ny, nx = N_list[L - 1]

            prior_wm_smm = 0
            tau_sum_smm = 0
            if gmm:
                prior_wm_gmm = 0
                tau_sum_gmm = 0

            n_sum = 0
            for l in range(L):
                if N_list[L - 1 - l][0] != ny:
                    prior_wm_smm = unpooling(
                        prior_wm_smm.reshape(ny, nx, kk), (2, 2, 1)
                    ).reshape(4 * ny * nx, kk)
                    if gmm:
                        prior_wm_gmm = unpooling(
                            prior_wm_gmm.reshape(ny, nx, kk), (2, 2, 1)
                        ).reshape(4 * ny * nx, kk)

                ny, nx = N_list[L - 1 - l]
                prior_wm_smm += prior_w_smm[L - 1 - l] * prior_means_smm[k, L - 1 - l]
                if gmm:
                    prior_wm_gmm += (
                        prior_w_gmm[L - 1 - l] * prior_means_gmm[k, L - 1 - l]
                    )

                # component selection (maybe add weights)
                tau_sum_smm += tau_smm[l].sum(0)
                if gmm:
                    tau_sum_gmm += tau_gmm[l].sum(0)
                n_sum += ny * nx

            for l in range(L):
                if N_list[l][0] != ny:
                    prior_wm_smm = pooling(
                        prior_wm_smm.reshape(ny, nx, kk), (2, 2)
                    ).reshape(ny * nx // 4, kk)
                    if gmm:
                        prior_wm_gmm = pooling(
                            prior_wm_gmm.reshape(ny, nx, kk), (2, 2)
                        ).reshape(ny * nx // 4, kk)

                ny, nx = N_list[l]

                # SMM
                # NOTE: does the work of "ext3"
                res[l, k, 2, 0].prior_means = prior_wm_smm
                res[l, k, 2, 0].prior_norm = prior_w_smm.sum()

                # cov_dist_smm = res[l,k,2,0].covars_.reshape(kk,ny*nx)
                # be_smm = (sp.spatial.distance.squareform(\
                #          sp.spatial.distance.pdist(cov_dist_smm,\
                #            metric='cosine'))<0.2).sum(0)-1
                res[l, k, 2, 0].q_weights_ = tau_sum_smm / (
                    tau_sum_smm + n_sum / (kk + 0.0)
                )
                res[l, k, 2, 0]._maximisation_step(Xpca[l], tau_smm[l], nu[l])

                proba_maps[i, l, k, 1] = res[l, k, 2, 0].weights_

                # GMM
                if gmm:
                    res[l, k, 1, 0].prior_means = prior_wm_gmm
                    res[l, k, 1, 0].prior_norm = prior_w_gmm.sum()
                    res[l, k, 1, 0].q_weights_ = tau_sum_gmm / (
                        tau_sum_gmm + n_sum / (8 * kk)
                    )
                    res[l, k, 1, 0]._maximisation_step(Xpca[l], tau_gmm[l])
                    proba_maps[i, l, k, 0] = res[l, k, 1, 0].weights_

            if i > 1:
                lkl_diff = np.abs(lkl_smm[k, :, i].mean() - lkl_smm[k, :, i - 1].mean())
                if lkl_diff < 1e-2:
                    i = n_iter
                else:
                    i += 1
            else:
                i += 1

            if verbose:
                print("Fitting ...")
                print("Iteration : (%i,%i)" % (k, i))
                if gmm:
                    print("GMM:", tau_sum_gmm[res[0, k, 1, 0].q_], n_sum / (8 * kk))
                print("SMM:", tau_sum_smm[res[0, k, 2, 0].q_], n_sum / (kk + 1.0))
                # print(prior_w_smm)
                sys.stdout.flush()

    if light:
        for k in range(K):
            for l in range(L):
                if gmm:
                    # del(res[l,k,1,0].Y)
                    # del(res[l,k,1,0].B_)
                    # del(res[l,k,1,0].S)
                    res[l, k, 1, 0].weights_ = np.float16(res[l, k, 1, 0].weights_)
                    res[l, k, 1, 0].means_ = np.float32(res[l, k, 1, 0].means_)
                    res[l, k, 1, 0].covars_ = np.float32(res[l, k, 1, 0].covars_)
                    # res[l,k,1,0].pcs = np.float32(res[l,k,1,0].pcs)
                    # res[l,k,1,0].taus = np.float32(res[l,k,1,0].taus)

                # del(res[l,k,2,0].Y)
                # del(res[l,k,2,0].B_)
                # del(res[l,k,2,0].S)
                res[l, k, 2, 0].weights_ = np.float16(res[l, k, 2, 0].weights_)
                res[l, k, 2, 0].means_ = np.float32(res[l, k, 2, 0].means_)
                res[l, k, 2, 0].covars_ = np.float32(res[l, k, 2, 0].covars_)
                res[l, k, 2, 0].degrees_ = np.float32(res[l, k, 2, 0].degrees_)
                # res[l,k,2,0].pcs = np.float32(res[l,k,2,0].pcs)
                # res[l,k,2,0].taus = np.float32(res[l,k,2,0].taus)

    return res  # , proba_maps #lkl_smm, lkl_gmm


"""
Model c: prior maps smoothed by previous and next layers
"""


def model_c(
    model,
    im,
    gt=None,
    gt_eps=None,
    n_components_best=None,
    kmeans=True,
    K_list=np.array([3, 4, 5]),
    L=16,
    d_list=np.array([64, 128]),
    N_list=np.array([256, 128]),
    neigh_size_list=np.tile(5, 16),
    n_iter=50,
    params="q",
    ppca=False,
    n_pca=0.95,
    gmm=False,
    light=True,
    verbose=True,
    keep=False,
    prior_weights="ext3",
    spatial_smoothing=True,
    layer_normalization=True,
    reshape_deep_layers=True,
):
    # do not use KMeans initialization if initial (groundtruth) map is provided
    if gt is not None:
        kmeans = False
    model = copy.deepcopy(model)
    Ny, Nx = im.shape[:2]  # image INPUT shape (doesn't change with layer)
    ny, nx = im.shape[:2]  # shape AT LAYER
    # initial n_components for model
    K = K_list.shape[0]
    # reshape image so color channels are the last dimension
    im_torch = (
        torch.from_numpy(np.moveaxis(im, [0, 1, 2], [1, 2, 0])).float().unsqueeze(0)
    )
    # CHANGED: #1 Try setting prior_weights to None to turn off spatial smoothing
    # DEBUG: SMM.neighbors is not initialized in SMM.__init__
    ## error in line 824
    prior_weights = "ext3"

    # Initializes the results for arrays used in FlexMM
    Xpca = np.zeros(L, dtype=object)
    res = np.zeros((L, K, 3, 1), dtype=object)
    proba_maps = np.zeros((n_iter, L, K, 8), dtype=object)

    # get deep features from VGG-19
    deep_features = get_deep_features(model, im_torch)

    # Initializes the FlexMM object for each layer and each number of components
    # for each layer...
    for l in range(L):
        # if not the first layer, mean pool using a 2x2 window
        if N_list[l][0] != ny and l > 0:
            Xpca0 = pooling(Xpca0.reshape((ny, nx, Xpca0.shape[-1])), (2, 2)).reshape(
                ny // 2 * nx // 2, Xpca0.shape[-1]
            )

        if l == 0:
            prior_init = False
        else:
            prior_init = True

        # initialize PCA Object from sklearn
        res[l, 0, 0, 0] = PCA(n_components=n_pca)
        # set embedding dimension of features
        d = d_list[l]
        # set im size dimenstion of features
        ny, nx = N_list[l]
        # reshape features to im size and embedding dimension at layer
        X = deep_features[l].reshape(d, ny * nx).T

        # fit PCA to deep_features from VGG-19
        Xpca[l] = res[l, 0, 0, 0].fit_transform(X)

        if l == 0:
            # Xpca0 is the PCA result from the first layer of features
            Xpca0 = np.copy(Xpca[0])
        else:
            # RGB features (Xpca0) in addition to features of every layer
            Xpca[l] = np.concatenate((Xpca[l], Xpca0), 1)

        k = 0
        # for each n_components
        for kk in K_list:
            # create a ny*nx by k vector of 1s (prior SMM object means)
            prior_means_init = np.ones((ny * nx, kk)) / kk
            # prior variance for the SMM object
            prior_var = 1.0
            # NOTE: for default case params is "wmcd" (weights, menas, covariances, dofs)
            if gmm:
                res[l, k, 1, 0] = GMM(
                    n_components=kk,
                    prior_weights=prior_weights,
                    n_init=1,
                    prior_means=prior_means_init,
                    prior_var=prior_var,
                    prior_init=prior_init,
                    im_shape=(ny, nx),
                    neigh_size=neigh_size_list[l],
                    tol=1e-3,
                    n_iter=200,
                    params="w" + params + "mc",
                    ppca=ppca,
                    n_pca=n_pca,
                )
            res[l, k, 2, 0] = SMM(
                n_components=kk,
                prior_weights=prior_weights,
                n_init=1,
                prior_means=prior_means_init,
                prior_var=prior_var,
                prior_init=prior_init,
                im_shape=(ny, nx),
                neigh_size=neigh_size_list[l],
                tol=1e-3,
                n_iter=200,
                params="w" + params + "mcd",
                ppca=ppca,
                n_pca=n_pca,
            )

            k += 1

    # NOTE: init
    if verbose:
        print("Initialization ...")
    for k in range(K):
        kk = K_list[k]

        for l in range(L):
            # NOTE: Mixture models are fit to either pooled or unpooled data depending on the layer
            if N_list[l][0] != ny:
                pool = True
            else:
                pool = False

            ny, nx = N_list[l]
            if l == 0:
                # SMM
                # NOTE: initialize the mixture model at the current layer
                res[l, k, 2, 0]._initialization_step(
                    Xpca[l],
                    gt=gt,
                    gt_eps=gt_eps,
                    n_components_best=n_components_best,
                    use_kmeans=kmeans,
                )
                # NOTE: calculates the responsibilities after the initialization step
                prior_param_smm = res[l, k, 2, 0]._posterior_proba(Xpca[l])  # Here
                # GMM
                if gmm:
                    res[l, k, 1, 0]._initialization_step(
                        Xpca[l],
                        gt=gt,
                        n_components_best=n_components_best,
                        use_kmeans=kmeans,
                    )
                    _, prior_param_gmm = res[l, k, 1, 0]._expectation_step(Xpca[l])
            else:
                if pool:
                    # SMM
                    res[l, k, 2, 0].prior_means = pooling(
                        prior_param_smm.reshape(2 * ny, 2 * nx, kk), (2, 2)
                    ).reshape(ny * nx, kk)
                    # GMM
                    if gmm:
                        res[l, k, 1, 0].prior_means = pooling(
                            prior_param_gmm.reshape(2 * ny, 2 * nx, kk), (2, 2)
                        ).reshape(ny * nx, kk)
                else:
                    # SMM
                    res[l, k, 2, 0].prior_means = prior_param_smm
                    # GMM
                    if gmm:
                        res[l, k, 1, 0].prior_means = prior_param_gmm

                # SMM
                res[l, k, 2, 0].prior_var = 1.0
                res[l, k, 2, 0].prior_norm = 2.0
                res[l, k, 2, 0]._initialization_step(Xpca[l])
                prior_param_smm = res[l, k, 2, 0]._posterior_proba(Xpca[l])

                # GMM
                if gmm:
                    res[l, k, 1, 0].prior_var = 1.0
                    res[l, k, 1, 0].prior_norm = 2.0
                    res[l, k, 1, 0]._initialization_step(Xpca[l])
                    _, prior_param_gmm = res[l, k, 1, 0]._expectation_step(Xpca[l])

    if gmm:
        lkl_gmm = np.zeros((K, L, n_iter))
        prior_means_gmm = np.zeros((K, L), dtype=object)
        prior_var_gmm = np.zeros((K, L))
        prior_wm_gmm = np.zeros(L, dtype=object)
        tau_gmm = np.zeros(L, dtype=object)

    # NOTE: likelihood array initialization
    lkl_smm = np.zeros((K, L, n_iter))
    # NOTE: prior means array initialization
    prior_means_smm = np.zeros((K, L), dtype=object)
    # NOTE: prior var array initialization
    prior_var_smm = np.zeros((K, L))
    # NOTE:?
    prior_wm_smm = np.zeros(L, dtype=object)
    # NOTE: prior tau array initilization
    # NOTE tau = E(class|observation)
    tau_smm = np.zeros(L, dtype=object)
    # NOTE: initialize prior degrees of freedom
    nu = np.zeros(L, dtype=object)
    # NOTE EXPECTATION-MAXIMIZATION STEPS:
    for k in range(K):
        kk = K_list[k]

        # SMM
        # NOTE: calculate responsibilities from the initial guess for the first layer
        prior_param_smm = res[0, k, 2, 0]._posterior_proba(Xpca[0])
        # GMM
        if gmm:
            _, prior_param_gmm = res[0, k, 1, 0]._expectation_step(Xpca[0])

        i = 0
        while i < n_iter:
            tau_sum_smm = 0
            if gmm:
                tau_sum_gmm = 0

            n_sum = 0
            for l in range(L):
                ny, nx = N_list[l]
                # SMM
                # NOTE: tau_smm are responsibilities of each mixture component
                # NOTE: nu are the gammaweights (GSM mixer) for a particular mixture component
                lkls_smm, tau_smm[l], nu[l] = res[l, k, 2, 0]._expectation_step(Xpca[l])
                rdl = lambda x: _reshape_deep_layers(x, l, N_list, d_list)
                # saves the likelihood per observation as output
                proba_maps[i, l, k, 5] = lkls_smm
                if ny != Ny:
                    proba_maps[i, l, k, 5] = rdl(lkls_smm)
                # NOTE: calculate the log-likelihood from the likelihood
                lkl_smm[k, l, i] = np.log(lkls_smm).mean()
                # NOTE: convolves responsibilites at every point with a 2D gaussian
                if spatial_smoothing == 1:
                    # each point gets the weighted average of the responibilities of each component
                    prior_means_smm[k, l] = sp.ndimage.convolve(
                        tau_smm[l].reshape(ny, nx, kk),
                        res[l, k, 2, 0].neighbors,
                        mode="nearest",
                    ).reshape(ny * nx, kk)
                    # each point gets the weighted variance of the responibilities of each component
                    prior_var = sp.ndimage.convolve(
                        (tau_smm[l] ** 2).reshape(ny, nx, kk),
                        res[l, k, 2, 0].neighbors,
                        mode="nearest",
                    ).reshape(ny * nx, kk)
                elif spatial_smoothing == 0:
                    # GAUSSIAN KERNEL IS INF
                    prior_means_smm[k, l] = (
                        np.mean(tau_smm[l], axis=0)
                        .T[np.newaxis, ...]
                        .repeat(ny * nx, axis=0)
                    ).reshape(ny * nx, kk)
                    prior_var = (
                        np.mean((tau_smm[l] ** 2), axis=0)
                        .T[np.newaxis, ...]
                        .repeat(ny * nx, axis=0)
                    ).reshape(ny * nx, kk)
                elif spatial_smoothing == -1:
                    # GAUSSIAN KERNEL IS EPS
                    prior_means_smm[k, l] = tau_smm[l].reshape(ny * nx, kk)
                    prior_var = (
                        np.mean((tau_smm[l] ** 2), axis=0)
                        .T[np.newaxis, ...]
                        .repeat(ny * nx, axis=0)
                    ).reshape(ny * nx, kk)

                prior_var -= prior_means_smm[k, l] ** 2
                prior_var_smm[k, l] = prior_var.mean()

                # GMM
                if gmm:
                    lkls_gmm, tau_gmm[l] = res[l, k, 1, 0]._expectation_step(Xpca[l])
                    lkl_gmm[k, l, i] = np.log(lkls_gmm).mean()
                    prior_means_gmm[k, l] = sp.ndimage.convolve(
                        tau_gmm[l].reshape(ny, nx, kk),
                        res[l, k, 1, 0].neighbors,
                        mode="nearest",
                    ).reshape(ny * nx, kk)
                    prior_var = sp.ndimage.convolve(
                        (tau_gmm[l] ** 2).reshape(ny, nx, kk),
                        res[l, k, 1, 0].neighbors,
                        mode="nearest",
                    ).reshape(ny * nx, kk)
                    prior_var -= prior_means_gmm[k, l] ** 2
                    prior_var_gmm[k, l] = prior_var.mean()

                # component selection (maybe add weights)
                tau_sum_smm += tau_smm[l].sum(0)
                # NOTE: calculates the sum of all responsibilities
                if gmm:
                    tau_sum_gmm += tau_gmm[l].sum(0)
                n_sum += ny * nx

            means_smm = np.pad(prior_means_smm[k], 1, mode="edge")
            var_smm = np.pad(prior_var_smm[k], 1, mode="edge")
            if gmm:
                means_gmm = np.pad(prior_means_gmm[k], 1, mode="edge")
                var_gmm = np.pad(prior_var_gmm[k], 1, mode="edge")

            n_list = np.pad(N_list, ((1, 1), (0, 0)), mode="edge")
            # NOTE: normalization across layers
            for l in range(1, L + 1):
                ny, nx = n_list[l]
                # NOTE: take the product of the variance of responsibilities across layers
                var_smm_prod = var_smm[l - 1 : l + 2]
                var_smm_prod = np.prod(
                    var_smm_prod[np.newaxis] * (1 - np.eye(3)) + np.eye(3), axis=1
                )

                if gmm:
                    var_gmm_prod = var_gmm[l - 1 : l + 2]
                    var_gmm_prod = np.prod(
                        var_gmm_prod[np.newaxis] * (1 - np.eye(3)) + np.eye(3),
                        axis=1,
                    )

                prior_wm_smm[l - 1] = 0
                if gmm:
                    prior_wm_gmm[l - 1] = 0

                if layer_normalization:
                    _range = range(-1, 2)
                else:
                    _range = [0]
                for j in _range:
                    if n_list[l + j][0] < ny:
                        means_smm_ = unpooling(
                            means_smm[j + l].reshape(ny // 2, nx // 2, kk),
                            (2, 2, 1),
                        ).reshape(ny * nx, kk)
                        if gmm:
                            means_gmm_ = unpooling(
                                means_gmm[j + l].reshape(ny // 2, nx // 2, kk),
                                (2, 2, 1),
                            ).reshape(ny * nx, kk)
                    elif n_list[l + j][0] > ny:
                        means_smm_ = pooling(
                            means_smm[j + l].reshape(2 * ny, 2 * nx, kk), (2, 2)
                        ).reshape(ny * nx, kk)
                        if gmm:
                            means_gmm_ = pooling(
                                means_gmm[j + l].reshape(2 * ny, 2 * nx, kk), (2, 2)
                            ).reshape(ny * nx, kk)
                    else:
                        means_smm_ = means_smm[j + l]
                        if gmm:
                            means_gmm_ = means_gmm[j + l]

                    prior_wm_smm[l - 1] += var_smm_prod[j + 1] * means_smm_
                    if gmm:
                        prior_wm_gmm[l - 1] += var_gmm_prod[j + 1] * means_gmm_

                prior_wm_smm[l - 1] /= var_smm_prod.sum()
                if gmm:
                    prior_wm_gmm[l - 1] /= var_gmm_prod.sum()

            for l in range(L):
                # SMM
                rdl = lambda x: _reshape_deep_layers(x, l, N_list, d_list)
                ny, nx = N_list[l]
                d = d_list[l]
                res[l, k, 2, 0].prior_means = prior_wm_smm[l]
                res[l, k, 2, 0].prior_norm = 1
                res[l, k, 2, 0].q_weights_ = tau_sum_smm / (
                    tau_sum_smm + n_sum / (kk + 0.0)
                )
                res[l, k, 2, 0]._maximisation_step(Xpca[l], tau_smm[l], nu[l])

                if spatial_smoothing == 1:
                    # TEMP CHANGE
                    proba_maps[i, l, k, 0] = res[l, k, 2, 0].weights_
                    if l > 1 and reshape_deep_layers:
                        reshaped_weights = rdl(res[l, k, 2, 0].weights_)
                        proba_maps[i, l, k, 0] = reshaped_weights

                else:  # use the posterior probaiblities for the spatial_smoothing==0 case because
                    # prior probabilty is scalar
                    regular_mixture_prior = res[l, k, 2, 0].weights_
                    _regular_mixture_posterior = res[l, k, 2, 0]._posterior_proba(
                        Xpca[l]
                    )
                    assert (
                        regular_mixture_prior.shape == _regular_mixture_posterior.shape
                    )
                    proba_maps[i, l, k, 0] = _regular_mixture_posterior
                    if l > 1 and reshape_deep_layers:
                        reshaped_weights = rdl(_regular_mixture_posterior)
                        proba_maps[i, l, k, 0] = reshaped_weights
                proba_maps[i, l, k, 1] = res[l, k, 2, 0].means_
                proba_maps[i, l, k, 2] = res[l, k, 2, 0].covars_
                proba_maps[i, l, k, 3] = res[l, k, 2, 0].degrees_
                proba_maps[i, l, k, 4] = tau_smm[l]
                proba_maps[0, l, k, 6] = Xpca[l]
                proba_maps[0, l, k, 7] = deep_features[l]
                if l > 1 and reshape_deep_layers:
                    # move the feature axis so that it's the last axis
                    deep_features = np.moveaxis(deep_features, 0, -1)
                    proba_maps[i, l, k, 4] = rdl(tau_smm[l])
                    proba_maps[0, l, k, 6] = rdl(Xpca[l])
                    proba_maps[0, l, k, 7] = rdl(deep_features[l])

                # GMM
                if gmm:
                    res[l, k, 1, 0].prior_means = prior_wm_gmm[l]
                    res[l, k, 1, 0].prior_norm = 1
                    res[l, k, 1, 0].q_weights_ = tau_sum_gmm / (
                        tau_sum_gmm + n_sum / (8 * kk)
                    )
                    res[l, k, 1, 0]._maximisation_step(Xpca[l], tau_gmm[l])

            if i > 1:
                lkl_diff = np.abs(lkl_smm[k, :, i].mean() - lkl_smm[k, :, i - 1].mean())
                if lkl_diff < 1e-2:
                    i = n_iter
                else:
                    i += 1
            else:
                i += 1

            if verbose:
                print("Fitting ...")
                print("Iteration : (%i,%i)" % (k, i))
                if gmm:
                    print("GMM:", tau_sum_gmm[res[0, k, 1, 0].q_], n_sum / (8 * kk))
                print("SMM:", tau_sum_smm[res[0, k, 2, 0].q_], n_sum / (kk + 1.0))
                # print(var_smm)
                sys.stdout.flush()

    if light:
        for k in range(K):
            for l in range(L):
                if gmm:
                    # del(res[l,k,1,0].Y)
                    # del(res[l,k,1,0].B_)
                    # del(res[l,k,1,0].S)
                    res[l, k, 1, 0].weights_ = np.float16(res[l, k, 1, 0].weights_)
                    res[l, k, 1, 0].means_ = np.float32(res[l, k, 1, 0].means_)
                    res[l, k, 1, 0].covars_ = np.float32(res[l, k, 1, 0].covars_)
                    # res[l,k,1,0].pcs = np.float32(res[l,k,1,0].pcs)
                    # res[l,k,1,0].taus = np.float32(res[l,k,1,0].taus)

                # del(res[l,k,2,0].Y)
                # del(res[l,k,2,0].B_)
                # del(res[l,k,2,0].S)
                res[l, k, 2, 0].weights_ = np.float16(res[l, k, 2, 0].weights_)
                res[l, k, 2, 0].means_ = np.float16(res[l, k, 2, 0].means_)
                res[l, k, 2, 0].covars_ = np.float32(res[l, k, 2, 0].covars_)
                res[l, k, 2, 0].degrees_ = np.float32(res[l, k, 2, 0].degrees_)
                # res[l,k,2,0].pcs = np.float32(res[l,k,2,0].pcs)
                # res[l,k,2,0].taus = np.float32(res[l,k,2,0].taus)

    if keep:
        # proba_maps are the output weights from each iteration of the M-step
        return res, proba_maps  # , lkl_smm, lkl_gmm
    else:
        return res
