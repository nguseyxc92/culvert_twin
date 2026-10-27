"""
Deep-Learning Surrogate Digital Twins for Predictive Capacity Auditing and Maintenance Optimization
Author : Neguse Solomon Mekonnen
Date   : October 2026
File   : src/culvert_twin/m6_surrogates/surrogates.py

Module 6: surrogate models for SWMM flood volume.
  ANN : ensemble of 5 feed-forward networks (3 hidden layers 64-64-32, ReLU, Adam, early
        stopping), scikit-learn MLPRegressor.  Ensemble spread = epistemic uncertainty.
  GPR : Gaussian process, Matern-5/2 ARD kernel + white noise, trained on a subsample
        (exact GPR does not scale to 50 000 rows).
  PCE : polynomial (degree 3) ridge regression on standardised inputs; a PCE-type baseline
        (a full orthogonal-polynomial PCE is not used; stated honestly in the paper).
All models predict y = log1p(flood volume, m3); inputs are standardised.
"""
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel


class ANNEnsemble:
    def __init__(self, k=5, hidden=(64, 64, 32), seed=0):
        self.k, self.hidden, self.seed = k, hidden, seed

    def fit(self, X, y):
        self.sc = StandardScaler().fit(X)
        Z = self.sc.transform(X)
        self.m = [MLPRegressor(hidden_layer_sizes=self.hidden, activation="relu", solver="adam", learning_rate_init=2e-3,
                               max_iter=300, early_stopping=True, validation_fraction=0.1,
                               n_iter_no_change=15, random_state=self.seed + j).fit(Z, y)
                  for j in range(self.k)]
        return self

    def predict(self, X, return_std=False):
        Z = self.sc.transform(X); P = np.column_stack([m.predict(Z) for m in self.m])
        return (P.mean(1), P.std(1)) if return_std else P.mean(1)


class GPR:
    def __init__(self, n_train=3000, seed=0):
        self.n_train, self.seed = n_train, seed

    def fit(self, X, y):
        rng = np.random.default_rng(self.seed)
        idx = rng.choice(len(X), min(self.n_train, len(X)), replace=False)
        self.sc = StandardScaler().fit(X)
        d = X.shape[1]
        k = ConstantKernel(1.0) * Matern(length_scale=np.ones(d), nu=2.5, length_scale_bounds=(1e-2, 1e3)) \
            + WhiteKernel(1e-2, (1e-6, 1))
        self.m = GaussianProcessRegressor(k, normalize_y=True, n_restarts_optimizer=0,
                                          random_state=self.seed).fit(self.sc.transform(X[idx]), y[idx])
        return self

    def predict(self, X, return_std=False):
        return self.m.predict(self.sc.transform(X), return_std=return_std)


class PCE:
    def __init__(self, degree=3, alpha=1e-3):
        self.degree, self.alpha = degree, alpha

    def fit(self, X, y):
        self.m = make_pipeline(StandardScaler(), PolynomialFeatures(self.degree), Ridge(alpha=self.alpha)).fit(X, y)
        return self

    def predict(self, X, return_std=False):
        return self.m.predict(X)
