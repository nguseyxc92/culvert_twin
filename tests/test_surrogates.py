"""
Deep-Learning Surrogate Digital Twins for Predictive Capacity Auditing and Maintenance Optimization
Author : Neguse Solomon Mekonnen
Date   : October 2026
File   : tests/test_surrogates.py
"""
import unittest
import numpy as np
from culvert_twin.m6_surrogates.surrogates import ANNEnsemble, PCE, GPR


def _data(n=600, seed=0):
    r = np.random.default_rng(seed); X = r.uniform(0, 1, (n, 3))
    return X, np.sin(2 * X[:, 0]) + X[:, 1] ** 2


class T(unittest.TestCase):
    def test_models_fit_smooth_function(self):
        X, y = _data(); Xt, yt = _data(200, 1)
        for m in (ANNEnsemble(k=2, hidden=(16, 16)), GPR(n_train=300), PCE()):
            p = m.fit(X, y).predict(Xt)
            self.assertGreater(np.corrcoef(p, yt)[0, 1], 0.95)


    def test_ann_std_shape(self):
        X, y = _data(); m = ANNEnsemble(k=3, hidden=(8,)).fit(X, y)
        mu, sd = m.predict(X[:10], return_std=True)
        self.assertEqual(mu.shape, sd.shape)

