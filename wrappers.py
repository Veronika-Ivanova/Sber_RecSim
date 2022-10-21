from scipy.sparse import coo_matrix, csr_matrix
from lightfm import LightFM
import lightgbm as lgbm
import numpy as np


class LGBMWrapper(object):
    def __init__(self, lgb_params={}, num_boost_round=1):
        self.estimator = None
        self.lgb_params = lgb_params
        self.num_boost_round = num_boost_round

    @property
    def is_fitted(self):
        return self.estimator is not None

    def partial_fit(self, X, y, classes=None, sample_weight=None):
        self.estimator = lgbm.train(
            self.lgb_params,
            init_model=self.estimator,
            train_set=lgbm.Dataset(X, y),
            keep_training_booster=True,
            num_boost_round=self.num_boost_round,
        )

    def predict_proba(self, X):
        p = self.estimator.predict(X)
        return np.array([p, p]).T

    def decision_function(self, X):
        return self.estimator.predict(X)


class LFMWrapper(object):
    def __init__(self, lfm_params={}, user_data=None, doc_data=None, epochs=1):
        self.model = None
        self.lgb_params = lfm_params
        self.nusers, self.nitems = user_data.shape[0], doc_data.shape[0]
        self.user_data = csr_matrix(user_data)
        self.doc_data = csr_matrix(doc_data)
        self.epochs = 1

    @property
    def is_fitted(self):
        return self.model is None

    def get_model(self):
        if not self.is_fitted:
            self.model = LightFM(loss="warp")
            x = coo_matrix(([0], ([0], [0])), shape=(self.nusers, self.nitems))
            self.model.fit_partial(
                x,
                user_features=self.user_data,
                item_features=self.doc_data,
                epochs=0,
            )
        return self.model

    def partial_fit(self, X, y, classes=None, sample_weight=None):
        model = self.get_model()
        x = coo_matrix(
            ([i * 2 - 1 for i in y], (X[:, 0], X[:, 1])),
            shape=(self.nusers, self.nitems),
        )
        model.fit_partial(
            x,
            user_features=self.user_data,
            item_features=self.doc_data,
            epochs=self.epochs,
        )
        self.model = model

    def predict_proba(self, X):
        p = self.model.predict(
            X[0, 0],
            X[:, 1],
            user_features=self.user_data,
            item_features=self.doc_data,
        )
        return np.array([p, p]).T

    def decision_function(self, X):
        return self.model.predict(
            X[0, 0],
            X[:, 1],
            user_features=self.user_data,
            item_features=self.doc_data,
        )
