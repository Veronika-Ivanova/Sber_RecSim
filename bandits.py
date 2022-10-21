from recsim.agent import AbstractEpisodicRecommenderAgent
from cbbench.utils import check_is_fitted
import numpy as np
import jax
import jax.numpy as jnp
from jax import jit
import time 
from jax import vmap
from functools import partial

# from joblib import Parallel, delayed
# import numba


class BaseIncrementalRecommenderAgent(AbstractEpisodicRecommenderAgent):
    def __init__(
        self, observation_space, action_space, base_model, policy_temp=1
    ):
        super(BaseIncrementalRecommenderAgent, self).__init__(action_space)
        self._base_model = base_model
        self.nitems = len(observation_space.spaces["doc"].spaces)
        self.item_names = list(observation_space.spaces["doc"].spaces)
        self.is_init = False
        self.policy_temp = policy_temp
        self.policy = dict(
            zip(self.item_names, [1 / self.nitems] * self.nitems)
        )
        self.is_docs_prepared = False
        np.random.seed(0)

    def begin_episode(self, observation=None):
        self._episode_num += 1
        return self.step(0, observation, replay=True)

    @staticmethod
    def prep_doc_vec(doc):
        if type(doc) is dict:
            return np.array([doc["quality"].item(), doc["cluster_id"]])
        return doc.reshape(-1)

    def _prepare_features(self, user, docs):
        if not self.is_docs_prepared:
            self.docs = np.array(
                [self.prep_doc_vec(docs[docid]) for docid in list(docs)]
            )
            self.is_docs_prepared = True
        user = (
            np.repeat(user, self.docs.shape[0])
            .reshape(-1, self.docs.shape[0])
            .T
        )
        X = np.concatenate([user, self.docs], axis=1)
        return X

    def _update(self, response):
        self.is_init = True
        self._last_slate = [
            self._last_slate[i]
            for i in range(len(self._last_slate))
            if response[i]["shown"]
        ]
        response = [i for i in response if i["shown"]]
        target = [i["click"] for i in response]
        weights = np.array([i.get("weight", 1) for i in response])
        self._base_model.partial_fit(
            self._lastX[self._last_slate],
            target,
            classes=[0, 1],
            sample_weight=weights,
        )

    def _infer(self):
        raw_preds = self._base_model.predict_proba(self._lastX)[:, 1]
        preds = raw_preds * self.policy_temp
        preds -= preds.min()
        preds = np.exp(preds)
        preds /= preds.sum()
        self.policy = dict(zip(self.item_names, preds))
        return raw_preds

    def _decision_function(self):
        return self._base_model.decision_function(self._lastX)

    def predict(self):
        preds = self._infer()
        return preds.argsort()[-self._slate_size :]

    def step(self, reward, observation, replay=False):
        self._lastX = self._prepare_features(
            observation["user"], observation["doc"]
        )
        if replay:
            if not self.is_init:
                self._last_slate = np.random.choice(
                    range(self.nitems), size=self._slate_size, replace=False
                )
            else:
                self._last_slate = self.predict()
            return self._last_slate
        if observation.get("response") is None:
            self._last_slate = np.random.choice(
                range(self.nitems), size=self._slate_size, replace=False
            )
            return self._last_slate
        self._update(observation["response"])
        self._last_slate = self.predict()
        return self._last_slate


class AdaptiveGreedy(BaseIncrementalRecommenderAgent):
    def __init__(
        self,
        observation_space,
        action_space,
        base_model,
        threshold=0.5,
        decay=0.999,
    ):
        super(AdaptiveGreedy, self).__init__(
            observation_space, action_space, base_model
        )
        self.threshold = threshold
        self.decay = decay

    # def predict(self):
    #     preds = self._infer()
    #     if any(preds > self.threshold):
    #         self.threshold *= self.decay
    #         return preds.argsort()[-self._slate_size:]
    #     self.threshold *= self.decay
    #     return np.random.choice(range(self.nitems), size=self._slate_size, replace=False)

    def predict(self):
        preds = self._infer()
        pos = np.flatnonzero(preds > self.threshold)
        self.threshold *= self.decay
        if len(pos) >= self._slate_size:
            return preds.argsort()[-self._slate_size :]
        else:
            return (
                pos.tolist()
                + np.random.choice(
                    list(set(range(self.nitems)) - set(pos)),
                    size=self._slate_size - len(pos),
                    replace=False,
                ).tolist()
            )


class AdaptiveGreedySliding(BaseIncrementalRecommenderAgent):
    def __init__(
        self,
        observation_space,
        action_space,
        base_model,
        threshold=0.5,
        decay=0.999,
        window=500,
        pct=0.8,
    ):
        super(AdaptiveGreedySliding, self).__init__(
            observation_space, action_space, base_model
        )
        self.threshold = threshold
        self.decay = decay
        self.window = window
        self.pct = pct
        self.story = []

    def update_threshold(self, preds):
        self.story.append(np.sort(preds)[-self._slate_size :])
        if len(self.story) < self.window:
            return self.threshold
        self.threshold = np.quantile(
            np.array(self.story[-self.window :]),
            self.pct,
            interpolation="higher",
        )
        self.pct *= self.decay

    def predict(self):
        preds = self._infer()
        pos = np.flatnonzero(preds > self.threshold)
        self.update_threshold(preds)
        if len(pos) >= self._slate_size:
            return preds.argsort()[-self._slate_size :]
        else:
            return (
                pos.tolist()
                + np.random.choice(
                    list(set(range(self.nitems)) - set(pos)),
                    size=self._slate_size - len(pos),
                    replace=False,
                ).tolist()
            )


class EpsilonGreedy(BaseIncrementalRecommenderAgent):
    def __init__(
        self,
        observation_space,
        action_space,
        base_model,
        threshold=0.1,
        decay=0.997,
    ):
        super(EpsilonGreedy, self).__init__(
            observation_space, action_space, base_model
        )
        self.threshold = threshold
        self.decay = decay

    def predict(self):
        if np.random.random() > self.threshold:
            self.threshold *= self.decay
            preds = self._infer()
            return preds.argsort()[-self._slate_size :]
        self.threshold *= self.decay
        return np.random.choice(
            range(self.nitems), size=self._slate_size, replace=False
        )


class SoftmaxExplorer(BaseIncrementalRecommenderAgent):
    def __init__(
        self,
        observation_space,
        action_space,
        base_model,
        init_t=1,
        inflation=1.01,
        max_t=100,
    ):
        super(SoftmaxExplorer, self).__init__(
            observation_space, action_space, base_model
        )
        self.t = init_t
        self.max_t = max_t
        self.inflation = inflation

    def predict(self):
        preds = self._decision_function()
        sm = preds - preds.min()
        sm = sm * self.t
        sm = np.exp(sm)
        sm /= sm.sum()
        if self.max_t > self.t:
            self.t *= self.inflation
        else:
            self.t = self.max_t
        self.policy = dict(zip(self.item_names, sm))
        return np.random.choice(
            range(self.nitems), p=sm, size=self._slate_size, replace=False
        )


class BootstrapedUCB(BaseIncrementalRecommenderAgent):
    def __init__(
        self,
        observation_space,
        action_space,
        base_model,
        story_length=10,
        pct=0.8,
    ):
        super(BootstrapedUCB, self).__init__(
            observation_space, action_space, base_model
        )
        self.story_length = story_length
        self.story = []
        self.targets = []
        self.story_slates = []
        self.nmodels = len(base_model)
        self.pct = pct
        self.story_weights = []

    def update(self):
        self.is_init = True
        X = []
        for x, sl in zip(self.story, self.story_slates):
            X.append(x[sl])
        X = np.concatenate(X)
        y = self.targets.copy()
        wgt = np.array(self.story_weights)
        self.story = []
        self.targets = []
        self.story_slates = []
        self.story_weights = []
        for model in self._base_model:
            weights = np.random.gamma(1, 1, len(wgt)) * wgt
            model.partial_fit(X, y, classes=[0, 1], sample_weight=weights)

    def _infer(self):
        preds = []
        for model in self._base_model:
            preds.append(model.predict_proba(self._lastX)[:, 1])
        preds = np.array(preds)
        return np.quantile(preds, self.pct, axis=0, interpolation="higher")

    def predict(self):
        preds = self._infer()
        preds *= self.policy_temp
        preds -= preds.min()
        preds = np.exp(preds)
        preds /= preds.sum()
        self.policy = dict(zip(self.item_names, preds))
        return preds.argsort()[-self._slate_size :]

    def step(self, reward, observation, replay=False):
        self._lastX = self._prepare_features(
            observation["user"], observation["doc"]
        )
        if replay:
            if not self.is_init:
                self._last_slate = np.random.choice(
                    range(self.nitems), size=self._slate_size, replace=False
                )
            else:
                self._last_slate = self.predict()
            return self._last_slate
        if observation.get("response") is None:
            self._last_slate = np.random.choice(
                range(self.nitems), size=self._slate_size, replace=False
            )
            return self._last_slate
        if len(self.story) < self.story_length:
            self.story.append(self._lastX.copy())
            self.targets.extend([i["click"] for i in observation["response"]])
            self.story_weights.extend(
                [i.get("weight", 1) for i in observation["response"]]
            )
            self._last_slate = np.random.choice(
                range(self.nitems), size=self._slate_size, replace=False
            )
        else:
            self.update()
        if check_is_fitted(self._base_model[0]):
            self._last_slate = self.predict()
        else:
            self._last_slate = np.random.choice(
                range(self.nitems), size=self._slate_size, replace=False
            )
        self.story_slates.append(self._last_slate)
        return self._last_slate


class LinUCB(BaseIncrementalRecommenderAgent):
    def __init__(self, observation_space, action_space, base_model, delta=0.1):
        super(LinUCB, self).__init__(
            observation_space, action_space, base_model
        )
        self.n_feats = (
            observation_space.spaces["user"].shape[0]
            + observation_space.spaces["doc"].spaces["0"].shape[0]
        )
        self.alpha = 1 + np.sqrt(np.log(1 / delta) / 2)
        self._init_bandit()
        self.i = 0

    def _init_bandit(self):
        self.A = jnp.array([jnp.identity(self.n_feats) for i in range(self.nitems)])
        self.b = jnp.array([jnp.zeros(self.n_feats) for i in range(self.nitems)])
        self.calc_thetas()

    @staticmethod
    def _calc_theta(inv, b):
        return jnp.dot(inv, b)

    @staticmethod
    def _calc_inv(A):
        return jnp.linalg.inv(A)
        
    def calc_thetas(self):
        self.invA = vmap(self._calc_inv, in_axes=(0), out_axes=0)(self.A)
        self.thetas = vmap(self._calc_theta, in_axes=(0, 0), out_axes=0)(self.invA, self.b)       
                
    @staticmethod
    @partial(jit, static_argnums=(3, ))
    def _calc_p(t, x, a, alpha):
        return jnp.dot(t, x) + alpha * jnp.sqrt(jnp.dot(jnp.dot(x, a), x))
            
    def calc_p(self):
        t, x, a, alpha = self.thetas, self._lastX, self.A, self.alpha
        p = vmap(self._calc_p, in_axes=(0, 0, 0, None), out_axes=0)(t, x, a, alpha)
        return p
              
    def _update(self, response):
        for i, r in enumerate(response):
            if not r.get("shown", True):
                continue
            arm = self._last_slate[i]
            x = jnp.array(self._lastX[arm])
            self.b = jax.ops.index_update(self.b, jnp.index_exp[arm], (2 * r.get("click", 0) - 1) * x * r.get("weight", 1) + self.b[arm])
            self.A = jax.ops.index_update(self.A, jnp.index_exp[arm], jnp.outer(x, x) * r.get("weight", 1) + self.A[arm]) 
                      
        self.calc_thetas()

    def _infer(self):
        self.p = self.calc_p()
        preds = self.p
        
        if preds.sum() == 0:
            preds += 1
        preds /= preds.sum()
        self.policy = dict(zip(self.item_names, preds))
        return jnp.array(self.p)

    def step(self, reward, observation, replay=False):
        self._lastX = self._prepare_features(
            observation["user"], observation["doc"]
        )
        self._lastX = jnp.array(self._lastX)
        if replay or (observation.get("response") is None):
            self._last_slate = self.predict()
            return self._last_slate
        self._update(observation["response"])
        self._last_slate = self.predict()
        return self._last_slate


class LinTS(LinUCB):
    def __init__(
        self,
        observation_space,
        action_space,
        base_model,
        delta=0.1,
        optimism=1,
    ):
        super(LinTS, self).__init__(
            observation_space, action_space, base_model, delta
        )
        self.n_feats = (
            observation_space.spaces["user"].shape[0]
            + observation_space.spaces["doc"].spaces["0"].shape[0]
        )
        self.alpha = 1 + np.sqrt(np.log(1 / delta) / 2)
        self._init_bandit()
        self.optimism = optimism ** 2

    def _calc_p(self):
        self.p = []
        for idx in range(len(self.thetas)):
            t, x, a = self.thetas[idx], self._lastX[idx], self.invA[idx]
            tt = np.random.multivariate_normal(t, a)
            self.p.append(tt.dot(x))


class HLinUCB(BaseIncrementalRecommenderAgent):
    def __init__(
        self,
        observation_space,
        action_space,
        base_model,
        delta=0.1,
    ):
        super(HLinUCB, self).__init__(
            observation_space,
            action_space,
            base_model,
        )
        self.d = observation_space.spaces["user"].shape[0]
        self.k = self.d * observation_space.spaces["doc"].spaces["0"].shape[0] 
        self.alpha = 1 + np.sqrt(np.log(1 / delta) / 2)
        self._init_bandits()
        self._init_bandit()
        self.i = 0
        

    def _init_bandits(self):
        self.A0 = jnp.identity(self.k)
        self.b0 = jnp.zeros(self.k)
        self.A0_inv = self._calc_inv(self.A0)
        self.beta = jnp.dot(self.A0_inv, self.b0)
        self.betas = jnp.array([self.beta for i in range(self.nitems)])

    def _init_bandit(self):
        self.A = jnp.array([jnp.identity(self.d) for i in range(self.nitems)])
        self.B = jnp.array([jnp.zeros((self.d, self.k)) for i in range(self.nitems)])
        self.b = jnp.array([jnp.zeros(self.d) for i in range(self.nitems)])
        self.calc_thetas()
    
    @staticmethod
    def _calc_ba(b, B, beta):
        return b - jnp.dot(B, beta)
    
    @staticmethod
    def _calc_theta(inv, ba):
        return jnp.dot(inv, ba)

    @staticmethod
    def _calc_inv(A):
        return jnp.linalg.inv(A)

           
    def calc_thetas(self):
        self.invA = vmap(self._calc_inv, in_axes=(0), out_axes=0)(self.A)
        self.ba = vmap(self._calc_ba, in_axes=(0, 0, 0), out_axes=0)(self.b, self.B, self.betas)
        self.thetas = vmap(self._calc_theta, in_axes=(0, 0), out_axes=0)(self.invA, self.ba)
                
    @staticmethod 
    @partial(jit, static_argnums=())
    
    def calc_s(A0_inv, z, x, a, Ba):
        dot_a_x = jnp.dot(a, x)
        dot_products = jnp.dot(A0_inv, jnp.dot(Ba.T, dot_a_x))
        x_t = x.T
        return jnp.dot(z.T, jnp.dot(A0_inv, z)) \
        - 2 * jnp.dot(z.T, dot_products) \
            + jnp.dot(x_t, dot_a_x) \
            + jnp.dot(x_t, jnp.dot(a, jnp.dot(Ba, dot_products)))

    @staticmethod 
    @partial(jit, static_argnums=(5, ))
    
    def calc_p(z, be, te, x, s, alpha):
        return jnp.dot(z.T, be) + jnp.dot(x.T, te) + alpha * jnp.sqrt(s)       

    def _calc_s_p(self):
        A0_inv, betas, alpha = self.A0_inv, self.betas, self.alpha
        z, x = self._lastZ, self._lastX
        invA, Ba, thetas = self.invA, self.B, self.thetas 
        s = vmap(self.calc_s, in_axes=(None, 0, 0, 0, 0), out_axes=0)(A0_inv, z, x, invA, Ba)
        p = vmap(self.calc_p, in_axes=(0, 0, 0, 0, 0, None), out_axes=0)(z, betas, thetas, x, s, alpha)
        return p

    def _update(self, response):
        for i, r in enumerate(response):
            if not r.get("shown", True):
                continue
            arm = self._last_slate[i]
            print(arm)
            x = jnp.array(self._lastX[arm])
            z = jnp.array(self._lastZ[arm])
            
            # global rewards update phase 1 
            inv_A = self._calc_inv(self.A[arm])
            dot_B_inv_A = jnp.dot(self.B[arm].T, inv_A)
            self.A0 += jnp.dot(dot_B_inv_A, self.B[arm]) 
            self.b0 += jnp.dot(dot_B_inv_A, self.b[arm])
            # local rewards update
            self.b = jax.ops.index_update(self.b, jnp.index_exp[arm], (2 * r.get("click", 0) - 1) * x * r.get("weight", 1) + self.b[arm])
            self.A = jax.ops.index_update(self.A, jnp.index_exp[arm], jnp.outer(x, x) * r.get("weight", 1) + self.A[arm]) 
            self.B = jax.ops.index_update(self.B, jnp.index_exp[arm], jnp.outer(x, z) * r.get("weight", 1) + self.B[arm]) 
            # global rewards update phase 2
            inv_A = self._calc_inv(self.A[arm])
            dot_B_inv_A = jnp.dot(self.B[arm].T, inv_A)
            self.A0 = self.A0 + jnp.outer(z, z) * r.get("weight", 1) - jnp.dot(dot_B_inv_A, self.B[arm]) 
            self.b0 = self.b0 + (2 * r.get("click", 0) - 1) * z * r.get("weight", 1) \
            - jnp.dot(dot_B_inv_A, self.b[arm])
        self.A0_inv = self._calc_inv(self.A0)
        self.beta = jnp.dot(self.A0_inv, self.b0)
        self.betas = jnp.array([self.beta for i in range(self.nitems)])

        self.calc_thetas()
        print('===', self.i)
        self.i += 1
        print(np.sum(self.beta))
        print(np.sum(self.thetas))

    def _infer(self):
        self.p = self._calc_s_p()
        
        preds = self.p
        if preds.sum() == 0:
            preds += 1
        preds /= preds.sum()
        self.policy = dict(zip(self.item_names, preds))
        return self.p

    def step(self, reward, observation, replay=False):
        self._lastXZ = self._prepare_features(
            user=observation["user"],
            docs=observation["doc"],
        )
        self._lastX = jnp.array(self._lastXZ[:, :self.d])
        zz = jnp.array(self._lastXZ[:, self.d:])
        lastZ = vmap(jnp.outer, in_axes=(0), out_axes=0)(self._lastX, zz)
        self._lastZ = lastZ.reshape(-1, self.k)

        if replay or (observation.get("response") is None):
            self._last_slate = self.predict()
            return self._last_slate
        self._update(observation["response"])
        self._last_slate = self.predict()
        return self._last_slate



