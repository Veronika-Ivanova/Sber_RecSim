import collections
import numpy as np
from gym import spaces
from cbbench import utils
from recsim import document, user
from recsim.simulator import environment, recsim_gym


class DatasetDocument(document.AbstractDocument):
    def __init__(self, doc_id, features):
        self.features = features
        super(DatasetDocument, self).__init__(doc_id)

    def create_observation(self):
        return self.features

    def observation_space(self):
        return spaces.Box(
            -np.inf, np.inf, shape=(self.features.shape[0],), dtype=np.float32
        )

    def __str__(self):
        return "Document {}.".format(self._doc_id)


class DatasetDocumentSampler(document.AbstractDocumentSampler):
    def __init__(self, doc_data, doc_ctor=DatasetDocument, seed=0, **kwargs):
        super(DatasetDocumentSampler, self).__init__(doc_ctor, **kwargs)
        self._doc_count = doc_data.shape[0]
        self.doc_data = doc_data
        np.random.seed(seed)
        self.num = 0

    def sample_document(self):
        doc_features = {}
        doc_features["doc_id"] = self.num
        doc_features["features"] = self.doc_data[doc_features["doc_id"]]
        self.num += 1
        if self.num == self._doc_count:
            self.num = 0
        return self._doc_ctor(**doc_features)


class DatasetUserState(user.AbstractUserState):
    def __init__(self, user_id, features, positives, show_id=False):
        self.features = features
        self.user_id = user_id
        self.positives = positives
        self.show_id = show_id

    def is_terminal(self):
        return not len(self.positives)

    def create_observation(self):
        if self.show_id:
            return self.user_id, self.features
        else:
            return self.features

    def observation_space(self):
        return spaces.Box(
            -np.inf, np.inf, shape=(self.features.shape[0],), dtype=np.float32
        )

    def score_document(self, doc_obs):
        return 0


class DatasetUserSampler(user.AbstractUserSampler):
    def __init__(
        self,
        user_data,
        positives,
        user_ctor=DatasetUserState,
        seed=0,
        **kwargs
    ):
        super(DatasetUserSampler, self).__init__(user_ctor, **kwargs)
        self.user_data = user_data
        self.positives = positives
        self._user_count = user_data.shape[0]
        np.random.seed(seed)

    def sample_user(self):
        user_features = {}
        user_features["user_id"] = np.random.randint(self._user_count)
        user_features["features"] = self.user_data[user_features["user_id"]]
        interactions = self.positives[user_features["user_id"]]
        cands = set(np.flatnonzero(interactions))
        user_features["positives"] = set(cands)
        # user_features['negatives'] = set(neg)
        return self._user_ctor(**user_features)


class DatasetResponse(user.AbstractResponse):
    def __init__(self, clicked=False, shown=True, weight=1):
        self.clicked = clicked
        self.shown = shown
        self.weight = weight

    def create_observation(self):
        return {
            "click": int(self.clicked),
            "shown": int(self.shown),
            "weight": self.weight,
        }

    @classmethod
    def response_space(cls):
        return spaces.Dict(
            {
                "click": spaces.Discrete(2),
                "shown": spaces.Discrete(2),
                "weight": spaces.Box(low=-np.inf, high=np.inf, shape=(1,)),
            }
        )


class DatasetChoiceModel(object):
    def __init__(self, positives):
        self.positives = positives

    def choose_item(self, user_state, doc_obs):
        return self.positives[user_state.user_id, doc_obs]


class DatasetUserModel(user.AbstractUserModel):
    def __init__(self, slate_size, user_data, positives, seed=0):
        super(DatasetUserModel, self).__init__(
            DatasetResponse,
            DatasetUserSampler(
                user_data, positives, DatasetUserState, seed=seed
            ),
            slate_size,
        )
        self.choice_model = DatasetChoiceModel(positives)

    def simulate_response(self, slate_documents):
        rewards = self.choice_model.choose_item(
            self._user_state, [doc.doc_id() for doc in slate_documents]
        )
        responses = self._generate_response(rewards)
        return responses

    def _generate_response(self, rewards):
        responses = []
        for i in rewards:
            responses.append(
                self._response_model_ctor(clicked=(i > 0), shown=(i != 0))
            )
        return responses

    def update_state(self, slate_documents, responses):
        doc_ids = [doc.doc_id() for doc in slate_documents]
        self._user_state.positives -= set(doc_ids)

    def is_terminal(self):
        return self._user_state.is_terminal()


def total_clicks_reward(responses):
    reward = 0.0
    for r in responses:
        reward += r.clicked
    return reward


class OrderedDatasetEnvironment(environment.SingleUserEnvironment):
    def __init__(
        self,
        dataset,
        user_data,
        user_model,
        document_sampler,
        num_candidates,
        slate_size,
        norm_policy=None,
        cap=None,
        resample_documents=False,
    ):
        super(OrderedDatasetEnvironment, self).__init__(
            user_model,
            document_sampler,
            num_candidates,
            slate_size,
            resample_documents,
        )
        self.dataset = dataset
        self.user_data = user_data
        self.current_row = 0
        self._state_clicked = set()
        self._state_shown = set()
        self.norm_policy = norm_policy
        self.agent_policy = None
        self.running_norm = 0
        self.cap = cap

    def normalize_obs(self, doc_id):
        if (self.norm_policy is None) or (self.agent_policy is None):
            return self.cap or 1
        else:
            w = self.agent_policy[doc_id] / self.norm_policy[doc_id]
            return min(w, self.cap) if self.cap is not None else w

    def set_agent_policy(self, agent_policy):
        self.agent_policy = agent_policy

    def game_over(self):
        return self.current_row + 1 >= self.dataset.shape[0]

    def _gen_obs(self, idx):
        user_id, self._state_clicked, self._state_shown = self.dataset[idx]
        user_obs = self.user_data[user_id]
        return user_obs

    def reset(self):
        user_obs = self._gen_obs(self.current_row)
        self.current_row += 1
        if not hasattr(self, "_current_documents"):
            self._current_documents = collections.OrderedDict(
                self._candidate_set.create_observation()
            )
        return user_obs, self._current_documents

    def reset_sampler(self):
        self._document_sampler.reset_sampler()
        self.current_row = 0

    def update_running_norm(self, responses):
        for response in responses:
            if response.shown:
                self.running_norm += response.weight

    def normalize_weights(self, responses):
        self.update_running_norm(responses)
        if self.running_norm == 0:
            return responses
        for response in responses:
            response.weight = response.weight / self.running_norm
        return responses

    def step(self, slate):
        assert (
            len(slate) <= self._slate_size
        ), "Received unexpectedly large slate size: expecting %s, got %s" % (
            self._slate_size,
            len(slate),
        )

        # Get the documents associated with the slate
        doc_ids = list(self._current_documents)
        mapped_slate = [doc_ids[x] for x in slate]
        # Simulate the user's response
        responses = [
            DatasetResponse(
                **{
                    "clicked": int(i) in self._state_clicked,
                    "shown": int(i) in self._state_shown,
                    "weight": self.normalize_obs(i),
                }
            )
            for i in mapped_slate
        ]
        responses = self.normalize_weights(responses)
        # Obtain next user state observation.
        if self.current_row + 1 >= self.dataset.shape[0]:
            return None, None, responses, True
        user_obs = self._gen_obs(self.current_row)
        # Check if reaches a terminal state and return.
        done = (self.current_row + 2) == self.dataset.shape[0]
        return user_obs, self._current_documents, responses, done


def create_dataset_environment(
    slate_size, user_data, doc_data, interactions, resample_documents=False
):
    env = environment.Environment(
        DatasetUserModel(slate_size, user_data, interactions),
        DatasetDocumentSampler(doc_data),
        doc_data.shape[0],
        slate_size,
        resample_documents=resample_documents,
    )
    return recsim_gym.RecSimGymEnv(
        env,
        total_clicks_reward,
        utils.aggregate_clicks_metrics,
        utils.write_clicks_metrics,
    )


def create_ordered_dataset_environment(
    slate_size,
    user_data,
    doc_data,
    interactions,
    norm_policy=None,
    cap=None,
    resample_documents=False,
):
    env = OrderedDatasetEnvironment(
        interactions,
        user_data,
        DatasetUserModel(slate_size, user_data, interactions),
        DatasetDocumentSampler(doc_data),
        doc_data.shape[0],
        slate_size,
        norm_policy=norm_policy,
        cap=cap,
        resample_documents=resample_documents,
    )
    return recsim_gym.RecSimGymEnv(
        env,
        total_clicks_reward,
        utils.aggregate_clicks_metrics,
        utils.write_clicks_metrics,
    )
