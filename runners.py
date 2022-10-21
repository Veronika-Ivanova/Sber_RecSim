from dopamine.discrete_domains import checkpointer
from recsim.simulator.runner_lib import Runner
import tensorflow.compat.v1 as tf
import time
import os


# TODO: create wrappers for other models like LGBM, LightFM, etc
# TODO: realize BootstrapedTS and other methods from paper
# TODO: realize LinUCB (may not require, because GLMModels already exists here)


class ReplayRunner(Runner):
    def __init__(
        self,
        max_eval_episodes=125000,
        test_mode=False,
        min_interval_secs=30,
        train_base_dir=None,
        normalization=None,
        method=None,
        **kwargs,
    ):
        tf.logging.info("max_eval_episodes = %s", max_eval_episodes)
        tf.disable_eager_execution()
        super(ReplayRunner, self).__init__(**kwargs)
        self._max_eval_episodes = max_eval_episodes
        self._test_mode = test_mode
        self._min_interval_secs = min_interval_secs
        self.normalization = normalization
        nm = f"eval_{method}_{time.time()}"
        tf.logging.info(nm)
        self._output_dir = os.path.join(self._base_dir, nm)
        tf.io.gfile.makedirs(self._output_dir)
        if train_base_dir is None:
            train_base_dir = self._base_dir
        self._checkpoint_dir = os.path.join(train_base_dir, "train")
        tf.io.gfile.makedirs(self._checkpoint_dir)
        self._checkpoint_dir = os.path.join(train_base_dir, "checkpoints")

        self._set_up(eval_mode=True)

    def _run_one_episode(self):
        """Executes a full trajectory of the agent interacting with the environment.

        Returns:
          The number of steps taken and the total reward.
        """
        step_number = 0
        total_reward = 0.0

        start_time = time.time()
        sequence_example = tf.train.SequenceExample()
        observation = self._env.reset()
        action = self._agent.begin_episode(observation)

        # Keep interacting until we reach a terminal state.
        while True:
            last_observation = observation
            if hasattr(self._agent, "policy"):
                self._env._environment.set_agent_policy(self._agent.policy)
            observation, reward, done, info = self._env.step(action)

            if done:
                break
            if not sum([i["shown"] for i in observation["response"]]):
                if self._env._environment.game_over():
                    break
                observation = self._env.reset()
                if not any(observation["user"] == last_observation["user"]):
                    break
                action = self._agent.begin_episode(observation)
                continue
            _ = self._agent.step(reward, observation)
            self._log_one_step(
                last_observation["user"],
                last_observation["doc"],
                action,
                observation["response"],
                reward,
                done,
                sequence_example,
            )
            # Update environment-specific metrics with responses to the slate.
            self._env.update_metrics(observation["response"], info)

            total_reward += reward
            step_number += 1
            if done:
                break
            elif step_number == self._max_steps_per_episode:
                # Stop the run loop once we reach the true end of episode.
                break
            else:
                action = self._agent.step(reward, observation)
        self._agent.end_episode(reward, observation)
        if self._episode_writer is not None:
            self._episode_writer.write(sequence_example.SerializeToString())

        time_diff = time.time() - start_time
        self._update_episode_metrics(
            episode_length=step_number,
            episode_time=time_diff,
            episode_reward=total_reward,
        )
        return step_number, total_reward

    def _run_eval_phase(self, total_steps):
        """Runs evaluation phase given model has been trained for total_steps."""

        self._env.reset_sampler()
        self._initialize_metrics()

        num_episodes = 0
        episode_rewards = []

        while num_episodes < self._max_eval_episodes:
            _, episode_reward = self._run_one_episode()
            episode_rewards.append(episode_reward)
            num_episodes += 1
            if hasattr(self._env._environment, "game_over"):
                if self._env._environment.game_over():
                    break
            if ((num_episodes + 1) % 100) == 0:
                self._write_metrics(num_episodes, suffix="eval")

        self._write_metrics(num_episodes, suffix="eval")

        output_file = os.path.join(
            self._output_dir, "returns_%s" % total_steps
        )
        tf.logging.info("eval_file: %s", output_file)
        with tf.io.gfile.GFile(output_file, "w+") as f:
            f.write(str(episode_rewards))

    def run_experiment(self):
        tf.logging.info("Beginning evaluation...")
        self._checkpointer = checkpointer.Checkpointer(
            self._checkpoint_dir, self._checkpoint_file_prefix
        )
        self._run_eval_phase(1000)
