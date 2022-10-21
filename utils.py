def check_is_fitted(estimator):
    if hasattr(estimator, "t_"):
        return True
    elif hasattr(estimator, "class_count_"):
        return True
    elif hasattr(estimator, "is_fitted"):
        return True
    return False


def aggregate_clicks_metrics(responses, metrics, info=None):
    """Aggregates the video cluster metrics with one step responses.

    Args:
      responses: a dictionary of names, observed responses.
      metrics: A dictionary mapping from metric_name to its value in float.
      info: Additional info for computing metrics (ignored here)

    Returns:
      A dictionary storing metrics after aggregation.
    """
    metrics["impression"] += sum(
        [response.get("shown", 1) for response in responses]
    )

    for response in responses:
        metrics["nis_down"] += response.get("weight", 1)
        if response["shown"]:
            metrics["nis_shown"] += response.get("weight", 1)
        if not response["click"]:
            continue
        metrics["click"] += 1
        metrics["nis_click"] += response.get("weight", 1)
        metrics["nis_click_shown"] += response.get("weight", 1) * response.get(
            "shown", 1
        )
    return metrics


def write_clicks_metrics(metrics, add_summary_fn):
    """Writes average video cluster metrics using add_summary_fn."""
    add_summary_fn("CTR", metrics["click"] / metrics["impression"])
    add_summary_fn("norm_CTR", metrics["nis_click"] / metrics["nis_shown"])
    add_summary_fn(
        "norm_CTR_shown", metrics["nis_click_shown"] / metrics["nis_shown"]
    )


def run_experiment(
    agent,
    env,
    Runner,
    base_model,
    mtype="AdaptiveGreedy",
    bm_params={},
    agent_params={},
    max_eval_episodes=1000,
    tmp_base_dir="./recsim/",
    max_steps_per_episode=1,
):
    def create_agent(sess, environment, eval_mode, summary_writer=None):
        if mtype in ["LinUCB", "LinTS", "HLinUCB"]:
            return agent(
                environment.observation_space,
                environment.action_space,
                base_model,
                **agent_params
            )
        elif mtype == "BootstrapedUCB":
            return agent(
                environment.observation_space,
                environment.action_space,
                [
                    base_model(**bm_params)
                    for i in range(bm_params.pop("nmodels", 10))
                ],
                **agent_params
            )
        return agent(
            environment.observation_space,
            environment.action_space,
            base_model(**bm_params),
            **agent_params
        )

    runner = Runner(
        max_steps_per_episode=max_steps_per_episode,
        max_eval_episodes=max_eval_episodes,
        base_dir=tmp_base_dir,
        create_agent_fn=create_agent,
        env=env,
        test_mode=True,
        method=mtype,
    )

    runner.run_experiment()
