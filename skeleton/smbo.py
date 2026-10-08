"""Optional SMBO interface for the shared forest search space.

Implement the method, connect a suitable package, or replace this interface.
Choose and explain the method's settings and how you retain search results.
"""

from __future__ import annotations
import numpy as np
from sklearn.ensemble import RandomForestRegressor


from typing import Any

from random_forest import Config, Evaluator, sample_configuration, make_classifier


def optimise_smbo(
    evaluator: Evaluator,
    n_trials: int,
    n_trees: int,
    seed: int,
) -> tuple[Config, Any]:
    """TODO: use SMBO to choose configurations based on previous evaluations.

    Use the shared search space and train each forest with n_trees trees.
    Use up to n_trials evaluations, including any initial evaluations.
    Select the best configuration using the validation objective, respecting
    whether higher or lower values are better.
    Return the selected configuration and results needed for your analysis.
    """

    attempted_config = []
    objectives = []

    n_initial_eval = min(5, n_trials)

    for n in range(n_initial_eval):
        rng = np.random.default_rng(seed + n)
        config = sample_configuration(rng)

        if config in attempted_config:
            continue

        result = evaluator(config, n_trees, seed + n)
        
        attempted_config.append(config)
        objectives.append(result["objective"])
        #print("Result bij smbo is ", result)

    for n in range(n_trials - n_initial_eval):
        X = np.array([
            [
                -1 if config["max_depth"] is None
                else config["max_depth"],

                0 if config["max_features"] == "sqrt"
                else config["max_features"],

                config["min_samples_leaf"],
            ]
            for config in attempted_config
        ])

        y = np.array(objectives)

        surrogate = RandomForestRegressor(n_estimators=1000, random_state=seed+n)

        surrogate.fit(X, y)

        candidates = []

        for _ in range(1000):

            candidate = sample_configuration(rng)

            if candidate in attempted_config:
                continue

            candidates.append(candidate)

        X_candidates = np.array([
                [
                    -1 if config["max_depth"] is None
                    else config["max_depth"],

                    0 if config["max_features"] == "sqrt"
                    else config["max_features"],

                    config["min_samples_leaf"],
                ]
                for config in candidates
            ])

        predictions = surrogate.predict(X_candidates)

        best_index = np.argmax(objectives)

        next_config = candidates[best_index]

        result = evaluator(next_config, n_trees, seed + n)


        attempted_config.append(next_config)
        objectives.append(result["objective"])

    best_index = np.argmax(objectives)

    return attempted_config[best_index], {
        "objective": objectives[best_index],
        "configurations": attempted_config,
        "objectives": objectives,
    }



        