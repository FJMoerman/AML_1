"""Optional interface for Random Search in the common forest space.

Implement this loop, connect a package, or use another organisation. Choose how
to retain the results needed to analyse search progress and computational effort.
"""

from __future__ import annotations
import numpy as np
from typing import Any
from sklearn.ensemble import RandomForestRegressor

from random_forest import Config, Evaluator, sample_configuration


def optimise_random_search(
    evaluator: Evaluator,
    n_trials: int,
    n_trees: int,
    seed: int,
) -> tuple[Config, Any]:
    """TODO: randomly sample and evaluate up to n_trials configurations.

    Use the shared search space and train each forest with n_trees trees.
    Select the best configuration using the validation objective, respecting
    whether higher or lower values are better.
    Return the selected configuration and results needed for your analysis.
    """
    attempted_config = []
    for n in range(n_trials):
        rng = np.random.default_rng(seed + n)
        config = sample_configuration(rng)
        result = evaluator(config, n_trees, seed + n)
        attempted_config.append(result)
    best_result = max(attempted_config, key=lambda result: result["objective"])
    return best_result["configuration"], attempted_config
    #raise NotImplementedError
