"""Optional Hyperband interface for random forests.

Implement the method, connect a suitable package, or replace this interface.
Choose and justify the allocation schedule and how you retain search results.
"""

from __future__ import annotations

from typing import Any
import math
from random_forest import Config, Evaluator, sample_configuration
import numpy as np

def optimise_hyperband(
    evaluator: Evaluator,
    min_trees: int,
    max_trees: int,
    seed: int,
    reduction_factor: int = 3,
) -> tuple[Config, Any]:
    """TODO: implement or configure multiple successive-halving brackets.

    Use the shared search space. Start brackets with different numbers of
    configurations and trees per forest, between min_trees and max_trees.
    At each stage, keep the better configurations and give them more trees,
    keeping other settings fixed. Use reduction_factor for the decrease in
    configuration count and increase in trees.
    Compare validation objectives consistently: respect whether higher or lower
    values are better. Explain your schedule, refitting or warm starts, and
    how validation results determine the final selection.
    Return the selected configuration and results needed for your analysis.
    """
    #Resource: per-configuration maximum, in units of the minimum resource
    R = max_trees // min_trees
    #
    s_max = math.floor(math.log(R,reduction_factor))
    #the total budget of one bracket  
    B = (s_max + 1) * R
    rng = np.random.default_rng(seed)

    #parameter to compare the scores to, to get best config
    best_score = -math.inf
    #parameter containing the best config so far
    best_config: Config | None = None
    results = []

    for s in range(s_max, -1, -1):
        n = math.ceil((B // R) * ((reduction_factor ** s) / (s + 1)))
        r = int(round(R * (reduction_factor ** (-s))))
        T = []
        for _ in range(n):
            T.append(sample_configuration(rng))
        for i in range(s + 1):
            n_i = math.floor(n * (reduction_factor**-i))
            r_i = int(round(r * (reduction_factor ** i)))
            n_trees = int(round(min_trees * r_i))
            score = []
            for config in T:
                result = evaluator(config, n_trees, seed)
                score.append(result)
                results.append(result)

            for result in score:
                if result["objective"] > best_score:
                    best_score = result["objective"]
                    best_config = result["configuration"]
            
            n_keep = math.floor(n_i / reduction_factor)
            T = [result["configuration"] for result in score[:n_keep]]

            score.sort(key=lambda result: result["objective"], reverse=True)
            T = [result["configuration"] for result in score[: math.floor(n_i / reduction_factor)]]

    return best_config, results
    #raise NotImplementedError
