"""Optional interface for a pre-trained model on the largest selected dataset.

Use a suitable package directly, adapt this interface, or organise your own experiment.
"""

from __future__ import annotations
from typing import Any
import numpy as np
from data_loading import DataSplits
from tabpfn import TabPFNClassifier, TabPFNRegressor
from time import perf_counter

def run_foundation_model(splits: DataSplits, seed: int) -> Any:
    """TODO: evaluate a pre-trained tabular foundation model.

    Choose and justify the model, how you use it, and the data available to it.
    Evaluate on the same complete test set as the forests, keeping it separate
    from training, adaptation, and model selection.
    Retain results for comparing performance and compute with the baseline
    and each tuned forest. See Section 3.5 of the assignment.
    """
    #no pretraining --> too many parameters --> much work
    #in context learning
    X_train = splits.X_train
    y_train = splits.y_train

    X_test = splits.X_test
    y_test = splits.y_test

    start = perf_counter()
    clf = TabPFNClassifier(random_state=seed)
    clf.fit(X_train, y_train)
    fit_seconds = perf_counter() - start

    start = perf_counter()
    pred_y_test = clf.predict(X_test)
    prediction_seconds = perf_counter() - start

    return {
        "predictions": pred_y_test,
        "y_test": splits.y_test,
        "fit_seconds": fit_seconds,
        "prediction_seconds": prediction_seconds,
        "total_seconds": fit_seconds + prediction_seconds,
        "context_size": len(X_train),
    }
    #data --> statified sampling
    raise NotImplementedError
