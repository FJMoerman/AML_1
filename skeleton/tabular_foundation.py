"""Optional interface for a pre-trained model on the largest selected dataset.

Use a suitable package directly, adapt this interface, or organise your own experiment.
"""

from __future__ import annotations
from typing import Any
import numpy as np
from data_loading import DataSplits
from tabpfn import TabPFNClassifier, TabPFNRegressor
from time import perf_counter
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

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
    X_train = splits.X_train[:1024]
    y_train = splits.y_train[:1024]

    X_test = splits.X_test
    y_test = splits.y_test

    start = perf_counter()
    clf = TabPFNClassifier(random_state=seed, device="cuda", show_progress_bar=True)
    clf.fit(X_train, y_train)
    fit_seconds = perf_counter() - start

    start = perf_counter()
    batch_size = 1024
    predictions = []

    for start_idx in range(0, len(X_test), batch_size):
        end_idx = start_idx + batch_size
        X_batch = X_test[start_idx:end_idx]

        batch_predictions = clf.predict(X_batch)
        predictions.append(batch_predictions)

    pred_y_test = np.concatenate(predictions)
    #pred_y_test = clf.predict(X_test)
    prediction_seconds = perf_counter() - start

    metrics = {
        "accuracy": accuracy_score(y_test, pred_y_test),
        "precision": precision_score(
            y_test, pred_y_test, average="macro", zero_division=0
        ),
        "recall": recall_score(
            y_test, pred_y_test, average="macro", zero_division=0
        ),
        "f1": f1_score(
            y_test, pred_y_test, average="macro", zero_division=0
        ),
        "total_seconds": fit_seconds + prediction_seconds,
        "fit_seconds": fit_seconds,
        "prediction_seconds": prediction_seconds
    }

    return metrics

    #return {
    #    "predictions": pred_y_test,
    #    "y_test": splits.y_test,
    #    "fit_seconds": fit_seconds,
    #    "prediction_seconds": prediction_seconds,
    #    "total_seconds": fit_seconds + prediction_seconds,
    #    "context_size": len(X_train),
    #}
    #data --> statified sampling
    raise NotImplementedError
