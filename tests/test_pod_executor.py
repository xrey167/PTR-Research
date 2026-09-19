import json
from pathlib import Path

import pytest

from neural_pods.pod_executor import (ExecutorFactory, GovernedExecutorPool,
                                      PodExecutor, TreeExecutor)


@pytest.fixture(scope="module")
def model_path(tmp_path_factory):
    import xgboost as xgb
    import numpy as np
    rng = np.random.default_rng(7)
    X = rng.random((200, 4))
    y = (X[:, 0] + X[:, 1] > 1.0).astype(int)
    model = xgb.XGBClassifier(n_estimators=8, max_depth=3, eval_metric="logloss")
    model.fit(X, y)
    path = tmp_path_factory.mktemp("xgb") / "tiny.json"
    model.get_booster().save_model(str(path))
    return path


def test_tree_executor_deterministic_inference(model_path):
    executor = TreeExecutor(model_path)
    executor.activate()
    payload = {"features": [0.9, 0.8, 0.1, 0.2]}
    first = executor.infer(payload)
    second = executor.infer(payload)
    assert first == second  # deterministic — verifiable by the guard
    assert first["prediction"] in (0.0, 1.0)
    assert executor.stats()["inferences"] == 2
    executor.release()
    assert executor.health() is False


def test_infer_without_activate_raises(model_path):
    executor = TreeExecutor(model_path)
    with pytest.raises(RuntimeError, match="not activated"):
        executor.infer({"features": [1, 2, 3, 4]})


def test_factory_registers_and_creates(model_path):
    factory = ExecutorFactory()
    factory.register("xgboost", lambda: TreeExecutor(model_path))
    factory.register("xgboost2", lambda: TreeExecutor(model_path))
    assert factory.kinds() == ("xgboost", "xgboost2")
    executor = factory.create("xgboost")
    executor.activate()
    assert executor.runtime == "xgboost"
    with pytest.raises(KeyError):
        factory.create("unknown-runtime")
    with pytest.raises(ValueError):
        factory.register("xgboost", lambda: TreeExecutor(model_path))


def test_governed_pool_budget_enforced(model_path):
    factory = ExecutorFactory()
    factory.register("xgboost", lambda: TreeExecutor(model_path, ram_bytes=100))
    pool = GovernedExecutorPool(factory, budget_bytes=250)
    pool.activate("pod-1", "xgboost")
    pool.activate("pod-2", "xgboost")
    with pytest.raises(RuntimeError, match="budget exhausted"):
        pool.activate("pod-3", "xgboost")
    pool.release("pod-1")
    pool.activate("pod-3", "xgboost")  # freed budget allows activation
    assert pool.stats()["active_pods"] == ["pod-2", "pod-3"]


def test_predict_gate_shape(model_path):
    """The gate check pattern: deterministic replay, 0 leases after."""
    executor = TreeExecutor(model_path)
    executor.activate()
    payload = {"features": [0.3, 0.2, 0.9, 0.1]}
    outputs = {json.dumps(executor.infer(payload), sort_keys=True) for _ in range(5)}
    assert len(outputs) == 1
    executor.release()
