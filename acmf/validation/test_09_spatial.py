import numpy as np
from acmf.model.parameters import ModelParameters
from acmf.model.contagion import compute_normalized_network_weights, compute_spatial_spillover
from acmf.validation.result import TestResult


def run_test_09(params: ModelParameters) -> TestResult:
    """TEST 09 — Пространственный сетевой переток (Spillover) и волны заражения."""
    n_nodes = 4
    # Линейная топология: 0 <-> 1 <-> 2 <-> 3
    raw_adj = np.array([
        [0.0, 1.0, 0.0, 0.0],
        [1.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 0.0],
    ], dtype=np.float64)

    j_matrix = compute_normalized_network_weights(raw_adj)

    # Узел 0 инфицирован высоким дефицитом SID, остальные здоровы (SID = 0)
    all_sids = np.zeros((n_nodes, 3), dtype=np.float64)
    all_sids[0, :] = 2.5

    spillover_node1 = compute_spatial_spillover(1, all_sids, j_matrix, params)
    spillover_node3 = compute_spatial_spillover(3, all_sids, j_matrix, params)

    val_node1 = float(np.mean(spillover_node1))
    val_node3 = float(np.mean(spillover_node3))

    # Переток в прямой соседний узел 1 должен быть доминирующим (разница > 100x)
    is_spillover_correct = (val_node1 > 0.05) and (val_node1 > 100.0 * val_node3)
    status = "PASSED" if is_spillover_correct else "FAILED"

    return TestResult(
        test_id="TEST_09",
        name="Spatial Network Contagion and Spillover",
        status=status,
        details={
            "spillover_node1": [float(x) for x in spillover_node1],
            "spillover_node3": [float(x) for x in spillover_node3],
            "spillover_ratio": float(val_node1 / (val_node3 + 1e-12)),
        },
    )