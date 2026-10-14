"""Tests for the optional customer/merchant/device/IP heterogeneous GNN."""
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from src.models.gnn_fraud import EDGE_TYPES, HeterogeneousFraudRGCN, build_fraud_graph


@pytest.fixture
def transactions():
    return pd.DataFrame({
        "customer_id": ["c1", "c1", "c2", "c3"],
        "merchant_id": ["m1", "m2", "m1", "m3"],
        "device_id": ["d-shared", "d-a", "d-shared", "d-c"],
        "ip_address": ["ip-shared", "ip-a", "ip-shared", "ip-c"],
        "amount": [10.0, 12.0, 30.0, 8.0],
        "collusion_label": [1, 1, 1, 0],
        "rto_label": [0, 1, 0, 0],
    })


def test_builder_creates_expected_node_types_relations_and_masked_labels(transactions):
    graph = build_fraud_graph(transactions)
    assert set(graph.node_ids) == {"customer", "merchant", "device_id", "ip_address"}
    assert set(graph.data.edge_index_dict) == set(EDGE_TYPES)
    assert graph.data["customer"].x.shape == (3, 2)
    assert graph.customer_labels.shape == (3, 2)
    assert graph.customer_label_masks.all()
    # Target labels do not enter node features.
    assert graph.data["customer"].x.shape[1] == 2


def test_model_returns_customer_risk_probabilities_and_all_node_embeddings(transactions):
    graph = build_fraud_graph(transactions)
    model = HeterogeneousFraudRGCN(hidden_channels=8, dropout=0.0)
    probabilities, embeddings = model(graph.data.x_dict, graph.data.edge_index_dict)
    assert probabilities["collusion_probability"].shape == (3,)
    assert probabilities["rto_probability"].shape == (3,)
    assert set(embeddings) == set(graph.node_ids)
    assert all(torch.isfinite(value).all() for value in probabilities.values())


def test_builder_isolates_missing_context_ids(transactions):
    transactions.loc[0, "device_id"] = None
    transactions.loc[1, "device_id"] = None
    graph = build_fraud_graph(transactions)
    assert "__missing_device_id_0" in graph.node_ids["device_id"]
    assert "__missing_device_id_1" in graph.node_ids["device_id"]


def test_builder_rejects_non_binary_labels(transactions):
    transactions.loc[0, "rto_label"] = 2
    with pytest.raises(ValueError, match="binary 0/1"):
        build_fraud_graph(transactions)


def test_builder_requires_identifiers(transactions):
    with pytest.raises(ValueError, match="ip_address"):
        build_fraud_graph(transactions.drop(columns="ip_address"))
