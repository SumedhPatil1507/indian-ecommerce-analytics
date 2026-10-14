"""Heterogeneous relational GNN for customer collusion and RTO risk.

The graph is built from transaction-level records. Target columns are used only
as customer labels and never enter the feature matrix.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

try:
    import torch
    from torch import nn
    import torch.nn.functional as F
    from torch_geometric.data import HeteroData
    from torch_geometric.nn import HeteroConv, SAGEConv
except ImportError as exc:  # Optional graph dependencies
    torch = nn = F = HeteroData = HeteroConv = SAGEConv = None
    _PYG_IMPORT_ERROR = exc
else:
    _PYG_IMPORT_ERROR = None

NODE_TYPES = ("customer", "merchant", "device_id", "ip_address")
ID_COLUMNS = {
    "customer": "customer_id",
    "merchant": "merchant_id",
    "device_id": "device_id",
    "ip_address": "ip_address",
}
EDGE_TYPES = (
    ("customer", "transacted_with", "merchant"),
    ("merchant", "serves", "customer"),
    ("customer", "used", "device_id"),
    ("device_id", "used_by", "customer"),
    ("customer", "connected_from", "ip_address"),
    ("ip_address", "connected_customer", "customer"),
)
LABEL_COLUMNS = ("collusion_label", "rto_label")


def _require_pyg() -> None:
    if _PYG_IMPORT_ERROR is not None:
        raise ImportError("Fraud GNN requires torch and torch_geometric; install requirements-advanced.txt") from _PYG_IMPORT_ERROR


@dataclass
class FraudGraphBundle:
    data: Any
    node_ids: dict[str, list[str]]
    customer_labels: Any | None
    customer_label_masks: Any | None


def build_fraud_graph(transactions: pd.DataFrame) -> FraudGraphBundle:
    """Build a customer-context graph and masked collusion/RTO labels.

    Required identifiers are customer_id, merchant_id, device_id and
    ip_address. Missing contextual IDs are made row-specific to avoid creating
    artificial shared-device or shared-IP edges for null values.
    """
    _require_pyg()
    required = tuple(ID_COLUMNS.values())
    missing = [column for column in required if column not in transactions.columns]
    if missing:
        raise ValueError(f"Missing required graph identifiers: {', '.join(missing)}")
    if transactions.empty:
        raise ValueError("At least one transaction is required to build a fraud graph")

    frame = transactions.copy().reset_index(drop=True)
    frame["customer_id"] = frame["customer_id"].astype("string")
    if frame["customer_id"].isna().any() or (frame["customer_id"].str.strip() == "").any():
        raise ValueError("customer_id must be populated for every transaction")
    for column in required[1:]:
        values = frame[column].astype("string")
        missing_values = values.isna() | (values.str.strip() == "")
        values.loc[missing_values] = [f"__missing_{column}_{i}" for i in frame.index[missing_values]]
        frame[column] = values.astype(str)
    for column in ("amount", "transaction_amount"):
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
    amount_column = "amount" if "amount" in frame else "transaction_amount"
    if amount_column not in frame:
        frame["amount"] = 0.0
        amount_column = "amount"
    if (frame[amount_column] < 0).any():
        raise ValueError("transaction amounts cannot be negative")

    node_ids = {kind: sorted(frame[column].unique().tolist()) for kind, column in ID_COLUMNS.items()}
    index = {kind: {value: i for i, value in enumerate(ids)} for kind, ids in node_ids.items()}
    data = HeteroData()

    customer_stats = frame.groupby("customer_id", sort=True).agg(
        transaction_count=("customer_id", "size"),
        mean_amount=(amount_column, "mean"),
    ).reindex(node_ids["customer"]).fillna(0.0)
    customer_features = customer_stats[["transaction_count", "mean_amount"]].to_numpy(dtype=np.float32)
    customer_features[:, :2] = np.log1p(np.maximum(customer_features[:, :2], 0))
    data["customer"].x = torch.tensor(customer_features, dtype=torch.float32)

    for kind, column in list(ID_COLUMNS.items())[1:]:
        counts = frame[column].value_counts().reindex(node_ids[kind]).fillna(0).to_numpy(dtype=np.float32)
        data[kind].x = torch.tensor(np.log1p(counts).reshape(-1, 1), dtype=torch.float32)

    relations = (
        ("transacted_with", "merchant_id", "merchant"),
        ("used", "device_id", "device_id"),
        ("connected_from", "ip_address", "ip_address"),
    )
    for relation, target_column, target_type in relations:
        pairs = frame[["customer_id", target_column]].drop_duplicates()
        edge_index = torch.tensor([
            pairs["customer_id"].map(index["customer"]).to_numpy(dtype=np.int64),
            pairs[target_column].map(index[target_type]).to_numpy(dtype=np.int64),
        ], dtype=torch.long)
        data[("customer", relation, target_type)].edge_index = edge_index
        reverse_name = {"transacted_with": "serves", "used": "used_by", "connected_from": "connected_customer"}[relation]
        data[(target_type, reverse_name, "customer")].edge_index = edge_index.flip(0)

    labels = masks = None
    if any(column in frame for column in LABEL_COLUMNS):
        label_data = pd.DataFrame(index=frame.index)
        for column in LABEL_COLUMNS:
            label_data[column] = pd.to_numeric(frame[column], errors="coerce") if column in frame else np.nan
        grouped = label_data.assign(customer_id=frame["customer_id"]).groupby("customer_id")[list(LABEL_COLUMNS)].max()
        grouped = grouped.reindex(node_ids["customer"])
        observed = grouped.stack(dropna=True)
        if not observed.isin([0, 1]).all():
            raise ValueError("collusion_label and rto_label must contain only binary 0/1 labels")
        masks = torch.tensor(grouped.notna().to_numpy(), dtype=torch.bool)
        labels = torch.tensor(grouped.fillna(0).to_numpy(dtype=np.float32), dtype=torch.float32)
    return FraudGraphBundle(data, node_ids, labels, masks)


if nn is not None:
    class HeterogeneousFraudRGCN(nn.Module):
        """Two relation-specific heterogeneous convolution layers and risk heads."""

        def __init__(self, hidden_channels: int = 64, dropout: float = 0.2):
            super().__init__()
            self.projections = nn.ModuleDict({
                "customer": nn.Linear(2, hidden_channels),
                "merchant": nn.Linear(1, hidden_channels),
                "device_id": nn.Linear(1, hidden_channels),
                "ip_address": nn.Linear(1, hidden_channels),
            })
            self.conv1 = HeteroConv({edge: SAGEConv((hidden_channels, hidden_channels), hidden_channels) for edge in EDGE_TYPES}, aggr="sum")
            self.conv2 = HeteroConv({edge: SAGEConv((hidden_channels, hidden_channels), hidden_channels) for edge in EDGE_TYPES}, aggr="sum")
            self.dropout = dropout
            self.collusion_head = nn.Linear(hidden_channels, 1)
            self.rto_head = nn.Linear(hidden_channels, 1)

        def forward(self, x_dict: dict[str, Any], edge_index_dict: dict[Any, Any]):
            hidden = {kind: F.relu(self.projections[kind](features)) for kind, features in x_dict.items()}
            hidden = self.conv1(hidden, edge_index_dict)
            hidden = {kind: F.dropout(F.relu(value), p=self.dropout, training=self.training) for kind, value in hidden.items()}
            hidden = self.conv2(hidden, edge_index_dict)
            embeddings = {kind: F.relu(value) for kind, value in hidden.items()}
            customer_embedding = embeddings["customer"]
            probabilities = {
                "collusion_probability": torch.sigmoid(self.collusion_head(customer_embedding)).squeeze(-1),
                "rto_probability": torch.sigmoid(self.rto_head(customer_embedding)).squeeze(-1),
            }
            return probabilities, embeddings
else:
    class HeterogeneousFraudRGCN:  # Import-safe when optional dependencies are absent
        def __init__(self, *args: Any, **kwargs: Any):
            _require_pyg()
