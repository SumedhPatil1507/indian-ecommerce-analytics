"""Heterogeneous R-GCN style model for buyer RTO and promotion fraud risk.

PyTorch Geometric is an optional dependency. The tabular analytics app remains
usable when it is absent or when no trained checkpoint has been configured.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import os

import numpy as np
import pandas as pd

try:
    import torch
    from torch import nn
    import torch.nn.functional as F
    from torch_geometric.data import HeteroData
    from torch_geometric.nn import HeteroConv, SAGEConv
except ImportError as exc:  # pragma: no cover - depends on optional runtime
    torch = nn = F = HeteroData = HeteroConv = SAGEConv = None
    _PYG_IMPORT_ERROR = exc
else:
    _PYG_IMPORT_ERROR = None


NODE_TYPES = ("buyer", "pincode", "device_id", "payment_method")
EDGE_TYPES = (
    ("buyer", "placed_order", "payment_method"),
    ("buyer", "delivered_to", "pincode"),
    ("buyer", "used_device", "device_id"),
    ("payment_method", "rev_placed_order", "buyer"),
    ("pincode", "rev_delivered_to", "buyer"),
    ("device_id", "rev_used_device", "buyer"),
)
ID_COLUMNS = {
    "buyer": "buyer_id", "pincode": "pincode",
    "device_id": "device_id", "payment_method": "payment_method",
}
NUMERIC_FEATURES = ("order_count", "mean_discount", "mean_revenue", "mean_units")


def _require_pyg() -> None:
    if _PYG_IMPORT_ERROR is not None:
        raise ImportError("RTO graph modeling requires torch and torch_geometric; install the graph extra") from _PYG_IMPORT_ERROR


@dataclass
class GraphBundle:
    data: Any
    node_ids: dict[str, list[str]]
    buyer_labels: Any | None
    buyer_label_masks: Any | None


def build_heterogeneous_graph(orders: pd.DataFrame) -> GraphBundle:
    """Create buyer-to-context edges and leakage-safe buyer input aggregates.

    Labels are read from explicit ``rto_label`` and ``promo_fraud_label`` columns;
    they are never included in node features.
    """
    _require_pyg()
    required = list(ID_COLUMNS.values())
    missing = [col for col in required if col not in orders.columns]
    if missing:
        raise ValueError(f"Missing graph identifiers: {', '.join(missing)}")
    frame = orders.copy()
    for col in ("discount_percent", "revenue", "units_sold"):
        if col not in frame:
            frame[col] = 0.0
        frame[col] = pd.to_numeric(frame[col], errors="coerce").fillna(0.0)
    for col in required:
        frame[col] = frame[col].fillna("unknown").astype(str)

    node_ids = {kind: sorted(frame[col].unique().tolist()) for kind, col in ID_COLUMNS.items()}
    index = {kind: {value: idx for idx, value in enumerate(ids)} for kind, ids in node_ids.items()}
    data = HeteroData()
    buyer_agg = frame.groupby("buyer_id").agg(
        order_count=("buyer_id", "size"), mean_discount=("discount_percent", "mean"),
        mean_revenue=("revenue", "mean"), mean_units=("units_sold", "mean"),
    ).reindex(node_ids["buyer"]).fillna(0)
    bx = buyer_agg.loc[:, NUMERIC_FEATURES].to_numpy(dtype=np.float32)
    bx[:, 0] = np.log1p(bx[:, 0])
    bx[:, 1] /= 100.0
    bx[:, 2:] = np.log1p(np.maximum(bx[:, 2:], 0))
    data["buyer"].x = torch.tensor(bx, dtype=torch.float32)

    # Context nodes have frequency features, independent of any risk outcome.
    for kind, col in list(ID_COLUMNS.items())[1:]:
        counts = frame[col].value_counts().reindex(node_ids[kind]).fillna(0).to_numpy(dtype=np.float32)
        data[kind].x = torch.tensor(np.log1p(counts).reshape(-1, 1), dtype=torch.float32)

    for relation, target_col in (("placed_order", "payment_method"), ("delivered_to", "pincode"), ("used_device", "device_id")):
        pairs = frame[["buyer_id", target_col]].drop_duplicates()
        target = {"placed_order": "payment_method", "delivered_to": "pincode", "used_device": "device_id"}[relation]
        edge_index = torch.tensor([
            pairs["buyer_id"].map(index["buyer"]).to_numpy(),
            pairs[target_col].map(index[target]).to_numpy(),
        ], dtype=torch.long)
        data[("buyer", relation, target)].edge_index = edge_index
        data[(target, f"rev_{relation}", "buyer")].edge_index = edge_index.flip(0)

    labels = masks = None
    label_columns = ("rto_label", "promo_fraud_label")
    if all(col in frame for col in label_columns):
        grouped = frame.groupby("buyer_id")[list(label_columns)].max().reindex(node_ids["buyer"])
        masks_np = grouped.notna().to_numpy()
        labels_np = grouped.fillna(0).to_numpy(dtype=np.float32)
        labels = torch.tensor(labels_np, dtype=torch.float32)
        masks = torch.tensor(masks_np, dtype=torch.bool)
    return GraphBundle(data, node_ids, labels, masks)


if nn is not None:
    class RTOHeteroGNN(nn.Module):
        """Two relation-aware HeteroConv layers with RTO and fraud heads."""
        def __init__(self, hidden_channels: int = 64, dropout: float = 0.2):
            super().__init__()
            self.conv1 = HeteroConv({edge: SAGEConv((-1, -1), hidden_channels) for edge in EDGE_TYPES}, aggr="sum")
            self.conv2 = HeteroConv({edge: SAGEConv((-1, -1), hidden_channels) for edge in EDGE_TYPES}, aggr="sum")
            self.dropout = dropout
            self.rto_head = nn.Linear(hidden_channels, 1)
            self.fraud_head = nn.Linear(hidden_channels, 1)

        def forward(self, x_dict: dict[str, Any], edge_index_dict: dict[Any, Any]):
            x_dict = self.conv1(x_dict, edge_index_dict)
            x_dict = {key: F.relu(value) for key, value in x_dict.items()}
            x_dict = {key: F.dropout(value, p=self.dropout, training=self.training) for key, value in x_dict.items()}
            x_dict = self.conv2(x_dict, edge_index_dict)
            embeddings = {key: F.relu(value) for key, value in x_dict.items()}
            buyer_embedding = embeddings["buyer"]
            return {
                "rto_probability": torch.sigmoid(self.rto_head(buyer_embedding)).squeeze(-1),
                "promo_fraud_risk": torch.sigmoid(self.fraud_head(buyer_embedding)).squeeze(-1),
            }, embeddings
else:
    class RTOHeteroGNN:  # import-safe placeholder
        def __init__(self, *args: Any, **kwargs: Any):
            _require_pyg()


def predict_risk_scores(orders: pd.DataFrame, checkpoint: str | None = None) -> pd.DataFrame | None:
    """Return row-aligned risk scores when a trained checkpoint is configured."""
    path = checkpoint or os.getenv("RTO_GNN_CHECKPOINT")
    if not path or not Path(path).is_file():
        return None
    _require_pyg()
    graph = build_heterogeneous_graph(orders)
    state = torch.load(path, map_location="cpu", weights_only=False)
    model = RTOHeteroGNN(state.get("hidden_channels", 64))
    model(graph.data.x_dict, graph.data.edge_index_dict)  # initialize lazy layers
    model.load_state_dict(state["model_state"])
    model.eval()
    with torch.no_grad():
        scores, _ = model(graph.data.x_dict, graph.data.edge_index_dict)
    buyers = orders.get("buyer_id", pd.Series("unknown", index=orders.index)).fillna("unknown").astype(str)
    mapping = {buyer: idx for idx, buyer in enumerate(graph.node_ids["buyer"])}
    return pd.DataFrame({
        "rto_probability": buyers.map(lambda buyer: float(scores["rto_probability"][mapping[buyer]])),
        "promo_fraud_risk": buyers.map(lambda buyer: float(scores["promo_fraud_risk"][mapping[buyer]])),
    }, index=orders.index)
