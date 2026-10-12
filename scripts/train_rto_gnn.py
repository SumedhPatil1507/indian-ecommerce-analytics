"""Train the buyer RTO/fraud hetero-GNN and persist node embeddings to Neo4j.

Input CSV needs buyer_id, pincode, device_id, payment_method,
rto_label, and promo_fraud_label. Neo4j credentials are read from environment.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import pandas as pd
import torch
import torch.nn.functional as F

# Support both ``python scripts/train_rto_gnn.py`` and module execution.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from modules.rto_graph import RTOHeteroGNN, build_heterogeneous_graph


def train(args: argparse.Namespace) -> None:
    orders = pd.read_csv(args.input)
    graph = build_heterogeneous_graph(orders)
    if graph.buyer_labels is None or graph.buyer_label_masks is None:
        raise ValueError("Input must include rto_label and promo_fraud_label")
    if not graph.buyer_label_masks.any():
        raise ValueError("No labeled buyers found")
    model = RTOHeteroGNN(hidden_channels=args.hidden_channels, dropout=args.dropout)
    model(graph.data.x_dict, graph.data.edge_index_dict)  # initialize lazy relation layers
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    labels, masks = graph.buyer_labels, graph.buyer_label_masks
    model.train()
    for epoch in range(1, args.epochs + 1):
        optimizer.zero_grad()
        predictions, _ = model(graph.data.x_dict, graph.data.edge_index_dict)
        losses = []
        if masks[:, 0].any():
            losses.append(F.binary_cross_entropy(predictions["rto_probability"][masks[:, 0]], labels[masks[:, 0], 0]))
        if masks[:, 1].any():
            losses.append(F.binary_cross_entropy(predictions["promo_fraud_risk"][masks[:, 1]], labels[masks[:, 1], 1]))
        loss = torch.stack(losses).mean()
        loss.backward()
        optimizer.step()
        if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
            print(f"epoch={epoch}/{args.epochs} loss={loss.item():.5f}")

    model.eval()
    with torch.no_grad():
        scores, embeddings = model(graph.data.x_dict, graph.data.edge_index_dict)
    checkpoint = Path(args.checkpoint)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "hidden_channels": args.hidden_channels}, checkpoint)
    _write_neo4j(graph.node_ids, embeddings, scores, args)
    print(f"Saved checkpoint to {checkpoint}")


def _write_neo4j(node_ids, embeddings, scores, args) -> None:
    from neo4j import GraphDatabase

    uri = args.neo4j_uri or os.environ["NEO4J_URI"]
    username = args.neo4j_user or os.environ["NEO4J_USER"]
    password = args.neo4j_password or os.environ["NEO4J_PASSWORD"]
    labels = {"buyer": "Buyer", "pincode": "Pincode", "device_id": "Device", "payment_method": "PaymentMethod"}
    key_properties = {"buyer": "buyer_id", "pincode": "pincode", "device_id": "device_id", "payment_method": "payment_method"}
    with GraphDatabase.driver(uri, auth=(username, password)) as driver:
        with driver.session(database=args.neo4j_database or None) as session:
            for node_type, ids in node_ids.items():
                vectors = embeddings[node_type].detach().cpu().tolist()
                rows = [{"id": node_id, "embedding": vector} for node_id, vector in zip(ids, vectors)]
                if node_type == "buyer":
                    rows = [{**row,
                             "rto_probability": float(scores["rto_probability"][i]),
                             "promo_fraud_risk": float(scores["promo_fraud_risk"][i])}
                            for i, row in enumerate(rows)]
                props = key_properties[node_type]
                query = (
                    f"UNWIND $rows AS row MERGE (n:{labels[node_type]} {{{props}: row.id}}) "
                    "SET n.graph_embedding = row.embedding, n.graph_embedding_model = 'rto_rgcn_v1'"
                )
                if node_type == "buyer":
                    query += ", n.rto_probability = row.rto_probability, n.promo_fraud_risk = row.promo_fraud_risk"
                session.run(query, rows=rows).consume()
        driver.verify_connectivity()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="CSV of order-level graph rows and labels")
    parser.add_argument("--checkpoint", default=os.getenv("RTO_GNN_CHECKPOINT", "artifacts/rto_gnn.pt"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--hidden-channels", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--neo4j-uri", default=None)
    parser.add_argument("--neo4j-user", default=None)
    parser.add_argument("--neo4j-password", default=None)
    parser.add_argument("--neo4j-database", default=None)
    train(parser.parse_args())


if __name__ == "__main__":
    main()
