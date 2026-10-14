"""Train the heterogeneous collusion/RTO model and write embeddings to Neo4j.

CSV requires customer_id, merchant_id, device_id and ip_address. Optional
labels are collusion_label and rto_label (0/1); only labeled customers train
the corresponding prediction head. NEO4J_URI, NEO4J_USER and NEO4J_PASSWORD
are required unless supplied as command-line arguments.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.models.gnn_fraud import HeterogeneousFraudRGCN, build_fraud_graph


def _write_neo4j(node_ids, embeddings, probabilities, args) -> None:
    from neo4j import GraphDatabase

    uri = args.neo4j_uri or os.environ["NEO4J_URI"]
    username = args.neo4j_user or os.environ["NEO4J_USER"]
    password = args.neo4j_password or os.environ["NEO4J_PASSWORD"]
    labels = {"customer": "Customer", "merchant": "Merchant", "device_id": "DeviceID", "ip_address": "IPAddress"}
    key_properties = {"customer": "customer_id", "merchant": "merchant_id", "device_id": "device_id", "ip_address": "ip_address"}
    with GraphDatabase.driver(uri, auth=(username, password)) as driver:
        driver.verify_connectivity()
        with driver.session(database=args.neo4j_database or None) as session:
            for node_type, ids in node_ids.items():
                vectors = embeddings[node_type].detach().cpu().tolist()
                node_rows = [{"id": node_id, "embedding": vector} for node_id, vector in zip(ids, vectors)]
                if node_type == "customer":
                    node_rows = [{
                        **row,
                        "collusion_probability": float(probabilities["collusion_probability"][i]),
                        "rto_probability": float(probabilities["rto_probability"][i]),
                    } for i, row in enumerate(node_rows)]
                label, prop = labels[node_type], key_properties[node_type]
                query = (
                    f"UNWIND $rows AS row MERGE (n:{label} {{{prop}: row.id}}) "
                    "SET n.gnn_embedding = row.embedding, n.gnn_model = 'fraud_rgcn_v1'"
                )
                if node_type == "customer":
                    query += ", n.collusion_probability = row.collusion_probability, n.rto_probability = row.rto_probability"
                for start in range(0, len(node_rows), args.neo4j_batch_size):
                    session.run(query, rows=node_rows[start:start + args.neo4j_batch_size]).consume()


def train(args: argparse.Namespace) -> None:
    graph = build_fraud_graph(pd.read_csv(args.input))
    if graph.customer_labels is None or graph.customer_label_masks is None or not graph.customer_label_masks.any():
        raise ValueError("CSV must contain at least one labeled customer via collusion_label and/or rto_label")
    model = HeterogeneousFraudRGCN(args.hidden_channels, args.dropout)
    model(graph.data.x_dict, graph.data.edge_index_dict)  # initialize relation layers
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    labels, masks = graph.customer_labels, graph.customer_label_masks
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        probabilities, _ = model(graph.data.x_dict, graph.data.edge_index_dict)
        losses = []
        for label_index, output_name in enumerate(("collusion_probability", "rto_probability")):
            mask = masks[:, label_index]
            if mask.any():
                losses.append(F.binary_cross_entropy(probabilities[output_name][mask], labels[mask, label_index]))
        if not losses:
            raise ValueError("No labeled customers found for collusion_label or rto_label")
        loss = torch.stack(losses).mean()
        loss.backward()
        optimizer.step()
        if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
            print(f"epoch={epoch}/{args.epochs} loss={loss.item():.5f}")

    model.eval()
    with torch.no_grad():
        probabilities, embeddings = model(graph.data.x_dict, graph.data.edge_index_dict)
    checkpoint = Path(args.checkpoint)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "hidden_channels": args.hidden_channels,
        "node_types": list(graph.node_ids),
        "node_ids": graph.node_ids,
    }, checkpoint)
    _write_neo4j(graph.node_ids, embeddings, probabilities, args)
    print(f"Saved checkpoint to {checkpoint}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Transaction CSV with heterogeneous identifiers and optional labels")
    parser.add_argument("--checkpoint", default=os.getenv("FRAUD_GNN_CHECKPOINT", "artifacts/fraud_rgcn.pt"))
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
    parser.add_argument("--neo4j-batch-size", type=int, default=1000)
    train(parser.parse_args())


if __name__ == "__main__":
    main()
