"""
modules/models.py
ML model training, evaluation, and comparison plots.
Models: Linear Regression, Decision Tree, Random Forest, XGBoost, PyTorch MLP.
"""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import learning_curve, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor
import xgboost as xgb

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    _TORCH_OK = True
except ImportError:
    _TORCH_OK = False

# ── feature config ────────────────────────────────────────────────────────────
CAT_FEATURES = [
    "state", "zone", "category", "brand_type", "customer_gender",
    "sales_event", "competition_intensity", "inventory_pressure",
]
NUM_FEATURES = ["customer_age", "base_price", "discount_percent", "year", "month", "weekday"]
ALL_FEATURES = CAT_FEATURES + NUM_FEATURES
TARGET       = "revenue"


def _preprocessor() -> ColumnTransformer:
    return ColumnTransformer([
        ("num", StandardScaler(), NUM_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_FEATURES),
    ])


# ── MLP ───────────────────────────────────────────────────────────────────────

class _MLP(nn.Module):
    def __init__(self, n_in: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 128),  nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64),   nn.ReLU(),
            nn.Linear(64, 1),
        )
    def forward(self, x):
        return self.net(x)


# ── train / evaluate ──────────────────────────────────────────────────────────

def train_all(df: pd.DataFrame) -> dict:
    """
    Train all models and return a dict with results + predictions.
    """
    X = df[ALL_FEATURES].copy()
    y = df[TARGET].copy()
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    prep = _preprocessor()
    X_tr_t = prep.fit_transform(X_tr)
    X_te_t  = prep.transform(X_te)
    feat_names = prep.get_feature_names_out()

    results, preds, pipes = [], {}, {}

    # sklearn models
    sk_models = [
        ("Linear Regression", LinearRegression()),
        ("Decision Tree",     DecisionTreeRegressor(max_depth=8, random_state=42)),
        ("Random Forest",     RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)),
        ("XGBoost",           xgb.XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)),
    ]
    for name, model in sk_models:
        pipe = Pipeline([("prep", prep), ("model", model)])
        pipe.fit(X_tr, y_tr)
        yp = pipe.predict(X_te)
        results.append(_metrics(name, y_te, yp))
        preds[name] = yp
        pipes[name] = pipe

    # PyTorch MLP (skip if torch not installed)
    if _TORCH_OK:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        mlp    = _MLP(X_tr_t.shape[1]).to(device)
        opt    = optim.Adam(mlp.parameters(), lr=1e-3)
        crit   = nn.MSELoss()
        ds_tr  = TensorDataset(torch.tensor(X_tr_t, dtype=torch.float32).to(device),
                               torch.tensor(y_tr.values, dtype=torch.float32).unsqueeze(1).to(device))
        loader = DataLoader(ds_tr, batch_size=128, shuffle=True)
        for _ in range(60):
            mlp.train()
            for bx, by in loader:
                opt.zero_grad(); crit(mlp(bx), by).backward(); opt.step()
        mlp.eval()
        with torch.no_grad():
            nn_pred = mlp(torch.tensor(X_te_t, dtype=torch.float32).to(device)).cpu().numpy().flatten()
        results.append(_metrics("Neural Network", y_te, nn_pred))
        preds["Neural Network"] = nn_pred
    else:
        print("PyTorch not installed – Neural Network skipped.")

    return {
        "results":    pd.DataFrame(results),
        "preds":      preds,
        "pipes":      pipes,
        "y_test":     y_te,
        "X_test_t":   X_te_t,
        "feat_names": feat_names,
        "prep":       prep,
    }


def _metrics(name, y_true, y_pred) -> dict:
    mse = mean_squared_error(y_true, y_pred)
    return {
        "name": name,
        "RMSE": np.sqrt(mse),
        "MAE":  mean_absolute_error(y_true, y_pred),
        "R²":   r2_score(y_true, y_pred),
    }


# ── comparison plots ──────────────────────────────────────────────────────────

def plot_comparison(output: dict) -> None:
    res = output["results"]

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=["RMSE (lower = better)", "R² (higher = better)"])
    fig.add_trace(go.Bar(x=res["name"], y=res["RMSE"], name="RMSE",
                         marker_color="indianred"), row=1, col=1)
    fig.add_trace(go.Bar(x=res["name"], y=res["R²"],   name="R²",
                         marker_color="steelblue"), row=1, col=2)
    fig.update_layout(title="Model Comparison", template="plotly_white",
                      showlegend=False)
    fig.show()

    # actual vs predicted
    y_te = output["y_test"]
    for name, yp in output["preds"].items():
        fig = px.scatter(x=y_te, y=yp, opacity=0.4,
                         labels={"x": "Actual Revenue", "y": "Predicted Revenue"},
                         title=f"Actual vs Predicted – {name}",
                         template="plotly_white")
        mn, mx = y_te.min(), y_te.max()
        fig.add_trace(go.Scatter(x=[mn, mx], y=[mn, mx],
                                 mode="lines", line=dict(color="red", dash="dash"),
                                 name="Perfect fit"))
        fig.show()

    # residuals for XGBoost
    xgb_pred = output["preds"].get("XGBoost")
    if xgb_pred is not None:
        resid = y_te - xgb_pred
        fig = px.scatter(x=xgb_pred, y=resid, opacity=0.4,
                         labels={"x": "Predicted", "y": "Residual"},
                         title="Residual Plot – XGBoost",
                         template="plotly_white")
        fig.add_hline(y=0, line_dash="dash", line_color="red")
        fig.show()
