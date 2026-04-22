"""
modules/explainability.py
SHAP (KernelExplainer + TreeExplainer), Permutation Importance, LIME.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from sklearn.inspection import permutation_importance


def plot_permutation_importance(model, X_test, y_test, feature_names, top_n=15):
    perm = permutation_importance(model, X_test, y_test, n_repeats=7,
                                  random_state=42, n_jobs=-1)
    idx  = perm.importances_mean.argsort()[-top_n:]
    fig  = go.Figure(go.Bar(
        x=perm.importances_mean[idx],
        y=[feature_names[i] for i in idx],
        orientation="h",
        error_x=dict(type="data", array=perm.importances_std[idx]),
        marker_color="steelblue",
    ))
    fig.update_layout(title=f"Permutation Importance – Top {top_n}",
                      xaxis_title="Mean increase in MSE",
                      template="plotly_white")
    fig.show()


def run_shap(model, X_train, X_test, feature_names, n_background=50, n_explain=300):
    try:
        import shap  # type: ignore
    except ImportError:
        print("shap not installed. Run: pip install shap")
        return None

    # Use TreeExplainer for tree-based models (much faster)
    try:
        explainer   = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test[:n_explain])
    except Exception:
        background  = shap.kmeans(X_train, n_background).data
        explainer   = shap.KernelExplainer(model.predict, background)
        shap_values = explainer.shap_values(X_test[:n_explain])

    # interactive summary via pareto module
    from modules.pareto import plot_shap_summary
    plot_shap_summary(shap_values, list(feature_names))
    return shap_values


def run_lime(model, X_train, X_test, y_test, feature_names, n_samples=3):
    try:
        import lime.lime_tabular  # type: ignore
    except ImportError:
        print("lime not installed. Run: pip install lime")
        return

    explainer = lime.lime_tabular.LimeTabularExplainer(
        X_train, feature_names=list(feature_names),
        mode="regression", discretize_continuous=True,
    )
    idxs = [np.argmin(y_test.values), len(y_test)//2, np.argmax(y_test.values)]
    for i, idx in enumerate(idxs[:n_samples], 1):
        exp = explainer.explain_instance(X_test[idx], model.predict, num_features=10)
        print(f"\nLIME – Sample {i}  (true revenue = ₹{y_test.iloc[idx]:,.0f})")
        for feat, weight in exp.as_list():
            print(f"  {feat:<55} {weight:+.4f}")
