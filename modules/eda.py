"""
modules/eda.py
Exploratory Data Analysis  distributions, categorical plots, pie charts,
boxplots, violin plots.  All plots are interactive (Plotly).
"""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


#  Histograms 

def plot_distributions(df: pd.DataFrame) -> list:
    figs = []
    cols = {
        "customer_age":    "Customer Age (years)",
        "discount_percent":"Discount Percentage (%)",
        "units_sold":      "Units Sold per Order",
    }
    for col, label in cols.items():
        fig = px.histogram(
            df, x=col, nbins=40, marginal="violin",
            title=f"Distribution of {label}",
            labels={col: label},
            template="plotly_white",
        )
        fig.update_traces(marker_line_width=0.4)
        figs.append(fig)

    # base vs final price overlay
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=df["base_price"],  name="Base Price",  opacity=0.65, nbinsx=60))
    fig.add_trace(go.Histogram(x=df["final_price"], name="Final Price", opacity=0.65, nbinsx=60))
    fig.update_layout(
        barmode="overlay", title="Base Price vs Final Price Distribution",
        xaxis_title="Price ()", yaxis_title="Count", template="plotly_white",
    )
    figs.append(fig)
    return figs


#  Categorical bar plots 

def plot_categorical(df: pd.DataFrame) -> list:
    figs = []
    cat_cols   = ["category", "zone", "brand_type", "sales_event",
                  "competition_intensity", "inventory_pressure"]
    num_cols   = ["revenue", "final_price", "units_sold", "discount_percent"]

    for cat in cat_cols:
        for num in num_cols:
            agg = df.groupby(cat)[num].mean().reset_index().sort_values(num, ascending=False)
            fig = px.bar(
                agg, x=cat, y=num,
                title=f"Average {num.replace('_',' ').title()} by {cat.title()}",
                labels={cat: cat.title(), num: f"Mean {num.replace('_',' ').title()}"},
                template="plotly_white", color=cat,
            )
            figs.append(fig)
    return figs


#  Count plots 

def plot_counts(df: pd.DataFrame) -> list:
    figs = []
    count_cols = ["category", "zone", "brand_type", "sales_event",
                  "competition_intensity", "inventory_pressure", "customer_gender"]
    for col in count_cols:
        vc = df[col].value_counts().reset_index()
        vc.columns = [col, "count"]
        fig = px.bar(
            vc, x="count", y=col, orientation="h",
            title=f"Order Count by {col.title()}",
            template="plotly_white", color=col,
        )
        figs.append(fig)

    # top 12 states
    top_states = df["state"].value_counts().head(12).reset_index()
    top_states.columns = ["state", "count"]
    fig = px.bar(
        top_states, x="count", y="state", orientation="h",
        title="Top 12 States by Order Volume",
        template="plotly_white", color="state",
    )
    figs.append(fig)
    return figs


#  Pie charts 

def plot_pies(df: pd.DataFrame) -> list:
    figs = []
    pie_specs = [
        ("category",             "Revenue Share by Product Category"),
        ("zone",                 "Revenue Share by Zone"),
        ("sales_event",          "Revenue  Festival vs Normal"),
        ("brand_type",           "Revenue Share  Mass vs Premium"),
        ("customer_gender",      "Order Distribution by Gender"),
        ("competition_intensity","Revenue Share by Competition Intensity"),
    ]
    for col, title in pie_specs:
        agg = df.groupby(col)["revenue"].sum().reset_index()
        fig = px.pie(agg, names=col, values="revenue", title=title,
                     hole=0.4, template="plotly_white")
        figs.append(fig)

    # top 8 states
    state_rev = df.groupby("state")["revenue"].sum().nlargest(8).reset_index()
    fig = px.pie(state_rev, names="state", values="revenue",
                 title="Revenue Share  Top 8 States", hole=0.4,
                 template="plotly_white")
    figs.append(fig)
    return figs


#  Boxplots 

def plot_boxplots(df: pd.DataFrame) -> list:
    figs = []
    specs = [
        ("category",             "final_price",      "Final Price by Category (log)"),
        ("sales_event",          "discount_percent",  "Discount %  Normal vs Festival"),
        ("zone",                 "revenue",           "Revenue by Zone (log)"),
        ("competition_intensity","final_price",       "Final Price by Competition Level"),
    ]
    for x_col, y_col, title in specs:
        fig = px.box(df, x=x_col, y=y_col, title=title,
                     log_y=(y_col in ["final_price", "revenue"]),
                     template="plotly_white", color=x_col)
        figs.append(fig)

    # units sold  brand  event
    fig = px.box(df, x="brand_type", y="units_sold", color="sales_event",
                 title="Units Sold  Mass vs Premium  Normal/Festival",
                 template="plotly_white")
    figs.append(fig)
    return figs


#  Violin plots 

def plot_violins(df: pd.DataFrame) -> list:
    figs = []
    specs = [
        ("category",             "final_price",      "Final Price by Category"),
        ("sales_event",          "discount_percent",  "Discount %  Normal vs Festival"),
        ("zone",                 "revenue",           "Revenue by Zone"),
        ("competition_intensity","final_price",       "Final Price by Competition"),
    ]
    for x_col, y_col, title in specs:
        fig = px.violin(df, x=x_col, y=y_col, box=True, points=False,
                        title=title, template="plotly_white", color=x_col,
                        log_y=(y_col in ["final_price", "revenue"]))
        figs.append(fig)

    # split violin  units sold by brand  event
    fig = px.violin(df, x="brand_type", y="units_sold", color="sales_event",
                    box=True, points=False,
                    title="Units Sold  Mass vs Premium  Normal/Festival",
                    template="plotly_white")
    figs.append(fig)
    return figs


if __name__ == "__main__":
    import random

    rng = np.random.default_rng(42)
    n = 500
    categories = ["Electronics", "Clothing", "Home", "Sports"]
    zones = ["North", "South", "East", "West"]
    brand_types = ["Mass", "Premium"]
    sales_events = ["Normal", "Festival"]
    competition = ["Low", "Medium", "High"]
    inventory = ["Low", "Medium", "High"]
    genders = ["Male", "Female"]
    states = ["Maharashtra", "Delhi", "Karnataka", "Tamil Nadu", "Gujarat",
              "Rajasthan", "West Bengal", "Uttar Pradesh", "Telangana", "Kerala",
              "Punjab", "Haryana"]

    df_sample = pd.DataFrame({
        "customer_age":       rng.integers(18, 65, n),
        "discount_percent":   rng.uniform(0, 40, n),
        "units_sold":         rng.integers(1, 10, n),
        "base_price":         rng.uniform(200, 5000, n),
        "final_price":        rng.uniform(150, 4800, n),
        "revenue":            rng.uniform(500, 20000, n),
        "category":           rng.choice(categories, n),
        "zone":               rng.choice(zones, n),
        "brand_type":         rng.choice(brand_types, n),
        "sales_event":        rng.choice(sales_events, n),
        "competition_intensity": rng.choice(competition, n),
        "inventory_pressure": rng.choice(inventory, n),
        "customer_gender":    rng.choice(genders, n),
        "state":              rng.choice(states, n),
    })

    print("Testing plot_distributions ...")
    for fig in plot_distributions(df_sample):
        fig.show()

    print("Testing plot_categorical ...")
    for fig in plot_categorical(df_sample):
        fig.show()

    print("Testing plot_counts ...")
    for fig in plot_counts(df_sample):
        fig.show()

    print("Testing plot_pies ...")
    for fig in plot_pies(df_sample):
        fig.show()

    print("Testing plot_boxplots ...")
    for fig in plot_boxplots(df_sample):
        fig.show()

    print("Testing plot_violins ...")
    for fig in plot_violins(df_sample):
        fig.show()

    print("All EDA plots rendered.")
