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

def plot_distributions(df: pd.DataFrame) -> None:
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
        fig.show()

    # base vs final price overlay
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=df["base_price"],  name="Base Price",  opacity=0.65, nbinsx=60))
    fig.add_trace(go.Histogram(x=df["final_price"], name="Final Price", opacity=0.65, nbinsx=60))
    fig.update_layout(
        barmode="overlay", title="Base Price vs Final Price Distribution",
        xaxis_title="Price ()", yaxis_title="Count", template="plotly_white",
    )
    fig.show()


#  Categorical bar plots 

def plot_categorical(df: pd.DataFrame) -> None:
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
            fig.show()


#  Count plots 

def plot_counts(df: pd.DataFrame) -> None:
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
        fig.show()

    # top 12 states
    top_states = df["state"].value_counts().head(12).reset_index()
    top_states.columns = ["state", "count"]
    fig = px.bar(
        top_states, x="count", y="state", orientation="h",
        title="Top 12 States by Order Volume",
        template="plotly_white", color="state",
    )
    fig.show()


#  Pie charts 

def plot_pies(df: pd.DataFrame) -> None:
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
        fig.show()

    # top 8 states
    state_rev = df.groupby("state")["revenue"].sum().nlargest(8).reset_index()
    fig = px.pie(state_rev, names="state", values="revenue",
                 title="Revenue Share  Top 8 States", hole=0.4,
                 template="plotly_white")
    fig.show()


#  Boxplots 

def plot_boxplots(df: pd.DataFrame) -> None:
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
        fig.show()

    # units sold  brand  event
    fig = px.box(df, x="brand_type", y="units_sold", color="sales_event",
                 title="Units Sold  Mass vs Premium  Normal/Festival",
                 template="plotly_white")
    fig.show()


#  Violin plots 

def plot_violins(df: pd.DataFrame) -> None:
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
        fig.show()

    # split violin  units sold by brand  event
    fig = px.violin(df, x="brand_type", y="units_sold", color="sales_event",
                    box=True, points=False,
                    title="Units Sold  Mass vs Premium  Normal/Festival",
                    template="plotly_white")
    fig.show()
