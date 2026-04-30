"""
modules/time_series.py
Time-series analysis, seasonal decomposition, Prophet + SARIMA forecasting.
All trend plots are interactive (Plotly).
"""
import warnings
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")


#  Monthly aggregation helper 

def _monthly(df: pd.DataFrame, col: str = "revenue", agg: str = "sum") -> pd.Series:
    s = df.groupby(df["order_date"].dt.to_period("M"))[col].agg(agg)
    s.index = s.index.to_timestamp()
    return s.asfreq("MS").interpolate("linear")


#  Line plots 

def plot_trends(df: pd.DataFrame) -> None:
    monthly_rev  = _monthly(df, "revenue", "sum").reset_index()
    monthly_rev.columns = ["date", "revenue"]

    fig = px.line(monthly_rev, x="date", y="revenue", markers=True,
                  title="Total Monthly Revenue Trend (36 months)",
                  labels={"revenue": "Total Revenue ()", "date": "Month"},
                  template="plotly_white")
    fig.show()

    # AOV
    aov = _monthly(df, "revenue", "mean").reset_index()
    aov.columns = ["date", "aov"]
    fig = px.line(aov, x="date", y="aov", markers=True,
                  title="Average Order Value (AOV) Trend",
                  labels={"aov": "Avg Revenue per Order ()", "date": "Month"},
                  template="plotly_white")
    fig.show()

    # order count
    cnt = df.groupby(df["order_date"].dt.to_period("M")).size()
    cnt.index = cnt.index.to_timestamp()
    cnt = cnt.reset_index()
    cnt.columns = ["date", "orders"]
    fig = px.line(cnt, x="date", y="orders", markers=True,
                  title="Number of Orders per Month",
                  labels={"orders": "Order Count", "date": "Month"},
                  template="plotly_white")
    fig.show()

    # avg discount
    disc = _monthly(df, "discount_percent", "mean").reset_index()
    disc.columns = ["date", "discount"]
    fig = px.line(disc, x="date", y="discount", markers=True,
                  title="Average Discount % Trend",
                  labels={"discount": "Avg Discount %", "date": "Month"},
                  template="plotly_white")
    fig.show()

    # zone comparison
    zone_m = (
        df.groupby([df["order_date"].dt.to_period("M"), "zone"])["revenue"]
        .sum().reset_index()
    )
    zone_m["date"] = zone_m["order_date"].dt.to_timestamp()
    fig = px.line(zone_m, x="date", y="revenue", color="zone", markers=True,
                  title="Monthly Revenue by Zone",
                  labels={"revenue": "Revenue ()", "date": "Month"},
                  template="plotly_white")
    fig.show()

    # brand type
    brand_m = (
        df.groupby([df["order_date"].dt.to_period("M"), "brand_type"])["revenue"]
        .sum().reset_index()
    )
    brand_m["date"] = brand_m["order_date"].dt.to_timestamp()
    fig = px.line(brand_m, x="date", y="revenue", color="brand_type", markers=True,
                  title="Revenue Trend: Mass vs Premium Brands",
                  labels={"revenue": "Revenue ()", "date": "Month"},
                  template="plotly_white")
    fig.show()


#  Seasonal decomposition 

def plot_decomposition(df: pd.DataFrame) -> None:
    from statsmodels.tsa.seasonal import seasonal_decompose

    monthly = _monthly(df)
    decomp  = seasonal_decompose(monthly, model="additive", period=12)

    fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                        subplot_titles=["Observed", "Trend", "Seasonal", "Residual"])
    for i, (comp, name) in enumerate(
        [(decomp.observed, "Observed"), (decomp.trend, "Trend"),
         (decomp.seasonal, "Seasonal"), (decomp.resid, "Residual")], 1
    ):
        fig.add_trace(go.Scatter(x=comp.index, y=comp.values, name=name), row=i, col=1)

    fig.update_layout(
        title="Monthly Revenue  Additive Decomposition",
        height=800, template="plotly_white", showlegend=False,
    )
    fig.show()


#  Prophet forecast 

def forecast_prophet(df: pd.DataFrame, periods: int = 60) -> pd.DataFrame:
    """
    Fit a Prophet model and forecast `periods` months ahead.
    Returns the full forecast DataFrame.
    """
    from prophet import Prophet  # type: ignore

    monthly = _monthly(df).reset_index()
    monthly.columns = ["ds", "y"]

    m = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        seasonality_mode="multiplicative",
        changepoint_prior_scale=0.05,
    )
    m.add_seasonality(name="monthly", period=1, fourier_order=5)
    m.fit(monthly)

    future   = m.make_future_dataframe(periods=periods, freq="MS")
    forecast = m.predict(future)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=monthly["ds"], y=monthly["y"],
                             mode="markers+lines", name="Historical",
                             line=dict(color="darkred")))
    fig.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat"],
                             mode="lines", name="Forecast",
                             line=dict(color="royalblue", width=2.5)))
    fig.add_trace(go.Scatter(
        x=pd.concat([forecast["ds"], forecast["ds"][::-1]]),
        y=pd.concat([forecast["yhat_upper"], forecast["yhat_lower"][::-1]]),
        fill="toself", fillcolor="rgba(65,105,225,0.15)",
        line=dict(color="rgba(255,255,255,0)"), name="80% CI",
    ))
    fig.add_vline(x=monthly["ds"].max(), line_dash="dash", line_color="gray")
    fig.update_layout(
        title="Revenue Forecast  Prophet (next 5 years)",
        xaxis_title="Date", yaxis_title="Monthly Revenue ()",
        template="plotly_white",
    )
    fig.show()
    return forecast


#  SARIMA forecast 

def forecast_sarima(df: pd.DataFrame, periods: int = 60) -> pd.Series:
    """
    Fit SARIMA(1,1,1)(1,1,1)[12] and forecast `periods` months ahead.
    Returns the forecast mean Series.
    """
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    import pandas.tseries.offsets as offsets

    monthly = _monthly(df)
    model   = SARIMAX(monthly, order=(1,1,1), seasonal_order=(1,1,1,12),
                      enforce_stationarity=False, enforce_invertibility=False)
    fit     = model.fit(disp=False)

    fc      = fit.get_forecast(steps=periods)
    mean    = fc.predicted_mean
    ci      = fc.conf_int(alpha=0.2)
    dates   = pd.date_range(monthly.index[-1] + offsets.MonthBegin(1),
                            periods=periods, freq="MS")
    mean.index = dates
    ci.index   = dates

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=monthly.index, y=monthly.values,
                             mode="markers+lines", name="Historical",
                             line=dict(color="black")))
    fig.add_trace(go.Scatter(x=dates, y=mean.values,
                             mode="lines", name="SARIMA Forecast",
                             line=dict(color="darkgreen", dash="dash", width=2.3)))
    fig.add_trace(go.Scatter(
        x=list(dates) + list(dates[::-1]),
        y=list(ci.iloc[:,1]) + list(ci.iloc[:,0])[::-1],
        fill="toself", fillcolor="rgba(0,100,0,0.10)",
        line=dict(color="rgba(255,255,255,0)"), name="80% CI",
    ))
    fig.update_layout(
        title="Revenue Forecast  SARIMA(1,1,1)(1,1,1)[12]",
        xaxis_title="Date", yaxis_title="Monthly Revenue ()",
        template="plotly_white",
    )
    fig.show()
    return mean
