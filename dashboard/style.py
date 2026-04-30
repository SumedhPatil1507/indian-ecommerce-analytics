CUSTOM_CSS = '''<style>
/*  Global  */
[data-testid="stAppViewContainer"] { background: #f8f9fc; }
[data-testid="stSidebar"] { background: #1a1f36; }
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
[data-testid="stSidebar"] .stRadio label { color: #e2e8f0 !important; }
[data-testid="stSidebar"] hr { border-color: #2d3561 !important; }

/*  Metric cards  */
[data-testid="metric-container"] {
    background: #ffffff;
    border: 1px solid #e8ecf4;
    border-radius: 12px;
    padding: 16px 20px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
}

/*  Tabs  */
.stTabs [data-baseweb="tab-list"] {
    background: #ffffff;
    border-radius: 10px;
    padding: 4px;
    border: 1px solid #e8ecf4;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    font-weight: 500;
    color: #64748b;
}
.stTabs [aria-selected="true"] {
    background: #4f46e5 !important;
    color: white !important;
}

/*  Buttons  */
.stButton > button[kind="primary"] {
    background: #4f46e5;
    border: none;
    border-radius: 8px;
    font-weight: 600;
}
.stButton > button[kind="primary"]:hover { background: #4338ca; }

/*  Insight cards  */
.insight-card {
    background: #ffffff;
    border-left: 4px solid #4f46e5;
    border-radius: 8px;
    padding: 14px 18px;
    margin: 8px 0;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    font-size: 0.95rem;
}
.risk-card {
    background: #fff8f8;
    border-left: 4px solid #ef4444;
    border-radius: 8px;
    padding: 14px 18px;
    margin: 8px 0;
}
.opp-card {
    background: #f0fdf4;
    border-left: 4px solid #22c55e;
    border-radius: 8px;
    padding: 14px 18px;
    margin: 8px 0;
}
.rec-card {
    background: #ffffff;
    border: 1px solid #e8ecf4;
    border-radius: 10px;
    padding: 16px 20px;
    margin: 10px 0;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05);
}

/*  Page title  */
.page-title {
    font-size: 1.8rem;
    font-weight: 700;
    color: #1a1f36;
    margin-bottom: 2px;
}
.page-subtitle {
    font-size: 0.9rem;
    color: #64748b;
    margin-bottom: 20px;
}
</style>'''
