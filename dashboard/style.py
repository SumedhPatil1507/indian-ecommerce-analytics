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

import plotly.graph_objects as go

THEME_CSS = """
<style>
/* Base & background */
[data-testid="stAppViewContainer"] { background: #0F172A !important; }
[data-testid="stSidebar"]          { background: #1E293B !important; border-right: 1px solid #334155 !important; }

/* Tab bar */
button[data-baseweb="tab"] {
    background: #1E293B !important;
    color: #94A3B8 !important;
    border-radius: 8px 8px 0 0 !important;
    font-weight: 600 !important;
    font-size: 0.82rem !important;
    padding: 8px 14px !important;
    border: 1px solid #334155 !important;
    border-bottom: none !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    background: #6366F1 !important;
    color: #FFFFFF !important;
    border-color: #6366F1 !important;
}
button[data-baseweb="tab"]:hover {
    background: #334155 !important;
    color: #E2E8F0 !important;
}

/* Metric cards */
[data-testid="metric-container"] {
    background: #1E293B !important;
    border: 1px solid #334155 !important;
    border-radius: 12px !important;
    padding: 12px 16px !important;
}
[data-testid="metric-container"] label { color: #94A3B8 !important; font-size: 0.78rem !important; }
[data-testid="metric-container"] [data-testid="stMetricValue"] { color: #F1F5F9 !important; font-size: 1.6rem !important; font-weight: 700 !important; }
[data-testid="metric-container"] [data-testid="stMetricDelta"] { font-size: 0.82rem !important; }

/* Headings & text */
h1, h2, h3 { color: #F1F5F9 !important; }
p, li, label { color: #CBD5E1 !important; }
.stCaption  { color: #64748B !important; }

/* Selectbox, multiselect, input */
[data-baseweb="select"] > div,
[data-baseweb="input"]  > div  { background: #1E293B !important; border-color: #475569 !important; color: #F1F5F9 !important; }

/* Dataframe */
[data-testid="stDataFrame"] { border: 1px solid #334155 !important; border-radius: 8px !important; }

/* Expander */
[data-testid="stExpander"] { background: #1E293B !important; border: 1px solid #334155 !important; border-radius: 8px !important; }

/* Button */
.stButton > button {
    background: #6366F1 !important;
    color: #FFFFFF !important;
    border-radius: 8px !important;
    border: none !important;
    font-weight: 600 !important;
    padding: 8px 20px !important;
}
.stButton > button:hover { background: #4F46E5 !important; }

/* Info / warning / error boxes */
[data-testid="stAlert"] { border-radius: 8px !important; }

/* Sidebar labels */
[data-testid="stSidebar"] label { color: #CBD5E1 !important; }
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2 { color: #F1F5F9 !important; }

/* Card divider */
hr { border-color: #334155 !important; }

/* General text override for dark background */
html, body, [class*="css"], .stApp, .main, .block-container,
p, span, div, label, li, td, th, h4, h5, h6,
.stMarkdown, .stMarkdown p, .stMarkdown span,
[data-testid="stMarkdownContainer"],
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] span,
[data-testid="stMarkdownContainer"] li,
.stSelectbox label, .stMultiSelect label,
.stSlider label, .stFileUploader label,
.stCheckbox label, .stRadio label,
.stExpander summary, .stExpander p,
[data-testid="stExpander"] p,
[data-testid="stExpander"] span,
[data-testid="stCaptionContainer"] p {
  color: #CBD5E1 !important;
}
</style>
"""


def apply_dark_theme(fig: go.Figure) -> go.Figure:
    """Apply consistent dark Plotly theme to any figure."""
    fig.update_layout(
        paper_bgcolor="#1E293B",
        plot_bgcolor="#0F172A",
        font=dict(
            color="#F1F5F9",
            family="Inter, system-ui, sans-serif",
        ),
        legend=dict(
            bgcolor="#1E293B",
            bordercolor="#475569",
            borderwidth=1,
            font=dict(color="#F1F5F9"),
        ),
    )
    fig.update_xaxes(gridcolor="#334155", zerolinecolor="#334155", tickfont=dict(color="#94A3B8"), title_font=dict(color="#CBD5E1"))
    fig.update_yaxes(gridcolor="#334155", zerolinecolor="#334155", tickfont=dict(color="#94A3B8"), title_font=dict(color="#CBD5E1"))
    return fig
