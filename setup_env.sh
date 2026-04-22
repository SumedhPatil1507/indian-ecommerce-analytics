#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# One-Click Setup Script
# Usage:  bash setup_env.sh
# ─────────────────────────────────────────────────────────────────────────────
set -e

echo "============================================================"
echo "  Indian E-Commerce Analytics – One-Click Setup"
echo "============================================================"

# 1. Python version check
python3 --version || { echo "Python 3.10+ required"; exit 1; }

# 2. Create virtual environment
if [ ! -d ".venv" ]; then
  echo "[1/4] Creating virtual environment..."
  python3 -m venv .venv
fi

# 3. Activate
source .venv/bin/activate

# 4. Install dependencies
echo "[2/4] Installing dependencies..."
pip install --upgrade pip -q
pip install -r requirements.txt -q

# 5. Install package in editable mode
echo "[3/4] Installing package..."
pip install -e . -q

# 6. Generate sample fixture for tests
echo "[4/4] Generating test fixture..."
python3 - <<'EOF'
import pandas as pd, numpy as np, os
os.makedirs("tests/fixtures", exist_ok=True)
np.random.seed(42)
n = 500
df = pd.DataFrame({
    "order_id":             [f"ORD{i:05d}" for i in range(n)],
    "order_date":           pd.date_range("2022-01-01", periods=n, freq="D").astype(str),
    "state":                np.random.choice(["Maharashtra","Delhi","Karnataka","Tamil Nadu"], n),
    "zone":                 np.random.choice(["North","South","East","West","Central"], n),
    "category":             np.random.choice(["Electronics","Fashion","Grocery Essentials","Premium Lifestyle"], n),
    "brand_type":           np.random.choice(["Mass","Premium"], n),
    "customer_gender":      np.random.choice(["Male","Female"], n),
    "customer_age":         np.random.randint(18, 60, n),
    "base_price":           np.random.uniform(500, 50000, n).round(2),
    "discount_percent":     np.random.choice([10,20,30,40,50,65], n).astype(float),
    "sales_event":          np.random.choice(["Normal","Festival"], n),
    "competition_intensity":np.random.choice(["Low","Medium","High"], n),
    "inventory_pressure":   np.random.choice(["Low","High"], n),
    "units_sold":           np.random.randint(1, 100, n),
})
df["final_price"] = (df["base_price"] * (1 - df["discount_percent"]/100)).round(2)
df["revenue"]     = (df["final_price"] * df["units_sold"]).round(2)
df.to_csv("tests/fixtures/sample.csv", index=False)
print("  ✅ tests/fixtures/sample.csv created")
EOF

echo ""
echo "============================================================"
echo "  Setup complete!  Next steps:"
echo ""
echo "  Activate env  :  source .venv/bin/activate"
echo "  Run dashboard :  streamlit run dashboard/app.py"
echo "  Run API       :  uvicorn api.main:app --reload --port 8000"
echo "  Run tests     :  pytest tests/ -v"
echo "============================================================"
