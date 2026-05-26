"""
data/connectors.py
Data Connector Matrix - standardised ingestion from multiple e-commerce sources.

Supported connectors:
  1. Shopify Webhooks       - order/create webhook payload -> standard schema
  2. Amazon Seller Central  - Orders API response -> standard schema
  3. WooCommerce DB Dump    - MySQL/CSV export -> standard schema
  4. Generic CSV/Excel      - Any file matching the standard schema
  5. Simulation Sandbox     - Synthetic data for demos (NOT for production use)

All connectors normalise to the same internal schema so every downstream
module works identically regardless of source.

Standard internal schema columns:
  order_id, order_date, state, zone, category, brand_type,
  customer_gender, customer_age, base_price, discount_percent,
  final_price, units_sold, revenue, sales_event,
  competition_intensity, inventory_pressure
"""
from __future__ import annotations
import io
import json
import logging
import random
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Standard schema definition ────────────────────────────────────────────────

REQUIRED_COLUMNS = [
    "order_id", "order_date", "state", "zone", "category", "brand_type",
    "customer_gender", "customer_age", "base_price", "discount_percent",
    "final_price", "units_sold", "revenue", "sales_event",
    "competition_intensity", "inventory_pressure",
]

OPTIONAL_COLUMNS = ["customer_id", "product_id", "sku", "channel", "source"]

COLUMN_TYPES = {
    "order_date":           "datetime",
    "customer_age":         "int",
    "base_price":           "float",
    "discount_percent":     "float",
    "final_price":          "float",
    "units_sold":           "int",
    "revenue":              "float",
}

# ── Shopify Webhook Connector ─────────────────────────────────────────────────

SHOPIFY_MOCK_SCHEMA = {
    "description": "Shopify order/create webhook payload",
    "source": "Shopify Webhooks (POST /webhooks/orders/create)",
    "docs": "https://shopify.dev/docs/api/admin-rest/2024-01/resources/webhook",
    "required_fields": [
        "id", "created_at", "total_price", "line_items",
        "customer", "billing_address", "discount_codes",
    ],
    "sample": {
        "id": 820982911946154500,
        "created_at": "2024-01-15T10:30:00+05:30",
        "total_price": "2499.00",
        "subtotal_price": "2999.00",
        "discount_codes": [{"code": "SALE20", "amount": "500.00", "type": "fixed_amount"}],
        "line_items": [{"quantity": 2, "price": "1499.50", "product_type": "Electronics"}],
        "customer": {"id": 207119551, "email": "customer@example.com"},
        "billing_address": {"city": "Mumbai", "province": "Maharashtra"},
    }
}


def from_shopify_webhook(payload: dict | list) -> pd.DataFrame:
    """
    Normalise Shopify order/create webhook payload(s) to standard schema.

    Accepts a single order dict or a list of orders.
    Maps Shopify fields to internal schema with sensible defaults.
    """
    if isinstance(payload, dict):
        payload = [payload]

    rows = []
    for order in payload:
        try:
            line_items = order.get("line_items", [{}])
            first_item = line_items[0] if line_items else {}
            billing    = order.get("billing_address", {})
            customer   = order.get("customer", {})
            discounts  = order.get("discount_codes", [])

            subtotal   = float(order.get("subtotal_price", order.get("total_price", 0)))
            total      = float(order.get("total_price", subtotal))
            disc_amt   = sum(float(d.get("amount", 0)) for d in discounts)
            disc_pct   = (disc_amt / subtotal * 100) if subtotal > 0 else 0
            qty        = sum(int(li.get("quantity", 1)) for li in line_items)
            unit_price = subtotal / max(qty, 1)

            rows.append({
                "order_id":             str(order.get("id", "")),
                "order_date":           order.get("created_at", datetime.now().isoformat()),
                "state":                billing.get("province", "Unknown"),
                "zone":                 _state_to_zone(billing.get("province", "")),
                "category":             _shopify_product_type(first_item.get("product_type", "")),
                "brand_type":           "Premium" if unit_price > 5000 else "Mass",
                "customer_gender":      "Unknown",
                "customer_age":         30,
                "base_price":           round(subtotal / max(qty, 1), 2),
                "discount_percent":     round(disc_pct, 1),
                "final_price":          round(total / max(qty, 1), 2),
                "units_sold":           qty,
                "revenue":              round(total, 2),
                "sales_event":          "Festival" if disc_pct > 30 else "Normal",
                "competition_intensity":"Medium",
                "inventory_pressure":   "Low",
                "source":               "shopify",
                "customer_id":          str(customer.get("id", "")),
            })
        except Exception as e:
            logger.warning("Shopify row parse error: %s", e)
            continue

    return _finalise(pd.DataFrame(rows))


# ── Amazon Seller Central Connector ──────────────────────────────────────────

AMAZON_MOCK_SCHEMA = {
    "description": "Amazon Seller Central Orders API response",
    "source": "Amazon SP-API GET /orders/v0/orders",
    "docs": "https://developer-docs.amazon.com/sp-api/docs/orders-api-v0-reference",
    "required_fields": [
        "AmazonOrderId", "PurchaseDate", "OrderTotal",
        "OrderItems", "ShippingAddress",
    ],
    "sample": {
        "AmazonOrderId": "402-1234567-1234567",
        "PurchaseDate":  "2024-01-15T05:00:00Z",
        "OrderTotal":    {"Amount": "1999.00", "CurrencyCode": "INR"},
        "OrderStatus":   "Shipped",
        "ShippingAddress": {"StateOrRegion": "Karnataka", "City": "Bengaluru"},
        "OrderItems": [{
            "ASIN": "B08N5WRWNW",
            "Title": "Electronics Product",
            "QuantityOrdered": 1,
            "ItemPrice": {"Amount": "2499.00", "CurrencyCode": "INR"},
            "PromotionDiscount": {"Amount": "500.00", "CurrencyCode": "INR"},
        }],
    }
}


def from_amazon_orders(payload: dict | list) -> pd.DataFrame:
    """
    Normalise Amazon SP-API Orders response to standard schema.

    Accepts the Orders array from the API response or a list of order dicts.
    """
    if isinstance(payload, dict):
        orders = payload.get("Orders", payload.get("orders", [payload]))
    else:
        orders = payload

    rows = []
    for order in orders:
        try:
            items    = order.get("OrderItems", [{}])
            first    = items[0] if items else {}
            addr     = order.get("ShippingAddress", {})
            total    = float(order.get("OrderTotal", {}).get("Amount", 0))
            qty      = sum(int(i.get("QuantityOrdered", 1)) for i in items)
            item_price = float(first.get("ItemPrice", {}).get("Amount", total))
            promo    = float(first.get("PromotionDiscount", {}).get("Amount", 0))
            base     = item_price + promo
            disc_pct = (promo / base * 100) if base > 0 else 0

            rows.append({
                "order_id":             order.get("AmazonOrderId", ""),
                "order_date":           order.get("PurchaseDate", datetime.now().isoformat()),
                "state":                addr.get("StateOrRegion", "Unknown"),
                "zone":                 _state_to_zone(addr.get("StateOrRegion", "")),
                "category":             _amazon_category(first.get("Title", "")),
                "brand_type":           "Premium" if item_price > 5000 else "Mass",
                "customer_gender":      "Unknown",
                "customer_age":         30,
                "base_price":           round(base, 2),
                "discount_percent":     round(disc_pct, 1),
                "final_price":          round(item_price, 2),
                "units_sold":           qty,
                "revenue":              round(total, 2),
                "sales_event":          "Festival" if disc_pct > 25 else "Normal",
                "competition_intensity":"High",
                "inventory_pressure":   "Low",
                "source":               "amazon",
            })
        except Exception as e:
            logger.warning("Amazon row parse error: %s", e)
            continue

    return _finalise(pd.DataFrame(rows))


# ── WooCommerce DB Dump Connector ─────────────────────────────────────────────

WOOCOMMERCE_MOCK_SCHEMA = {
    "description": "WooCommerce MySQL/CSV export",
    "source": "WooCommerce DB export or REST API /wp-json/wc/v3/orders",
    "docs": "https://woocommerce.github.io/woocommerce-rest-api-docs/#orders",
    "required_fields": [
        "id", "date_created", "total", "line_items",
        "billing", "discount_total", "status",
    ],
    "sample": {
        "id": 1234,
        "date_created": "2024-01-15T10:30:00",
        "status": "completed",
        "total": "1799.00",
        "subtotal": "1999.00",
        "discount_total": "200.00",
        "billing": {"state": "DL", "city": "New Delhi"},
        "line_items": [{"quantity": 1, "total": "1799.00", "name": "Fashion Product"}],
    }
}


def from_woocommerce(payload: dict | list) -> pd.DataFrame:
    """
    Normalise WooCommerce REST API orders or CSV export to standard schema.
    """
    if isinstance(payload, dict):
        payload = [payload]

    rows = []
    for order in payload:
        try:
            items    = order.get("line_items", [{}])
            first    = items[0] if items else {}
            billing  = order.get("billing", {})
            total    = float(order.get("total", 0))
            subtotal = float(order.get("subtotal", total))
            disc     = float(order.get("discount_total", 0))
            disc_pct = (disc / subtotal * 100) if subtotal > 0 else 0
            qty      = sum(int(i.get("quantity", 1)) for i in items)

            rows.append({
                "order_id":             str(order.get("id", "")),
                "order_date":           order.get("date_created", datetime.now().isoformat()),
                "state":                _woo_state(billing.get("state", billing.get("city", "Unknown"))),
                "zone":                 _state_to_zone(billing.get("state", "")),
                "category":             _woo_category(first.get("name", "")),
                "brand_type":           "Premium" if (subtotal / max(qty, 1)) > 5000 else "Mass",
                "customer_gender":      "Unknown",
                "customer_age":         30,
                "base_price":           round(subtotal / max(qty, 1), 2),
                "discount_percent":     round(disc_pct, 1),
                "final_price":          round(total / max(qty, 1), 2),
                "units_sold":           qty,
                "revenue":              round(total, 2),
                "sales_event":          "Festival" if disc_pct > 20 else "Normal",
                "competition_intensity":"Medium",
                "inventory_pressure":   "Low",
                "source":               "woocommerce",
            })
        except Exception as e:
            logger.warning("WooCommerce row parse error: %s", e)
            continue

    return _finalise(pd.DataFrame(rows))


# ── Simulation Sandbox ────────────────────────────────────────────────────────

SIMULATION_CONFIG = {
    "description": (
        "Simulation Sandbox - generates realistic synthetic data to showcase "
        "system capabilities. NOT for production use. "
        "Use real connectors (Shopify/Amazon/WooCommerce) for live deployments."
    ),
    "use_case": "Demo, testing, and capability showcase only",
    "categories":  ["Electronics", "Fashion", "Grocery Essentials",
                    "Premium Lifestyle", "Home & Kitchen", "Sports & Fitness"],
    "zones":       ["North", "South", "East", "West", "Central"],
    "states":      ["Maharashtra", "Delhi", "Karnataka", "Tamil Nadu",
                    "Uttar Pradesh", "Gujarat", "West Bengal", "Rajasthan",
                    "Telangana", "Kerala", "Punjab", "Haryana"],
    "price_bands": {
        "Electronics":        (8000,  80000),
        "Fashion":            (500,   8000),
        "Grocery Essentials": (100,   2000),
        "Premium Lifestyle":  (5000,  150000),
        "Home & Kitchen":     (800,   25000),
        "Sports & Fitness":   (600,   15000),
    },
}


def generate_simulation(
    n_rows: int = 3000,
    months: int = 24,
    gdp_growth: float = 0.07,
    cpi_rate: float = 0.05,
    fx_rate: float = 84.0,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic e-commerce data for the Simulation Sandbox.

    Parameters are calibrated to live macro signals when available.
    This function is ONLY for demonstrating system capabilities.
    All downstream modules work identically with real connector data.
    """
    rng = np.random.default_rng(seed)
    cfg = SIMULATION_CONFIG

    end_date   = pd.Timestamp.today().normalize()
    start_date = end_date - pd.DateOffset(months=months)
    dates      = pd.date_range(start_date, end_date, freq="D")
    festival_months = {10, 11, 1, 8}
    price_mult = 1 + cpi_rate

    rows = []
    for i in range(n_rows):
        order_date  = pd.Timestamp(rng.choice(dates))
        month       = order_date.month
        category    = rng.choice(cfg["categories"])
        brand       = rng.choice(["Mass", "Premium"])
        zone        = rng.choice(cfg["zones"])
        state       = rng.choice(cfg["states"])
        competition = rng.choice(["Low", "Medium", "High"])
        is_festival = month in festival_months
        event       = "Festival" if (is_festival and rng.random() < 0.65) else "Normal"

        lo, hi     = cfg["price_bands"][category]
        base_price = round(float(rng.uniform(lo, hi)) * price_mult, 2)
        disc_base  = 20 if event == "Normal" else 40
        if competition == "High": disc_base += 10
        if brand == "Premium":    disc_base -= 5
        discount    = float(np.clip(rng.normal(disc_base, 12), 0, 65))
        final_price = round(base_price * (1 - discount / 100), 2)

        months_elapsed = max((order_date - start_date).days / 30, 0)
        growth_factor  = 1 + gdp_growth * (months_elapsed / months)
        units_base     = 30 if brand == "Mass" else 10
        if event == "Festival": units_base *= 2
        units_sold = max(1, int(rng.normal(units_base, units_base * 0.4) * growth_factor))
        revenue    = round(final_price * units_sold, 2)

        rows.append({
            "order_id":             f"SIM{i:07d}",
            "order_date":           order_date.strftime("%Y-%m-%d"),
            "state":                state,
            "zone":                 zone,
            "category":             category,
            "brand_type":           brand,
            "customer_gender":      rng.choice(["Male", "Female"]),
            "customer_age":         int(rng.integers(18, 62)),
            "base_price":           base_price,
            "discount_percent":     round(discount, 1),
            "final_price":          final_price,
            "units_sold":           units_sold,
            "revenue":              revenue,
            "sales_event":          event,
            "competition_intensity":competition,
            "inventory_pressure":   rng.choice(["Low", "High"]),
            "source":               "simulation",
        })

    df = pd.DataFrame(rows)
    df["revenue_usd"]  = (df["revenue"] / fx_rate).round(2)
    df["usd_inr_rate"] = fx_rate
    return df


# ── Schema validation ─────────────────────────────────────────────────────────

def validate_schema(df: pd.DataFrame) -> dict:
    """
    Validate a DataFrame against the standard internal schema.
    Returns a dict with 'valid', 'missing_cols', 'type_errors', 'warnings'.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    type_errors = []
    warnings = []

    for col, expected_type in COLUMN_TYPES.items():
        if col not in df.columns:
            continue
        try:
            if expected_type == "datetime":
                pd.to_datetime(df[col], errors="raise")
            elif expected_type == "float":
                pd.to_numeric(df[col], errors="raise")
            elif expected_type == "int":
                pd.to_numeric(df[col], errors="raise")
        except Exception:
            type_errors.append(f"{col}: expected {expected_type}")

    if "discount_percent" in df.columns:
        out_of_range = ((df["discount_percent"] < 0) | (df["discount_percent"] > 100)).sum()
        if out_of_range > 0:
            warnings.append(f"{out_of_range} rows have discount_percent outside 0-100 (will be clipped)")

    if "revenue" in df.columns and "final_price" in df.columns and "units_sold" in df.columns:
        calc_rev = df["final_price"] * df["units_sold"]
        mismatch = (~np.isclose(df["revenue"], calc_rev, rtol=0.05)).sum()
        if mismatch > len(df) * 0.1:
            warnings.append(f"{mismatch} rows: revenue != final_price * units_sold (>10% mismatch)")

    return {
        "valid":        len(missing) == 0 and len(type_errors) == 0,
        "missing_cols": missing,
        "type_errors":  type_errors,
        "warnings":     warnings,
        "row_count":    len(df),
        "col_count":    len(df.columns),
    }


# ── Helper functions ──────────────────────────────────────────────────────────

_STATE_ZONE_MAP = {
    "Maharashtra": "West",  "Gujarat": "West",    "Rajasthan": "West",
    "Delhi": "North",       "Haryana": "North",   "Punjab": "North",
    "Uttar Pradesh": "North","Uttarakhand": "North",
    "Karnataka": "South",   "Tamil Nadu": "South","Kerala": "South",
    "Telangana": "South",   "Andhra Pradesh": "South",
    "West Bengal": "East",  "Odisha": "East",     "Bihar": "East",
    "Jharkhand": "East",    "Assam": "East",
    "Madhya Pradesh": "Central","Chhattisgarh": "Central",
}

_SHOPIFY_CATEGORY_MAP = {
    "electronics": "Electronics", "mobile": "Electronics", "laptop": "Electronics",
    "fashion": "Fashion", "clothing": "Fashion", "apparel": "Fashion",
    "grocery": "Grocery Essentials", "food": "Grocery Essentials",
    "luxury": "Premium Lifestyle", "premium": "Premium Lifestyle",
    "home": "Home & Kitchen", "kitchen": "Home & Kitchen",
    "sports": "Sports & Fitness", "fitness": "Sports & Fitness",
}


def _state_to_zone(state: str) -> str:
    for k, v in _STATE_ZONE_MAP.items():
        if k.lower() in state.lower():
            return v
    return "Central"


def _shopify_product_type(product_type: str) -> str:
    pt = product_type.lower()
    for k, v in _SHOPIFY_CATEGORY_MAP.items():
        if k in pt:
            return v
    return "Fashion"


def _amazon_category(title: str) -> str:
    t = title.lower()
    if any(w in t for w in ["phone", "laptop", "tv", "camera", "electronic"]):
        return "Electronics"
    if any(w in t for w in ["shirt", "dress", "shoe", "fashion", "cloth"]):
        return "Fashion"
    if any(w in t for w in ["food", "grocery", "snack", "beverage"]):
        return "Grocery Essentials"
    if any(w in t for w in ["luxury", "premium", "gold", "diamond"]):
        return "Premium Lifestyle"
    if any(w in t for w in ["home", "kitchen", "furniture", "decor"]):
        return "Home & Kitchen"
    if any(w in t for w in ["sport", "gym", "fitness", "yoga"]):
        return "Sports & Fitness"
    return "Fashion"


def _woo_category(name: str) -> str:
    return _amazon_category(name)


def _woo_state(state_code: str) -> str:
    woo_state_map = {
        "MH": "Maharashtra", "DL": "Delhi", "KA": "Karnataka",
        "TN": "Tamil Nadu",  "UP": "Uttar Pradesh", "GJ": "Gujarat",
        "WB": "West Bengal", "RJ": "Rajasthan", "TS": "Telangana",
        "KL": "Kerala",      "PB": "Punjab",    "HR": "Haryana",
    }
    return woo_state_map.get(state_code.upper(), state_code)


def _finalise(df: pd.DataFrame) -> pd.DataFrame:
    """Apply cleaning and engineering to a connector-normalised DataFrame."""
    if df.empty:
        return df
    from data.loader import _clean, _engineer
    return _engineer(_clean(df))
