-- ============================================================
-- IndiaCommerce Analytics - Supabase Schema v4.0
-- Run this entire file in Supabase SQL Editor
-- ============================================================

-- ── Core tables ───────────────────────────────────────────────

create table if not exists public.profiles (
  id         uuid references auth.users(id) on delete cascade primary key,
  email      text,
  plan       text default 'starter',
  created_at timestamptz default now()
);

create table if not exists public.datasets (
  id          bigserial primary key,
  name        text,
  row_count   int,
  source      text default 'upload',
  uploaded_at timestamptz default now()
);

create table if not exists public.audit_log (
  id         bigserial primary key,
  action     text,
  detail     text,
  created_at timestamptz default now()
);

-- ── Operational Actions (approve/dismiss from UI) ─────────────

create table if not exists public.operational_actions (
  id          bigserial primary key,
  action_type text not null,
  payload     jsonb,
  status      text default 'pending',
  source      text default 'dashboard',
  created_at  timestamptz default now(),
  updated_at  timestamptz
);

create index if not exists idx_ops_status     on public.operational_actions(status);
create index if not exists idx_ops_type       on public.operational_actions(action_type);
create index if not exists idx_ops_created    on public.operational_actions(created_at desc);

-- ── Model Result Cache ────────────────────────────────────────

create table if not exists public.model_results (
  cache_key   text primary key,
  model_name  text not null,
  result_json text,
  computed_at timestamptz default now()
);

create index if not exists idx_model_name on public.model_results(model_name);

-- ── CLV Cache ─────────────────────────────────────────────────

create table if not exists public.clv_cache (
  dataset_hash text primary key,
  result_json  text,
  row_count    int,
  computed_at  timestamptz default now()
);

-- ── Anomaly Score Cache ───────────────────────────────────────

create table if not exists public.anomaly_cache (
  dataset_hash  text primary key,
  anomaly_count int,
  anomaly_pct   float,
  result_json   text,
  computed_at   timestamptz default now()
);

-- ── Price Recommendations ─────────────────────────────────────

create table if not exists public.price_recommendations (
  id                 bigserial primary key,
  category           text,
  current_discount   float,
  optimal_discount   float,
  direction          text,
  revenue_impact_pct float,
  approved           boolean default false,
  created_at         timestamptz default now()
);

-- ── At-Risk Alerts ────────────────────────────────────────────

create table if not exists public.at_risk_alerts (
  id                  bigserial primary key,
  customer_id         text,
  churn_risk_score    float,
  risk_label          text,
  value_tier          text,
  days_since_order    int,
  total_revenue       float,
  recommended_action  text,
  exported            boolean default false,
  created_at          timestamptz default now()
);

-- ── Drift Reports ─────────────────────────────────────────────

create table if not exists public.drift_reports (
  id               bigserial primary key,
  features_drifted int,
  max_psi          float,
  pred_r2_drop     float,
  drift_alert      boolean,
  report_json      text,
  created_at       timestamptz default now()
);

-- ── Storage bucket (run separately if needed) ─────────────────
-- create bucket 'datasets' in Supabase Storage UI (set to private)

-- Storage RLS
drop policy if exists "storage: shared folder" on storage.objects;
create policy "storage: shared folder"
  on storage.objects for all
  using (bucket_id = 'datasets');
