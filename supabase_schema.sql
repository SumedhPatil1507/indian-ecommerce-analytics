-- ============================================================
-- IndiaCommerce Analytics - Supabase Schema v5.0 (Multi-Tenant)
-- Run this entire file in Supabase SQL Editor
-- ============================================================

-- ── Core tables ───────────────────────────────────────────────

create table if not exists public.profiles (
  id             uuid references auth.users(id) on delete cascade primary key,
  email          text,
  plan           text default 'starter',
  webhook_secret text default 'global_secret',
  created_at     timestamptz default now()
);

create table if not exists public.orders (
  id                     bigserial primary key,
  tenant_id              uuid references auth.users(id) on delete cascade not null,
  order_id               text not null,
  order_date             timestamptz not null,
  state                  text,
  zone                   text,
  category               text,
  brand_type             text,
  customer_gender        text,
  customer_age           int,
  base_price             float,
  discount_percent       float,
  final_price            float,
  units_sold             int,
  revenue                float,
  sales_event            text,
  competition_intensity  text,
  inventory_pressure     text,
  source                 text,
  customer_id            text,
  created_at             timestamptz default now()
);

create index if not exists idx_orders_tenant_date on public.orders(tenant_id, order_date desc);
create index if not exists idx_orders_tenant_cat on public.orders(tenant_id, category);

create table if not exists public.datasets (
  id          bigserial primary key,
  tenant_id   uuid references auth.users(id) on delete cascade not null,
  name        text,
  row_count   int,
  source      text default 'upload',
  uploaded_at timestamptz default now()
);

create table if not exists public.audit_log (
  id         bigserial primary key,
  tenant_id  uuid references auth.users(id) on delete cascade not null,
  action     text,
  detail     text,
  created_at timestamptz default now()
);

-- ── Operational Actions (approve/dismiss from UI) ─────────────

create table if not exists public.operational_actions (
  id          bigserial primary key,
  tenant_id   uuid references auth.users(id) on delete cascade not null,
  action_type text not null,
  payload     jsonb,
  status      text default 'pending',
  source      text default 'dashboard',
  created_at  timestamptz default now(),
  updated_at  timestamptz
);

create index if not exists idx_ops_tenant_status  on public.operational_actions(tenant_id, status);
create index if not exists idx_ops_tenant_type    on public.operational_actions(tenant_id, action_type);

-- ── Model Result Cache ────────────────────────────────────────

create table if not exists public.model_results (
  tenant_id   uuid references auth.users(id) on delete cascade not null,
  cache_key   text not null,
  model_name  text not null,
  result_json text,
  computed_at timestamptz default now(),
  primary key (tenant_id, cache_key)
);

-- ── CLV Cache ─────────────────────────────────────────────────

create table if not exists public.clv_cache (
  tenant_id    uuid references auth.users(id) on delete cascade not null,
  dataset_hash text not null,
  result_json  text,
  row_count    int,
  computed_at  timestamptz default now(),
  primary key (tenant_id, dataset_hash)
);

-- ── Anomaly Score Cache ───────────────────────────────────────

create table if not exists public.anomaly_cache (
  tenant_id     uuid references auth.users(id) on delete cascade not null,
  dataset_hash  text not null,
  anomaly_count int,
  anomaly_pct   float,
  result_json   text,
  computed_at   timestamptz default now(),
  primary key (tenant_id, dataset_hash)
);

-- ── Price Recommendations ─────────────────────────────────────

create table if not exists public.price_recommendations (
  id                 bigserial primary key,
  tenant_id          uuid references auth.users(id) on delete cascade not null,
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
  tenant_id           uuid references auth.users(id) on delete cascade not null,
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
  tenant_id        uuid references auth.users(id) on delete cascade not null,
  features_drifted int,
  max_psi          float,
  pred_r2_drop     float,
  drift_alert      boolean,
  report_json      text,
  created_at       timestamptz default now()
);

-- ── Row Level Security (RLS) Policies ─────────────────────────

-- Enable RLS
alter table public.profiles enable row level security;
alter table public.orders enable row level security;
alter table public.datasets enable row level security;
alter table public.audit_log enable row level security;
alter table public.operational_actions enable row level security;
alter table public.model_results enable row level security;
alter table public.clv_cache enable row level security;
alter table public.anomaly_cache enable row level security;
alter table public.price_recommendations enable row level security;
alter table public.at_risk_alerts enable row level security;
alter table public.drift_reports enable row level security;

-- Drop existing policies if they exist
drop policy if exists "Profiles are owner-only" on public.profiles;
drop policy if exists "Orders are tenant-isolated" on public.orders;
drop policy if exists "Datasets are tenant-isolated" on public.datasets;
drop policy if exists "Audit log is tenant-isolated" on public.audit_log;
drop policy if exists "Operational actions are tenant-isolated" on public.operational_actions;
drop policy if exists "Model results are tenant-isolated" on public.model_results;
drop policy if exists "CLV cache is tenant-isolated" on public.clv_cache;
drop policy if exists "Anomaly cache is tenant-isolated" on public.anomaly_cache;
drop policy if exists "Price recommendations are tenant-isolated" on public.price_recommendations;
drop policy if exists "At-risk alerts are tenant-isolated" on public.at_risk_alerts;
drop policy if exists "Drift reports are tenant-isolated" on public.drift_reports;

-- Create Tenant Isolation Policies (tied to auth.uid())
create policy "Profiles are owner-only" on public.profiles
  for all using (id = auth.uid());

create policy "Orders are tenant-isolated" on public.orders
  for all using (tenant_id = auth.uid());

create policy "Datasets are tenant-isolated" on public.datasets
  for all using (tenant_id = auth.uid());

create policy "Audit log is tenant-isolated" on public.audit_log
  for all using (tenant_id = auth.uid());

create policy "Operational actions are tenant-isolated" on public.operational_actions
  for all using (tenant_id = auth.uid());

create policy "Model results are tenant-isolated" on public.model_results
  for all using (tenant_id = auth.uid());

create policy "CLV cache is tenant-isolated" on public.clv_cache
  for all using (tenant_id = auth.uid());

create policy "Anomaly cache is tenant-isolated" on public.anomaly_cache
  for all using (tenant_id = auth.uid());

create policy "Price recommendations are tenant-isolated" on public.price_recommendations
  for all using (tenant_id = auth.uid());

create policy "At-risk alerts are tenant-isolated" on public.at_risk_alerts
  for all using (tenant_id = auth.uid());

create policy "Drift reports are tenant-isolated" on public.drift_reports
  for all using (tenant_id = auth.uid());

-- ── Storage Bucket & Policies ─────────────────────────────────

-- Storage RLS: allow authenticated users to read/write their own folder in bucket 'datasets'
drop policy if exists "storage: tenant folder isolation" on storage.objects;
create policy "storage: tenant folder isolation"
  on storage.objects for all
  using (bucket_id = 'datasets' and (auth.uid()::text = (storage.foldername(name))[1]));
