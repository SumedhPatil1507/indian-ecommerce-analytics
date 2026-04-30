-- ============================================================
-- IndiaCommerce Analytics — Supabase Schema v3.0
-- Run this entire file in Supabase SQL Editor
-- ============================================================

-- Profiles
create table if not exists public.profiles (
  id uuid references auth.users(id) on delete cascade primary key,
  email text, plan text default 'starter',
  created_at timestamptz default now()
);

-- Datasets metadata
create table if not exists public.datasets (
  id bigserial primary key,
  user_id uuid references auth.users(id) on delete cascade,
  name text, row_count int,
  uploaded_at timestamptz default now()
);

-- Audit log
create table if not exists public.audit_log (
  id bigserial primary key,
  user_id uuid references auth.users(id) on delete cascade,
  action text, detail text,
  created_at timestamptz default now()
);

-- Price recommendations (Price Optimizer output)
create table if not exists public.price_recommendations (
  id bigserial primary key,
  user_id uuid references auth.users(id) on delete cascade,
  category text,
  current_discount float,
  optimal_discount float,
  direction text,
  revenue_impact_pct float,
  created_at timestamptz default now()
);

-- At-risk customer alerts
create table if not exists public.at_risk_alerts (
  id bigserial primary key,
  user_id uuid references auth.users(id) on delete cascade,
  customer_id text,
  churn_risk_score float,
  risk_label text,
  value_tier text,
  days_since_order int,
  total_revenue float,
  recommended_action text,
  created_at timestamptz default now()
);

-- Model drift reports
create table if not exists public.drift_reports (
  id bigserial primary key,
  user_id uuid references auth.users(id) on delete cascade,
  features_drifted int,
  max_psi float,
  pred_r2_drop float,
  drift_alert boolean,
  report_json text,
  created_at timestamptz default now()
);

-- ── RLS ──────────────────────────────────────────────────────
alter table public.profiles             enable row level security;
alter table public.datasets             enable row level security;
alter table public.audit_log            enable row level security;
alter table public.price_recommendations enable row level security;
alter table public.at_risk_alerts       enable row level security;
alter table public.drift_reports        enable row level security;

create policy "own profile"       on public.profiles              for all using (auth.uid()=id);
create policy "own datasets"      on public.datasets              for all using (auth.uid()=user_id);
create policy "own audit"         on public.audit_log             for all using (auth.uid()=user_id);
create policy "own price recs"    on public.price_recommendations for all using (auth.uid()=user_id);
create policy "own at risk"       on public.at_risk_alerts        for all using (auth.uid()=user_id);
create policy "own drift"         on public.drift_reports         for all using (auth.uid()=user_id);

-- ── Auto-create profile on signup ────────────────────────────
create or replace function public.handle_new_user()
returns trigger language plpgsql security definer as $$
begin
  insert into public.profiles(id,email) values(new.id,new.email)
  on conflict(id) do nothing; return new;
end;$$;

drop trigger if exists on_auth_user_created on auth.users;
create trigger on_auth_user_created after insert on auth.users
  for each row execute procedure public.handle_new_user();

-- ── Storage RLS ───────────────────────────────────────────────
drop policy if exists "storage: own folder" on storage.objects;
create policy "storage: own folder" on storage.objects for all
  using (auth.uid()::text = (storage.foldername(name))[1]::text);
