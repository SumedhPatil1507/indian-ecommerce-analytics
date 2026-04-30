"""
core/auth.py - Supabase authentication for Streamlit.
Falls back to demo mode when Supabase is not configured.
"""
from __future__ import annotations
import streamlit as st
from core.config import settings


def _client():
    if not settings.supabase_ready:
        return None
    try:
        from supabase import create_client
        return create_client(settings.supabase_url, settings.supabase_anon_key)
    except Exception:
        return None


def is_authenticated() -> bool:
    return st.session_state.get("user") is not None


def current_user() -> dict:
    return st.session_state.get("user", {})


def render_auth_wall() -> bool:
    """Returns True if user is authenticated or in demo mode."""
    if not settings.supabase_ready:
        st.session_state["user"] = {
            "email": "demo@indiacommerce.app",
            "id": "demo",
            "role": "admin",
            "demo": True,
        }
        return True

    if is_authenticated():
        return True

    # Centre the login card
    _, col, _ = st.columns([1, 2, 1])
    with col:
        st.markdown("""
        <div style='padding:32px;background:#fff;border-radius:16px;
             box-shadow:0 4px 24px rgba(0,0,0,0.10);margin-top:60px;'>
        """, unsafe_allow_html=True)
        st.markdown(f"## {settings.app_name}")
        st.markdown("Sign in to access your analytics dashboard.")

        tab_in, tab_up = st.tabs(["Sign In", "Create Account"])
        with tab_in:
            email = st.text_input("Email", key="li_email")
            pwd   = st.text_input("Password", type="password", key="li_pwd")
            if st.button("Sign In", use_container_width=True, type="primary"):
                _do_login(email, pwd)
        with tab_up:
            email2 = st.text_input("Email", key="su_email")
            pwd2   = st.text_input("Password (min 8 chars)", type="password", key="su_pwd")
            if st.button("Create Account", use_container_width=True):
                _do_signup(email2, pwd2)

        st.markdown("</div>", unsafe_allow_html=True)
    return False


def _do_login(email: str, password: str) -> None:
    if not email or not password:
        st.warning("Enter email and password.")
        return
    client = _client()
    if not client:
        st.error("Supabase not configured.")
        return
    try:
        res = client.auth.sign_in_with_password({"email": email, "password": password})
        # supabase-py v2: res.user and res.session
        st.session_state["user"] = {
            "email": res.user.email,
            "id":    str(res.user.id),
            "role":  "admin",
            "demo":  False,
        }
        st.session_state["access_token"] = res.session.access_token
        st.rerun()
    except Exception as e:
        st.error(f"Login failed: {e}")


def _do_signup(email: str, password: str) -> None:
    if not email or not password:
        st.warning("Enter email and password.")
        return
    client = _client()
    if not client:
        st.error("Supabase not configured.")
        return
    try:
        client.auth.sign_up({"email": email, "password": password})
        st.success("Account created! Sign in below.")
    except Exception as e:
        st.error(f"Sign-up failed: {e}")


def logout() -> None:
    for k in ["user", "access_token", "df"]:
        st.session_state.pop(k, None)
    st.rerun()
