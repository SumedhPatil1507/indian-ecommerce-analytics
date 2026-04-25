"""
core/auth.py — Supabase authentication for Streamlit.
Handles sign-up, login, logout, and session persistence via st.session_state.
Falls back gracefully when Supabase is not configured (demo mode).
"""
from __future__ import annotations
import streamlit as st
from core.config import settings


def _client():
    """Return a Supabase client or None if not configured."""
    if not settings.supabase_ready:
        return None
    try:
        from supabase import create_client  # type: ignore
        return create_client(settings.supabase_url, settings.supabase_anon_key)
    except ImportError:
        return None


def is_authenticated() -> bool:
    return st.session_state.get("user") is not None


def current_user() -> dict | None:
    return st.session_state.get("user")


def render_auth_wall() -> bool:
    """
    Render login/signup UI. Returns True if user is authenticated.
    In demo mode (no Supabase configured) always returns True.
    """
    if not settings.supabase_ready:
        # Demo mode — no auth required
        st.session_state["user"] = {"email": "demo@indiacommerce.app", "role": "admin", "demo": True}
        return True

    if is_authenticated():
        return True

    st.markdown("""
    <div style='max-width:420px;margin:80px auto;padding:40px;
         background:#fff;border-radius:16px;
         box-shadow:0 4px 24px rgba(0,0,0,0.10);'>
    """, unsafe_allow_html=True)

    st.image("https://img.icons8.com/fluency/96/combo-chart.png", width=64)
    st.markdown(f"### {settings.app_name}")
    st.markdown("Sign in to access your analytics dashboard.")

    tab_login, tab_signup = st.tabs(["Sign In", "Create Account"])

    with tab_login:
        email    = st.text_input("Email", key="login_email")
        password = st.text_input("Password", type="password", key="login_pass")
        if st.button("Sign In", use_container_width=True, type="primary"):
            _do_login(email, password)

    with tab_signup:
        email2 = st.text_input("Email", key="signup_email")
        pass2  = st.text_input("Password (min 8 chars)", type="password", key="signup_pass")
        if st.button("Create Account", use_container_width=True):
            _do_signup(email2, pass2)

    st.markdown("</div>", unsafe_allow_html=True)
    return False


def _do_login(email: str, password: str) -> None:
    client = _client()
    if not client:
        st.error("Auth not configured.")
        return
    try:
        res = client.auth.sign_in_with_password({"email": email, "password": password})
        st.session_state["user"] = {
            "email": res.user.email,
            "id":    res.user.id,
            "role":  "admin",
        }
        st.session_state["access_token"] = res.session.access_token
        st.rerun()
    except Exception as e:
        st.error(f"Login failed: {e}")


def _do_signup(email: str, password: str) -> None:
    client = _client()
    if not client:
        st.error("Auth not configured.")
        return
    try:
        client.auth.sign_up({"email": email, "password": password})
        st.success("Account created! Check your email to confirm, then sign in.")
    except Exception as e:
        st.error(f"Sign-up failed: {e}")


def logout() -> None:
    for key in ["user", "access_token", "df"]:
        st.session_state.pop(key, None)
    st.rerun()
