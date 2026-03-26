"""
Auth Service
Wraps Supabase Python client for authentication and user data operations.
Uses the service-role key so the backend has full admin access (bypasses RLS).
"""
import logging
from typing import Dict, Any, Optional, Tuple
from supabase import create_client, Client

logger = logging.getLogger(__name__)


class AuthService:
    """Service layer for Supabase auth and user data operations"""

    def __init__(self, supabase_url: str, supabase_service_key: str):
        """
        Initialize the auth service with Supabase credentials.
        Uses the SERVICE ROLE key so all database operations bypass RLS.
        """
        self.url = supabase_url
        self._client: Client = create_client(supabase_url, supabase_service_key)
        logger.info("AuthService initialized with Supabase service-role client")

    # ── Auth Operations ─────────────────────────────────────────────────

    def sign_in(self, email: str, password: str) -> Tuple[bool, Dict[str, Any]]:
        """
        Sign in with email + password.
        Returns (success, data_or_error).
        """
        try:
            res = self._client.auth.sign_in_with_password({
                "email": email,
                "password": password,
            })
            user = res.user
            session = res.session

            if not user or not session:
                return False, {"error": "Invalid credentials"}

            return True, {
                "user": {
                    "id": user.id,
                    "email": user.email,
                    "name": user.user_metadata.get("full_name", user.email.split("@")[0]),
                    "email_confirmed": user.email_confirmed_at is not None,
                },
                "access_token": session.access_token,
                "refresh_token": session.refresh_token,
                "expires_in": session.expires_in,
            }

        except Exception as e:
            msg = str(e)
            logger.error(f"[AUTH] sign_in error: {msg}")
            # Extract readable message from Supabase errors
            if "Invalid login credentials" in msg:
                return False, {"error": "Invalid email or password"}
            return False, {"error": msg}

    def sign_up(self, email: str, password: str, full_name: str,
                organization: str = None, purpose: str = None) -> Tuple[bool, Dict[str, Any]]:
        """
        Register a new user.
        Also upserts their profile row so extra fields are stored.
        """
        try:
            res = self._client.auth.sign_up({
                "email": email,
                "password": password,
                "options": {
                    "data": {"full_name": full_name},
                },
            })
            user = res.user
            session = res.session
            needs_verification = session is None

            if not user:
                return False, {"error": "Sign-up failed – no user returned"}

            # Upsert profile (the DB trigger also creates one, but we want
            # to store organization & purpose immediately)
            self._upsert_profile(user.id, {
                "full_name": full_name,
                "email": email,
                "organization": organization,
                "purpose": purpose,
            })

            result: Dict[str, Any] = {
                "user": {
                    "id": user.id,
                    "email": user.email,
                    "name": full_name,
                },
                "needs_verification": needs_verification,
            }

            if session:
                result["access_token"] = session.access_token
                result["refresh_token"] = session.refresh_token
                result["expires_in"] = session.expires_in

            return True, result

        except Exception as e:
            msg = str(e)
            logger.error(f"[AUTH] sign_up error: {msg}")
            if "already registered" in msg.lower() or "already been registered" in msg.lower():
                return False, {"error": "An account with this email already exists"}
            return False, {"error": msg}

    def verify_token(self, access_token: str) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        Verify a Supabase JWT and return the user info.
        Returns (is_valid, user_dict_or_None).
        """
        try:
            res = self._client.auth.get_user(access_token)
            user = res.user
            if not user:
                return False, None
            return True, {
                "id": user.id,
                "email": user.email,
                "name": user.user_metadata.get("full_name", user.email.split("@")[0]),
                "email_confirmed": user.email_confirmed_at is not None,
            }
        except Exception as e:
            logger.warning(f"[AUTH] token verification failed: {e}")
            return False, None

    def sign_out(self, access_token: str) -> bool:
        """Sign out (invalidate the session on Supabase side)."""
        try:
            # Admin sign-out requires the user's JWT
            self._client.auth.admin.sign_out(access_token)
            return True
        except Exception:
            # Even if server-side sign-out fails, the frontend clears the token
            return True

    def refresh_session(self, refresh_token: str) -> Tuple[bool, Dict[str, Any]]:
        """Refresh an expired session using a refresh token."""
        try:
            res = self._client.auth.refresh_session(refresh_token)
            session = res.session
            user = res.user
            if not session or not user:
                return False, {"error": "Refresh failed"}
            return True, {
                "user": {
                    "id": user.id,
                    "email": user.email,
                    "name": user.user_metadata.get("full_name", user.email.split("@")[0]),
                    "email_confirmed": user.email_confirmed_at is not None,
                },
                "access_token": session.access_token,
                "refresh_token": session.refresh_token,
                "expires_in": session.expires_in,
            }
        except Exception as e:
            logger.error(f"[AUTH] refresh error: {e}")
            return False, {"error": str(e)}

    # ── Profile Operations ──────────────────────────────────────────────

    def get_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Fetch a user's profile row."""
        try:
            res = (
                self._client.table("user_profiles")
                .select("*")
                .eq("user_id", user_id)
                .single()
                .execute()
            )
            return res.data
        except Exception as e:
            logger.error(f"[AUTH] get_profile error: {e}")
            return None

    def update_profile(self, user_id: str, updates: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Update profile fields for a user."""
        try:
            self._client.table("user_profiles").update(updates).eq("user_id", user_id).execute()
            return True, None
        except Exception as e:
            logger.error(f"[AUTH] update_profile error: {e}")
            return False, str(e)

    def _upsert_profile(self, user_id: str, data: Dict[str, Any]):
        """Internal: upsert profile during signup."""
        try:
            row = {
                "user_id": user_id,
                "full_name": data.get("full_name", ""),
                "email": data.get("email", ""),
                "organization": data.get("organization"),
                "purpose": data.get("purpose"),
            }
            self._client.table("user_profiles").upsert(
                row, on_conflict="user_id"
            ).execute()
        except Exception as e:
            logger.warning(f"[AUTH] _upsert_profile warning (non-fatal): {e}")

    # ── Email Verification ───────────────────────────────────────────────

    def resend_verification_email(self, email: str) -> Tuple[bool, Optional[str]]:
        """
        Resend the signup confirmation / verification email.
        Uses the Supabase auth.resend() method.
        """
        try:
            self._client.auth.resend({
                "type": "signup",
                "email": email,
            })
            logger.info(f"[AUTH] Verification email resent to {email}")
            return True, None
        except Exception as e:
            msg = str(e)
            logger.error(f"[AUTH] resend_verification error: {msg}")
            if "rate" in msg.lower() or "limit" in msg.lower():
                return False, "Please wait a few minutes before requesting another verification email."
            return False, msg

    # ── Activity Log Operations ─────────────────────────────────────────

    def log_activity(self, user_id: str, activity_type: str,
                     description: str = None, metadata: dict = None,
                     device_info: str = None) -> bool:
        """Insert an activity log row."""
        try:
            self._client.table("activity_logs").insert({
                "user_id": user_id,
                "activity_type": activity_type,
                "description": description,
                "metadata": metadata or {},
                "device_info": device_info,
            }).execute()
            return True
        except Exception as e:
            logger.warning(f"[AUTH] log_activity warning: {e}")
            return False

    def get_activity_logs(self, user_id: str, limit: int = 50) -> list:
        """Fetch recent activity logs for a user."""
        try:
            res = (
                self._client.table("activity_logs")
                .select("*")
                .eq("user_id", user_id)
                .order("created_at", desc=True)
                .limit(limit)
                .execute()
            )
            return res.data or []
        except Exception as e:
            logger.error(f"[AUTH] get_activity_logs error: {e}")
            return []

    # ── Service Status ──────────────────────────────────────────────────

    def get_service_status(self) -> Dict[str, Any]:
        return {
            "service": "AuthService",
            "status": "healthy",
            "supabase_url": self.url,
        }
