"""
Auth Data Models
Defines authentication and user data structures
"""
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from datetime import datetime


@dataclass
class LoginRequest:
    """Login request parameters"""
    email: str
    password: str

    def validate(self) -> List[str]:
        errors = []
        if not self.email or not self.email.strip():
            errors.append("Email is required")
        elif "@" not in self.email:
            errors.append("Invalid email format")
        if not self.password or len(self.password) < 6:
            errors.append("Password must be at least 6 characters")
        return errors


@dataclass
class SignUpRequest:
    """Sign-up request parameters"""
    email: str
    password: str
    full_name: str
    organization: Optional[str] = None
    purpose: Optional[str] = None

    def validate(self) -> List[str]:
        errors = []
        if not self.email or not self.email.strip():
            errors.append("Email is required")
        elif "@" not in self.email:
            errors.append("Invalid email format")
        if not self.password or len(self.password) < 6:
            errors.append("Password must be at least 6 characters")
        if not self.full_name or not self.full_name.strip():
            errors.append("Full name is required")
        return errors


@dataclass
class ProfileUpdate:
    """Profile update parameters"""
    full_name: Optional[str] = None
    organization: Optional[str] = None
    purpose: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Return only non-None fields"""
        d = {}
        if self.full_name is not None:
            d["full_name"] = self.full_name
        if self.organization is not None:
            d["organization"] = self.organization
        if self.purpose is not None:
            d["purpose"] = self.purpose
        return d


VALID_ACTIVITY_TYPES = [
    "login", "logout", "signup",
    "prediction_run", "weather_forecast",
    "chatbot_query", "profile_update",
    "settings_change", "dataset_view",
]


@dataclass
class ActivityLogEntry:
    """Activity log entry"""
    activity_type: str
    description: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    device_info: Optional[str] = None

    def validate(self) -> List[str]:
        errors = []
        if not self.activity_type:
            errors.append("activity_type is required")
        elif self.activity_type not in VALID_ACTIVITY_TYPES:
            errors.append(f"Invalid activity_type. Must be one of: {', '.join(VALID_ACTIVITY_TYPES)}")
        return errors
