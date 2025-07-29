import os
import json
import hashlib
import base64
import secrets
from typing import Dict, Tuple, Optional

"""
auth.py
--------

This module contains utility functions to handle user authentication for the
financial portfolio manager.  Users can register with a username and password,
and their credentials are stored with a salted SHA‑256 hash.  Authentication
checks a provided password against the stored salted hash.

Passwords are never stored in plain text.  Each password is hashed using the
PBKDF2‑HMAC algorithm (SHA‑256) with a per‑user random salt and 100,000
iterations.  The salt and hashed password are encoded using base64 so they can
be persisted in JSON.
"""

# File paths for user data.  The users.json file stores a mapping from
# usernames to {'salt': str, 'hash': str}.  Use a dedicated directory under
# user_data to keep user‑related files isolated from code.
USER_DATA_DIR = os.path.join(os.path.dirname(__file__), "user_data")
USERS_FILE = os.path.join(USER_DATA_DIR, "users.json")

def _ensure_user_data_dir() -> None:
    """Ensure the directory for storing user data exists."""
    if not os.path.exists(USER_DATA_DIR):
        os.makedirs(USER_DATA_DIR, exist_ok=True)

def load_users() -> Dict[str, Dict[str, str]]:
    """
    Load the existing user database from disk.

    Returns
    -------
    dict
        A mapping from username to a dict with 'salt' and 'hash' keys.
    """
    _ensure_user_data_dir()
    if not os.path.isfile(USERS_FILE):
        return {}
    try:
        with open(USERS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        # If the file is corrupted, start over with an empty dict.
        return {}

def save_users(users: Dict[str, Dict[str, str]]) -> None:
    """
    Persist the users dictionary to disk.

    Parameters
    ----------
    users : dict
        Mapping of username to credential information.
    """
    _ensure_user_data_dir()
    with open(USERS_FILE, "w", encoding="utf-8") as f:
        json.dump(users, f, indent=2)

def _hash_password(password: str, salt: Optional[bytes] = None) -> Tuple[str, str]:
    """
    Hash a password using PBKDF2‑HMAC with SHA‑256.

    Parameters
    ----------
    password : str
        The plain text password to hash.
    salt : bytes, optional
        A random salt.  If None, a new salt is generated.

    Returns
    -------
    tuple of (str, str)
        Returns a tuple containing the base64‑encoded salt and the base64‑encoded hash.
    """
    if salt is None:
        salt = secrets.token_bytes(16)
    # Use PBKDF2 with 100k iterations.  This is a widely accepted standard.
    key = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 100_000)
    b64_salt = base64.b64encode(salt).decode("utf-8")
    b64_hash = base64.b64encode(key).decode("utf-8")
    return b64_salt, b64_hash

def register_user(username: str, password: str) -> bool:
    """
    Register a new user.

    Parameters
    ----------
    username : str
        Desired username.
    password : str
        Plain text password.

    Returns
    -------
    bool
        True if the user was registered successfully, False if the username already exists.
    """
    users = load_users()
    if username in users:
        return False
    salt, hashed = _hash_password(password)
    users[username] = {"salt": salt, "hash": hashed}
    save_users(users)
    return True

def authenticate_user(username: str, password: str) -> bool:
    """
    Verify a username and password combination.

    Parameters
    ----------
    username : str
        The username.
    password : str
        The plain text password supplied at login.

    Returns
    -------
    bool
        True if the credentials match a stored user, False otherwise.
    """
    users = load_users()
    user_record = users.get(username)
    if not user_record:
        return False
    try:
        salt = base64.b64decode(user_record["salt"])
        # stored_hash = base64.b64decode(user_record["hash"])
    except (KeyError, ValueError, TypeError):
        return False
    # Recompute the hash with the stored salt
    _, computed_hash_b64 = _hash_password(password, salt)
    return computed_hash_b64 == user_record["hash"]
