"""
Enhanced Authentication Module
==============================

Secure user authentication system for the portfolio manager with:
- PBKDF2-HMAC password hashing with salt
- JSON-based user storage with error handling
- Input validation and sanitization
- Comprehensive logging for security events
- Rate limiting protection (basic implementation)

Security Features:
- Passwords hashed with PBKDF2-HMAC-SHA256
- Individual salt per user (16 bytes)
- 100,000 iterations for hash computation
- Base64 encoding for storage compatibility
- No plaintext password storage ever

Author: Enhanced by AI Assistant
"""

import os
import json
import hashlib
import base64
import secrets
import logging
import time
from typing import Dict, Tuple, Optional, Union
from typing import List
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# Configuration Constants
# ============================================================================

# Security parameters
PBKDF2_ITERATIONS = 100_000
SALT_LENGTH = 16  # bytes
HASH_ALGORITHM = 'sha256'

# Rate limiting (basic implementation)
MAX_LOGIN_ATTEMPTS = 5
LOCKOUT_DURATION_MINUTES = 15

# File storage configuration
USER_DATA_DIR = os.path.join(os.path.dirname(__file__), "user_data")
USERS_FILE = os.path.join(USER_DATA_DIR, "users.json")
LOGIN_ATTEMPTS_FILE = os.path.join(USER_DATA_DIR, "login_attempts.json")

# ============================================================================
# Directory and File Management
# ============================================================================

def _ensure_user_data_dir() -> None:
    """Ensure the user data directory exists with proper permissions."""
    try:
        os.makedirs(USER_DATA_DIR, mode=0o700, exist_ok=True)
        logger.debug(f"User data directory ensured: {USER_DATA_DIR}")
    except Exception as e:
        logger.error(f"Failed to create user data directory: {e}")
        raise

def _create_backup_if_needed(file_path: str) -> None:
    """Create a backup of the file before modifying it."""
    if os.path.exists(file_path):
        try:
            backup_path = f"{file_path}.backup"
            with open(file_path, 'r', encoding='utf-8') as original:
                with open(backup_path, 'w', encoding='utf-8') as backup:
                    backup.write(original.read())
            logger.debug(f"Backup created: {backup_path}")
        except Exception as e:
            logger.warning(f"Failed to create backup for {file_path}: {e}")

# ============================================================================
# User Data Operations
# ============================================================================

def load_users() -> Dict[str, Dict[str, str]]:
    """
    Load the user database from disk with error handling.
    
    Returns
    -------
    Dict[str, Dict[str, str]]
        Mapping of username to user credentials and metadata
    """
    _ensure_user_data_dir()
    
    if not os.path.isfile(USERS_FILE):
        logger.info("Users file does not exist - returning empty user database")
        return {}
    
    try:
        with open(USERS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Validate the loaded data structure
        if not isinstance(data, dict):
            logger.error("Invalid users file format - not a dictionary")
            return {}
        
        # Validate each user record
        valid_users = {}
        for username, user_data in data.items():
            if isinstance(user_data, dict) and 'salt' in user_data and 'hash' in user_data:
                valid_users[username] = user_data
            else:
                logger.warning(f"Invalid user record for {username} - skipping")
        
        logger.info(f"Loaded {len(valid_users)} users from database")
        return valid_users
        
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error in users file: {e}")
        # Create backup and return empty dict
        _create_backup_if_needed(USERS_FILE)
        return {}
    except Exception as e:
        logger.error(f"Error loading users file: {e}")
        return {}

def save_users(users: Dict[str, Dict[str, str]]) -> bool:
    """
    Save the user database to disk with atomic operations.
    
    Parameters
    ----------
    users : Dict[str, Dict[str, str]]
        User database to save
        
    Returns
    -------
    bool
        True if save was successful, False otherwise
    """
    _ensure_user_data_dir()
    
    # Validate input
    if not isinstance(users, dict):
        logger.error("Invalid users data - not a dictionary")
        return False
    
    try:
        # Create backup of existing file
        _create_backup_if_needed(USERS_FILE)
        
        # Write to temporary file first (atomic operation)
        temp_file = f"{USERS_FILE}.tmp"
        
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(users, f, indent=2, ensure_ascii=False)
        
        # Atomic rename (on most systems)
        if os.name == 'nt':  # Windows
            if os.path.exists(USERS_FILE):
                os.remove(USERS_FILE)
        os.rename(temp_file, USERS_FILE)
        
        # Set restrictive permissions
        os.chmod(USERS_FILE, 0o600)
        
        logger.info(f"Successfully saved {len(users)} users to database")
        return True
        
    except Exception as e:
        logger.error(f"Error saving users file: {e}")
        # Clean up temp file if it exists
        try:
            if os.path.exists(temp_file):
                os.remove(temp_file)
        except:
            pass
        return False

# ============================================================================
# Password Hashing and Verification
# ============================================================================

def _hash_password(password: str, salt: Optional[bytes] = None) -> Tuple[str, str]:
    """
    Hash a password using PBKDF2-HMAC with SHA-256.
    
    Parameters
    ----------
    password : str
        Plain text password to hash
    salt : bytes, optional
        Cryptographic salt. If None, generates a new one.
        
    Returns
    -------
    Tuple[str, str]
        Base64-encoded salt and hash
        
    Raises
    ------
    ValueError
        If password is empty or invalid
    """
    if not password:
        raise ValueError("Password cannot be empty")
    
    if not isinstance(password, str):
        raise ValueError("Password must be a string")
    
    try:
        # Generate salt if not provided
        if salt is None:
            salt = secrets.token_bytes(SALT_LENGTH)
        
        # Validate salt length
        if len(salt) != SALT_LENGTH:
            raise ValueError(f"Salt must be {SALT_LENGTH} bytes long")
        
        # Hash the password
        key = hashlib.pbkdf2_hmac(
            HASH_ALGORITHM,
            password.encode('utf-8'),
            salt,
            PBKDF2_ITERATIONS
        )
        
        # Encode for storage
        b64_salt = base64.b64encode(salt).decode('utf-8')
        b64_hash = base64.b64encode(key).decode('utf-8')
        
        logger.debug("Password hashed successfully")
        return b64_salt, b64_hash
        
    except Exception as e:
        logger.error(f"Error hashing password: {e}")
        raise

def _verify_password(password: str, stored_salt: str, stored_hash: str) -> bool:
    """
    Verify a password against stored hash and salt.
    
    Parameters
    ----------
    password : str
        Plain text password to verify
    stored_salt : str
        Base64-encoded salt from storage
    stored_hash : str
        Base64-encoded hash from storage
        
    Returns
    -------
    bool
        True if password matches, False otherwise
    """
    try:
        # Decode stored values
        salt = base64.b64decode(stored_salt)
        expected_hash = stored_hash
        
        # Hash the provided password with the stored salt
        _, computed_hash = _hash_password(password, salt)
        
        # Constant-time comparison to prevent timing attacks
        return secrets.compare_digest(computed_hash, expected_hash)
        
    except Exception as e:
        logger.error(f"Error verifying password: {e}")
        return False

# ============================================================================
# Rate Limiting and Security
# ============================================================================

def load_login_attempts() -> Dict[str, Dict[str, Union[int, str]]]:
    """Load login attempt tracking data."""
    if not os.path.exists(LOGIN_ATTEMPTS_FILE):
        return {}
    
    try:
        with open(LOGIN_ATTEMPTS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading login attempts: {e}")
        return {}

def save_login_attempts(attempts: Dict[str, Dict[str, Union[int, str]]]) -> None:
    """Save login attempt tracking data."""
    _ensure_user_data_dir()
    
    try:
        with open(LOGIN_ATTEMPTS_FILE, 'w', encoding='utf-8') as f:
            json.dump(attempts, f, indent=2)
        os.chmod(LOGIN_ATTEMPTS_FILE, 0o600)
    except Exception as e:
        logger.error(f"Error saving login attempts: {e}")

def is_account_locked(username: str) -> bool:
    """
    Check if an account is temporarily locked due to failed attempts.
    
    Parameters
    ----------
    username : str
        Username to check
        
    Returns
    -------
    bool
        True if account is locked, False otherwise
    """
    attempts_data = load_login_attempts()
    user_attempts = attempts_data.get(username, {})
    
    if not user_attempts:
        return False
    
    try:
        failed_count = user_attempts.get('failed_count', 0)
        last_attempt_str = user_attempts.get('last_attempt', '')
        
        if failed_count < MAX_LOGIN_ATTEMPTS:
            return False
        
        if not last_attempt_str:
            return False
        
        # Parse last attempt time
        last_attempt = datetime.fromisoformat(last_attempt_str)
        lockout_duration = timedelta(minutes=LOCKOUT_DURATION_MINUTES)
        
        # Check if lockout period has expired
        if datetime.now() - last_attempt > lockout_duration:
            # Reset attempts count
            user_attempts['failed_count'] = 0
            attempts_data[username] = user_attempts
            save_login_attempts(attempts_data)
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"Error checking account lock status: {e}")
        return False

def record_login_attempt(username: str, success: bool) -> None:
    """
    Record a login attempt for rate limiting.
    
    Parameters
    ----------
    username : str
        Username that attempted login
    success : bool
        Whether the login was successful
    """
    attempts_data = load_login_attempts()
    
    if username not in attempts_data:
        attempts_data[username] = {'failed_count': 0, 'last_attempt': ''}
    
    user_attempts = attempts_data[username]
    user_attempts['last_attempt'] = datetime.now().isoformat()
    
    if success:
        user_attempts['failed_count'] = 0
        logger.info(f"Successful login recorded for {username}")
    else:
        user_attempts['failed_count'] = user_attempts.get('failed_count', 0) + 1
        logger.warning(f"Failed login attempt recorded for {username} (count: {user_attempts['failed_count']})")
    
    attempts_data[username] = user_attempts
    save_login_attempts(attempts_data)

# ============================================================================
# Input Validation and Sanitization
# ============================================================================

def validate_username(username: str) -> Tuple[bool, str]:
    """
    Validate username according to security requirements.
    
    Parameters
    ----------
    username : str
        Username to validate
        
    Returns
    -------
    Tuple[bool, str]
        (is_valid, error_message)
    """
    if not username:
        return False, "Username cannot be empty"
    
    if not isinstance(username, str):
        return False, "Username must be a string"
    
    # Clean the username
    username = username.strip()
    
    if len(username) < 3:
        return False, "Username must be at least 3 characters long"
    
    if len(username) > 50:
        return False, "Username must be less than 50 characters long"
    
    # Allow alphanumeric characters, underscores, and hyphens
    allowed_chars = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-')
    if not all(c in allowed_chars for c in username):
        return False, "Username can only contain letters, numbers, underscores, and hyphens"
    
    # Cannot start or end with special characters
    if username.startswith(('_', '-')) or username.endswith(('_', '-')):
        return False, "Username cannot start or end with special characters"
    
    return True, ""

def validate_password(password: str) -> Tuple[bool, str]:
    """
    Validate password according to security requirements.
    
    Parameters
    ----------
    password : str
        Password to validate
        
    Returns
    -------
    Tuple[bool, str]
        (is_valid, error_message)
    """
    if not password:
        return False, "Password cannot be empty"
    
    if not isinstance(password, str):
        return False, "Password must be a string"
    
    if len(password) < 6:
        return False, "Password must be at least 6 characters long"
    
    if len(password) > 128:
        return False, "Password must be less than 128 characters long"
    
    # Check for at least one letter and one number
    has_letter = any(c.isalpha() for c in password)
    has_number = any(c.isdigit() for c in password)
    
    if not (has_letter and has_number):
        return False, "Password must contain at least one letter and one number"
    
    return True, ""

def sanitize_username(username: str) -> str:
    """
    Sanitize username input.
    
    Parameters
    ----------
    username : str
        Raw username input
        
    Returns
    -------
    str
        Sanitized username
    """
    if not isinstance(username, str):
        return ""
    
    # Strip whitespace and convert to lowercase for consistency
    return username.strip().lower()

# ============================================================================
# Public Authentication Functions
# ============================================================================

def register_user(username: str, password: str) -> bool:
    """
    Register a new user with comprehensive validation.
    
    Parameters
    ----------
    username : str
        Desired username
    password : str
        Plain text password
        
    Returns
    -------
    bool
        True if registration successful, False otherwise
    """
    try:
        # Sanitize input
        username_clean = sanitize_username(username)
        
        # Validate inputs
        username_valid, username_error = validate_username(username_clean)
        if not username_valid:
            logger.warning(f"Registration failed - invalid username: {username_error}")
            return False
        
        password_valid, password_error = validate_password(password)
        if not password_valid:
            logger.warning(f"Registration failed - invalid password: {password_error}")
            return False
        
        # Load existing users
        users = load_users()
        
        # Check if username already exists
        if username_clean in users:
            logger.warning(f"Registration failed - username already exists: {username_clean}")
            return False
        
        # Hash the password
        salt, hashed_password = _hash_password(password)
        
        # Create user record with metadata
        users[username_clean] = {
            'salt': salt,
            'hash': hashed_password,
            'created_at': datetime.now().isoformat(),
            'last_login': '',
            'login_count': 0
        }
        
        # Save to database
        if save_users(users):
            logger.info(f"New user registered successfully: {username_clean}")
            return True
        else:
            logger.error(f"Failed to save new user: {username_clean}")
            return False
        
    except Exception as e:
        logger.error(f"Error during user registration: {e}")
        return False

def authenticate_user(username: str, password: str) -> bool:
    """
    Authenticate a user with comprehensive security checks.
    
    Parameters
    ----------
    username : str
        Username
    password : str
        Plain text password
        
    Returns
    -------
    bool
        True if authentication successful, False otherwise
    """
    try:
        # Sanitize input
        username_clean = sanitize_username(username)
        
        # Basic validation
        if not username_clean or not password:
            logger.warning("Authentication failed - empty username or password")
            record_login_attempt(username_clean, False)
            return False
        
        # Check if account is locked
        if is_account_locked(username_clean):
            logger.warning(f"Authentication failed - account locked: {username_clean}")
            return False
        
        # Load users database
        users = load_users()
        
        # Check if user exists
        if username_clean not in users:
            logger.warning(f"Authentication failed - user not found: {username_clean}")
            record_login_attempt(username_clean, False)
            return False
        
        user_record = users[username_clean]
        
        # Validate user record structure
        required_fields = ['salt', 'hash']
        if not all(field in user_record for field in required_fields):
            logger.error(f"Authentication failed - invalid user record: {username_clean}")
            record_login_attempt(username_clean, False)
            return False
        
        # Verify password
        if _verify_password(password, user_record['salt'], user_record['hash']):
            # Update user metadata
            user_record['last_login'] = datetime.now().isoformat()
            user_record['login_count'] = user_record.get('login_count', 0) + 1
            users[username_clean] = user_record
            save_users(users)
            
            # Record successful attempt
            record_login_attempt(username_clean, True)
            
            logger.info(f"Successful authentication: {username_clean}")
            return True
        else:
            logger.warning(f"Authentication failed - invalid password: {username_clean}")
            record_login_attempt(username_clean, False)
            return False
        
    except Exception as e:
        logger.error(f"Error during authentication: {e}")
        if username:
            record_login_attempt(sanitize_username(username), False)
        return False

# ============================================================================
# User Management Functions
# ============================================================================

def get_user_info(username: str) -> Optional[Dict[str, str]]:
    """
    Get user information (excluding sensitive data).
    
    Parameters
    ----------
    username : str
        Username to query
        
    Returns
    -------
    Optional[Dict[str, str]]
        User info or None if not found
    """
    try:
        username_clean = sanitize_username(username)
        users = load_users()
        
        if username_clean not in users:
            return None
        
        user_record = users[username_clean]
        
        # Return only non-sensitive information
        return {
            'username': username_clean,
            'created_at': user_record.get('created_at', ''),
            'last_login': user_record.get('last_login', ''),
            'login_count': str(user_record.get('login_count', 0))
        }
        
    except Exception as e:
        logger.error(f"Error getting user info: {e}")
        return None

def change_password(username: str, old_password: str, new_password: str) -> bool:
    """
    Change user password with validation.
    
    Parameters
    ----------
    username : str
        Username
    old_password : str
        Current password
    new_password : str
        New password
        
    Returns
    -------
    bool
        True if password change successful, False otherwise
    """
    try:
        # First authenticate with old password
        if not authenticate_user(username, old_password):
            logger.warning(f"Password change failed - authentication failed: {username}")
            return False
        
        # Validate new password
        password_valid, password_error = validate_password(new_password)
        if not password_valid:
            logger.warning(f"Password change failed - invalid new password: {password_error}")
            return False
        
        # Load users and update password
        username_clean = sanitize_username(username)
        users = load_users()
        
        if username_clean not in users:
            return False
        
        # Hash new password
        salt, hashed_password = _hash_password(new_password)
        
        # Update user record
        user_record = users[username_clean]
        user_record['salt'] = salt
        user_record['hash'] = hashed_password
        user_record['password_changed_at'] = datetime.now().isoformat()
        
        users[username_clean] = user_record
        
        # Save changes
        if save_users(users):
            logger.info(f"Password changed successfully: {username_clean}")
            return True
        else:
            logger.error(f"Failed to save password change: {username_clean}")
            return False
        
    except Exception as e:
        logger.error(f"Error changing password: {e}")
        return False

def delete_user(username: str, password: str) -> bool:
    """
    Delete a user account with authentication.
    
    Parameters
    ----------
    username : str
        Username to delete
    password : str
        Password for confirmation
        
    Returns
    -------
    bool
        True if deletion successful, False otherwise
    """
    try:
        # Authenticate first
        if not authenticate_user(username, password):
            logger.warning(f"User deletion failed - authentication failed: {username}")
            return False
        
        username_clean = sanitize_username(username)
        users = load_users()
        
        if username_clean not in users:
            return False
        
        # Remove user
        del users[username_clean]
        
        # Save changes
        if save_users(users):
            logger.info(f"User deleted successfully: {username_clean}")
            
            # Also clean up login attempts
            attempts_data = load_login_attempts()
            if username_clean in attempts_data:
                del attempts_data[username_clean]
                save_login_attempts(attempts_data)
            
            return True
        else:
            logger.error(f"Failed to save user deletion: {username_clean}")
            return False
        
    except Exception as e:
        logger.error(f"Error deleting user: {e}")
        return False

# ============================================================================
# Administrative Functions
# ============================================================================

def list_users() -> List[str]:
    """
    List all registered usernames (admin function).
    
    Returns
    -------
    List[str]
        List of usernames
    """
    try:
        users = load_users()
        return list(users.keys())
    except Exception as e:
        logger.error(f"Error listing users: {e}")
        return []

def get_system_stats() -> Dict[str, Union[int, str]]:
    """
    Get system statistics (admin function).
    
    Returns
    -------
    Dict[str, Union[int, str]]
        System statistics
    """
    try:
        users = load_users()
        attempts_data = load_login_attempts()
        
        locked_accounts = sum(1 for username in users.keys() if is_account_locked(username))
        
        return {
            'total_users': len(users),
            'locked_accounts': locked_accounts,
            'users_file_exists': os.path.exists(USERS_FILE),
            'last_backup': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting system stats: {e}")
        return {'error': str(e)}

# ============================================================================
# Module Initialization and Cleanup
# ============================================================================

def cleanup_old_login_attempts(days_old: int = 30) -> None:
    """
    Clean up old login attempt records.
    
    Parameters
    ----------
    days_old : int, default 30
        Remove attempts older than this many days
    """
    try:
        attempts_data = load_login_attempts()
        cutoff_date = datetime.now() - timedelta(days=days_old)
        
        users_to_remove = []
        for username, user_attempts in attempts_data.items():
            try:
                last_attempt_str = user_attempts.get('last_attempt', '')
                if last_attempt_str:
                    last_attempt = datetime.fromisoformat(last_attempt_str)
                    if last_attempt < cutoff_date:
                        users_to_remove.append(username)
            except:
                # If we can't parse the date, remove it
                users_to_remove.append(username)
        
        for username in users_to_remove:
            del attempts_data[username]
        
        if users_to_remove:
            save_login_attempts(attempts_data)
            logger.info(f"Cleaned up {len(users_to_remove)} old login attempt records")
        
    except Exception as e:
        logger.error(f"Error cleaning up login attempts: {e}")

def initialize_auth_module() -> bool:
    """
    Initialize the authentication module.
    
    Returns
    -------
    bool
        True if initialization successful
    """
    try:
        _ensure_user_data_dir()
        
        # Clean up old data periodically
        cleanup_old_login_attempts()
        
        logger.info("Authentication module initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize authentication module: {e}")
        return False

# Initialize on import
if __name__ != "__main__":
    initialize_auth_module()
