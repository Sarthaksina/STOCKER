"""
Authentication service for STOCKER Pro.

This module provides services for user management, authentication,
and authorization.
"""
import jwt
import bcrypt
from datetime import datetime, timedelta
from typing import Dict, Optional, Union, Any

from src.core.config import config
from src.core.logging import logger
from src.db.models import User
from src.core.exceptions import StockerBaseException


class AuthError(StockerBaseException):
    """Exception raised for authentication errors."""
    pass


class AuthService:
    """
    Authentication service for user management and authentication.
    
    Provides methods for user registration, login, token generation,
    and user management.
    """
    
    def __init__(self):
        """Initialize the authentication service."""
        from src.db.session import get_mongodb_db
        
        try:
            db = get_mongodb_db()
            self.users_collection = db[config.database.user_collection]
            logger.info("Auth service initialized")
        except Exception as e:
            logger.error(f"Failed to initialize auth service: {e}")
            raise
    
    def register_user(self, username: str, email: str, password: str, 
                     first_name: Optional[str] = None, last_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Register a new user.
        
        Args:
            username: Username
            email: Email address
            password: Password (will be hashed)
            first_name: First name (optional)
            last_name: Last name (optional)
            
        Returns:
            User data without password
            
        Raises:
            AuthError: If user already exists or registration fails
        """
        try:
            # Check if user already exists
            existing_user = self.users_collection.find_one({"$or": [{"username": username}, {"email": email}]})
            if existing_user:
                logger.warning(f"User registration failed: User already exists (username={username}, email={email})")
                raise AuthError("User with that username or email already exists")
            
            # Hash password
            salt = bcrypt.gensalt()
            hashed_password = bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')
            
            # Create user
            now = datetime.now().isoformat()
            user = User(
                username=username,
                email=email,
                hashed_password=hashed_password,
                first_name=first_name,
                last_name=last_name,
                created_at=now,
                updated_at=now,
                is_active=True
            )
            
            # Insert into database
            result = self.users_collection.insert_one(user.dict())
            
            logger.info(f"User registered successfully: {username}")
            
            # Return user data without password
            user_data = user.dict()
            user_data.pop("hashed_password")
            return user_data
            
        except AuthError:
            raise
        except Exception as e:
            logger.error(f"User registration failed: {e}")
            raise AuthError(f"Failed to register user: {e}")
    
    def authenticate_user(self, username_or_email: str, password: str) -> Dict[str, Any]:
        """
        Authenticate a user.
        
        Args:
            username_or_email: Username or email
            password: Password
            
        Returns:
            User data without password
            
        Raises:
            AuthError: If authentication fails
        """
        try:
            # Find user
            user = self.users_collection.find_one({
                "$or": [{"username": username_or_email}, {"email": username_or_email}]
            })
            
            if not user:
                logger.warning(f"Authentication failed: User not found ({username_or_email})")
                raise AuthError("Invalid username/email or password")
            
            # Check if user is active
            if not user.get("is_active", True):
                logger.warning(f"Authentication failed: User account is inactive ({username_or_email})")
                raise AuthError("User account is inactive")
            
            # Verify password
            hashed_password = user.get("hashed_password", "")
            if not bcrypt.checkpw(password.encode('utf-8'), hashed_password.encode('utf-8')):
                logger.warning(f"Authentication failed: Invalid password for user ({username_or_email})")
                raise AuthError("Invalid username/email or password")
            
            # Return user data without password
            user_data = user.copy()
            user_data.pop("hashed_password", None)
            user_data.pop("_id", None)  # Remove MongoDB ObjectId
            
            logger.info(f"User authenticated successfully: {username_or_email}")
            return user_data
            
        except AuthError:
            raise
        except Exception as e:
            logger.error(f"Authentication failed: {e}")
            raise AuthError(f"Failed to authenticate user: {e}")
    
    def generate_token(self, user_data: Dict[str, Any], expires_in_minutes: int = 60) -> str:
        """
        Generate a JWT token for a user.
        
        Args:
            user_data: User data
            expires_in_minutes: Token expiration time in minutes
            
        Returns:
            JWT token
        """
        try:
            expiration = datetime.utcnow() + timedelta(minutes=expires_in_minutes)
            
            # Create token payload
            payload = {
                "sub": user_data.get("id") or user_data.get("username"),
                "username": user_data.get("username"),
                "email": user_data.get("email"),
                "is_admin": user_data.get("is_admin", False),
                "exp": expiration
            }
            
            # Generate token
            token = jwt.encode(
                payload,
                config.api.jwt_secret_key or "stocker_secret",
                algorithm="HS256"
            )
            
            logger.debug(f"Generated JWT token for user: {user_data.get('username')}")
            return token
            
        except Exception as e:
            logger.error(f"Failed to generate token: {e}")
            raise AuthError(f"Failed to generate token: {e}")
    
    def verify_token(self, token: str) -> Dict[str, Any]:
        """
        Verify and decode a JWT token.
        
        Args:
            token: JWT token
            
        Returns:
            Decoded token payload
            
        Raises:
            AuthError: If token is invalid or expired
        """
        try:
            payload = jwt.decode(
                token,
                config.api.jwt_secret_key or "stocker_secret",
                algorithms=["HS256"]
            )
            
            return payload
            
        except jwt.ExpiredSignatureError:
            logger.warning("Token verification failed: Token has expired")
            raise AuthError("Token has expired")
        except jwt.InvalidTokenError:
            logger.warning("Token verification failed: Invalid token")
            raise AuthError("Invalid token")
        except Exception as e:
            logger.error(f"Token verification failed: {e}")
            raise AuthError(f"Failed to verify token: {e}")
    
    def get_user_by_id(self, user_id: str) -> Dict[str, Any]:
        """
        Get user by ID.
        
        Args:
            user_id: User ID
            
        Returns:
            User data without password
            
        Raises:
            AuthError: If user not found
        """
        try:
            user = self.users_collection.find_one({"id": user_id})
            
            if not user:
                logger.warning(f"User not found: {user_id}")
                raise AuthError("User not found")
            
            # Remove sensitive data
            user_data = user.copy()
            user_data.pop("hashed_password", None)
            user_data.pop("_id", None)
            
            return user_data
            
        except AuthError:
            raise
        except Exception as e:
            logger.error(f"Failed to get user: {e}")
            raise AuthError(f"Failed to get user: {e}")
    
    def update_user(self, user_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update user information.
        
        Args:
            user_id: User ID
            updates: Fields to update
            
        Returns:
            Updated user data without password
            
        Raises:
            AuthError: If user not found or update fails
        """
        try:
            # Don't allow updating sensitive fields directly
            updates.pop("id", None)
            updates.pop("hashed_password", None)
            updates.pop("is_admin", None)
            
            # Update timestamp
            updates["updated_at"] = datetime.now().isoformat()
            
            # Update user
            result = self.users_collection.update_one(
                {"id": user_id},
                {"$set": updates}
            )
            
            if result.matched_count == 0:
                logger.warning(f"User update failed: User not found ({user_id})")
                raise AuthError("User not found")
            
            # Return updated user
            updated_user = self.get_user_by_id(user_id)
            logger.info(f"User updated successfully: {user_id}")
            
            return updated_user
            
        except AuthError:
            raise
        except Exception as e:
            logger.error(f"User update failed: {e}")
            raise AuthError(f"Failed to update user: {e}")
    
    def delete_user(self, user_id: str) -> bool:
        """
        Delete a user.
        
        Args:
            user_id: User ID
            
        Returns:
            True if successful
            
        Raises:
            AuthError: If user not found or deletion fails
        """
        try:
            result = self.users_collection.delete_one({"id": user_id})
            
            if result.deleted_count == 0:
                logger.warning(f"User deletion failed: User not found ({user_id})")
                raise AuthError("User not found")
            
            logger.info(f"User deleted successfully: {user_id}")
            return True
            
        except AuthError:
            raise
        except Exception as e:
            logger.error(f"User deletion failed: {e}")
            raise AuthError(f"Failed to delete user: {e}") 


# Utility functions
def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify a password against a hash.
    
    Args:
        plain_password: Plain text password
        hashed_password: Hashed password
        
    Returns:
        True if password matches hash, False otherwise
    """
    try:
        return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))
    except Exception as e:
        logger.error(f"Password verification failed: {e}")
        return False


def create_access_token(user_data: Dict[str, Any], expires_in_minutes: int = 60) -> str:
    """
    Create a JWT access token for a user.
    
    This is a convenience wrapper around AuthService.generate_token.
    
    Args:
        user_data: User data dictionary
        expires_in_minutes: Token expiration time in minutes
        
    Returns:
        JWT token string
    """
    auth_service = get_auth_service()
    return auth_service.generate_token(user_data, expires_in_minutes)


# Singleton instance
_auth_service_instance = None


def get_auth_service() -> AuthService:
    """
    Get the singleton instance of the AuthService.
    
    Returns:
        AuthService instance
    """
    global _auth_service_instance
    
    if _auth_service_instance is None:
        _auth_service_instance = AuthService()
        
    return _auth_service_instance