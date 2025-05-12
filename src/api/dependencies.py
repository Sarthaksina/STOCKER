
"""
FastAPI dependencies for STOCKER Pro API.

This module provides dependency injection functions for the FastAPI application.
"""
from fastapi import Depends, HTTPException, Header
from typing import Optional
import jwt

from src.core.config import config
from src.services.auth import AuthService
from src.services.portfolio import PortfolioService
from src.services.prediction import PredictionService
from src.services.training import TrainingService
from src.data.manager import DataManager


# Database dependencies
def get_db():
    """Get database session.
    
    Returns:
        Session: SQLAlchemy database session
    """
    from src.db.session import get_session
    session = get_session()
    try:
        yield session
    finally:
        session.close()

# Service dependencies
def get_auth_service() -> AuthService:
    """Get authentication service instance."""
    return AuthService()

def get_portfolio_service() -> PortfolioService:
    """Get portfolio service instance."""
    return PortfolioService()

def get_prediction_service() -> PredictionService:
    """Get prediction service instance."""
    return PredictionService()

def get_training_service() -> TrainingService:
    """Get training service instance."""
    return TrainingService()

def get_data_manager() -> DataManager:
    """Get data manager instance."""
    return DataManager()


# Authentication dependencies
def get_token_payload(authorization: Optional[str] = Header(None)) -> dict:
    """
    Extract and verify JWT token from Authorization header.
    
    Args:
        authorization: Authorization header
        
    Returns:
        Token payload
        
    Raises:
        HTTPException: If token is invalid or missing
    """
    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization header missing")
        
    try:
        # Extract token from "Bearer <token>"
        scheme, token = authorization.split()
        if scheme.lower() != "bearer":
            raise HTTPException(status_code=401, detail="Invalid authentication scheme")
            
        # Decode token
        payload = jwt.decode(
            token,
            config.api.jwt_secret_key or "stocker_secret",
            algorithms=["HS256"]
        )
        
        return payload
        
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token has expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Authentication error: {str(e)}")

def get_current_user(payload: dict = Depends(get_token_payload), auth_service: AuthService = Depends(get_auth_service)) -> dict:
    """
    Get current authenticated user.
    
    Args:
        payload: Token payload
        auth_service: Authentication service
        
    Returns:
        User data
        
    Raises:
        HTTPException: If user not found
    """
    try:
        user_id = payload.get("sub")
        return auth_service.get_user_by_id(user_id)
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Invalid credentials: {str(e)}")

def get_admin_user(user: dict = Depends(get_current_user)) -> dict:
    """
    Check if current user is an admin.
    
    Args:
        user: Current user
        
    Returns:
        User data
        
    Raises:
        HTTPException: If user is not an admin
    """
    if not user.get("is_admin", False):
        raise HTTPException(status_code=403, detail="Admin access required")
    return user 