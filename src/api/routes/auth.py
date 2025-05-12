"""
Authentication routes for STOCKER Pro API.

This module provides routes for user authentication and management.
"""
from fastapi import APIRouter, Depends, HTTPException, Body
from pydantic import BaseModel, EmailStr
from typing import Optional, Dict, Any

from src.api.dependencies import get_auth_service, get_current_user
from src.services.auth import AuthService, AuthError


# Pydantic models for request validation
class UserRegisterRequest(BaseModel):
    """User registration request."""
    username: str
    email: EmailStr
    password: str
    first_name: Optional[str] = None
    last_name: Optional[str] = None


class UserLoginRequest(BaseModel):
    """User login request."""
    username_or_email: str
    password: str


class TokenResponse(BaseModel):
    """Token response."""
    access_token: str
    token_type: str
    user: Dict[str, Any]


class UserUpdateRequest(BaseModel):
    """User update request."""
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    email: Optional[EmailStr] = None


# Create router
router = APIRouter(
    prefix="/auth",
    tags=["authentication"],
    responses={
        401: {"description": "Not authenticated"},
        403: {"description": "Not authorized"}
    }
)


@router.post("/register", response_model=Dict[str, Any])
async def register_user(
    request: UserRegisterRequest,
    auth_service: AuthService = Depends(get_auth_service)
):
    """
    Register a new user.
    
    Args:
        request: User registration request
        auth_service: Authentication service
        
    Returns:
        Registered user data
        
    Raises:
        HTTPException: If registration fails
    """
    try:
        user = auth_service.register_user(
            username=request.username,
            email=request.email,
            password=request.password,
            first_name=request.first_name,
            last_name=request.last_name
        )
        return user
    except AuthError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Registration failed: {str(e)}")


@router.post("/login", response_model=TokenResponse)
async def login(
    request: UserLoginRequest,
    auth_service: AuthService = Depends(get_auth_service)
):
    """
    Authenticate a user.
    
    Args:
        request: User login request
        auth_service: Authentication service
        
    Returns:
        Token and user data
        
    Raises:
        HTTPException: If authentication fails
    """
    try:
        user = auth_service.authenticate_user(
            username_or_email=request.username_or_email,
            password=request.password
        )
        token = auth_service.generate_token(user)
        
        return {
            "access_token": token,
            "token_type": "bearer",
            "user": user
        }
    except AuthError as e:
        raise HTTPException(status_code=401, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Login failed: {str(e)}")


@router.get("/me", response_model=Dict[str, Any])
async def get_current_user_info(
    user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Get current user info.
    
    Args:
        user: Current user
        
    Returns:
        User data
    """
    return user


@router.put("/me", response_model=Dict[str, Any])
async def update_user_info(
    request: UserUpdateRequest,
    user: Dict[str, Any] = Depends(get_current_user),
    auth_service: AuthService = Depends(get_auth_service)
):
    """
    Update current user info.
    
    Args:
        request: User update request
        user: Current user
        auth_service: Authentication service
        
    Returns:
        Updated user data
        
    Raises:
        HTTPException: If update fails
    """
    try:
        updates = request.dict(exclude_unset=True)
        if not updates:
            return user
            
        updated_user = auth_service.update_user(user["id"], updates)
        return updated_user
    except AuthError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Update failed: {str(e)}")


@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(
    user: Dict[str, Any] = Depends(get_current_user),
    auth_service: AuthService = Depends(get_auth_service)
):
    """
    Refresh access token.
    
    Args:
        user: Current user
        auth_service: Authentication service
        
    Returns:
        New token and user data
    """
    token = auth_service.generate_token(user)
    
    return {
        "access_token": token,
        "token_type": "bearer",
        "user": user
    } 