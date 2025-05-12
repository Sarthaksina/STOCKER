"""Agent routes for STOCKER Pro API.

This module provides API endpoints for interacting with the STOCKER agent system.
"""
from fastapi import APIRouter, Depends
from typing import Dict, Any

from src.features.mega_agent import MegaAgent

router = APIRouter(prefix="/agent", tags=["agent"])
mega_agent = MegaAgent()

@router.post("/execute")
def api_agent(task: str, params: Dict[str, Any]):
    """Execute any agent task by name using MegaAgent."""
    return mega_agent.execute(task, params)
