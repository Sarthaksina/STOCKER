"""
Portfolio Real-time Updates Module for STOCKER Pro

This module provides real-time portfolio monitoring and alerts.
"""

import numpy as np
import pandas as pd
import logging
import threading
import time
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from datetime import datetime, timedelta
import queue

from stocker.cloud.portfolio_config import PortfolioConfig

# Configure logging
logger = logging.getLogger(__name__)

class PortfolioMonitor:
    """
    Real-time portfolio monitoring and alerts
    """
    
    def __init__(self, config: Optional[PortfolioConfig] = None):
        """
        Initialize portfolio monitor
        
        Args:
            config: Portfolio configuration
        """
        self.config = config or PortfolioConfig()
        
        # Monitor state
        self.running = False
        self.update_thread = None
        self.update_interval = 60  # Default update interval in seconds
        self.data_queue = queue.Queue()
        self.last_update = None
        self.portfolio_data = None
        self.alerts = []
        self.alert_callbacks = []
        self.alert_thresholds = {}
        
    def start_monitoring(self, 
                        portfolio_data: Dict[str, Any],
                        update_interval: int = 60) -> None:
        """
        Start real-time portfolio monitoring
        
        Args:
            portfolio_data: Initial portfolio data
            update_interval: Update interval in seconds
        """
        if self.running:
            logger.warning("Portfolio monitoring already running")
            return
            
        self.portfolio_data = portfolio_data
        self.update_interval = update_interval
        self.running = True
        self.last_update = datetime.now()
        
        # Start monitoring thread
        self.update_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self.update_thread.start()
        
        logger.info(f"Started portfolio monitoring with {update_interval}s interval")
    
    def stop_monitoring(self) -> None:
        """Stop real-time portfolio monitoring"""
        if not self.running:
            logger.warning("Portfolio monitoring not running")
            return
            
        self.running = False
        if self.update_thread:
            self.update_thread.join(timeout=2.0)
            
        logger.info("Stopped portfolio monitoring")
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop"""
        while self.running:
            try:
                # Fetch latest data
                self._fetch_latest_data()
                
                # Process alerts
                self._check_alerts()
                
                # Update last update time
                self.last_update = datetime.now()
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                
            # Wait for next update
            time.sleep(self.update_interval)
    
    def _fetch_latest_data(self) -> None:
        """Fetch latest portfolio data"""
        # This would typically call an API or data service
        # For now, we'll just simulate with a placeholder
        
        # In a real implementation, this would update prices and recalculate
        # portfolio values, returns, etc.
        
        # Put updated data in queue for consumers
        if self.portfolio_data:
            # Simulate updating portfolio value with small random change
            if 'value' in self.portfolio_data:
                change_pct = np.random.normal(0, 0.001)  # Small random change
                self.portfolio_data['value'] *= (1 + change_pct)
                self.portfolio_data['last_change_pct'] = change_pct
                self.portfolio_data['last_update'] = datetime.now().isoformat()
                
                # Put in queue for consumers
                self.data_queue.put(self.portfolio_data.copy())
    
    def _check_alerts(self) -> None:
        """Check for alert conditions"""
        if not self.portfolio_data or not self.alert_thresholds:
            return
            
        # Check each alert threshold
        for alert_type, threshold in self.alert_thresholds.items():
            if alert_type == 'value_change_pct':
                if 'last_change_pct' in self.portfolio_data:
                    change_pct = self.portfolio_data['last_change_pct'] * 100  # Convert to percentage
                    if abs(change_pct) > threshold:
                        self._trigger_alert(
                            alert_type=alert_type,
                            message=f"Portfolio value changed by {change_pct:.2f}% (threshold: {threshold:.2f}%)",
                            data={
                                'change_pct': change_pct,
                                'threshold': threshold,
                                'value': self.portfolio_data.get('value')
                            }
                        )
    
    def _trigger_alert(self, alert_type: str, message: str, data: Dict[str, Any]) -> None:
        """Trigger an alert"""
        alert = {
            'type': alert_type,
            'message': message,
            'timestamp': datetime.now().isoformat(),
            'data': data
        }
        
        # Add to alerts list
        self.alerts.append(alert)
        
        # Call alert callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")
                
        logger.info(f"Alert triggered: {message}")
    
    def register_alert_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """
        Register a callback function for alerts
        
        Args:
            callback: Function to call when an alert is triggered
        """
        self.alert_callbacks.append(callback)
        
    def set_alert_threshold(self, alert_type: str, threshold: float) -> None:
        """
        Set an alert threshold
        
        Args:
            alert_type: Type of alert ('value_change_pct', 'drawdown', etc.)
            threshold: Threshold value
        """
        self.alert_thresholds[alert_type] = threshold
        logger.info(f"Set {alert_type} alert threshold to {threshold}")
    
    def get_latest_data(self, timeout: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """
        Get latest portfolio data
        
        Args:
            timeout: Optional timeout in seconds
            
        Returns:
            Latest portfolio data or None if timeout
        """
        try:
            return self.data_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def get_alerts(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent alerts
        
        Args:
            limit: Maximum number of alerts to return
            
        Returns:
            List of recent alerts
        """