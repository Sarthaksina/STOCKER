import logging
from typing import Any, Dict, List, Optional

# Import all agent modules
from src.features.data_agents import fetch_news, analyze_concall_agent, fetch_events_agent
from src.features.analytics_agents import AnalyticsAgents
from src.features.reporting_agents import ReportingAgents

logger = logging.getLogger(__name__)

# Instantiate consolidated agents
analytics = AnalyticsAgents()
reporting = ReportingAgents()

class MegaAgent:
    """
    Unified agent combining data, analytics, and reporting features.
    Provides a single interface for all tasks previously handled by crew_agents, data_agents, and reporting_agents.
    """
    def __init__(self):
        self.tasks: Dict[str, Any] = {
            # Data/News/Events/Concall
            'news': fetch_news,
            'concall': analyze_concall_agent,
            'events': fetch_events_agent,
            # Analytics
            'portfolio': analytics.portfolio,
            'self_assess': analytics.self_assess,
            'rebalance': analytics.rebalance,
            'suggest_hq': analytics.suggest_hq,
            'peer': analytics.peer,
            'top_peers': analytics.top_peers,
            'charts': analytics.charts,
            'risk_var': analytics.risk_var,
            'risk_drawdown': analytics.risk_drawdown,
            'risk_sharpe': analytics.risk_sharpe,
            'sentiment_agg': analytics.sentiment_agg,
            'holdings': analytics.holdings,
            'analysis_performance': analytics.analysis_performance,
            'analysis_sharpe': analytics.analysis_sharpe,
            'analysis_valuation': analytics.analysis_valuation,
            'analysis_alpha_beta': analytics.analysis_alpha_beta,
            'analysis_attribution': analytics.analysis_attribution,
            'analysis_momentum': analytics.analysis_momentum,
            'vector_search': analytics.vector_search,
            # Reporting
            'report': reporting.report,
            'text_report': reporting.text_report,
            'visual_report': reporting.visual_report
        }

    def execute(self, task: str, params: Dict[str, Any]) -> Any:
        """
        Execute a task by delegating to the appropriate function/agent.
        """
        key = task.replace(' ', '_').lower()
        if key in self.tasks:
            try:
                return self.tasks[key](params)
            except Exception as e:
                logger.error(f"Error executing task '{key}': {e}")
                raise
        else:
            raise ValueError(f"Unknown task: {task}")
