# src/my_project/__init__.py
"""
AI Web Scraper - AI-powered web scraping system with real-time progress tracking
"""

__version__ = "0.1.0"
__author__ = "AI Web Scraper Team"
__description__ = "AI-powered web scraping system using Crew AI and FastAPI"

from .crew import WebScrapingCrew, ProgressTracker, ProgressUpdate, ScrapeResult
from .main import app

__all__ = [
    "WebScrapingCrew", 
    "ProgressTracker", 
    "ProgressUpdate", 
    "ScrapeResult",
    "app"
]


# src/my_project/tools/__init__.py (already created, but here's the enhanced version)
"""
Tools module for AI Web Scraper
Contains all the specialized tools used by AI agents
"""

from .scraper_tool import ScraperTool
from .processor_tool import ProcessorTool  
from .query_tool import QueryTool

__all__ = ["ScraperTool", "ProcessorTool", "QueryTool"]


# src/my_project/config/__init__.py
"""
Configuration module for AI Web Scraper
Contains agent and task definitions
"""

import os
import yaml
from pathlib import Path

def load_config(config_name: str) -> dict:
    """Load configuration from YAML file"""
    config_path = Path(__file__).parent / f"{config_name}.yaml"
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)

def get_agents_config() -> dict:
    """Get agents configuration"""
    return load_config('agents')

def get_tasks_config() -> dict:
    """Get tasks configuration"""  
    return load_config('tasks')

__all__ = ["load_config", "get_agents_config", "get_tasks_config"]