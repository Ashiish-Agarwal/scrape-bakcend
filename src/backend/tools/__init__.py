"""
AI Web Scraper Tools Package

This module provides a centralized import interface for all scraping tools
used by the Crew AI agents in the web scraping system.

Tools Available:
- ScraperTool: Handles website content extraction
- ProcessorTool: Processes and structures scraped content  
- QueryTool: Handles user queries against scraped content

Usage:
    from my_project.tools import ScraperTool, ProcessorTool, QueryTool
    
    # Initialize tools
    scraper = ScraperTool()
    processor = ProcessorTool()
    query_handler = QueryTool()
"""

from .scraper_tool import ScraperTool
from .processor_tool import ProcessorTool
from .query_tool import QueryTool

# Make all tools available at package level
__all__ = [
    "ScraperTool",
    "ProcessorTool", 
    "QueryTool"
]

# Version information
__version__ = "0.1.0"

# Tool registry for dynamic access
TOOL_REGISTRY = {
    "scraper": ScraperTool,
    "processor": ProcessorTool,
    "query": QueryTool
}

def get_tool(tool_name: str):
    """
    Factory function to get tool instances by name.
    
    Args:
        tool_name (str): Name of the tool ('scraper', 'processor', 'query')
        
    Returns:
        Tool instance
        
    Raises:
        ValueError: If tool_name is not recognized
    """
    if tool_name not in TOOL_REGISTRY:
        available_tools = ", ".join(TOOL_REGISTRY.keys())
        raise ValueError(
            f"Unknown tool '{tool_name}'. Available tools: {available_tools}"
        )
    
    return TOOL_REGISTRY[tool_name]()

def get_all_tools():
    """
    Get instances of all available tools.
    
    Returns:
        dict: Dictionary mapping tool names to tool instances
    """
    return {name: tool_class() for name, tool_class in TOOL_REGISTRY.items()}

def list_available_tools():
    """
    Get list of all available tool names.
    
    Returns:
        list: List of available tool names
    """
    return list(TOOL_REGISTRY.keys())