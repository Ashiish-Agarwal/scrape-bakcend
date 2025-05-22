"""
Enhanced crew.py with WebSocket progress tracking integration
Step 13: Add Progress Tracking to existing crew.py
"""

import asyncio
import json
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, asdict
from enum import Enum

from crewai import Agent, Task, Crew
import yaml
import os

from .tools import ScraperTool, ProcessorTool, QueryTool


class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ProgressUpdate:
    """Progress update data structure for WebSocket communication"""
    scrape_id: str
    status: TaskStatus
    progress_percentage: int
    current_task: str
    message: str
    timestamp: str
    error: Optional[str] = None
    
    def to_json(self) -> str:
        """Convert progress update to JSON string"""
        data = asdict(self)
        data['status'] = self.status.value
        return json.dumps(data)


class ProgressTracker:
    """Manages progress tracking and WebSocket communication"""
    
    def __init__(self):
        self.active_sessions: Dict[str, Dict] = {}
        self.websocket_callbacks: Dict[str, Callable] = {}
    
    def register_websocket_callback(self, scrape_id: str, callback: Callable):
        """Register WebSocket callback for progress updates"""
        self.websocket_callbacks[scrape_id] = callback
    
    def unregister_websocket_callback(self, scrape_id: str):
        """Remove WebSocket callback"""
        if scrape_id in self.websocket_callbacks:
            del self.websocket_callbacks[scrape_id]
    
    async def send_progress_update(self, progress_update: ProgressUpdate):
        """Send progress update via WebSocket if callback exists"""
        scrape_id = progress_update.scrape_id
        
        # Update session data
        self.active_sessions[scrape_id] = {
            'status': progress_update.status,
            'progress': progress_update.progress_percentage,
            'current_task': progress_update.current_task,
            'last_update': progress_update.timestamp,
            'error': progress_update.error
        }
        
        # Send via WebSocket if callback exists
        if scrape_id in self.websocket_callbacks:
            try:
                callback = self.websocket_callbacks[scrape_id]
                await callback(progress_update.to_json())
            except Exception as e:
                print(f"Error sending WebSocket update: {e}")
    
    def get_session_status(self, scrape_id: str) -> Optional[Dict]:
        """Get current status of a scraping session"""
        return self.active_sessions.get(scrape_id)


# Global progress tracker instance
progress_tracker = ProgressTracker()


class WebScrapingCrew:
    """Enhanced Web Scraping Crew with progress tracking"""
    
    def __init__(self):
        self.agents = self._load_agents()
        self.tasks = self._load_tasks()
        self.crew = None
        self.scraper_tool = ScraperTool()
        self.processor_tool = ProcessorTool()
        self.query_tool = QueryTool()
    
    def _load_agents(self) -> Dict[str, Agent]:
        """Load agent configurations from YAML"""
        config_path = os.path.join(os.path.dirname(__file__), 'config', 'agents.yaml')
        
        with open(config_path, 'r') as file:
            agents_config = yaml.safe_load(file)
        
        agents = {}
        for agent_name, config in agents_config.items():
            # Assign tools based on agent type
            tools = []
            if agent_name == 'dynamic_scraper':
                tools = [self.scraper_tool]
            elif agent_name == 'content_analyzer':
                tools = [self.processor_tool]
            elif agent_name == 'query_handler':
                tools = [self.query_tool]
            
            agents[agent_name] = Agent(
                role=config['role'],
                goal=config['goal'],
                backstory=config['backstory'],
                tools=tools,
                verbose=True,
                allow_delegation=False
            )
        
        return agents
    
    def _load_tasks(self) -> Dict[str, dict]:
        """Load task configurations from YAML"""
        config_path = os.path.join(os.path.dirname(__file__), 'config', 'tasks.yaml')
        
        with open(config_path, 'r') as file:
            tasks_config = yaml.safe_load(file)
        
        return tasks_config
    
    async def _send_progress_update(self, scrape_id: str, status: TaskStatus, 
                                  progress: int, task: str, message: str, 
                                  error: Optional[str] = None):
        """Send progress update via progress tracker"""
        update = ProgressUpdate(
            scrape_id=scrape_id,
            status=status,
            progress_percentage=progress,
            current_task=task,
            message=message,
            timestamp=datetime.now().isoformat(),
            error=error
        )
        await progress_tracker.send_progress_update(update)
    
    async def scrape_website(self, url: str, options: Dict = None) -> Dict[str, Any]:
        """Execute website scraping with progress tracking"""
        scrape_id = str(uuid.uuid4())
        options = options or {}
        
        try:
            # Initialize progress tracking
            await self._send_progress_update(
                scrape_id, TaskStatus.RUNNING, 10, 
                "scraping", f"Starting to scrape {url}"
            )
            
            # Create scraping task
            scraping_task = Task(
                description=self.tasks['scraping_task']['description'].format(
                    url=url, options=str(options)
                ),
                expected_output=self.tasks['scraping_task']['expected_output'],
                agent=self.agents['dynamic_scraper']
            )
            
            # Update progress
            await self._send_progress_update(
                scrape_id, TaskStatus.RUNNING, 30,
                "scraping", "Extracting content from website"
            )
            
            # Create processing task
            processing_task = Task(
                description=self.tasks['processing_task']['description'],
                expected_output=self.tasks['processing_task']['expected_output'],
                agent=self.agents['content_analyzer']
            )
            
            # Update progress
            await self._send_progress_update(
                scrape_id, TaskStatus.RUNNING, 60,
                "processing", "Analyzing and structuring content"
            )
            
            # Create and execute crew
            crew = Crew(
                agents=[self.agents['dynamic_scraper'], self.agents['content_analyzer']],
                tasks=[scraping_task, processing_task],
                verbose=True
            )
            
            # Execute the crew workflow
            result = crew.kickoff()
            
            # Final progress update
            await self._send_progress_update(
                scrape_id, TaskStatus.COMPLETED, 100,
                "completed", f"Successfully scraped and processed {url}"
            )
            
            return {
                'scrape_id': scrape_id,
                'url': url,
                'status': 'completed',
                'result': str(result),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            error_msg = f"Scraping failed: {str(e)}"
            await self._send_progress_update(
                scrape_id, TaskStatus.FAILED, 0,
                "failed", error_msg, error=str(e)
            )
            
            return {
                'scrape_id': scrape_id,
                'url': url,
                'status': 'failed',
                'error': error_msg,
                'timestamp': datetime.now().isoformat()
            }
    
    async def query_content(self, query: str, scrape_id: Optional[str] = None) -> Dict[str, Any]:
        """Execute content query with progress tracking"""
        query_id = str(uuid.uuid4())
        
        try:
            # Create query task
            query_task = Task(
                description=self.tasks['query_task']['description'].format(
                    query=query, scrape_id=scrape_id or "all"
                ),
                expected_output=self.tasks['query_task']['expected_output'],
                agent=self.agents['query_handler']
            )
            
            # Create and execute crew
            crew = Crew(
                agents=[self.agents['query_handler']],
                tasks=[query_task],
                verbose=True
            )
            
            result = crew.kickoff()
            
            return {
                'query_id': query_id,
                'query': query,
                'status': 'completed',
                'answer': str(result),
                'scrape_id': scrape_id,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'query_id': query_id,
                'query': query,
                'status': 'failed',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }


class ScrapeTask:
    """Individual scraping operation with progress tracking"""
    
    def __init__(self, scrape_id: str, url: str, options: Dict = None):
        self.scrape_id = scrape_id
        self.url = url
        self.options = options or {}
        self.status = TaskStatus.PENDING
        self.progress = 0
        self.result = None
        self.error = None
        self.created_at = datetime.now()
        self.completed_at = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary representation"""
        return {
            'scrape_id': self.scrape_id,
            'url': self.url,
            'options': self.options,
            'status': self.status.value,
            'progress': self.progress,
            'error': self.error,
            'created_at': self.created_at.isoformat(),
            'completed_at': self.completed_at.isoformat() if self.completed_at else None
        }


class QueryTask:
    """Individual query operation"""
    
    def __init__(self, query_id: str, query: str, scrape_id: Optional[str] = None):
        self.query_id = query_id
        self.query = query
        self.scrape_id = scrape_id
        self.status = TaskStatus.PENDING
        self.result = None
        self.error = None
        self.created_at = datetime.now()
        self.completed_at = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary representation"""
        return {
            'query_id': self.query_id,
            'query': self.query,
            'scrape_id': self.scrape_id,
            'status': self.status.value,
            'result': self.result,
            'error': self.error,
            'created_at': self.created_at.isoformat(),
            'completed_at': self.completed_at.isoformat() if self.completed_at else None
        }