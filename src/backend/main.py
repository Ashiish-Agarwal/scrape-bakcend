"""
main.py - Basic FastAPI setup with initial configuration
Step 14: Basic FastAPI Setup
"""

import os
import asyncio
from datetime import datetime
from typing import Dict, Any, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import uvicorn

from .crew import WebScrapingCrew, progress_tracker, ScrapeTask, QueryTask


# Load environment variables
load_dotenv()


# Pydantic models for request/response validation
class HealthResponse(BaseModel):
    """Health check response model"""
    status: str = Field(..., description="Application status")
    timestamp: str = Field(..., description="Current timestamp")
    version: str = Field(..., description="Application version")


class ScrapeRequest(BaseModel):
    """Website scraping request model"""
    url: str = Field(..., description="URL to scrape", min_length=1)
    options: Optional[Dict[str, Any]] = Field(default={}, description="Scraping options")


class ScrapeResponse(BaseModel):
    """Website scraping response model"""
    scrape_id: str = Field(..., description="Unique scraping session ID")
    url: str = Field(..., description="URL being scraped")
    status: str = Field(..., description="Current scraping status")
    websocket_url: str = Field(..., description="WebSocket URL for progress updates")
    message: str = Field(..., description="Status message")


class QueryRequest(BaseModel):
    """Content query request model"""
    query: str = Field(..., description="Question to ask about scraped content", min_length=1)
    scrape_id: Optional[str] = Field(default=None, description="Specific scrape ID to query")


class QueryResponse(BaseModel):
    """Content query response model"""
    query_id: str = Field(..., description="Unique query ID")
    query: str = Field(..., description="Original query")
    answer: str = Field(..., description="AI-generated answer")
    sources: Optional[list] = Field(default=[], description="Source references")
    scrape_id: Optional[str] = Field(default=None, description="Related scrape ID")


class ErrorResponse(BaseModel):
    """Error response model"""
    error: Dict[str, Any] = Field(..., description="Error details")


# Application state management
class AppState:
    """Application state manager"""
    
    def __init__(self):
        self.web_scraping_crew = None
        self.active_websockets: Dict[str, WebSocket] = {}
        self.scrape_tasks: Dict[str, ScrapeTask] = {}
        self.query_tasks: Dict[str, QueryTask] = {}
    
    async def initialize(self):
        """Initialize application components"""
        try:
            self.web_scraping_crew = WebScrapingCrew()
            print("✅ WebScrapingCrew initialized successfully")
        except Exception as e:
            print(f"❌ Failed to initialize WebScrapingCrew: {e}")
            raise


# Global application state
app_state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    print("🚀 Starting AI Web Scraper application...")
    await app_state.initialize()
    print("✅ Application startup complete")
    
    yield
    
    # Shutdown
    print("🛑 Shutting down AI Web Scraper application...")
    # Close any active WebSocket connections
    for websocket in app_state.active_websockets.values():
        try:
            await websocket.close()
        except:
            pass
    print("✅ Application shutdown complete")


# Initialize FastAPI app
app = FastAPI(
    title="AI Web Scraper API",
    description="AI-powered web scraping system with real-time progress tracking",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)


# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Health check endpoint
@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health Check",
    description="Check application health and status"
)
async def health_check():
    """Application health check endpoint"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        version="0.1.0"
    )


# WebSocket connection manager
class WebSocketManager:
    """Manages WebSocket connections for progress tracking"""
    
    @staticmethod
    async def websocket_callback(scrape_id: str, websocket: WebSocket):
        """WebSocket callback function for sending progress updates"""
        async def send_update(message: str):
            try:
                await websocket.send_text(message)
            except Exception as e:
                print(f"Error sending WebSocket message: {e}")
                # Remove from active connections if send fails
                if scrape_id in app_state.active_websockets:
                    del app_state.active_websockets[scrape_id]
        
        return send_update


# WebSocket endpoint for scraping progress
@app.websocket("/ws/scraping-progress/{scrape_id}")
async def websocket_scraping_progress(websocket: WebSocket, scrape_id: str):
    """WebSocket endpoint for real-time scraping progress updates"""
    await websocket.accept()
    
    try:
        # Store WebSocket connection
        app_state.active_websockets[scrape_id] = websocket
        
        # Register progress callback
        callback = await WebSocketManager.websocket_callback(scrape_id, websocket)
        progress_tracker.register_websocket_callback(scrape_id, callback)
        
        # Send initial connection confirmation
        await websocket.send_text(f'{{"status": "connected", "scrape_id": "{scrape_id}"}}')
        
        # Keep connection alive
        while True:
            try:
                # Wait for any messages from client (heartbeat, etc.)
                await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
            except asyncio.TimeoutError:
                # Send heartbeat if no message received
                await websocket.send_text('{"type": "heartbeat"}')
                
    except WebSocketDisconnect:
        print(f"WebSocket disconnected for scrape_id: {scrape_id}")
    except Exception as e:
        print(f"WebSocket error for scrape_id {scrape_id}: {e}")
    finally:
        # Cleanup
        if scrape_id in app_state.active_websockets:
            del app_state.active_websockets[scrape_id]
        progress_tracker.unregister_websocket_callback(scrape_id)


# Error handling
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Global HTTP exception handler"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": f"HTTP_{exc.status_code}",
                "message": exc.detail,
                "timestamp": datetime.now().isoformat()
            }
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Global exception handler for unhandled exceptions"""
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "code": "INTERNAL_SERVER_ERROR",
                "message": "An unexpected error occurred",
                "details": str(exc),
                "timestamp": datetime.now().isoformat()
            }
        }
    )


# Development server runner
def run_development_server():
    """Run the development server"""
    host = os.getenv("FASTAPI_HOST", "localhost")
    port = int(os.getenv("FASTAPI_PORT", "8000"))
    debug = os.getenv("FASTAPI_DEBUG", "true").lower() == "true"
    
    print(f"🚀 Starting FastAPI server on {host}:{port}")
    print(f"📚 API Documentation: http://{host}:{port}/docs")
    print(f"🔄 ReDoc Documentation: http://{host}:{port}/redoc")
    
    uvicorn.run(
        "src.my_project.main:app",
        host=host,
        port=port,
        reload=debug,
        log_level="info" if debug else "warning"
    )


if __name__ == "__main__":
    run_development_server()


# Core Endpoints - Step 15

# Scrape Website Endpoint
@app.post(
    "/scrape-website",
    response_model=ScrapeResponse,
    summary="Initiate Website Scraping",
    description="Start scraping a website with optional configuration",
    response_description="Scraping job details with WebSocket URL for progress tracking"
)
async def scrape_website(request: ScrapeRequest):
    """Initiate website scraping with real-time progress tracking"""
    try:
        # Generate unique scrape ID
        scrape_id = str(uuid.uuid4())
        
        # Create scrape task
        scrape_task = ScrapeTask(scrape_id, request.url, request.options)
        app_state.scrape_tasks[scrape_id] = scrape_task
        
        # Start background task
        asyncio.create_task(
            _execute_scrape_task(scrape_id, request.url, request.options)
        )
        
        # Generate WebSocket URL for progress tracking
        host = os.getenv("FASTAPI_HOST", "localhost")
        port = os.getenv("FASTAPI_PORT", "8000")
        websocket_url = f"ws://{host}:{port}/ws/scraping-progress/{scrape_id}"
        
        return ScrapeResponse(
            scrape_id=scrape_id,
            url=request.url,
            status="initiated",
            websocket_url=websocket_url,
            message=f"Scraping initiated for {request.url}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "code": "SCRAPING_INITIATION_FAILED",
                "message": "Failed to initiate scraping",
                "details": str(e),
                "timestamp": datetime.now().isoformat()
            }
        )


async def _execute_scrape_task(scrape_id: str, url: str, options: Dict[str, Any]):
    """Execute scraping task in background"""
    try:
        # Update task status
        app_state.scrape_tasks[scrape_id].status = TaskStatus.RUNNING
        
        # Execute scraping via crew
        result = await app_state.web_scraping_crew.scrape_website(url, options)
        
        # Update task with results
        app_state.scrape_tasks[scrape_id].status = TaskStatus.COMPLETED
        app_state.scrape_tasks[scrape_id].result = result
        app_state.scrape_tasks[scrape_id].progress = 100
        app_state.scrape_tasks[scrape_id].completed_at = datetime.now()
        
    except Exception as e:
        # Update task with error
        app_state.scrape_tasks[scrape_id].status = TaskStatus.FAILED
        app_state.scrape_tasks[scrape_id].error = str(e)
        app_state.scrape_tasks[scrape_id].completed_at = datetime.now()
        
        # Log error
        print(f"Error executing scrape task {scrape_id}: {e}")


# Query Endpoint
@app.post(
    "/query",
    response_model=QueryResponse,
    summary="Query Scraped Content",
    description="Ask questions about previously scraped content",
    response_description="AI-generated answer with source citations"
)
async def query_content(request: QueryRequest):
    """Query scraped content with AI-powered answering"""
    try:
        # Generate unique query ID
        query_id = str(uuid.uuid4())
        
        # Create query task
        query_task = QueryTask(query_id, request.query, request.scrape_id)
        app_state.query_tasks[query_id] = query_task
        
        # Execute query via crew
        result = await app_state.web_scraping_crew.query_content(
            request.query, request.scrape_id
        )
        
        # Update task with results
        app_state.query_tasks[query_id].status = TaskStatus.COMPLETED
        app_state.query_tasks[query_id].result = result
        app_state.query_tasks[query_id].completed_at = datetime.now()
        
        # Extract sources if available
        sources = []
        if isinstance(result.get('result'), dict) and 'sources' in result.get('result', {}):
            sources = result['result']['sources']
        
        return QueryResponse(
            query_id=query_id,
            query=request.query,
            answer=result.get('answer', str(result)),
            sources=sources,
            scrape_id=request.scrape_id
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "code": "QUERY_FAILED",
                "message": "Failed to process query",
                "details": str(e),
                "timestamp": datetime.now().isoformat()
            }
        )


# Management Endpoints - Step 15

# List all scraped sites
@app.get(
    "/scraped-sites",
    summary="List Scraped Sites",
    description="Get a list of all scraped websites"
)
async def list_scraped_sites():
    """List all scraped websites with their status"""
    try:
        sites = [
            {
                "scrape_id": scrape_id,
                "url": task.url,
                "status": task.status.value,
                "created_at": task.created_at.isoformat(),
                "completed_at": task.completed_at.isoformat() if task.completed_at else None
            }
            for scrape_id, task in app_state.scrape_tasks.items()
        ]
        
        return {"sites": sites, "count": len(sites)}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "code": "LIST_SITES_FAILED",
                "message": "Failed to list scraped sites",
                "details": str(e),
                "timestamp": datetime.now().isoformat()
            }
        )


# Check scraping status
@app.get(
    "/scrape-status/{scrape_id}",
    summary="Check Scraping Status",
    description="Get the current status of a scraping operation"
)
async def get_scrape_status(scrape_id: str):
    """Get detailed status of a specific scraping operation"""
    try:
        # Check if scrape task exists
        if scrape_id not in app_state.scrape_tasks:
            raise HTTPException(
                status_code=404,
                detail={
                    "code": "SCRAPE_NOT_FOUND",
                    "message": f"No scraping task found with ID: {scrape_id}",
                    "timestamp": datetime.now().isoformat()
                }
            )
        
        # Get task details
        task = app_state.scrape_tasks[scrape_id]
        
        # Get progress tracker status
        tracker_status = progress_tracker.get_session_status(scrape_id)
        
        return {
            "scrape_id": scrape_id,
            "url": task.url,
            "status": task.status.value,
            "progress": task.progress,
            "created_at": task.created_at.isoformat(),
            "completed_at": task.completed_at.isoformat() if task.completed_at else None,
            "error": task.error,
            "tracker_status": tracker_status
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "code": "STATUS_CHECK_FAILED",
                "message": "Failed to check scraping status",
                "details": str(e),
                "timestamp": datetime.now().isoformat()
            }
        )


# Remove scraped content
@app.delete(
    "/scraped-site/{scrape_id}",
    summary="Remove Scraped Content",
    description="Delete scraped content for a specific site"
)
async def delete_scraped_site(scrape_id: str):
    """Remove scraped content and associated data"""
    try:
        # Check if scrape task exists
        if scrape_id not in app_state.scrape_tasks:
            raise HTTPException(
                status_code=404,
                detail={
                    "code": "SCRAPE_NOT_FOUND",
                    "message": f"No scraping task found with ID: {scrape_id}",
                    "timestamp": datetime.now().isoformat()
                }
            )
        
        # Remove from tasks dictionary
        del app_state.scrape_tasks[scrape_id]
        
        # Remove from progress tracker
        if scrape_id in progress_tracker.active_sessions:
            del progress_tracker.active_sessions[scrape_id]
        
        # Remove WebSocket callback if exists
        progress_tracker.unregister_websocket_callback(scrape_id)
        
        # TODO: In a production system, also remove associated files from storage
        
        return {
            "status": "success",
            "message": f"Scraped content for ID {scrape_id} has been removed",
            "timestamp": datetime.now().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "code": "DELETE_FAILED",
                "message": "Failed to delete scraped content",
                "details": str(e),
                "timestamp": datetime.now().isoformat()
            }
        )


# Utility Endpoints - Step 15

# Test scraping functionality
@app.post(
    "/test-scrape",
    summary="Test Scraping Functionality",
    description="Test the scraping functionality with a sample URL"
)
async def test_scrape(request: ScrapeRequest):
    """Test scraping functionality with a sample URL"""
    try:
        # Generate unique scrape ID with test prefix
        scrape_id = f"test-{str(uuid.uuid4())}"
        
        # Create scrape task
        scrape_task = ScrapeTask(scrape_id, request.url, request.options)
        app_state.scrape_tasks[scrape_id] = scrape_task
        
        # Execute scraping synchronously for testing
        try:
            # Update task status
            app_state.scrape_tasks[scrape_id].status = TaskStatus.RUNNING
            
            # Execute limited scraping via crew
            # Add test flag to options to limit scraping depth/scope
            test_options = request.options.copy() if request.options else {}
            test_options["test_mode"] = True
            test_options["max_pages"] = test_options.get("max_pages", 1)  # Limit to 1 page by default
            
            # Execute scraping
            result = await app_state.web_scraping_crew.scrape_website(request.url, test_options)
            
            # Update task with results
            app_state.scrape_tasks[scrape_id].status = TaskStatus.COMPLETED
            app_state.scrape_tasks[scrape_id].result = result
            app_state.scrape_tasks[scrape_id].progress = 100
            app_state.scrape_tasks[scrape_id].completed_at = datetime.now()
            
            # Return test results
            return {
                "scrape_id": scrape_id,
                "url": request.url,
                "status": "completed",
                "test_result": "success",
                "message": "Test scraping completed successfully",
                "sample_data": result.get("result", "")[:500] + "..." if result.get("result") else "",
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            # Update task with error
            app_state.scrape_tasks[scrape_id].status = TaskStatus.FAILED
            app_state.scrape_tasks[scrape_id].error = str(e)
            app_state.scrape_tasks[scrape_id].completed_at = datetime.now()
            
            # Return error details
            return {
                "scrape_id": scrape_id,
                "url": request.url,
                "status": "failed",
                "test_result": "failed",
                "message": "Test scraping failed",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
            
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "code": "TEST_SCRAPE_FAILED",
                "message": "Failed to execute test scraping",
                "details": str(e),
                "timestamp": datetime.now().isoformat()
            }
        )