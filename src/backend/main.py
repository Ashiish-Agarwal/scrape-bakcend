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