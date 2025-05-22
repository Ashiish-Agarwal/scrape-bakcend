# AI Web Scraper Project Rules

## Project Overview
This is an AI-powered web scraping system using **Crew AI** for agent orchestration and **FastAPI** as the backend framework. The system enables real-time website scraping with AI-powered content querying capabilities.

## Project Structure Standards

### Required Directory Structure
```
my_project/
├── .gitignore
├── .env                         # Environment variables
├── pyproject.toml               # Dependencies and project config
├── README.md
├── project-rules.md             # This file - development guidelines
├── knowledge/                   # Scraped content storage
└── src/
    └── my_project/
        ├── __init__.py
        ├── main.py              # FastAPI app and endpoints
        ├── crew.py              # Crew AI orchestration
        ├── tools/               # Custom scraping tools
        │   ├── __init__.py
        │   ├── scraper_tool.py  # Web scraping functionality
        │   ├── processor_tool.py # Content processing
        │   └── query_tool.py    # Query handling
        └── config/              # Agent configurations
            ├── agents.yaml      # AI agent definitions
            └── tasks.yaml       # Workflow definitions
```

### File Naming Conventions
- Use **snake_case** for all Python files and directories
- Configuration files use **lowercase** with extensions (.yaml, .toml, .env)
- Keep file names descriptive and purpose-specific
- Tool files must end with `_tool.py`

## Development Rules

### 1. Code Organization
- **Single Responsibility**: Each file/class/function should have one clear purpose
- **Separation of Concerns**: API logic (main.py) separate from AI logic (crew.py)
- **Tool Isolation**: Each tool handles one specific domain (scraping, processing, querying)
- **Configuration Driven**: Use YAML files for agent and task configurations

### 2. API Development Standards
- All endpoints must include comprehensive **request/response validation**
- Use **Pydantic models** for request/response schemas
- Implement **proper HTTP status codes** (200, 201, 400, 404, 500, etc.)
- Include **detailed error messages** with helpful context
- Add **request logging** for debugging and monitoring

### 3. Crew AI Implementation Rules
- **Three Agents Only**: Dynamic Scraper, Content Analyzer, Query Handler
- Each agent gets **dedicated tools** - no tool sharing between agents
- All agent configurations must be in **agents.yaml**
- All task definitions must be in **tasks.yaml**
- Use **sequential task execution** for predictable workflows

### 4. Real-time Features Requirements
- **WebSocket** connections for scraping progress updates
- **Background task processing** for non-blocking operations
- **Progress tracking** with percentage completion and status messages
- **Graceful error handling** with user-friendly messages

### 5. Content Management Rules
- Store scraped content in `knowledge/` directory during development
- Use **unique identifiers** (UUIDs) for each scraping session
- Implement **content indexing** for fast search and retrieval
- **Clean and structure** all scraped content before storage

## Required API Endpoints

### Core Functionality
| Method | Endpoint | Purpose | WebSocket |
|--------|----------|---------|-----------|
| POST | `/scrape-website` | Initiate website scraping | Yes |
| WS | `/ws/scraping-progress/{scrape_id}` | Real-time progress | - |
| POST | `/query` | Query scraped content | No |

### Management
| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/scraped-sites` | List all scraped sites |
| GET | `/scrape-status/{scrape_id}` | Check scraping status |
| DELETE | `/scraped-site/{scrape_id}` | Remove scraped content |

### Utilities
| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/health` | Application health check |
| POST | `/test-scrape` | Test scraping functionality |

## Agent Architecture Rules

### Dynamic Scraper Agent
- **Responsibility**: Website content extraction
- **Tool**: `scraper_tool.py` only
- **Requirements**: Handle various website structures, JavaScript content, rate limiting
- **Output**: Raw scraped content with metadata

### Content Analyzer Agent
- **Responsibility**: Content processing and structuring
- **Tool**: `processor_tool.py` only
- **Requirements**: Clean data, extract key information, categorize content
- **Output**: Structured, searchable content

### Query Handler Agent
- **Responsibility**: User question processing
- **Tool**: `query_tool.py` only
- **Requirements**: Search content, provide contextual answers with sources
- **Output**: AI-generated responses with citations

## Error Handling Standards

### Required Error Handling
- **Network failures** during scraping (timeouts, connection errors)
- **Invalid URLs** and malformed requests
- **Rate limiting** and anti-bot measures
- **Content processing failures** (malformed HTML, encoding issues)
- **AI model failures** and API timeouts
- **WebSocket disconnections** and reconnection logic

### Error Response Format
```json
{
    "error": {
        "code": "SCRAPING_FAILED",
        "message": "User-friendly error description",
        "details": "Technical error details for debugging",
        "scrape_id": "uuid-if-applicable",
        "timestamp": "2024-01-15T10:30:00Z"
    }
}
```

## Performance Requirements

### Response Time Targets
- Health check: < 100ms
- Scrape initiation: < 500ms
- Query responses: < 2 seconds
- WebSocket updates: < 100ms latency

### Scalability Considerations
- Design for **horizontal scaling**
- Use **asynchronous processing** where possible
- Implement **caching** for frequently accessed content
- **Queue-based task processing** for heavy operations

## Security Rules

### Data Protection
- **Never log sensitive data** (API keys, user tokens)
- **Validate all input** to prevent injection attacks
- **Rate limit** all endpoints to prevent abuse
- **Sanitize scraped content** before storage

### Access Control
- Use **environment variables** for all configuration
- **No hardcoded secrets** in source code
- Implement **request size limits** to prevent DoS
- **CORS configuration** for web client access

## Testing Requirements

### Test Coverage Areas
- **Unit tests** for all tools (scraper, processor, query)
- **Integration tests** for Crew AI workflows
- **API endpoint tests** for all endpoints
- **WebSocket functionality** testing
- **Error scenario testing** for edge cases

### Test Data Management
- Use **mock websites** or test URLs for development
- **Separate test environment** configuration
- **Clean test data** between test runs
- **Performance benchmarks** for scraping operations

## Development Workflow

### Phase-by-Phase Development
1. **Foundation**: Project structure, dependencies, basic FastAPI setup
2. **Scraping Core**: Dynamic Scraper Agent + scraper_tool.py
3. **Real-time Features**: WebSocket progress tracking
4. **Content Processing**: Content Analyzer Agent + processor_tool.py
5. **AI Querying**: Query Handler Agent + query_tool.py
6. **API Completion**: All management and utility endpoints
7. **Error Handling**: Comprehensive error scenarios
8. **Testing & Optimization**: Performance tuning and testing

### Git Workflow
- **Feature branches** for each development phase
- **Descriptive commit messages** with issue references
- **Code review** before merging to main
- **Tag releases** for major milestones

## Environment Configuration

### Required Environment Variables
```bash
# API Configuration
FASTAPI_HOST=localhost
FASTAPI_PORT=8000
FASTAPI_DEBUG=true

# AI Model Configuration
gemini_api_key=your_gemini_api_key
CREW_AI_MODEL=gpt-4

# Scraping Configuration
MAX_SCRAPING_DEPTH=3
SCRAPING_DELAY=1
MAX_PAGES_PER_SITE=100

# Storage Configuration
KNOWLEDGE_BASE_PATH=./knowledge
TEMP_STORAGE_PATH=./temp
```

## Code Quality Standards

### Required Code Practices
- **Type hints** for all function parameters and returns
- **Docstrings** for all classes and functions
- **Consistent formatting** using Black formatter
- **Import organization** with isort
- **Variable naming** must be descriptive and clear
- **No TODO comments** in production code

### Documentation Requirements
- **README.md** with setup and usage instructions
- **API documentation** auto-generated from FastAPI
- **Agent configuration** documented in YAML comments
- **Tool documentation** with usage examples

## Monitoring and Logging

### Required Logging
- **Scraping progress** and completion status
- **API request/response** logging (without sensitive data)
- **Error tracking** with stack traces
- **Performance metrics** (response times, success rates)
- **WebSocket connection** status and events

### Log Format
```
[TIMESTAMP] [LEVEL] [COMPONENT] [SCRAPE_ID] MESSAGE
Example: [2024-01-15 10:30:00] [INFO] [SCRAPER] [uuid-123] Started scraping https://example.com
```

## Deployment Preparation

### Production Readiness Checklist
- [ ] Environment variables properly configured
- [ ] Error handling covers all scenarios
- [ ] Rate limiting implemented
- [ ] Security headers configured
- [ ] Performance optimized
- [ ] Monitoring and logging operational
- [ ] Database integration ready (for future scaling)
- [ ] Docker containerization prepared
- [ ] Load testing completed

---

**Remember**: This project prioritizes **real-time user experience**, **reliable scraping**, and **intelligent content querying**. Every development decision should support these core objectives.