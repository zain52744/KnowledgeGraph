# Knowledge Graph RAG System

A hybrid Retrieval Augmented Generation (RAG) system combining Knowledge Graphs (NetworkX + MongoDB) with FAISS vector search for intelligent document understanding and querying.

## Features

✨ **Multi-source ingestion**: Parse PDFs and load structured CSV data  
🔍 **Intelligent extraction**: Extract entities and relations using GPT via LangChain  
🚀 **Hybrid retrieval**:
- Vector search: FAISS-based semantic similarity for text chunks
- Graph traversal: NetworkX graph for structured entity relationships

📊 **Production-ready API**: FastAPI with async MongoDB storage  
🔬 **LangFuse tracing**: Monitor and trace all LLM operations  
🐳 **Docker-ready**: Complete Docker Compose setup with MongoDB and the API

## Quick Start

### Prerequisites
- Docker and Docker Compose
- Python 3.11+ (for local development)
- OpenAI API key

### Run with Docker (Recommended)

```bash
# Clone and setup
git clone <repo>
cd KnowledgeGraph

# Create .env file with your configuration
cp .env.example .env
# Edit .env with your OpenAI API key and other settings

# Start services
docker-compose up -d

# Wait for services to be healthy
docker-compose ps

# API will be available at http://localhost:8000
```

### Run Locally

```bash
# Install dependencies with uv
uv sync

# Setup MongoDB locally or update MONGODB_URI in .env
# Set OPENAI_API_KEY in environment

# Run the application
uv run python -m uvicorn api.main:app --host 0.0.0.0 --port 8000

# Run tests
uv run pytest tests/ -v
```

## API Endpoints

### Health & Info

#### GET `/health`
Check API health status and MongoDB connection.

```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "status": "healthy",
  "mongodb": "connected"
}
```

#### GET `/`
API information and available endpoints.

```bash
curl http://localhost:8000/
```

### Data Ingestion

#### POST `/ingest/pdf`
Upload and process a PDF file for knowledge extraction.

```bash
curl -X POST http://localhost:8000/ingest/pdf \
  -F "file=@/path/to/document.pdf"
```

Response:
```json
{
  "status": "success",
  "chunks_processed": 42,
  "entities_extracted": 156
}
```

#### POST `/ingest/csv`
Upload and process a CSV file with entity relations.

Expected CSV format (source, relation, target):
```csv
source,relation,target
Apple,founded_by,Steve Jobs
Steve Jobs,works_at,Apple
```

```bash
curl -X POST http://localhost:8000/ingest/csv \
  -F "file=@/path/to/triples.csv"
```

Response:
```json
{
  "status": "success",
  "chunks_processed": 25,
  "entities_extracted": 50
}
```

### Hybrid RAG Query

#### POST `/query`
Query the knowledge graph with hybrid RAG retrieval.

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the relationship between entity A and entity B?"
  }'
```

Response:
```json
{
  "answer": "Entity A is connected to Entity B through...",
  "sources": [
    "document_chunk_1.txt",
    "document_chunk_2.txt"
  ],
  "graph_context": [
    {
      "entity": "Entity A",
      "relation": "related_to",
      "target": "Entity B"
    }
  ]
}
```

### Graph Exploration

#### GET `/graph/entity/{entity_name}`
Retrieve an entity and its direct neighbors from the knowledge graph.

```bash
curl http://localhost:8000/graph/entity/Apple
```

Response:
```json
{
  "entity": "Apple",
  "data": {
    "type": "Company",
    "attributes": {}
  },
  "neighbors": ["Steve Jobs", "Tim Cook", "iPhone"],
  "neighbor_count": 3
}
```

#### GET `/graph/path?source=Entity1&target=Entity2`
Find the shortest path between two entities in the knowledge graph.

```bash
curl "http://localhost:8000/graph/path?source=Apple&target=iPhone"
```

Response:
```json
{
  "source": "Apple",
  "target": "iPhone",
  "path": ["Apple", "products", "iPhone"],
  "path_length": 2
}
```

No path response:
```json
{
  "source": "Apple",
  "target": "UnrelatedEntity",
  "path": null,
  "path_length": -1,
  "message": "No path exists between entities"
}
```

## How It Works

### The RAG Pipeline

```
1. DATA INGESTION
   ├─ PDF parsing: Extract text with 1000-char chunks, 100-char overlap
   ├─ CSV loading: Load structured triples (source, relation, target)
   └─ Store raw content for retrieval

2. EXTRACTION & INDEXING
   ├─ Entity extraction: Use GPT to identify entities in chunks
   ├─ Relation extraction: Extract relationships between entities
   ├─ Build knowledge graph: Store in NetworkX DiGraph
   └─ Index vectors: Create FAISS embeddings with text-embedding-3-small

3. RETRIEVAL (Hybrid)
   ├─ Vector Search:
   │  ├─ Embed user question with text-embedding-3-small
   │  ├─ Find top-5 similar text chunks in FAISS
   │  └─ Combine chunks as context
   │
   └─ Graph Search:
      ├─ Extract entities from question via LLM
      ├─ Find neighbors in NetworkX graph (1-hop)
      └─ Collect neighbor node information

4. ANSWER GENERATION
   ├─ Combine vector context + graph context
   ├─ Build prompt with combined context
   ├─ Call GPT-4o-mini with temperature=0 (deterministic)
   └─ Return grounded answer

5. MONITORING & TRACING
   ├─ Trace all LLM calls with LangFuse
   ├─ Log input/output/latency for each operation
   └─ Store in MongoDB for analysis
```

### System Components

| Component | Purpose | Technology |
|-----------|---------|-----------|
| **PDF Parser** | Extract text from PDFs | pypdf, LangChain |
| **CSV Loader** | Load structured knowledge | pandas |
| **Extractor** | Extract entities & relations | ChatOpenAI (GPT-4o-mini) |
| **Graph Storage** | Store entity relationships | NetworkX DiGraph + MongoDB |
| **Vector Store** | Store semantic embeddings | FAISS (text-embedding-3-small) |
| **Orchestration** | Coordinate pipeline | LangGraph StateGraph |
| **API** | HTTP interface | FastAPI |
| **Monitoring** | Trace LLM operations | LangFuse |

## Environment Configuration

Create a `.env` file in the project root:

```bash
# OpenAI Configuration
OPENAI_API_KEY=sk-your-key-here
OPENAI_MODEL=gpt-4o-mini

# MongoDB Configuration
MONGODB_URI=mongodb://admin:password@mongodb:27017/knowledge_graph?authSource=admin

# LangFuse Configuration (optional)
LANGFUSE_PUBLIC_KEY=pk-your-key
LANGFUSE_SECRET_KEY=sk-your-key
LANGFUSE_HOST=https://cloud.langfuse.com

# Application Configuration
FAISS_INDEX_PATH=./faiss_index
LOG_LEVEL=INFO
```

## Testing

Run the comprehensive test suite:

```bash
# Run all tests
uv run pytest tests/ -v

# Run specific test module
uv run pytest tests/test_rag_retriever.py -v

# Run with coverage
uv run pytest tests/ --cov=. --cov-report=html
```

Test coverage includes:
- ✅ Hybrid retrieval (13 tests)
- ✅ Orchestration pipeline (16 tests)
- ✅ LangFuse monitoring (28 tests)
- **Total**: 57 tests passing

## Deployment

### Docker Compose (Development)

```bash
docker-compose up -d
docker-compose logs -f app
docker-compose down  # Cleanup
```

### Production Deployment

1. Use environment-specific `.env` files
2. Configure MongoDB with authentication
3. Set up LangFuse for production tracing
4. Use a production ASGI server (Gunicorn with Uvicorn workers)
5. Set up monitoring and alerting

Example production docker-compose override:
```yaml
services:
  app:
    command: >
      gunicorn api.main:app
      --workers=4
      --worker-class uvicorn.workers.UvicornWorker
      --bind 0.0.0.0:8000
```

## Development

### Project Structure

```
KnowledgeGraph/
├── api/                          # FastAPI application
│   ├── main.py                  # App initialization, CORS, startup
│   └── routes/
│       ├── ingest.py            # PDF and CSV ingestion endpoints
│       └── query.py             # Query and graph exploration endpoints
├── ingestion/                    # Data loading
│   ├── pdf_parser.py            # PDF text extraction
│   └── csv_loader.py            # CSV triple loading
├── extraction/                   # Entity/relation extraction
│   ├── extractor.py             # LLM-based extraction
│   └── schemas.py               # Pydantic models
├── graph/                        # Knowledge graph storage
│   ├── graph_builder.py         # NetworkX graph operations
│   └── mongo_store.py           # MongoDB persistence
├── vector_store/                 # Semantic search
│   └── faiss_store.py           # FAISS index management
├── rag/                          # Retrieval & answering
│   └── retriever.py             # Hybrid search and answer generation
├── orchestration/                # LangGraph pipelines
│   └── pipeline.py              # Ingestion and query pipelines
├── monitoring/                   # Observability
│   └── langfuse_setup.py        # LangFuse tracing
└── tests/                        # Comprehensive test suite
    ├── test_rag_retriever.py
    ├── test_orchestration_pipeline.py
    └── test_langfuse_monitoring.py
```

### Adding New Features

1. **New ingestion format**: Add parser in `ingestion/`, update `/ingest/<format>` route
2. **New extraction method**: Extend `extraction/extractor.py`, update pipeline node
3. **Custom retrieval logic**: Modify `rag/retriever.py` hybrid search strategy
4. **New graph queries**: Add GET endpoint in `api/routes/query.py`

## Monitoring & Debugging

### Access Swagger UI
```
http://localhost:8000/docs
```

### View Logs
```bash
# Docker logs
docker-compose logs -f app

# LangFuse traces (if configured)
https://cloud.langfuse.com
```

### MongoDB Queries
```bash
# Connect to MongoDB
mongosh mongodb://admin:password@localhost:27017

# View knowledge graphs
use knowledge_graph
db.graph_snapshots.find()

# View extraction results
db.extractions.find()
```

## Contributing

1. Create a feature branch: `git checkout -b feature/xyz`
2. Make changes and run tests: `uv run pytest tests/ -v`
3. Ensure all tests pass (57/57)
4. Submit a pull request

## License

MIT

    PDF/CSV Input
         │
    ┌────▼────────────────┐
    │ Parse & Chunk       │
    │ (PDFParser/CSVLoad) │
    └────┬────────────────┘
         │
    ┌────▼──────────────────────────┐
    │ Extract Entities & Relations  │
    │ (OpenAI via LangChain)         │
    └────┬──────────────────────────┘
         │
    ┌────┴──────────────┬───────────────┐
    │                   │               │
┌───▼────────┐   ┌────▼──────┐  ┌──────▼────┐
│ NetworkX   │   │ FAISS     │  │ MongoDB   │
│ Graph      │   │ Vectors   │  │ Persistence
└────────────┘   └───────────┘  └───────────┘
```

## Setup Instructions

### Prerequisites

- Python 3.11+
- Docker & Docker Compose
- OpenAI API key

### 1. Clone Repository

```bash
cd d:\KnowledgeGraph
```

### 2. Environment Configuration

```bash
# Copy example to .env
cp .env.example .env

# Edit .env and add your credentials:
OPENAI_API_KEY=sk-your-key-here
MONGODB_URI=mongodb://localhost:27017
LANGFUSE_PUBLIC_KEY=your_key
LANGFUSE_SECRET_KEY=your_secret
```

### 3. Install Dependencies (Local Development)

#### Option A: Using `uv` (Recommended)

```bash
# Install uv (if not already installed)
# On Windows:
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Create and activate virtual environment with uv
uv venv
source .venv/Scripts/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
uv sync

# Install dev dependencies (for testing)
uv sync --all-extras
```

#### Option B: Using pip

```bash
# Create virtual environment
python -m venv .venv
source .venv/Scripts/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# For development/testing
pip install pytest pytest-cov pytest-mock black isort flake8
```

### 4. Start Services with Docker

```bash
docker-compose up -d
```

This starts:
- **FastAPI app** on `http://localhost:8000`
- **MongoDB** on `localhost:27017`

### 5. Verify Setup

```bash
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "healthy",
  "mongodb": "connected"
}
```

## API Usage Examples

### Health Check

```bash
curl http://localhost:8000/health
```

### Ingest PDF

```bash
curl -X POST http://localhost:8000/ingest/pdf \
  -F "file=@sample_data/sample.pdf"
```

Response:
```json
{
  "success": true,
  "message": "Successfully ingested sample.pdf",
  "data_count": 42
}
```

### Ingest CSV

```bash
curl -X POST http://localhost:8000/ingest/csv \
  -F "file=@sample_data/sample.csv"
```

Expected CSV format (source, relation, target):
```csv
source,relation,target
Albert Einstein,discovered,Theory of Relativity
Marie Curie,collaborated_with,Pierre Curie
```

Response:
```json
{
  "success": true,
  "message": "Successfully ingested sample.csv",
  "data_count": 5
}
```

### Query the Knowledge Graph

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Who discovered radium?",
    "vector_k": 5,
    "graph_hops": 1
  }'
```

Response:
```json
{
  "answer": "Based on the knowledge graph, Marie Curie discovered Radium in 1898.",
  "vector_sources": [
    {
      "text": "Marie Curie's pioneering research...",
      "source": "sample.csv",
      "distance": 0.234
    }
  ],
  "graph_context": [
    {
      "entity": "Marie Curie",
      "type": "Person",
      "neighbors": {
        "outgoing": [
          ["Radium", "discovered"],
          ["Nobel Prize", "won_prize"]
        ],
        "incoming": []
      }
    }
  ],
  "combined_context": "## Vector Search Results:..."
}
```

## Why FAISS?

- **Lightweight**: No external vector database service needed
- **Fast**: CPU-optimized similarity search via L2 distance
- **Self-contained**: Index stored as local files (faiss.index + metadata.json)
- **Scalable**: Handles millions of vectors efficiently
- **Integrated**: Works seamlessly with LangChain OpenAI embeddings
- **Cost-effective**: No cloud DB fees, runs locally or in container

## Project Structure

```
d:\KnowledgeGraph\
├── ingestion/              # PDF parsing & CSV loading
│   ├── pdf_parser.py      # PyMuPDF text extraction
│   └── csv_loader.py      # Pandas CSV to triples
├── extraction/            # Entity/relation extraction
│   ├── extractor.py       # OpenAI via LangChain
│   └── schemas.py         # Pydantic models
├── graph/                 # Knowledge graph operations
│   ├── graph_builder.py   # NetworkX directed graph
│   └── mongo_store.py     # Async MongoDB CRUD
├── vector_store/          # Vector embeddings
│   └── faiss_store.py     # FAISS index wrapper
├── rag/                   # Retrieval & generation
│   └── retriever.py       # Hybrid retriever
├── api/                   # FastAPI application
│   ├── main.py            # App setup & lifespan
│   └── routes/            # API endpoints
│       ├── ingest.py      # PDF/CSV ingestion
│       └── query.py       # Query endpoint
├── orchestration/         # LangGraph workflows
│   └── pipeline.py        # Ingest → Extract → Store
├── monitoring/            # Observability
│   └── langfuse_setup.py  # LLM tracing
├── sample_data/           # Example files
├── requirements.txt       # Python dependencies
├── docker-compose.yml     # Services config
├── Dockerfile             # Container image
└── README.md             # This file
```

## Key Features

### ✅ Async/Await Throughout

All FastAPI routes and MongoDB calls use async/await for high concurrency:

```python
@router.post("/query")
async def query(request: QueryRequest):
    # Fully async - no blocking
    return await retriever.retrieve(...)
```

### ✅ Comprehensive Docstrings

All classes and functions include docstrings with type hints:

```python
def extract(self, text: str) -> KnowledgeGraph:
    """
    Extract entities and relations from text.
    
    Args:
        text: Input text to analyze.
        
    Returns:
        KnowledgeGraph object with extracted entities and relations.
    """
```

### ✅ Environment Variable Management

Uses `python-dotenv` for configuration:

```python
load_dotenv()
mongo_uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
```

### ✅ Production-Ready

- CORS enabled for cross-origin requests
- Health check endpoint for monitoring
- Structured error handling with HTTP exceptions
- Logging throughout for debugging
- Docker support for containerization

## Development

### Run Tests

```bash
pytest tests/
```

### Local Development (without Docker)

```bash
# Install dependencies
pip install -r requirements.txt

# Start MongoDB separately
mongod --dbpath ./data/mongo

# Run API
uvicorn api.main:app --reload
```

## Testing

The project includes comprehensive unit tests for all storage layers:

### Run All Tests

```bash
# Using pytest
pytest tests/ -v

# With coverage
pytest tests/ -v --cov=extraction --cov=ingestion --cov=graph --cov=vector_store --cov-report=html

# Using uv (if installed)
uv run pytest tests/ -v
```

### Test Files

- **test_extraction.py** - Entity/Relation extraction and validation
- **test_ingestion.py** - PDF parsing and CSV loading  
- **test_graph_storage.py** - NetworkX graph operations and queries
- **test_mongo_store.py** - MongoDB persistence and retrieval
- **test_faiss_store.py** - Vector storage and semantic search

### Run Specific Tests

```bash
# Graph storage tests only
pytest tests/test_graph_storage.py -v

# MongoDB tests
pytest tests/test_mongo_store.py -v

# Vector storage tests
pytest tests/test_faiss_store.py -v

# Single test class
pytest tests/test_graph_storage.py::TestGraphBuilder -v

# Single test method
pytest tests/test_graph_storage.py::TestGraphBuilder::test_initialization -v
```

### View Logs

```bash
docker-compose logs -f app
```

### Access MongoDB

```bash
docker exec -it knowledge-graph-mongodb mongosh
# Use database
use knowledge_graph
# Check collections
show collections
```

## Monitoring with Langfuse

When configured with Langfuse credentials, all LLM calls are automatically traced:

1. Configure `.env` with Langfuse keys
2. LLM calls appear in Langfuse dashboard
3. Track latency, token usage, and errors

## Performance Tips

1. **Batch Processing**: Process large CSV files in chunks
2. **Vector Index**: Re-build FAISS index periodically with `faiss_store.save()`
3. **MongoDB Indexes**: Already created on app startup
4. **CORS**: Restrict `allow_origins` in production
5. **Rate Limiting**: Add FastAPI `SlowAPI` for production deployments

## Troubleshooting

### MongoDB Connection Failed

```
Check MONGODB_URI in .env
Ensure MongoDB container is running: docker ps
View logs: docker-compose logs mongodb
```

### OpenAI API Errors

```
Verify OPENAI_API_KEY in .env
Check API key permissions
Ensure rate limits not exceeded
```

### FAISS Index Not Loading

```
Verify FAISS_INDEX_PATH exists
Check file permissions
Rebuild index: faiss_store.load(path)
```

## License

MIT License - See LICENSE file for details

## Support

For issues or questions, refer to the docstrings and inline comments throughout the codebase.
