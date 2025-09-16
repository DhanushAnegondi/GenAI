# Generative AI Knowledge Extraction

A document processing system that uses AI to make large document collections searchable. Built for a hackathon to solve the problem of finding information across millions of documents quickly.

## Problem

Organizations have massive document repositories but struggle to find specific information. Traditional search only matches keywords, missing related concepts and context. We needed something that actually understands the content.

## Solution

We built a system that reads documents, understands their meaning using AI, and stores this understanding in a way that allows near-instant retrieval. When you search, it finds information based on meaning, not just matching words.

## How it works

1. Documents get uploaded to the system
2. Each document is split into manageable chunks while preserving context
3. AI generates embeddings (mathematical representations) for each chunk
4. These embeddings are stored in a vector database
5. Search queries get converted to the same format and matched against stored embeddings
6. Relevant chunks are retrieved and used to build comprehensive answers

## Tech Stack

- **FastAPI** for REST APIs
- **OpenAI embeddings** for document understanding
- **Pinecone** as vector database
- **AWS Lambda** for serverless processing
- **Apache Airflow** for workflow orchestration
- **Redis** for caching frequent queries
- **Docker** for containerization
- **Snowflake** for analytics and reporting

## Setup

Clone the repository:
```bash
git clone <repo>
cd genai-knowledge-extraction
```

Create environment file:
```bash
cp .env.example .env
```

Add your API keys to `.env`:
- OpenAI API key
- Pinecone API key
- AWS credentials

Run with Docker:
```bash
docker-compose up
```

The API will be available at `http://localhost:8000`

## Usage

Upload a document:
```bash
POST /api/v1/upload/document
Content-Type: multipart/form-data
Body: file
```

Search for information:
```bash
POST /api/v1/search/query
Content-Type: application/json
Body: {"query": "your question", "top_k": 10}
```

## Architecture

The system consists of several microservices:

- **Upload Service**: Handles document ingestion and validation
- **Processing Service**: Chunks documents and generates embeddings
- **Search Service**: Handles queries and retrieval
- **Analytics Service**: Tracks usage and performance metrics

Data flows through these services:
```
Document → Upload → Processing → Vector Store
Query → Search Service → Vector Store → Results
```

Background jobs are managed by Airflow for:
- Batch document processing
- Model retraining
- Data synchronization with Snowflake

## Performance

The system handles:
- Search latency under 250ms
- Document processing at 15 seconds per document
- Support for millions of documents
- 2000+ concurrent users

## Project Structure

```
/api            # FastAPI application and routes
/services       # Core business logic
/pipelines      # Airflow DAGs and Lambda functions
/infrastructure # Docker and Kubernetes configs
/tests          # Unit and integration tests
/scripts        # Deployment and utility scripts
```

## Key Features

**Semantic Search**: Understands context and meaning, not just keywords. A search for "revenue growth" will also find documents mentioning "sales increase" or "profit expansion".

**Smart Chunking**: Documents are split intelligently to preserve context. The system understands that tables, code blocks, and paragraphs need different handling.

**Distributed Processing**: Uses AWS Lambda for parallel processing of multiple documents, automatically scaling based on load.

**Analytics Dashboard**: Snowflake integration provides insights into search patterns, popular documents, and system performance.

## Testing

Run tests:
```bash
pytest tests/
```

The test suite includes:
- Unit tests for individual components
- Integration tests for API endpoints
- Performance benchmarks

## Deployment

The system is designed for cloud deployment with:
- Docker containers for all services
- Kubernetes manifests for orchestration
- Terraform scripts for infrastructure
- CI/CD pipeline configuration

Production deployment:
```bash
./scripts/deploy.sh
```

## Monitoring

Prometheus and Grafana provide monitoring for:
- API response times
- Document processing queue
- Vector database performance
- System resource usage

Access Grafana at `http://localhost:3000`

## Future Improvements

- Support for more document formats
- Multi-language support
- Real-time collaborative features
- On-premise deployment option

## License

MIT
