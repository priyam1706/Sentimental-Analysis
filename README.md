AI Social Media Command Center
A backend-focused AI system that ingests social media content, runs it through a locally hosted Mistral LLM via Ollama, and returns AI-driven analysis including sentiment tagging and summarization — all through a unified REST API.
Features
FastAPI backend with clean REST endpoints for content ingestion and analysis
Local LLM inference using Ollama (Mistral) — no external API calls required
Web scraping scripts to collect content from public social media sources
Pydantic schemas for strict request validation
SQLAlchemy ORM with SQLite for persistent storage of posts and analysis results
Modular service architecture separating ingestion, inference, and storage layers
Tech Stack
Language: Python
Backend: FastAPI, REST APIs
LLM: Ollama (Mistral), local inference
Validation: Pydantic
Database: SQLite, SQLAlchemy ORM
Scraping: BeautifulSoup / custom scripts
Architecture
```
Ingestion Layer     →    Inference Layer     →    Storage Layer
(Web Scraper +           (Ollama / Mistral        (SQLAlchemy +
 FastAPI endpoints)       LLM analysis)            SQLite)
```
How to Run
1. Install dependencies
```bash
pip install -r requirements.txt
```
2. Install and start Ollama with Mistral
```bash
ollama pull mistral
ollama serve
```
3. Start the FastAPI server
```bash
uvicorn main:app --reload
```
4. API available at
```
http://localhost:8000/docs
```
API Endpoints
Method	Endpoint	Description
POST	`/ingest`	Submit social media content for analysis
GET	`/posts`	Retrieve stored posts and their analysis
POST	`/analyze`	Run LLM sentiment tagging on a post
GET	`/summary`	Get aggregated content summary
Project Structure
```
SentimentalAnalysis/
├── main.py           # FastAPI app entry point
├── models.py         # SQLAlchemy database models
├── schemas.py        # Pydantic request/response schemas
├── services/
│   ├── ingestion.py  # Web scraping and data collection
│   ├── inference.py  # Ollama LLM integration
│   └── storage.py    # Database operations
└── requirements.txt  # Dependencies
```
