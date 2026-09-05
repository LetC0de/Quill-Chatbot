# Quill

## Project Description

Quill is an AI-powered Enterprise Knowledge Assistant that transforms your documents into an intelligent, searchable knowledge base. Upload PDFs and chat with them naturally — Quill understands context, remembers conversations, and provides accurate answers with source citations.

## Features

- **Document Intelligence** — Upload PDFs, auto-chunk and embed them into a vector store for semantic search
- **AI-Powered Q&A** — Ask questions in natural language and get context-aware answers grounded in your documents
- **Streaming Responses** — Real-time token-by-token answer streaming via Server-Sent Events
- **Conversation Memory** — Multi-turn conversations with persistent history powered by LangGraph checkpoints
- **Source Citations** — Each answer includes page-level references from the original document
- **Auto-Generated Titles** — Conversations are automatically named based on the first question
- **Dual Mode** — Document-specific RAG answers or general "concierge" chat when no document is selected
- **Secure Authentication** — JWT-based user registration and login system
- **Responsive Design** — Fully responsive UI that works on desktop and mobile
- **Dockerized Deployment** — One-command setup with Docker Compose

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | React 19, TypeScript, Vite 8, Motion (Framer Motion) |
| Backend | Python 3.13, FastAPI, SQLAlchemy, Alembic |
| Database | PostgreSQL (metadata + conversation history) |
| AI / RAG | LangChain, LangGraph, Gemini 2.5 Flash (LLM), Mistral (embeddings) |
| Vector Store | Qdrant Cloud (semantic document search) |
| Authentication | JWT (PyJWT + Argon2 password hashing) |
| Infrastructure | Docker, Docker Compose |
| Document Processing | PyPDF, LangChain Text Splitters |

## Project Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        FRONTEND                             │
│           React 19 + TypeScript + Vite + Motion             │
│                                                             │
│    Landing -> Auth -> Sidebar + ChatArea + UploadModal      │
│              |             |                                │
│              v             v                                │
│         JWT Token      SSE Stream (/chat/query)             │
└───────────────┬───────────────┬─────────────────────────────┘
                │ REST API       │ Server-Sent Events
                v                v
┌─────────────────────────────────────────────────────────────┐
│                    BACKEND (FastAPI)                        │
│                                                             │
│  ┌──────────┐  ┌───────────┐  ┌────────────────────────┐    │
│  │  Auth    │  │ Document  │  │   LangGraph Pipeline   │    │
│  │  Module  │  │ Module    │  │                        │    │
│  └──────────┘  └───────────┘  │  retrieve_documents    │    │
│       │             │         │         v              │    │
│       v             v         │   build_context        │    │
│  ┌──────────┐  ┌───────────┐  │         v              │    │
│  │ User     │  │ Upload    │  │  generate_answer       │    │
│  │(Postgres)│  │ (PyPDF -> │  │         v              │    │
│  └──────────┘  │  chunks)  │  │  checkpoint memory     │    │
│                └───────────┘  └────────────────────────┘    │
│                                       │                     │
│                  ┌────────────────────┼──────────┐          │
│                  v                    v          v          │
│            ┌───────────┐      ┌───────────┐ ┌────────┐      │
│            │ PostgreSQL│      │   Qdrant  │ │ Gemini │      │
│            │ (metadata │      │ (vectors) │ │ 2.5    │      │
│            │  + memory)│      │           │ │ Flash  │      │
│            └───────────┘      └───────────┘ └────────┘      │
└─────────────────────────────────────────────────────────────┘
```

**Request Flow:**
1. User uploads a PDF → PyPDF extracts text → LangChain splits into chunks → Mistral embeddings → stored in Qdrant
2. User asks a question → Qdrant retrieves relevant chunks → LangGraph builds context → Gemini generates an answer
3. Response streams token-by-token via SSE → PostgresSaver checkpoints conversation for memory
4. On the next turn, LangGraph loads prior history from checkpoint, giving the model conversational context

## Project Structure

```
rag-project/
├── backend/
│   ├── app.py                          # FastAPI application entry point + lifespan
│   ├── requirements.txt
│   ├── Dockerfile
│   ├── .env.example
│   ├── migrations/                     # Alembic database migrations
│   │   └── versions/
│   └── src/
│       ├── rag/                        # Core RAG pipeline
│       │   ├── llm.py                  # Gemini LLM instance + title generation
│       │   ├── embeddings.py           # Mistral embedding model
│       │   ├── vector_store.py         # Qdrant vector store init
│       │   ├── retriever.py            # Semantic retriever (filtered by document)
│       │   ├── loader.py              # PDF loader + text splitter
│       │   └── prompt.py              # System/human prompts (RAG + concierge)
│       ├── graph/                      # LangGraph orchestration
│       │   ├── graph.py               # Compiled StateGraph (retrieve → context → generate)
│       │   ├── nodes.py               # Graph node functions
│       │   ├── state.py               # RAGState TypedDict
│       │   ├── streaming.py           # SSE streaming orchestrator
│       │   └── checkpointer.py        # PostgresSaver singleton
│       ├── user/                       # User auth module
│       │   ├── model.py               # SQLAlchemy User model
│       │   ├── schema.py              # Pydantic request/response schemas
│       │   ├── controller.py          # Register/login logic
│       │   └── router.py             # FastAPI routes (/auth/*)
│       ├── document/                   # Document management
│       │   ├── model.py               # SQLAlchemy Document model
│       │   ├── schema.py
│       │   └── router.py             # CRUD routes (/documents/*)
│       ├── upload/                     # PDF upload + ingestion
│       │   ├── controller.py          # Extract → chunk → embed → store
│       │   ├── schema.py
│       │   └── router.py             # Upload route
│       ├── conversation/               # Conversation history
│       │   ├── model.py               # SQLAlchemy Conversation model
│       │   ├── schema.py
│       │   └── router.py             # CRUD routes (/conversations/*)
│       ├── query/                      # Chat query endpoint
│       │   ├── schema.py              # QueryRequestSchema
│       │   └── router.py             # SSE /chat/query endpoint
│       ├── admin/                      # Admin module
│       ├── qdrant/                     # Qdrant setup
│       │   ├── client.py              # Async Qdrant client
│       │   └── collection.py          # Collection creation/validation
│       └── utils/
│           ├── settings.py            # Pydantic Settings (env config)
│           ├── db.py                  # SQLAlchemy engine + session
│           └── helpers.py
├── frontend/
│   ├── package.json
│   ├── Dockerfile
│   ├── index.html
│   └── src/
│       ├── App.tsx                     # Root app with auth routing
│       ├── main.tsx                    # Entry point
│       ├── index.css / App.css
│       ├── components/
│       │   ├── Landing.tsx            # Marketing landing page
│       │   ├── AuthScreen.tsx         # Login / Register form
│       │   ├── Sidebar.tsx            # Conversation list + new chat
│       │   ├── ChatArea.tsx           # Main chat view + SSE handling
│       │   ├── ChatMessage.tsx        # Message bubble (markdown render)
│       │   ├── Composer.tsx           # Input field + send button
│       │   ├── DocumentPicker.tsx     # PDF selector sidebar
│       │   ├── UploadModal.tsx        # File upload dialog
│       │   └── Icons.tsx             # SVG icon components
│       └── lib/
│           ├── api.ts                # API client + SSE helpers
│           ├── auth.tsx              # Auth context + token management
│           ├── types.ts              # TypeScript interfaces
│           ├── palette.ts            # Theme color palette
│           └── markdown.tsx          # Markdown-to-JSX renderer
└── docker-compose.yml                 # Backend + Frontend services
```

## Demo / Live Link

https://enterpriseassistant.vercel.app/

## Screenshots

### Landing Page
<p align="center">
  <img src="./screenshots/landing.png" width="85%" alt="Landing Page"/>
</p>

The first impression of Quill — a clean, modern landing page that introduces the product, highlights key features like document intelligence and AI-powered Q&A, and guides users to get started.

### Login
<p align="center">
  <img src="./screenshots/login.png" width="85%" alt="Login Page"/>
</p>

A minimal and secure JWT-based authentication screen. Users can register or log in to access their personal document workspace, keeping conversations and files private.

### Chat Interface
<p align="center">
  <img src="./screenshots/home.png" width="85%" alt="Chat Interface"/>
</p>

The heart of Quill — an intuitive chat UI where users upload PDFs and ask questions in natural language. Responses stream in real-time with inline page citations, and conversations are automatically named and saved for future reference.

---

## Installation / Setup

### Prerequisites

Before you begin, make sure you have these installed and ready:

| Tool | Version | Purpose |
|------|---------|---------|
| **Python** | 3.13+ | Backend runtime |
| **Node.js** | 20+ | Frontend runtime |
| **PostgreSQL** | 14+ | User data, conversations, LangGraph checkpoints |
| **Qdrant Cloud** | — | Vector database for document embeddings |
| **Gemini API Key** | — | LLM for chat responses and title generation |
| **Mistral API Key** | — | Embedding model for document vectors |
---

### Step 1 — Clone the Repository

```bash
git clone https://github.com/LetC0de/Quill-Chatbot.git
cd project
```

---

### Step 2 — Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv venv

# Activate it
# Windows (PowerShell)
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

# Install all Python dependencies
pip install -r requirements.txt
```

---

### Step 3 — Configure Environment Variables


Now open `.env` and fill in your actual API keys and database URL:

```env
# LLM — Gemini (chat responses + title generation)
GEMINI_API_KEY=your_gemini_api_key

# Embeddings — Mistral (document vectorisation)
MISTRAL_API_KEY=your_mistral_api_key
MISTRAL_MODEL=mistral-embed-2312

# Vector Store — Qdrant Cloud
QDRANT_URL=https://your-cluster.cloud.qdrant.io
QDRANT_API_KEY=your_qdrant_api_key

# PostgreSQL — metadata, users, conversations, LangGraph checkpoints
DB_CONNECTION=postgresql://user:pass@host/dbname?sslmode=require

# CORS — allowed frontend origins
CORS_ORIGINS=["http://localhost:5173","http://127.0.0.1:5173"]

# Primary frontend URL
FRONTEND_URL=http://localhost:5173

# JWT Authentication
SECRET_KEY=your_jwt_secret_key
ALGORITHM=HS256
EXP_TIME=60
```
---

### Step 4 — Run Database Migrations

Quill uses **Alembic** to manage PostgreSQL schema. Run this to create all tables (users, documents, conversations):

```bash
alembic upgrade head
```

This creates the following tables in your database:
- `users` — registered accounts
- `documents` — uploaded PDF records
- `conversations` — chat session metadata
- LangGraph checkpoint tables (auto-created at first boot)

---

### Step 5 — Frontend Setup

Open a **new terminal** (keep the backend terminal running):

```bash
cd frontend

# Install Node.js dependencies
npm install
```

---

### Step 6 — Start the Servers

You need **two terminals** running side by side:

**Terminal 1 — Backend:**

```bash
cd backend
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux

uvicorn app:app --reload --port 8000
```

Backend runs at → **http://localhost:8000**
Interactive API docs → **http://localhost:8000/docs**

**Terminal 2 — Frontend:**

```bash
cd frontend
npm run dev
```

Frontend runs at → **http://localhost:5173**

---

### Step 7 — Use the App

1. Open **http://localhost:5173** in your browser
2. **Register** a new account (first screen)
3. **Upload a PDF** using the upload button
4. **Ask questions** about the document — answers stream in real-time with page citations

---

## Docker Setup

No need to install Python, Node.js, or manage virtual environments. Just **Docker Desktop** and one command.

### Prerequisites

- **Docker Desktop** installed and running — [Install Docker](https://www.docker.com/products/docker-desktop/)

### Quick Start

```bash
# Make sure your backend/.env file exists and is filled with API keys
# Then from the project root:
docker compose up
```

That's it. Both backend and frontend start automatically.

### docker-compose.yml

```yaml
services:

  backend:
    image: abhishekdevdocker392/quill-backend:latest
    ports:
      - "8000:8000"
    env_file:
      - ./backend/.env

  frontend:
    image: abhishekdevdocker392/quill-frontend:latest
    ports:
      - "3000:80"
    depends_on:
      - backend
```

### Services

| Service | Image | Port | URL |
|---------|-------|------|-----|
| Frontend | `quill-frontend:latest` | 3000 → 80 | http://localhost:3000 |
| Backend | `quill-backend:latest` | 8000 | http://localhost:8000 |
| API Docs | (built into backend) | — | http://localhost:8000/docs |


> **Note:** The `.env` file must be present at `backend/.env` before running Docker Compose. The backend container reads environment variables from this file at startup.

---

## Author

**Abhishek** — Full-Stack Ai Engineer

- GitHub: [@LetC0de](https://github.com/LetC0de)
- LinkedIn: [LinkedIn](https://www.linkedin.com/in/abhishek8at/)
- Docker Hub: [abhishekdevdocker392](https://hub.docker.com/u/abhishekdevdocker392)

---

<p align="center">Made with ❤️ </p>
