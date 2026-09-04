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
| Frontend | React 19, TypeScript, Vite, Motion |
| Backend | FastAPI, SQLAlchemy, PostgreSQL |
| AI / RAG | LangGraph, LangChain, Mistral LLM, Qdrant (vector DB) |
| Infrastructure | Docker, Docker Compose |

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
