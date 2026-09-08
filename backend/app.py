from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.qdrant.collection import create_collection
from src.conversation.model import ConversationModel  # noqa: F401  registers table with base.metadata
from src.conversation.router import conversation_router
from src.document.model import DocumentModel  # noqa: F401
from src.document.router import document_router
from src.query.router import chat_router
from src.upload.router import upload_router
from src.user.model import UserModel  # noqa: F401
from src.user.router import user_router
from src.utils.db import base, engine
from src.utils.settings import settings



base.metadata.create_all(bind=engine)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup — Qdrant collection + LangGraph checkpointer tables
    # (PostgresSaver.setup() is idempotent; safe to run on every boot).
    create_collection()

    # Initialise the LangGraph PostgresSaver singleton + run setup() once.
    # This creates the LangGraph checkpoint tables (checkpoint, writes, blobs)
    # if they don't exist. Per the architecture spec: setup() runs at DB
    # initialisation time, NOT per request.
    from src.graph.checkpointer import close_checkpointer, init_checkpointer
    from src.graph.graph import get_compiled_graph
    # Initialise the async PostgresSaver singleton + run setup() once (creates
    # the LangGraph checkpoint tables). This must run before the first request.
    await init_checkpointer()
    # Compile the graph once and keep it warm for the lifetime of the app.
    get_compiled_graph()

    yield

    # Release the long-lived checkpointer connection on shutdown.
    await close_checkpointer()


app = FastAPI(title="Enterprise Knowledge Asistant", lifespan=lifespan)


app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(upload_router)
app.include_router(chat_router)
app.include_router(document_router)
app.include_router(user_router)
app.include_router(conversation_router)


@app.get("/")
async def root():
    return {"message": "Enterprise Knowledge Assistant API"}
    