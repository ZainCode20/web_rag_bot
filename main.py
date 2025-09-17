import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from resume_qa import ResumeQABot
from typing import Any

# -----------------------------
# Logging
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# -----------------------------
# Global bot
# -----------------------------
bot = None   # will be set inside lifespan

# -----------------------------
# Lifespan context (startup + shutdown)
# -----------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Initialize the Jina AI powered bot on startup
    and clean up on shutdown.
    """
    global bot
    try:
        logger.info("🚀 Starting Ali Akbar AI Graphics Specialist Bot...")

        # Validate environment variables
        required_vars = {
            "JINA_API_KEY": "Jina AI embeddings API key",
            "QDRANT_URL": "Qdrant vector database URL",
            "GROQ_API_KEY": "Groq LLM API key",
        }
        missing = [
            f"{var} ({desc})"
            for var, desc in required_vars.items()
            if not os.getenv(var)
        ]
        if missing:
            raise ValueError(
                "❌ Missing required environment variables:\n"
                + "\n".join(f"- {var}" for var in missing)
            )

        # Check PDF
        pdf_path = "zains_rag_data.pdf"
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"Profile data PDF not found: {pdf_path}")
        logger.info(f"📄 Found profile PDF: {pdf_path}")

        # Initialize bot
        bot_instance = ResumeQABot(pdf_path)
        test_embedding = bot_instance.embedder.embed_query("test connection")
        logger.info(f"✅ Jina AI connected (embedding dim {len(test_embedding)})")

        if bot_instance.ingest():
            logger.info("🎉 Document ingestion completed")
            logger.info(bot_instance.get_usage_summary())
        else:
            raise RuntimeError("Failed to ingest profile documents")

        bot = bot_instance
        logger.info("🎯 Bot ready!")
        yield  # ---- App runs here ----

    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        bot = None
        yield  # app still starts so /health shows unhealthy

    finally:
        if bot:
            logger.info("🛑 Shutting down bot gracefully...")
            # bot.close() if needed

# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(
    title="Zain Ali - AI/ML Engineer Chatbot",
    description="Chat with Zain Ali about AI/ML projects, automation workflows, and n8n integrations, powered by Jina AI embeddings",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan  # ✅ now defined above
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=500)

class ChatResponse(BaseModel):
    response: str
    status: str = "success"
    usage_info: dict[str, Any] | None = None   # <--- changed from str to dict

class StatusResponse(BaseModel):
    status: str
    message: str | None = None
    details: dict | None = None
# -----------------------------
# Routes (unchanged)
# -----------------------------
@app.get("/")
async def root():
    return {
        "service": "Ali Akbar - AI Graphics Specialist Chatbot",
        "powered_by": "Jina AI + Groq LLM",
        "status": "ready" if bot else "initializing",
        "version": "2.0.0",
    }

@app.get("/health", response_model=StatusResponse)
async def health_check():
    if not bot:
        return StatusResponse(status="unhealthy", message="Bot not initialized")
    try:
        s = bot.get_status()
        return StatusResponse(
            status="healthy" if s["status"] == "healthy" else "unhealthy",
            message="All systems operational" if s["status"] == "healthy" else "System components not healthy",
            details=s,
        )
    except Exception as e:
        return StatusResponse(status="unhealthy", message=str(e))

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if not bot:
        raise HTTPException(status_code=503, detail="Bot is still initializing")

    # Use the correct method and gather usage info
    response_text = bot.ask(request.message)
    usage_info = bot.embedder.get_usage_stats()

    return ChatResponse(response=response_text, usage_info=usage_info)


@app.post("/reset", response_model=StatusResponse)
async def reset_conversation(background_tasks: BackgroundTasks):
    if not bot:
        raise HTTPException(503, "Bot not initialized")
    background_tasks.add_task(bot.reset_context)
    return StatusResponse(status="success", message="Conversation context reset")
