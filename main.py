import os
import logging

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
def initialize_bot():
    global bot
    if bot is not None:
        return bot
    
    try:
        logger.info("🚀 Starting Zain Ali AI/ML Bot...")
        
        # Validate environment variables
        required_vars = {
            "JINA_API_KEY": "Jina AI embeddings API key",
            "QDRANT_URL": "Qdrant vector database URL",
            "GROQ_API_KEY": "Groq LLM API key",
        }
        missing = [f"{var} ({desc})" for var, desc in required_vars.items() if not os.getenv(var)]
        if missing:
            logger.error(f"❌ Missing env vars: {missing}")
            return None

        # Check PDF
        pdf_path = "zains_rag_data.pdf"
        if not os.path.exists(pdf_path):
            logger.error(f"❌ PDF not found: {pdf_path}")
            return None
            
        logger.info(f"📄 Found profile PDF: {pdf_path}")

        # Initialize bot
        bot_instance = ResumeQABot(pdf_path)
        test_embedding = bot_instance.embedder.embed_query("test connection")
        logger.info(f"✅ Jina AI connected (embedding dim {len(test_embedding)})")

        if bot_instance.ingest():
            logger.info("🎉 Document ingestion completed")
            bot = bot_instance
            logger.info("🎯 Bot ready!")
            return bot
        else:
            logger.error("❌ Failed to ingest documents")
            return None
            
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        return None            # bot.close() if needed

# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(
    title="Zain Ali - AI/ML Engineer Chatbot",
    description="Chat with Zain Ali about AI/ML projects, automation workflows, and n8n integrations, powered by Jina AI embeddings",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    # lifespan=lifespan  # ✅ now defined above
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
        "service": "Zain Ali - AI/ML Engineer Chatbot",
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
    current_bot = initialize_bot()
    if not current_bot:
        raise HTTPException(status_code=503, detail="Bot is still initializing")

    response_text = current_bot.ask(request.message)
    usage_info = current_bot.embedder.get_usage_stats() if hasattr(current_bot, 'embedder') else {}

    return ChatResponse(response=response_text, usage_info=usage_info)


@app.post("/reset", response_model=StatusResponse)
async def reset_conversation(background_tasks: BackgroundTasks):
    if not bot:
        raise HTTPException(503, "Bot not initialized")
    background_tasks.add_task(bot.reset_context)
    return StatusResponse(status="success", message="Conversation context reset")

from mangum import Mangum
handler = Mangum(app, lifespan="off")
