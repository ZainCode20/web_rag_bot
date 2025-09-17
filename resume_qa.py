import os
import logging
import requests
from typing import List, Dict, Any
import time
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_qdrant import QdrantVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableMap
from langchain_core.embeddings import Embeddings
import random

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Jina AI Configuration
JINA_CONFIG = {
    "base_url": "https://api.jina.ai/v1/embeddings", # remove the spaces
    "model": "jina-embeddings-v2-base-en",
    "dimension": 768,
    "max_tokens_per_request": 8192,
    "max_requests_per_minute": 200,
    "encoding": "cl100k_base"
}

class JinaEmbeddings(Embeddings):
    """Optimized Jina AI embeddings with smart batching and error handling"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("JINA_API_KEY")
        if not self.api_key:
            raise ValueError("JINA_API_KEY environment variable is required")
        
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Track usage and performance
        self.requests_made = 0
        self.tokens_used = 0
        self.jina_failures = 0
        
        logger.info("🔍 Jina AI Embeddings initialized")
    
    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimation (1 token ≈ 4 characters for English)"""
        return len(text) // 4
    
    def _make_jina_request(self, texts: List[str], max_retries: int = 3) -> List[List[float]]:
        """Make request to Jina AI API with retry logic"""
        
        for attempt in range(max_retries):
            try:
                # Prepare payload
                payload = {
                    "model": JINA_CONFIG["model"],
                    "input": texts
                }
                response = requests.post(
                    JINA_CONFIG["base_url"],
                    headers=self.headers,
                    json=payload,
                    timeout=60
                )
                                
                logger.debug(f"📤 Jina API request: {len(texts)} texts")
                
                # response = requests.post(
                #     JINA_CONFIG["base_url"],
                #     headers=self.headers,
                #     json=payload,
                #     timeout=60
                # )
                
                # Handle rate limiting
                if response.status_code == 429:
                    wait_time = min(2 ** attempt * 5, 60)  # Max 60 seconds
                    logger.warning(f"⏳ Jina rate limited, waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                
                response.raise_for_status()
                data = response.json()
                
                # Extract embeddings
                embeddings = [item["embedding"] for item in data["data"]]
                
                # Update usage tracking
                self.requests_made += 1
                self.tokens_used += sum(self._estimate_tokens(text) for text in texts)
                
                logger.debug(f"✅ Jina success: {len(embeddings)} embeddings")
                return embeddings
                
            except requests.exceptions.Timeout:
                logger.warning(f"⏰ Jina timeout (attempt {attempt + 1})")
                time.sleep(2 ** attempt)
                continue
                
            except requests.exceptions.RequestException as e:
                logger.error(f"❌ Jina API error (attempt {attempt + 1}): {e}")
                self.jina_failures += 1
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                break
            
        return None
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple documents with smart batching"""
        if not texts:
            return []
        
        logger.info(f"🔢 Embedding {len(texts)} documents with Jina AI")
        
        # Process in batches to respect rate limits and token limits
        batch_size = 10  # Conservative batch size for Jina
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (len(texts) - 1) // batch_size + 1
            
            logger.info(f"📦 Processing batch {batch_num}/{total_batches} ({len(batch)} texts)")
            
            # Truncate texts that are too long
            truncated_batch = []
            for text in batch:
                if self._estimate_tokens(text) > JINA_CONFIG["max_tokens_per_request"]:
                    truncated_text = text[:JINA_CONFIG["max_tokens_per_request"] * 4]  # Rough truncation
                    truncated_batch.append(truncated_text)
                    logger.debug("✂️ Truncated long text")
                else:
                    truncated_batch.append(text)
            
            # Try Jina AI first
            embeddings = self._make_jina_request(truncated_batch)
            
            # Last resort: dummy embeddings
            if embeddings is None:
                embeddings = self._create_dummy_embeddings(len(truncated_batch))
            
            all_embeddings.extend(embeddings)
            
            # Rate limiting between batches
            if i + batch_size < len(texts):
                time.sleep(1)
        
        logger.info(f"✅ Completed embedding {len(all_embeddings)} documents")
        return all_embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """Embed single query text"""
        logger.debug(f"🔍 Embedding query: {text[:100]}...")
        
        # Truncate if too long
        if self._estimate_tokens(text) > JINA_CONFIG["max_tokens_per_request"]:
            text = text[:JINA_CONFIG["max_tokens_per_request"] * 4]
        
        # Try Jina AI
        embeddings = self._make_jina_request([text])
        if embeddings and len(embeddings) > 0:
            return embeddings[0]
        
        # Last resort: dummy embedding
        logger.warning("⚠️ Using dummy embedding for query")
        return [0.0] * JINA_CONFIG["dimension"]
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics"""
        return {
            "requests_made": self.requests_made,
            "estimated_tokens_used": self.tokens_used,
            "jina_failures": self.jina_failures,
            "tokens_remaining_estimate": max(0, 1_000_000 - self.tokens_used),
            "model": JINA_CONFIG["model"],
            "dimension": JINA_CONFIG["dimension"]
        }
    
    def _create_dummy_embeddings(self, count: int) -> List[List[float]]:
        """Create dummy embeddings with zero vectors."""
        logger.warning("⚠️ Generating dummy embeddings as fallback")
        return [[0.0] * JINA_CONFIG["dimension"] for _ in range(count)]


class ResumeQABot:
    """Resume QA Bot optimized for Jina AI embeddings on Render"""
    
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        
        # Initialize Jina embeddings
        self.embedder = JinaEmbeddings()
        logger.info(f"📏 Using {JINA_CONFIG['dimension']}-dimensional embeddings")
        
        # Qdrant setup
        qdrant_url = os.getenv("QDRANT_URL")
        qdrant_api_key = os.getenv("QDRANT_API_KEY")
        
        if not qdrant_url:
            raise ValueError("❌ QDRANT_URL environment variable is required")
        
        self.qdrant_client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        self.collection_name = "resume_jina_embeddings"
        
        # Create collection with Jina's 768-dimensional vectors
        if self.qdrant_client.collection_exists(self.collection_name):
          self.qdrant_client.delete_collection(self.collection_name)
          logger.info(f"🗑️  Deleted old collection ‘{self.collection_name}’")

    # ➜  then create the new 768-d collection (your existing code)
        self.qdrant_client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=768, distance=Distance.COSINE),
        )

        # Initialize vector store
        self.vectorstore = QdrantVectorStore(
            client=self.qdrant_client,
            collection_name=self.collection_name,
            embedding=self.embedder,  # ✅ correct parameter name
        )
                
        # Groq LLM setup
        groq_api_key = os.getenv('GROQ_API_KEY')
        if not groq_api_key:
            raise ValueError("❌ GROQ_API_KEY environment variable is required")
        
        self.llm = ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0.3,
            max_tokens=1024,
            api_key=groq_api_key,
        )
        
        # Conversation memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            max_token_limit=2000
        )
        
        # Optimized prompt for Ali Akbar's services
        self.qa_prompt = ChatPromptTemplate.from_messages([
           ("system",
    "You are Zain Ali, a Python AI/ML Engineer answering questions about MY services and expertise. "
    "I specialize in AI chatbots, automation workflows, RAG systems, and custom AI solutions. "
    "ONLY answer questions related to my AI/ML services, projects, or technical expertise. "
    
    "Examples of what I CAN help with:"
    "- 'Can you build a chatbot for my business?' → 'Yes, I create custom AI chatbots for businesses. Let me know your requirements!'"
    "- 'What's a RAG system?' → 'RAG systems help chatbots answer questions using your specific documents and data. I build these regularly.'"
    "- 'Do you do automation?' → 'Absolutely! I create automation workflows to streamline business processes using Python and AI.'"
    
    "Examples of what's OUTSIDE my scope:"
    "- 'What can generate images?' → 'That's outside my area of focus. I specialize in AI/ML solutions. How can I help you with chatbots, automation, or custom AI development?'"
    "- 'How to cook pasta?' → 'That's outside my area of focus. I specialize in AI/ML solutions. How can I help you with chatbots, automation, or custom AI development?'"
    "- 'What's the weather?' → 'That's outside my area of focus. I specialize in AI/ML solutions. How can I help you with chatbots, automation, or custom AI development?'"
    
    "Keep all responses brief and conversational - maximum 2-3 sentences, speaking as ME (Zain). "
    "Use first person (I, my, me) to make it personal. "
    "For project discussions or consultations, direct users to contact me: "
    "Email: zaincode20@gmail.com or WhatsApp: +923062020798. "
    "Don't provide lengthy explanations or generic information."
),
          
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{question}")
        ])
        
        self.parser = StrOutputParser()
        logger.info("🤖 ResumeQABot initialized successfully!")
    
    def ingest(self) -> bool:
        """Ingest PDF documents with Jina AI embeddings"""
        logger.info(f"📚 Starting document ingestion: {os.path.basename(self.pdf_path)}")
        
        if not os.path.exists(self.pdf_path):
            logger.error(f"❌ PDF file not found: {self.pdf_path}")
            return False
        
        try:
            # Check if documents already exist
            try:
                existing_docs = self.vectorstore.similarity_search("Zain Ali services", k=1)
                if existing_docs:
                    logger.info("✅ Documents already exist in vector store")
                    return True
            except Exception:
                logger.info("📁 Vector store is empty, proceeding with ingestion")
            
            # Load PDF
            loader = PyPDFLoader(self.pdf_path)
            documents = loader.load()
            logger.info(f"📄 Loaded {len(documents)} pages from PDF")
            
            # Optimize text splitting for Jina AI
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,  # Larger chunks for better context
                chunk_overlap=200,  # Good overlap for continuity
                separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " "],
                length_function=len,
            )
            
            chunks = splitter.split_documents(documents)
            logger.info(f"✂️ Created {len(chunks)} text chunks")
            
            # Add rich metadata
            for i, doc in enumerate(chunks):
                doc.metadata.update({
                    "chunk_id": i,
                    "source": os.path.basename(self.pdf_path),
                    "page": doc.metadata.get("page", 0),
                    "char_count": len(doc.page_content),
                    "estimated_tokens": len(doc.page_content) // 4
                })
            
            # Ingest in small batches (Jina free tier consideration)
            batch_size = 5  # Conservative for free tier
            total_batches = (len(chunks) - 1) // batch_size + 1
            
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                batch_num = i // batch_size + 1
                
                logger.info(f"📦 Ingesting batch {batch_num}/{total_batches}")
                
                try:
                    self.vectorstore.add_documents(batch)
                    logger.debug(f"✅ Batch {batch_num} completed")
                    
                    # Rate limiting for Jina API
                    if batch_num < total_batches:
                        time.sleep(2)
                        
                except Exception as e:
                    logger.error(f"❌ Failed to ingest batch {batch_num}: {e}")
                    return False
            
            # Log usage statistics
            stats = self.embedder.get_usage_stats()
            logger.info(f"📊 Ingestion complete! Usage: {stats['estimated_tokens_used']} tokens, "
                       f"{stats['requests_made']} requests")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Document ingestion failed: {e}")
            return False
    
    def ask(self, question: str) -> str:
        """Process user question with Jina-powered retrieval"""
        if not question.strip():
           return "Hello! I’m Zain Ali, a Python AI/ML Engineer. Ask me about AI automation, RAG applications, FastAPI APIs, n8n workflows, or advanced image pipelines."

        
        try:
            logger.info(f"💬 Processing question: {question[:100]}...")
            
            # Retrieve relevant context
            retriever = self.vectorstore.as_retriever(
                search_kwargs={
                    "k": 4,  # Get top 4 most relevant chunks
                    "score_threshold": 0.7  # Only high-quality matches
                }
            )
            
            relevant_docs = retriever.get_relevant_documents(question)
            logger.debug(f"🔍 Retrieved {len(relevant_docs)} relevant documents")
            
            # Format context
            context = "\n\n".join([
                f"[Source: Page {doc.metadata.get('page', '?')}]\n{doc.page_content}"
                for doc in relevant_docs[:3]  # Limit context to avoid token limits
            ])
            
            # Create conversation chain
            chain = (
                RunnableMap({
                    "question": lambda x: x["question"],
                    "context": lambda x: context,
                    "chat_history": lambda _: self.memory.chat_memory.messages[-6:]  # Last 3 exchanges
                })
                | self.qa_prompt
                | self.llm
                | self.parser
            )
            
            # Generate response
            response = chain.invoke({"question": question})
            
            # Update memory
            self.memory.chat_memory.add_user_message(question)
            self.memory.chat_memory.add_ai_message(response)
            
            # Add natural conversation flow
            final_response = self._make_conversational(response)
            
            logger.info("✅ Response generated successfully")
            return final_response
            
        except Exception as e:
            logger.error(f"❌ Error processing question: {e}")
            return("I'm having a bit of trouble right now, but I'm here to help! "
                    "Feel free to ask me about AI automation, RAG applications, FastAPI APIs, "
                    "n8n workflows, or advanced image processing with ComfyUI.")

    
    def _make_conversational(self, response: str) -> str:
        """Add natural conversation starters"""
        casual_starters = [
            "Great question! ", "Absolutely! ", "Sure thing! ", 
            "Yeah, ", "Well, ", "Good point! ", "I'd be happy to help! "
        ]
        
        # Don't add starter if response already sounds natural
        if (len(response) > 40 and 
            not any(response.lower().startswith(word) for word in 
                   ["great", "sure", "absolutely", "yeah", "well", "hi", "hello", "i"])):
            return random.choice(casual_starters) + response
        
        return response
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive bot status"""
        try:
            # Qdrant health
            collections = self.qdrant_client.get_collections()
            collection_exists = self.collection_name in [c.name for c in collections.collections]
            
            # Test embedding
            test_embedding = self.embedder.embed_query("test")
            
            # Jina usage stats
            jina_stats = self.embedder.get_usage_stats()
            
            return {
                "status": "healthy",
                "qdrant": {
                    "connected": True,
                    "collection_exists": collection_exists,
                    "collection_name": self.collection_name
                },
                "embeddings": {
                    "provider": "Jina AI",
                    "model": JINA_CONFIG["model"],
                    "dimension": len(test_embedding),
                    "fallback_available": self.embedder.fallback_to_hf,
                    **jina_stats
                },
                "memory_conversations": len(self.memory.chat_memory.messages) // 2
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }
    
    def reset_conversation(self):
        """Reset conversation memory"""
        self.memory.clear()
        logger.info("🔄 Conversation memory reset")
    
    def get_usage_summary(self) -> str:
        """Get human-readable usage summary"""
        stats = self.embedder.get_usage_stats()
        
        tokens_used = stats['estimated_tokens_used']
        tokens_remaining = stats['tokens_remaining_estimate']
        usage_percent = (tokens_used / 1_000_000) * 100
        
        return (f"📊 Jina AI Usage: {tokens_used:,} tokens used "
                f"({usage_percent:.1f}% of free tier), "
                f"{tokens_remaining:,} tokens remaining this month")
    







    #  (
    #             "system",
    #             "You're Zain Ali, a Python AI/ML Engineer specializing in backend automation, RAG applications, "
    #             "and AI-powered workflows.\n"
    #             "You excel in: FastAPI API development, LangChain/LangGraph RAG solutions, n8n automation (image & video "
    #             "generation bots, social media posting), and advanced image pipelines with ComfyUI.\n"
    #             "Chat naturally and professionally with potential clients or users. Be helpful, concise, and focus on your "
    #             "expertise in AI/ML engineering, automation, and agentic AI.\n"
    #             "If asked about topics outside your professional specialties, politely redirect the conversation back to your "
    #             "AI/ML services.\n\n"
    #             "Relevant context from your profile:\n{context}"
    #         ),