"""
FastAPI application for Conversation Insights Pipeline
"""
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import json
import os
from dotenv import load_dotenv

from services.processor import chunk_text, create_batch_requests, parse_batch_results
from services.openai_client import OpenAIBatchClient
from services.pinecone_client import PineconeClient

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="Conversation Insights API")

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize clients
openai_client = OpenAIBatchClient()
pinecone_client = PineconeClient()

# In-memory storage for batch tracking
# Format: {batch_id: {conversation_id, status, chunks, metadata}}
batch_storage: Dict[str, Dict[str, Any]] = {}


# Pydantic models for request/response validation
class ConversationUpload(BaseModel):
    id: str
    transcript: str
    metadata: Dict[str, Any] = {}


class QueryRequest(BaseModel):
    query: str
    filters: Optional[Dict[str, Any]] = None
    top_k: int = 50


class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    processing_time_ms: float


class UploadResponse(BaseModel):
    batch_id: str
    status: str
    chunks: int
    conversation_id: str


class StatusResponse(BaseModel):
    batch_id: str
    status: str
    progress: str
    conversation_id: str
    details: Optional[Dict[str, Any]] = None


@app.on_event("startup")
async def startup_event():
    """Initialize Pinecone index on startup"""
    try:
        print("Initializing Pinecone index...")
        pinecone_client.setup_index(dimension=1536, metric="cosine")
        print("Pinecone index ready")
    except Exception as e:
        print(f"Error initializing Pinecone: {e}")
        print("Application will continue, but vector storage may not work")


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "service": "Conversation Insights API",
        "status": "running",
        "version": "1.0.0"
    }


@app.post("/upload", response_model=UploadResponse)
async def upload_conversation(conversation: ConversationUpload):
    """
    Upload and process a conversation

    Steps:
    1. Chunk the transcript
    2. Create batch requests for OpenAI (embeddings + extraction)
    3. Submit to OpenAI Batch API
    4. Store batch_id and return immediately
    """
    try:
        print(f"Processing conversation: {conversation.id}")

        # Step 1: Chunk the transcript
        chunks = chunk_text(conversation.transcript, max_tokens=1500, overlap=200)
        print(f"Created {len(chunks)} chunks")

        # Step 2: Create batch requests
        batch_requests = create_batch_requests(
            conversation_id=conversation.id,
            chunks=chunks,
            metadata=conversation.metadata
        )
        print(f"Created {len(batch_requests)} batch requests")

        # Step 3: Submit to OpenAI Batch API
        batch_id = openai_client.submit_batch(batch_requests)

        # Step 4: Store batch info
        batch_storage[batch_id] = {
            "conversation_id": conversation.id,
            "status": "submitted",
            "chunks": chunks,
            "metadata": conversation.metadata,
            "total_chunks": len(chunks)
        }

        return UploadResponse(
            batch_id=batch_id,
            status="submitted",
            chunks=len(chunks),
            conversation_id=conversation.id
        )

    except Exception as e:
        print(f"Error processing conversation: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process conversation: {str(e)}")


@app.get("/status/{batch_id}", response_model=StatusResponse)
async def get_batch_status(batch_id: str):
    """
    Check status of a batch job

    If completed, process results and upsert to Pinecone
    """
    try:
        # Check if batch_id exists in our storage
        if batch_id not in batch_storage:
            raise HTTPException(status_code=404, detail=f"Batch {batch_id} not found")

        stored_info = batch_storage[batch_id]

        # Check status with OpenAI
        batch_status = openai_client.check_batch(batch_id)

        # Update our storage
        stored_info["status"] = batch_status["status"]

        # Calculate progress
        request_counts = batch_status["request_counts"]
        progress = f"{request_counts['completed']}/{request_counts['total']} requests"

        # If completed, process results
        if batch_status["status"] == "completed" and stored_info.get("processed") != True:
            print(f"Batch {batch_id} completed, processing results...")

            try:
                # Get results from OpenAI
                results = openai_client.get_results(batch_id)

                # Parse and combine results
                pinecone_records = parse_batch_results(
                    results=results,
                    conversation_id=stored_info["conversation_id"],
                    chunks=stored_info["chunks"],
                    metadata=stored_info["metadata"]
                )

                # Upsert to Pinecone
                pinecone_client.upsert_conversation(
                    records=pinecone_records,
                    namespace="conversations"
                )

                # Mark as processed
                stored_info["processed"] = True
                stored_info["status"] = "completed_and_stored"

                print(f"Successfully stored {len(pinecone_records)} records in Pinecone")

            except Exception as e:
                print(f"Error processing batch results: {e}")
                stored_info["status"] = "processing_failed"
                stored_info["error"] = str(e)

        return StatusResponse(
            batch_id=batch_id,
            status=stored_info["status"],
            progress=progress,
            conversation_id=stored_info["conversation_id"],
            details=batch_status
        )

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error checking batch status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to check status: {str(e)}")


@app.post("/query", response_model=QueryResponse)
async def query_conversations(request: QueryRequest):
    """
    Query conversations with natural language

    Steps:
    1. Create embedding for the query
    2. Search Pinecone for relevant chunks
    3. Send top results to GPT for synthesis
    4. Return answer with sources
    """
    try:
        import time
        start_time = time.time()

        print(f"Processing query: {request.query}")

        # Step 1: Create query embedding
        query_embedding = openai_client.create_embedding(request.query)

        # Step 2: Search Pinecone
        matches = pinecone_client.query(
            embedding=query_embedding,
            top_k=request.top_k,
            filter=request.filters,
            namespace="conversations",
            include_metadata=True
        )

        print(f"Found {len(matches)} matches")

        if not matches:
            return QueryResponse(
                answer="No relevant conversations found for your query.",
                sources=[],
                processing_time_ms=(time.time() - start_time) * 1000
            )

        # Step 3: Take top 20 for synthesis
        top_matches = matches[:20]
        context_chunks = [match["metadata"] for match in top_matches]

        # Synthesize answer with GPT
        answer = openai_client.synthesize_answer(
            query=request.query,
            context_chunks=context_chunks
        )

        # Format sources
        sources = []
        for match in top_matches[:10]:  # Return top 10 sources
            metadata = match["metadata"]
            sources.append({
                "conversation_id": metadata.get("conversation_id"),
                "date": metadata.get("date"),
                "score": match["score"],
                "text": metadata.get("text", "")[:200] + "...",  # Truncate for display
                "sentiment": metadata.get("sentiment"),
                "intents": metadata.get("intents", [])
            })

        processing_time = (time.time() - start_time) * 1000

        return QueryResponse(
            answer=answer,
            sources=sources,
            processing_time_ms=processing_time
        )

    except Exception as e:
        print(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process query: {str(e)}")


@app.get("/conversations")
async def get_conversations():
    """
    Get list of all processed conversations

    Returns conversation metadata from Pinecone
    """
    try:
        conversations = pinecone_client.get_all_conversations(namespace="conversations")
        return {
            "conversations": conversations,
            "total": len(conversations)
        }

    except Exception as e:
        print(f"Error fetching conversations: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch conversations: {str(e)}")


@app.get("/batches")
async def get_all_batches():
    """
    Get all tracked batch jobs (for debugging)
    """
    return {
        "batches": batch_storage,
        "total": len(batch_storage)
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
