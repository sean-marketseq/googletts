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
        pinecone_client.setup_index(dimension=1024, metric="cosine")
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

        # Step 2: Create batch requests (returns dict with separate embeddings/extractions)
        batch_requests = create_batch_requests(
            conversation_id=conversation.id,
            chunks=chunks,
            metadata=conversation.metadata
        )
        embedding_count = len(batch_requests["embeddings"])
        extraction_count = len(batch_requests["extractions"])
        print(f"Created {embedding_count} embedding requests and {extraction_count} extraction requests")

        # Step 3: Submit TWO separate batches (OpenAI requires same endpoint per batch)
        embedding_batch_id = openai_client.submit_batch(
            batch_requests["embeddings"],
            endpoint="/v1/embeddings"
        )
        extraction_batch_id = openai_client.submit_batch(
            batch_requests["extractions"],
            endpoint="/v1/chat/completions"
        )

        # Step 4: Store both batch infos with cross-references
        batch_storage[embedding_batch_id] = {
            "conversation_id": conversation.id,
            "status": "submitted",
            "chunks": chunks,
            "metadata": conversation.metadata,
            "total_chunks": len(chunks),
            "type": "embeddings",
            "paired_batch_id": extraction_batch_id
        }
        batch_storage[extraction_batch_id] = {
            "conversation_id": conversation.id,
            "status": "submitted",
            "chunks": chunks,
            "metadata": conversation.metadata,
            "total_chunks": len(chunks),
            "type": "extractions",
            "paired_batch_id": embedding_batch_id
        }

        # Return embedding batch_id as primary (client will poll this one)
        return UploadResponse(
            batch_id=embedding_batch_id,
            status="submitted",
            chunks=len(chunks),
            conversation_id=conversation.id
        )

    except Exception as e:
        print(f"Error processing conversation: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process conversation: {str(e)}")


@app.post("/upload-audio", response_model=UploadResponse)
async def upload_audio(
    file: UploadFile = File(...),
    conversation_id: str = None,
    metadata: str = "{}"
):
    """
    Upload and process an audio file

    Steps:
    1. Transcribe audio with Whisper API
    2. Chunk the transcript
    3. Create batch requests for OpenAI (embeddings + extraction)
    4. Submit to OpenAI Batch API
    5. Store batch_id and return immediately
    """
    try:
        # Validate file type
        allowed_extensions = ['.mp3', '.mp4', '.mpeg', '.mpga', '.m4a', '.wav', '.webm']
        file_ext = os.path.splitext(file.filename)[1].lower()

        if file_ext not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type. Allowed: {', '.join(allowed_extensions)}"
            )

        # Generate conversation ID if not provided
        if not conversation_id:
            import uuid
            conversation_id = f"conv_{uuid.uuid4().hex[:8]}"

        # Parse metadata
        try:
            metadata_dict = json.loads(metadata)
        except:
            metadata_dict = {}

        print(f"Processing audio file: {file.filename} for conversation: {conversation_id}")

        # Step 1: Transcribe audio
        audio_content = await file.read()
        transcript = openai_client.transcribe_audio(audio_content, file.filename)

        print(f"Transcription complete: {len(transcript)} characters")

        # Step 2: Chunk the transcript
        chunks = chunk_text(transcript, max_tokens=1500, overlap=200)
        print(f"Created {len(chunks)} chunks")

        # Step 3: Create batch requests (returns dict with separate embeddings/extractions)
        batch_requests = create_batch_requests(
            conversation_id=conversation_id,
            chunks=chunks,
            metadata=metadata_dict
        )
        embedding_count = len(batch_requests["embeddings"])
        extraction_count = len(batch_requests["extractions"])
        print(f"Created {embedding_count} embedding requests and {extraction_count} extraction requests")

        # Step 4: Submit TWO separate batches (OpenAI requires same endpoint per batch)
        embedding_batch_id = openai_client.submit_batch(
            batch_requests["embeddings"],
            endpoint="/v1/embeddings"
        )
        extraction_batch_id = openai_client.submit_batch(
            batch_requests["extractions"],
            endpoint="/v1/chat/completions"
        )

        # Step 5: Store both batch infos with cross-references
        batch_storage[embedding_batch_id] = {
            "conversation_id": conversation_id,
            "status": "submitted",
            "chunks": chunks,
            "metadata": metadata_dict,
            "total_chunks": len(chunks),
            "source": "audio",
            "filename": file.filename,
            "type": "embeddings",
            "paired_batch_id": extraction_batch_id
        }
        batch_storage[extraction_batch_id] = {
            "conversation_id": conversation_id,
            "status": "submitted",
            "chunks": chunks,
            "metadata": metadata_dict,
            "total_chunks": len(chunks),
            "source": "audio",
            "filename": file.filename,
            "type": "extractions",
            "paired_batch_id": embedding_batch_id
        }

        # Return embedding batch_id as primary (client will poll this one)
        return UploadResponse(
            batch_id=embedding_batch_id,
            status="submitted",
            chunks=len(chunks),
            conversation_id=conversation_id
        )

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error processing audio: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to process audio: {str(e)}")


@app.get("/status/{batch_id}", response_model=StatusResponse)
async def get_batch_status(batch_id: str):
    """
    Check status of a batch job

    Since we now submit TWO batches (embeddings + extractions),
    we wait for BOTH to complete before processing results
    """
    try:
        # Check if batch_id exists in our storage
        if batch_id not in batch_storage:
            raise HTTPException(status_code=404, detail=f"Batch {batch_id} not found")

        stored_info = batch_storage[batch_id]
        paired_batch_id = stored_info.get("paired_batch_id")

        # Check status of the requested batch
        batch_status = openai_client.check_batch(batch_id)
        stored_info["status"] = batch_status["status"]

        # Also check the paired batch if it exists
        paired_batch_status = None
        if paired_batch_id and paired_batch_id in batch_storage:
            paired_batch_status = openai_client.check_batch(paired_batch_id)
            batch_storage[paired_batch_id]["status"] = paired_batch_status["status"]

        # Calculate combined progress
        request_counts = batch_status["request_counts"]
        if paired_batch_status:
            paired_counts = paired_batch_status["request_counts"]
            total_completed = request_counts['completed'] + paired_counts['completed']
            total_requests = request_counts['total'] + paired_counts['total']
            progress = f"{total_completed}/{total_requests} requests (embeddings + extractions)"
        else:
            progress = f"{request_counts['completed']}/{request_counts['total']} requests"

        # Only process if BOTH batches are completed
        both_completed = (
            batch_status["status"] == "completed" and
            (not paired_batch_status or paired_batch_status["status"] == "completed")
        )

        if both_completed and stored_info.get("processed") != True:
            print(f"Both batches completed for {stored_info['conversation_id']}, processing results...")

            try:
                # Get results from BOTH batches
                results = openai_client.get_results(batch_id)

                if paired_batch_id:
                    paired_results = openai_client.get_results(paired_batch_id)
                    # Combine results from both batches
                    results.extend(paired_results)
                    print(f"Combined {len(results)} results from both batches")

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

                # Mark BOTH batches as processed
                stored_info["processed"] = True
                stored_info["status"] = "completed_and_stored"

                if paired_batch_id and paired_batch_id in batch_storage:
                    batch_storage[paired_batch_id]["processed"] = True
                    batch_storage[paired_batch_id]["status"] = "completed_and_stored"

                print(f"Successfully stored {len(pinecone_records)} records in Pinecone")

            except Exception as e:
                print(f"Error processing batch results: {e}")
                import traceback
                traceback.print_exc()
                stored_info["status"] = "processing_failed"
                stored_info["error"] = str(e)

        # Determine overall status to return
        if stored_info.get("processed"):
            overall_status = "completed_and_stored"
        elif both_completed:
            overall_status = "completed"
        elif batch_status["status"] == "failed" or (paired_batch_status and paired_batch_status["status"] == "failed"):
            overall_status = "failed"
        else:
            overall_status = "processing"

        return StatusResponse(
            batch_id=batch_id,
            status=overall_status,
            progress=progress,
            conversation_id=stored_info["conversation_id"],
            details=batch_status
        )

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error checking batch status: {e}")
        import traceback
        traceback.print_exc()
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


@app.delete("/purge-index")
async def purge_index():
    """
    ⚠️ DANGER: Delete ALL data from Pinecone index

    This is for testing purposes only. Use with caution!
    """
    try:
        print("⚠️  PURGE REQUEST RECEIVED")

        # Purge all data from Pinecone
        pinecone_client.purge_all_data(namespace="conversations")

        # Clear in-memory batch storage
        batch_storage.clear()

        return {
            "message": "All data purged successfully",
            "namespace": "conversations",
            "batches_cleared": True
        }

    except Exception as e:
        print(f"Error purging index: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to purge index: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
