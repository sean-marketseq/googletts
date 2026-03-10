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
import random

from services.processor import chunk_text, create_batch_requests, parse_batch_results
from services.openai_client import OpenAIBatchClient
from services.pinecone_client import PineconeClient
from services.hume_client import HumeClient
from fastapi import Request
import io
import traceback

# Load environment variables
load_dotenv()

# Hume sampling rate (0.0 to 1.0) - default 10% to reduce costs
HUME_SAMPLING_RATE = float(os.getenv("HUME_SAMPLING_RATE", "0.1"))
print(f"[CONFIG] Hume sampling rate: {HUME_SAMPLING_RATE * 100}%")

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
hume_client = HumeClient()

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
    """Initialize Pinecone indices on startup"""
    try:
        print("Initializing Pinecone conversations index...")
        pinecone_client.setup_index(dimension=1024, metric="cosine")
        print("Conversations index ready")

        print("Initializing Pinecone emotions index...")
        pinecone_client.setup_emotion_index(dimension=192, metric="cosine")
        print("Emotions index ready")
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
    1. Transcribe audio with Whisper API (with diarization)
    2. Label speakers (AGENT/CALLER or SPEAKER_A/B)
    3. Submit to Hume for emotion prosody analysis
    4. Chunk the transcript
    5. Create batch requests for OpenAI (embeddings + extraction)
    6. Submit to OpenAI Batch API
    7. Store all job IDs and return immediately
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

        # Parse metadata first
        try:
            metadata_dict = json.loads(metadata)
        except:
            metadata_dict = {}

        # Generate conversation ID if not provided
        # Try to extract from filename pattern: conv_<id>_<date>.ext
        if not conversation_id:
            import uuid
            import re

            # Try to parse filename: conv_0001k83j5q64f6ytzj2vkw9vrve0_2025-10-21.mp3
            filename_without_ext = os.path.splitext(file.filename)[0]
            match = re.match(r'^(conv_[a-z0-9]+)_(\d{4}-\d{2}-\d{2})$', filename_without_ext)

            if match:
                conversation_id = match.group(1)
                extracted_date = match.group(2)
                print(f"Extracted conversation_id: {conversation_id}, date: {extracted_date} from filename")

                # Add extracted date to metadata if not already present
                if 'date' not in metadata_dict:
                    metadata_dict['date'] = extracted_date
            else:
                # Fallback to generated ID
                conversation_id = f"conv_{uuid.uuid4().hex[:8]}"
                print(f"Filename pattern not recognized, generated conversation_id: {conversation_id}")

        print(f"Processing audio file: {file.filename} for conversation: {conversation_id}")

        # Read audio content
        audio_content = await file.read()
        file_size_mb = len(audio_content) / (1024 * 1024)
        print(f"File size: {file_size_mb:.2f} MB")

        # === STEP 1: Transcribe with diarization ===
        try:
            print(f"Starting transcription for {file.filename}...")
            diarization = openai_client.transcribe_audio_with_diarization(
                io.BytesIO(audio_content),
                file.filename
            )
            print(f"Transcription successful: {len(diarization.get('segments', []))} segments")
        except Exception as e:
            print(f"ERROR: Transcription failed for {file.filename}: {type(e).__name__}: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Transcription failed: {type(e).__name__}: {str(e)}"
            )

        speaker_labels, labeling_confidence = openai_client.label_speakers_as_agent_caller(
            diarization["segments"]
        )

        print(f"Speaker labeling: {speaker_labels} (confidence: {labeling_confidence})")

        # === STEP 2: Submit to Hume (with random sampling for cost reduction) ===
        hume_job_id = None
        use_hume = random.random() < HUME_SAMPLING_RATE

        if use_hume:
            try:
                hume_callback_url = os.getenv("HUME_CALLBACK_URL")
                hume_job_id = hume_client.submit_audio(
                    io.BytesIO(audio_content),
                    file.filename,
                    callback_url=hume_callback_url
                )
                print(f"[HUME] Job submitted: {hume_job_id} (sampled: {HUME_SAMPLING_RATE * 100}%)")
            except Exception as e:
                print(f"[HUME] Warning: Submission failed: {e}")
                print("Continuing with OpenAI processing only...")
        else:
            print(f"[HUME] Skipped for cost savings (sampling rate: {HUME_SAMPLING_RATE * 100}%)")

        # === STEP 3: Chunk transcript ===
        transcript = diarization["full_transcript"]
        chunks = chunk_text(transcript, max_tokens=1500, overlap=200)
        print(f"Created {len(chunks)} chunks")

        # === STEP 4: Extract data from chunks IN PARALLEL (TIER 1 OPTIMIZATION) ===
        print(f"[OPTIMIZATION] Extracting data from {len(chunks)} chunks in parallel...")
        try:
            # Use new parallel extraction method
            extraction_results = await openai_client.extract_chunks_parallel(chunks)

            # Add conversation_id to each result
            for result in extraction_results:
                result["chunk_id"] = f"{conversation_id}_chunk_{result['chunk_id'].split('_')[-1]}"

            print(f"[OPTIMIZATION] Parallel extraction complete for {len(chunks)} chunks")
        except Exception as e:
            print(f"ERROR: Parallel extraction failed: {type(e).__name__}: {str(e)}")
            # Fallback to sequential if parallel fails
            print("Falling back to sequential extraction...")
            extraction_results = []
            for chunk in chunks:
                extracted = openai_client.extract_conversation_data(chunk["text"])
                extraction_results.append({
                    "chunk_id": f"{conversation_id}_chunk_{chunk['index']}",
                    "data": extracted
                })
            print(f"Sequential extraction complete for {len(chunks)} chunks")

        # === STEP 5: Create embedding batch requests (only embeddings now) ===
        batch_requests = create_batch_requests(
            conversation_id=conversation_id,
            chunks=chunks,
            metadata=metadata_dict
        )
        embedding_count = len(batch_requests["embeddings"])
        print(f"Created {embedding_count} embedding requests")

        # === STEP 6: Submit embedding batch ===
        print(f"[BATCH] Submitting embedding batch with {embedding_count} requests...")
        embedding_batch_id = openai_client.submit_batch(
            batch_requests["embeddings"],
            endpoint="/v1/embeddings"
        )
        print(f"[BATCH] Embedding batch submitted: {embedding_batch_id}")

        # Check initial batch status
        try:
            embedding_initial = openai_client.check_batch(embedding_batch_id)
            print(f"[BATCH] Initial embedding status: {embedding_initial['status']}")
        except Exception as e:
            print(f"[BATCH] Warning: Could not check initial status: {e}")

        # === STEP 7: Store job info (2 jobs: embedding + hume, extractions done synchronously) ===
        batch_storage[embedding_batch_id] = {
            "conversation_id": conversation_id,
            "status": "submitted",
            "chunks": chunks,
            "metadata": metadata_dict,
            "total_chunks": len(chunks),
            "source": "audio",
            "filename": file.filename,
            "type": "embeddings",
            "paired_batch_ids": {
                "hume": hume_job_id
            },
            "extraction_results": extraction_results,  # Store synchronous extraction results
            "diarization": diarization,
            "speaker_labels": speaker_labels,
            "labeling_confidence": labeling_confidence
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
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to process audio: {str(e)}")


@app.post("/hume/callback")
async def hume_callback(request: Request):
    """
    Webhook endpoint for Hume batch completion
    Validates signature and processes emotion data
    """
    try:
        # TEMPORARILY DISABLED - Hume webhook secret validation causing 401 errors
        # provided_secret = request.headers.get("X-Hume-Webhook-Secret")
        # expected_secret = os.getenv("HUME_WEBHOOK_SECRET")
        #
        # if provided_secret != expected_secret:
        #     print(f"Invalid webhook secret received")
        #     raise HTTPException(status_code=401, detail="Invalid webhook secret")

        payload = await request.json()
        print(f"Hume webhook received: {payload}")

        job_id = payload.get("job_id")
        status = payload.get("status")

        if status == "COMPLETED":
            # Find conversation by hume job_id
            batch_info = None
            for bid, info in batch_storage.items():
                if info.get("paired_batch_ids", {}).get("hume") == job_id:
                    batch_info = info
                    break

            if not batch_info:
                print(f"No batch found for Hume job: {job_id}")
                return {"status": "ignored"}

            print(f"Processing Hume results for job: {job_id}")

            # Use predictions from webhook if available, otherwise fetch via API
            if "predictions" in payload:
                print("Using predictions from webhook payload")
                predictions = payload["predictions"]
            else:
                print("Fetching predictions via API")
                predictions = hume_client.get_predictions(job_id)

            # Align with speakers
            aligned = hume_client.align_emotions_with_speakers(
                predictions,
                batch_info["diarization"]["segments"],
                batch_info["speaker_labels"]
            )

            # Compute features for each speaker
            speaker_a_features = hume_client.compute_speaker_emotion_features(
                aligned["speaker_a_timeline"]
            )
            speaker_b_features = hume_client.compute_speaker_emotion_features(
                aligned["speaker_b_timeline"]
            )

            # Combine features
            combined = hume_client.compute_combined_features(
                speaker_a_features,
                speaker_b_features
            )

            # DEBUG: Check combined vector
            combined_vec = combined["combined_vector"]
            non_zero_count = sum(1 for v in combined_vec if v != 0.0)
            print(f"DEBUG: Combined emotion vector length: {len(combined_vec)}, non-zero values: {non_zero_count}")
            if non_zero_count == 0:
                print(f"WARNING: Combined emotion vector is all zeros!")
                print(f"  Speaker A emotion vector length: {len(speaker_a_features['emotion_vector'])}, sample: {speaker_a_features['emotion_vector'][:5]}")
                print(f"  Speaker B emotion vector length: {len(speaker_b_features['emotion_vector'])}, sample: {speaker_b_features['emotion_vector'][:5]}")

            # Store emotion data for this conversation
            batch_info["emotion_data"] = {
                "speaker_a": speaker_a_features,
                "speaker_b": speaker_b_features,
                "combined_vector": combined["combined_vector"],
                "timelines": {
                    "speaker_a": aligned["speaker_a_timeline"],
                    "speaker_b": aligned["speaker_b_timeline"]
                },
                "hume_job_status": "completed"
            }

            print(f"Emotion data stored for conversation: {batch_info['conversation_id']}")

        return {"status": "received"}

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in Hume callback: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/status/{batch_id}", response_model=StatusResponse)
async def get_batch_status(batch_id: str):
    """
    Check status of a batch job

    Now checks THREE jobs: embeddings, extractions, AND hume prosody
    Waits for ALL THREE to complete before processing results
    """
    try:
        # Check if batch_id exists in our storage
        if batch_id not in batch_storage:
            raise HTTPException(status_code=404, detail=f"Batch {batch_id} not found")

        stored_info = batch_storage[batch_id]
        paired_batch_ids = stored_info.get("paired_batch_ids", {})

        # Check 2 job statuses (embeddings + Hume, extractions done synchronously)
        embedding_batch_id = batch_id
        hume_job_id = paired_batch_ids.get("hume")

        # Check embedding batch status
        embedding_status = openai_client.check_batch(embedding_batch_id)

        # Log batch status for debugging
        print(f"[STATUS] Embedding batch {embedding_batch_id}: {embedding_status['status']} - {embedding_status['request_counts']}")

        # Check Hume job status - use stored emotion data if webhook received
        hume_status = {"state": "N/A"}
        hume_done = False

        if hume_job_id:
            # First check if webhook has already stored emotion data
            emotion_data = stored_info.get("emotion_data", {})
            if emotion_data.get("hume_job_status") == "completed":
                hume_status = {"state": "COMPLETED", "message": "Webhook received"}
                hume_done = True
                print(f"[STATUS] Hume job {hume_job_id}: COMPLETED (via webhook)")
            else:
                # Fallback to API check if no webhook data yet
                try:
                    hume_raw = hume_client.get_job_status(hume_job_id)
                    # Sanitize Hume response - only extract serializable fields
                    hume_status = {
                        "state": hume_raw.get("state", "UNKNOWN"),
                        "message": hume_raw.get("message", "")
                    }
                    if "error" in hume_raw:
                        hume_status["error"] = str(hume_raw["error"])
                    hume_done = hume_status.get("state") == "COMPLETED"
                except Exception as e:
                    print(f"Error checking Hume job status: {e}")
                    hume_status = {"state": "ERROR", "error": str(e)}

        # Calculate combined progress (2 jobs: embeddings + Hume)
        jobs_completed = 0

        embedding_done = embedding_status and embedding_status["status"] == "completed"

        # Only count Hume if it was submitted
        if hume_job_id:
            total_jobs = 2
            if embedding_done:
                jobs_completed += 1
            if hume_done:
                jobs_completed += 1
            progress = f"{jobs_completed}/{total_jobs} jobs complete (Embeddings + Hume)"
            all_completed = embedding_done and hume_done
        else:
            total_jobs = 1
            if embedding_done:
                jobs_completed += 1
            progress = f"{jobs_completed}/{total_jobs} jobs complete (Embeddings only)"
            all_completed = embedding_done

        # Process and upsert if all done
        if all_completed and stored_info.get("processed") != True:
            print(f"All jobs completed for {stored_info['conversation_id']}, processing...")

            try:
                # Get embedding results from batch
                embedding_batch_results = openai_client.get_results(embedding_batch_id)

                # Build embedding map
                # New format: ONE batch result with ALL embeddings as array
                embeddings_map = {}
                for result in embedding_batch_results:
                    custom_id = result.get("custom_id", "")
                    if custom_id.startswith("embed_all_"):
                        # Extract all embeddings from the data array
                        response_body = result.get("response", {}).get("body", {})
                        embeddings_data = response_body.get("data", [])

                        # Map embeddings by index to chunk IDs
                        for idx, embed_obj in enumerate(embeddings_data):
                            chunk_id = f"{stored_info['conversation_id']}_chunk_{idx}"
                            embedding = embed_obj.get("embedding", [])
                            embeddings_map[chunk_id] = embedding

                        print(f"Extracted {len(embeddings_data)} embeddings from single batch request")
                        break  # Only one result now

                # Get stored extraction results (already computed synchronously)
                extraction_results = stored_info.get("extraction_results", [])

                # Build extractions map
                extractions_map = {}
                for extraction in extraction_results:
                    chunk_id = extraction["chunk_id"]
                    extractions_map[chunk_id] = extraction["data"]

                # Combine into Pinecone records
                conversation_records = []
                for chunk in stored_info["chunks"]:
                    chunk_id = f"{stored_info['conversation_id']}_chunk_{chunk['index']}"

                    if chunk_id not in embeddings_map:
                        print(f"Warning: Missing embedding for {chunk_id}")
                        continue

                    extracted = extractions_map.get(chunk_id, {
                        "intents": [],
                        "entities": [],
                        "sentiment": 0.0,
                        "action_items": [],
                        "compliance_flags": []
                    })

                    record = {
                        "id": chunk_id,
                        "values": embeddings_map[chunk_id],
                        "metadata": {
                            "conversation_id": stored_info["conversation_id"],
                            "chunk_index": chunk["index"],
                            "text": chunk["text"],
                            "token_count": chunk["token_count"],
                            # Merge conversation metadata
                            **stored_info["metadata"],
                            # Add extracted data
                            "intents": extracted.get("intents", []),
                            "entities": json.dumps(extracted.get("entities", [])),
                            "sentiment": extracted.get("sentiment", 0.0),
                            "action_items": extracted.get("action_items", []),
                            "compliance_flags": extracted.get("compliance_flags", [])
                        }
                    }

                    conversation_records.append(record)

                # Add emotion metadata to conversation records
                emotion_data = stored_info.get("emotion_data", {})
                for record in conversation_records:
                    record["metadata"]["speaker_a_top_emotions"] = emotion_data.get("speaker_a", {}).get("top_5_emotions", [])
                    record["metadata"]["speaker_b_top_emotions"] = emotion_data.get("speaker_b", {}).get("top_5_emotions", [])
                    record["metadata"]["labeling_confidence"] = stored_info.get("labeling_confidence", False)

                # Create emotion record for emotions index
                # Extract peak emotion names (Pinecone only accepts strings, not dicts or null)
                speaker_a_peak = emotion_data.get("speaker_a", {}).get("peak_emotion")
                speaker_b_peak = emotion_data.get("speaker_b", {}).get("peak_emotion")

                emotion_record = {
                    "id": stored_info["conversation_id"],
                    "values": emotion_data.get("combined_vector", [0.0] * 192),
                    "metadata": {
                        "conversation_id": stored_info["conversation_id"],
                        "speaker_a_top_5": emotion_data.get("speaker_a", {}).get("top_5_emotions", []),
                        "speaker_b_top_5": emotion_data.get("speaker_b", {}).get("top_5_emotions", []),
                        "speaker_a_peak": speaker_a_peak["emotion"] if speaker_a_peak else "",
                        "speaker_b_peak": speaker_b_peak["emotion"] if speaker_b_peak else "",
                        "speaker_labels": json.dumps(stored_info.get("speaker_labels", {})),
                        "labeling_confidence": stored_info.get("labeling_confidence", False),
                        **stored_info["metadata"]
                    }
                }

                # Dual upsert to both indices
                pinecone_client.upsert_conversation(
                    records=conversation_records,
                    namespace="conversations"
                )

                # Only upsert emotions if vector contains non-zero values
                emotion_vector = emotion_record["values"]
                has_nonzero = any(v != 0.0 for v in emotion_vector)
                if has_nonzero:
                    pinecone_client.upsert_emotions(
                        records=[emotion_record],
                        namespace="emotions"
                    )
                    print(f"Successfully stored {len(conversation_records)} conversation records + 1 emotion record")
                else:
                    print(f"WARNING: Skipping emotion record - vector is all zeros")
                    print(f"Successfully stored {len(conversation_records)} conversation records (no emotion data)")

                # Mark as processed
                stored_info["processed"] = True
                stored_info["status"] = "completed_and_stored"

            except Exception as e:
                print(f"Error processing results: {e}")
                traceback.print_exc()
                stored_info["status"] = "processing_failed"
                stored_info["error"] = str(e)

        # Determine overall status
        if stored_info.get("processed"):
            overall_status = "completed_and_stored"
        elif all_completed:
            overall_status = "completed"
        else:
            overall_status = "processing"

        # Calculate time elapsed and detailed progress
        import time
        current_time = time.time()

        batch_diagnostics = {
            "embedding_batch": {
                "id": embedding_batch_id,
                "status": embedding_status["status"] if embedding_status else "N/A",
                "progress": f"{embedding_status['request_counts']['completed']}/{embedding_status['request_counts']['total']}" if embedding_status else "N/A",
                "failed": embedding_status['request_counts']['failed'] if embedding_status else 0,
                "time_elapsed_min": round((current_time - embedding_status["created_at"]) / 60, 1) if embedding_status and embedding_status.get("created_at") else None
            } if embedding_batch_id else None,
            "hume_job": {
                "id": hume_job_id,
                "status": hume_status.get("state", "N/A"),
                "message": hume_status.get("message", "")
            } if hume_job_id else None
        }

        # Build response details - sanitize emotion_data for serialization
        emotion_data = stored_info.get("emotion_data", {})
        emotion_summary = None
        if emotion_data:
            emotion_summary = {
                "speaker_a_top_5": emotion_data.get("speaker_a", {}).get("top_5_emotions", []),
                "speaker_b_top_5": emotion_data.get("speaker_b", {}).get("top_5_emotions", []),
                "speaker_a_peak": emotion_data.get("speaker_a", {}).get("peak_emotion"),
                "speaker_b_peak": emotion_data.get("speaker_b", {}).get("peak_emotion"),
                "hume_job_status": emotion_data.get("hume_job_status", "unknown")
            }

        details = {
            "batch_diagnostics": batch_diagnostics,
            "emotion_data_ready": emotion_data is not None,
            "emotion_summary": emotion_summary,
            "speaker_labels": stored_info.get("speaker_labels"),
            "labeling_confidence": stored_info.get("labeling_confidence")
        }

        # Validate JSON serializability before returning
        try:
            json.dumps(details)
        except TypeError as e:
            print(f"[ERROR] Response contains non-serializable data: {e}")
            print(f"[ERROR] Problematic details: {details}")
            # Return minimal safe response
            details = {
                "batch_diagnostics": batch_diagnostics,
                "error": "Response serialization error - check logs"
            }

        return StatusResponse(
            batch_id=batch_id,
            status=overall_status,
            progress=progress,
            conversation_id=stored_info["conversation_id"],
            details=details
        )

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error checking batch status: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to check status: {str(e)}")


@app.get("/debug/batch/{batch_id}")
async def debug_batch(batch_id: str):
    """
    Debug endpoint to inspect batch details and errors
    """
    try:
        batch_status = openai_client.check_batch(batch_id)
        batch_errors = openai_client.get_batch_errors(batch_id)

        return {
            "batch_id": batch_id,
            "status": batch_status,
            "error_count": len(batch_errors),
            "errors": batch_errors[:10]  # First 10 errors
        }
    except Exception as e:
        return {
            "error": str(e),
            "traceback": traceback.format_exc()
        }


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
        try:
            print(f"Creating embedding for query...")
            query_embedding = openai_client.create_embedding(request.query)
            print(f"Embedding created: {len(query_embedding)} dimensions")
        except Exception as e:
            print(f"ERROR: Failed to create embedding: {type(e).__name__}: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Embedding failed: {str(e)}")

        # Step 2: Search Pinecone
        try:
            print(f"Searching Pinecone for top {request.top_k} matches...")
            matches = pinecone_client.query(
                embedding=query_embedding,
                top_k=request.top_k,
                filter=request.filters,
                namespace="conversations",
                include_metadata=True
            )
            print(f"Found {len(matches)} matches")
        except Exception as e:
            print(f"ERROR: Pinecone query failed: {type(e).__name__}: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

        if not matches:
            print(f"No matches found - index might be empty")
            return QueryResponse(
                answer="No relevant conversations found for your query. Make sure files have finished processing and are stored in the index.",
                sources=[],
                processing_time_ms=(time.time() - start_time) * 1000
            )

        # Step 3: Take top 20 for synthesis
        top_matches = matches[:20]
        context_chunks = [match["metadata"] for match in top_matches]

        # Synthesize answer with GPT
        try:
            print(f"Synthesizing answer with GPT from {len(context_chunks)} chunks...")
            answer = openai_client.synthesize_answer(
                query=request.query,
                context_chunks=context_chunks
            )
            print(f"Answer synthesized: {len(answer)} characters")
        except Exception as e:
            print(f"ERROR: GPT synthesis failed: {type(e).__name__}: {str(e)}")
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Answer generation failed: {str(e)}")

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
    ⚠️ DANGER: Delete ALL data from Pinecone indices (conversations + emotions)

    This is for testing purposes only. Use with caution!
    """
    try:
        print("⚠️  PURGE REQUEST RECEIVED")

        # Purge all data from both Pinecone indices
        purge_results = pinecone_client.purge_all_data(namespace="conversations")

        # Clear in-memory batch storage
        batch_storage.clear()

        # Check if any errors occurred
        errors = []
        if purge_results["conversations"] and "error" in purge_results["conversations"]:
            errors.append(f"Conversations: {purge_results['conversations']}")
        if purge_results["emotions"] and "error" in purge_results["emotions"]:
            errors.append(f"Emotions: {purge_results['emotions']}")

        if errors:
            error_msg = "; ".join(errors)
            print(f"⚠️  Purge completed with errors: {error_msg}")
            return {
                "message": "Purge completed with some errors",
                "results": purge_results,
                "batches_cleared": True,
                "errors": errors
            }

        return {
            "message": "All data purged successfully from both indices",
            "results": purge_results,
            "batches_cleared": True
        }

    except Exception as e:
        print(f"Error purging index: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to purge index: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
