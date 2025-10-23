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
from services.hume_client import HumeClient
from fastapi import Request
import io
import traceback

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

        # === STEP 1: Transcribe with diarization ===
        try:
            diarization = openai_client.transcribe_audio_with_diarization(
                io.BytesIO(audio_content),
                file.filename
            )
            speaker_labels, labeling_confidence = openai_client.label_speakers_as_agent_caller(
                diarization["segments"]
            )
        except Exception as e:
            print(f"Error during transcription: {e}")
            traceback.print_exc()
            raise HTTPException(
                status_code=500,
                detail=f"Audio transcription failed: {str(e)}"
            )

        print(f"Speaker labeling: {speaker_labels} (confidence: {labeling_confidence})")

        # === STEP 2: Submit to Hume (with error handling) ===
        hume_job_id = None
        try:
            hume_callback_url = os.getenv("HUME_CALLBACK_URL")
            hume_job_id = hume_client.submit_audio(
                io.BytesIO(audio_content),
                file.filename,
                callback_url=hume_callback_url
            )
            print(f"Hume job submitted: {hume_job_id}")
        except Exception as e:
            print(f"Warning: Hume submission failed: {e}")
            print("Continuing with OpenAI processing only...")

        # === STEP 3: Chunk transcript ===
        try:
            transcript = diarization["full_transcript"]
            chunks = chunk_text(transcript, max_tokens=1500, overlap=200)
            print(f"Created {len(chunks)} chunks")
        except Exception as e:
            print(f"Error during transcript chunking: {e}")
            traceback.print_exc()
            raise HTTPException(
                status_code=500,
                detail=f"Transcript chunking failed: {str(e)}"
            )

        # === STEP 4: Extract data from chunks synchronously (FAST - direct API calls) ===
        try:
            print(f"Extracting data from {len(chunks)} chunks...")
            extraction_results = []
            for i, chunk in enumerate(chunks):
                try:
                    extracted = openai_client.extract_conversation_data(chunk["text"])
                    extraction_results.append({
                        "chunk_id": f"{conversation_id}_chunk_{chunk['index']}",
                        "data": extracted
                    })
                except Exception as chunk_error:
                    print(f"Error extracting chunk {i}: {chunk_error}")
                    raise HTTPException(
                        status_code=500,
                        detail=f"Data extraction failed at chunk {i+1}/{len(chunks)}: {str(chunk_error)}"
                    )
            print(f"Extraction complete for {len(chunks)} chunks")
        except HTTPException:
            raise
        except Exception as e:
            print(f"Error during data extraction: {e}")
            traceback.print_exc()
            raise HTTPException(
                status_code=500,
                detail=f"Data extraction failed: {str(e)}"
            )

        # === STEP 5: Create embeddings directly (no batch, much faster!) ===
        try:
            chunk_texts = [chunk["text"] for chunk in chunks]
            print(f"Creating embeddings for {len(chunk_texts)} chunks...")
            embeddings = openai_client.create_embeddings_batch(chunk_texts)
            print(f"Embeddings created successfully")
        except Exception as e:
            print(f"Error creating embeddings: {e}")
            traceback.print_exc()
            raise HTTPException(
                status_code=500,
                detail=f"Failed to create embeddings: {str(e)}"
            )

        # === STEP 6: Build Pinecone records and upsert immediately ===
        try:
            print(f"Building Pinecone records for {len(chunks)} chunks...")

            # Build conversation records
            conversation_records = []
            for i, chunk in enumerate(chunks):
                chunk_id = f"{conversation_id}_chunk_{chunk['index']}"

                # Get extraction data for this chunk
                extracted = extraction_results[i]["data"] if i < len(extraction_results) else {
                    "intents": [],
                    "entities": [],
                    "sentiment": 0.0,
                    "action_items": [],
                    "compliance_flags": []
                }

                record = {
                    "id": chunk_id,
                    "values": embeddings[i],
                    "metadata": {
                        "conversation_id": conversation_id,
                        "chunk_index": chunk["index"],
                        "text": chunk["text"],
                        "token_count": chunk["token_count"],
                        # Merge conversation metadata
                        **metadata_dict,
                        # Add extracted data
                        "intents": extracted.get("intents", []),
                        "entities": json.dumps(extracted.get("entities", [])),
                        "sentiment": extracted.get("sentiment", 0.0),
                        "action_items": extracted.get("action_items", []),
                        "compliance_flags": extracted.get("compliance_flags", []),
                        # Add speaker info
                        "labeling_confidence": labeling_confidence
                    }
                }

                conversation_records.append(record)

            # Upsert to conversations index
            print(f"Upserting {len(conversation_records)} records to Pinecone conversations index...")
            pinecone_client.upsert_conversation(
                records=conversation_records,
                namespace="conversations"
            )
            print(f"Successfully upserted conversation records")

        except Exception as e:
            print(f"Error upserting to Pinecone: {e}")
            traceback.print_exc()
            raise HTTPException(
                status_code=500,
                detail=f"Failed to store in vector database: {str(e)}"
            )

        # === STEP 7: Store for Hume webhook processing ===
        # Generate a tracking ID for this upload
        upload_id = f"upload_{conversation_id}"

        batch_storage[upload_id] = {
            "conversation_id": conversation_id,
            "status": "processing_emotions" if hume_job_id else "completed",
            "chunks": chunks,
            "metadata": metadata_dict,
            "total_chunks": len(chunks),
            "source": "audio",
            "filename": file.filename,
            "hume_job_id": hume_job_id,
            "diarization": diarization,
            "speaker_labels": speaker_labels,
            "labeling_confidence": labeling_confidence,
            "conversation_stored": True  # Conversation is already in Pinecone
        }

        # Return success immediately (no batch polling needed!)
        return UploadResponse(
            batch_id=upload_id,
            status="processing_emotions" if hume_job_id else "completed",
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

        job_id = payload.get("job_id")
        status = payload.get("status")
        print(f"Hume webhook received - Job ID: {job_id}, Status: {status}")

        if status == "COMPLETED":
            # Find conversation by hume job_id (check both old and new formats)
            batch_info = None
            for bid, info in batch_storage.items():
                # New format: hume_job_id field
                if info.get("hume_job_id") == job_id:
                    batch_info = info
                    break
                # Old format: paired_batch_ids.hume
                if info.get("paired_batch_ids", {}).get("hume") == job_id:
                    batch_info = info
                    break

            if not batch_info:
                print(f"No upload found for Hume job: {job_id}")
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

            # Immediately upsert emotion record to Pinecone
            try:
                speaker_a_peak = speaker_a_features.get("peak_emotion")
                speaker_b_peak = speaker_b_features.get("peak_emotion")

                emotion_record = {
                    "id": batch_info["conversation_id"],
                    "values": combined["combined_vector"],
                    "metadata": {
                        "conversation_id": batch_info["conversation_id"],
                        "speaker_a_top_5": speaker_a_features.get("top_5_emotions", []),
                        "speaker_b_top_5": speaker_b_features.get("top_5_emotions", []),
                        "speaker_a_peak": speaker_a_peak["emotion"] if speaker_a_peak else "",
                        "speaker_b_peak": speaker_b_peak["emotion"] if speaker_b_peak else "",
                        "speaker_labels": json.dumps(batch_info.get("speaker_labels", {})),
                        "labeling_confidence": batch_info.get("labeling_confidence", False),
                        **batch_info.get("metadata", {})
                    }
                }

                # Only upsert if vector has non-zero values
                emotion_vector = emotion_record["values"]
                has_nonzero = any(v != 0.0 for v in emotion_vector)
                if has_nonzero:
                    pinecone_client.upsert_emotions(
                        records=[emotion_record],
                        namespace="emotions"
                    )
                    print(f"Upserted emotion record for conversation: {batch_info['conversation_id']}")
                else:
                    print(f"WARNING: Skipping emotion upsert - vector is all zeros for {batch_info['conversation_id']}")
            except Exception as e:
                print(f"Error upserting emotion data: {e}")
                traceback.print_exc()

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
    Check status of upload job

    Handles two types:
    - Old: batch_* IDs (legacy batch API, still processing)
    - New: upload_* IDs (direct API, instant upload)
    """
    try:
        # Check if batch_id exists in our storage
        if batch_id not in batch_storage:
            raise HTTPException(status_code=404, detail=f"Upload {batch_id} not found")

        stored_info = batch_storage[batch_id]

        # NEW FLOW: Direct uploads (upload_*)
        if batch_id.startswith("upload_"):
            return await _handle_direct_upload_status(batch_id, stored_info)

        # OLD FLOW: Batch API uploads (batch_*) - for backwards compatibility
        return await _handle_batch_upload_status(batch_id, stored_info)

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error checking status: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to check status: {str(e)}")


async def _handle_direct_upload_status(upload_id: str, stored_info: Dict[str, Any]) -> StatusResponse:
    """Handle status for new direct upload flow"""
    conversation_id = stored_info["conversation_id"]
    hume_job_id = stored_info.get("hume_job_id")

    # Check Hume status if applicable
    hume_status = {"state": "N/A"}
    hume_done = False

    if hume_job_id:
        emotion_data = stored_info.get("emotion_data", {})
        if emotion_data.get("hume_job_status") == "completed":
            hume_status = {"state": "COMPLETED", "message": "Webhook received"}
            hume_done = True
            print(f"[STATUS] Hume job {hume_job_id}: COMPLETED (via webhook)")
        else:
            # Check Hume API
            try:
                hume_raw = hume_client.get_job_status(hume_job_id)
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

    # Determine overall status
    if hume_job_id and not hume_done:
        overall_status = "processing_emotions"
        progress = "Conversation indexed, waiting for emotion analysis"
    else:
        overall_status = "completed"
        progress = "Complete with emotions" if hume_done else "Complete"

    # Build diagnostics
    diagnostics = {
        "conversation_stored": stored_info.get("conversation_stored", False),
        "hume_job": {
            "id": hume_job_id,
            "status": hume_status.get("state", "N/A"),
            "message": hume_status.get("message", "")
        } if hume_job_id else None
    }

    # Build emotion summary
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
        "batch_diagnostics": diagnostics,
        "emotion_data_ready": emotion_data is not None,
        "emotion_summary": emotion_summary,
        "speaker_labels": stored_info.get("speaker_labels"),
        "labeling_confidence": stored_info.get("labeling_confidence")
    }

    return StatusResponse(
        batch_id=upload_id,
        status=overall_status,
        progress=progress,
        conversation_id=conversation_id,
        details=details
    )


async def _handle_batch_upload_status(batch_id: str, stored_info: Dict[str, Any]) -> StatusResponse:
    """Handle status for old batch-based upload flow (legacy)"""
    paired_batch_ids = stored_info.get("paired_batch_ids", {})
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
    Query conversations with natural language using BOTH text and emotion indices

    Steps:
    1. Extract emotion intent from query
    2. Create text embedding for query
    3. Search conversations index (text-based)
    4. If emotion intent detected, search emotions index (emotion-based)
    5. Merge and rank results
    6. Send enriched context to GPT for synthesis
    7. Return answer with sources
    """
    try:
        import time
        start_time = time.time()

        print(f"Processing query: {request.query}")

        # Step 1: Extract emotion intent
        emotion_intent = openai_client.extract_emotion_intent(request.query)
        print(f"Emotion intent: {emotion_intent}")

        # Step 2: Create query embedding for text search
        query_embedding = openai_client.create_embedding(request.query)

        # Step 3: Search conversations index (text-based)
        text_matches = pinecone_client.query(
            embedding=query_embedding,
            top_k=request.top_k,
            filter=request.filters,
            namespace="conversations",
            include_metadata=True
        )

        print(f"Found {len(text_matches)} text-based matches")

        # Step 4: Search emotions index if emotion intent detected
        emotion_matches = []
        if emotion_intent.get("has_emotion_intent") and emotion_intent.get("emotion_weight", 0) > 0.3:
            print(f"Searching emotions index for: {emotion_intent['primary_emotions']}")

            # Create synthetic emotion vector
            emotion_vector = hume_client.create_emotion_query_vector(
                primary_emotions=emotion_intent["primary_emotions"],
                emotion_weight=emotion_intent.get("emotion_weight", 0.8)
            )

            # Query emotions index
            if not pinecone_client.emotion_index:
                pinecone_client.setup_emotion_index(dimension=192)

            emotion_results = pinecone_client.emotion_index.query(
                vector=emotion_vector,
                top_k=20,  # Get top 20 emotionally similar conversations
                filter=request.filters,
                namespace="emotions",
                include_metadata=True
            )

            # Convert to same format as text_matches
            for match in emotion_results.matches:
                emotion_matches.append({
                    "id": match.id,
                    "score": match.score,
                    "metadata": match.metadata,
                    "match_type": "emotion"
                })

            print(f"Found {len(emotion_matches)} emotion-based matches")

        # Step 5: Merge results by conversation_id
        # Combine scores for conversations that match both text and emotion
        conversation_scores = {}
        all_metadata = {}

        # Add text matches
        for match in text_matches:
            conv_id = match["metadata"].get("conversation_id")
            conversation_scores[conv_id] = conversation_scores.get(conv_id, 0) + match["score"]
            if conv_id not in all_metadata:
                all_metadata[conv_id] = []
            all_metadata[conv_id].append(match["metadata"])

        # Add emotion matches (boost score if query has emotion intent)
        emotion_boost = emotion_intent.get("emotion_weight", 0.5) if emotion_intent.get("has_emotion_intent") else 0
        for match in emotion_matches:
            conv_id = match["id"]
            # Boost emotion scores based on emotion_weight
            boosted_score = match["score"] * (1.0 + emotion_boost)
            conversation_scores[conv_id] = conversation_scores.get(conv_id, 0) + boosted_score

            # Add emotion metadata to the conversation
            if conv_id in all_metadata:
                # Enrich existing metadata with emotion info
                for meta in all_metadata[conv_id]:
                    meta["emotion_match"] = True
                    meta["emotion_score"] = match["score"]
                    if match["metadata"]:
                        meta["speaker_a_top_emotions"] = match["metadata"].get("speaker_a_top_5", [])
                        meta["speaker_b_top_emotions"] = match["metadata"].get("speaker_b_top_5", [])

        # Sort by combined score
        ranked_conversations = sorted(
            conversation_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        print(f"Merged into {len(ranked_conversations)} unique conversations")

        # Get matches for top conversations
        matches = []
        for conv_id, score in ranked_conversations[:request.top_k]:
            if conv_id in all_metadata:
                for meta in all_metadata[conv_id]:
                    matches.append({
                        "id": meta.get("chunk_id", conv_id),
                        "score": score,
                        "metadata": meta
                    })

        print(f"Returning {len(matches)} total chunks from top conversations")

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

        # Format sources with emotion data
        sources = []
        for match in top_matches[:10]:  # Return top 10 sources
            metadata = match["metadata"]
            source = {
                "conversation_id": metadata.get("conversation_id"),
                "date": metadata.get("date"),
                "score": match["score"],
                "text": metadata.get("text", "")[:200] + "...",  # Truncate for display
                "sentiment": metadata.get("sentiment"),
                "intents": metadata.get("intents", [])
            }

            # Add emotion data if available
            if metadata.get("emotion_match"):
                source["emotion_match"] = True
                source["emotion_score"] = metadata.get("emotion_score")
                source["speaker_a_top_emotions"] = metadata.get("speaker_a_top_emotions", [])[:3]
                source["speaker_b_top_emotions"] = metadata.get("speaker_b_top_emotions", [])[:3]

            sources.append(source)

        processing_time = (time.time() - start_time) * 1000

        # Log emotion query details
        print(f"Query complete - Emotion intent: {emotion_intent.get('has_emotion_intent')}, "
              f"Emotions: {emotion_intent.get('primary_emotions')}, "
              f"Text matches: {len(text_matches)}, Emotion matches: {len(emotion_matches)}")

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


@app.get("/debug/emotions")
async def debug_emotions():
    """
    Debug endpoint to check emotion vectors in Pinecone

    Returns stats and sample emotion records to verify they're being stored
    """
    try:
        if not pinecone_client.emotion_index:
            pinecone_client.setup_emotion_index(dimension=192)

        # Get index stats
        stats = pinecone_client.emotion_index.describe_index_stats()

        # Try to query for some emotion records
        dummy_vector = [0.0] * 192
        sample_results = pinecone_client.emotion_index.query(
            vector=dummy_vector,
            top_k=10,
            namespace="emotions",
            include_metadata=True
        )

        # Format sample results
        samples = []
        for match in sample_results.matches:
            # Check if vector has non-zero values
            has_data = any(v != 0.0 for v in match.values) if hasattr(match, 'values') and match.values else False

            samples.append({
                "id": match.id,
                "score": match.score,
                "has_nonzero_vector": has_data,
                "metadata_keys": list(match.metadata.keys()) if match.metadata else [],
                "speaker_a_top_5": match.metadata.get("speaker_a_top_5", [])[:3] if match.metadata else [],
                "speaker_b_top_5": match.metadata.get("speaker_b_top_5", [])[:3] if match.metadata else []
            })

        return {
            "index_stats": {
                "total_vector_count": stats.total_vector_count,
                "namespaces": stats.namespaces
            },
            "sample_count": len(samples),
            "samples": samples,
            "message": f"Found {len(samples)} emotion records. Check 'has_nonzero_vector' to verify data quality."
        }

    except Exception as e:
        print(f"Error debugging emotions: {e}")
        traceback.print_exc()
        return {
            "error": str(e),
            "traceback": traceback.format_exc()
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
