"""
Text processing and batch request creation for OpenAI Batch API
"""
from typing import List, Dict, Any
import tiktoken
import json


def chunk_text(text: str, max_tokens: int = 1500, overlap: int = 200) -> List[Dict[str, Any]]:
    """
    Chunk text into smaller pieces with token-based splitting

    Args:
        text: The conversation transcript to chunk
        max_tokens: Maximum tokens per chunk
        overlap: Number of overlapping tokens between chunks

    Returns:
        List of chunks with metadata
    """
    # Initialize tokenizer (cl100k_base is used by gpt-4 and text-embedding-3-small)
    encoding = tiktoken.get_encoding("cl100k_base")

    # Tokenize the entire text
    tokens = encoding.encode(text)
    chunks = []

    start = 0
    chunk_index = 0

    while start < len(tokens):
        # Calculate end position for this chunk
        end = min(start + max_tokens, len(tokens))

        # Get the chunk tokens
        chunk_tokens = tokens[start:end]

        # Decode back to text
        chunk_text = encoding.decode(chunk_tokens)

        chunks.append({
            "index": chunk_index,
            "text": chunk_text,
            "token_count": len(chunk_tokens),
            "start_token": start,
            "end_token": end
        })

        chunk_index += 1

        # Move start position forward, accounting for overlap
        start = end - overlap

        # If we're at the end, break to avoid infinite loop
        if end >= len(tokens):
            break

    return chunks


def create_batch_requests(conversation_id: str, chunks: List[Dict[str, Any]], metadata: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Create OpenAI Batch API request format for embeddings and extraction

    Returns separate lists since OpenAI Batch API requires same endpoint per batch

    Args:
        conversation_id: Unique conversation identifier
        chunks: List of text chunks from chunk_text()
        metadata: Conversation metadata (account_id, date, agent_id, etc.)

    Returns:
        Dict with 'embeddings' and 'extractions' lists of batch requests
    """
    embedding_requests = []
    extraction_requests = []

    # Extraction prompt template
    extraction_prompt = """Extract from this conversation chunk:
- intents: list of customer intents (e.g., ["cancel_subscription", "request_discount"])
- entities: list of {type, value} objects (e.g., [{"type": "product", "value": "subscription"}])
- sentiment: numeric score from -1 (negative) to 1 (positive)
- action_items: list of action items mentioned (e.g., ["check for discounts", "process cancellation"])
- compliance_flags: list of any compliance issues detected (e.g., ["missing data privacy notice"])

Return only valid JSON matching this schema:
{
  "intents": ["string"],
  "entities": [{"type": "string", "value": "string"}],
  "sentiment": 0.0,
  "action_items": ["string"],
  "compliance_flags": ["string"]
}

Conversation chunk:
"""

    # Create ONE embedding request with ALL chunks as an array (much more efficient!)
    # The embeddings endpoint can handle up to 2048 inputs in a single request
    chunk_texts = [chunk["text"] for chunk in chunks]

    embedding_requests.append({
        "custom_id": f"embed_all_{conversation_id}",
        "method": "POST",
        "url": "/v1/embeddings",
        "body": {
            "model": "text-embedding-3-small",
            "input": chunk_texts,  # Array of all chunk texts
            "dimensions": 1024
        }
    })

    # Note: Extractions are now done synchronously during upload (not in batch)
    # This loop is kept for backwards compatibility but extractions list will be empty

    return {
        "embeddings": embedding_requests,
        "extractions": extraction_requests
    }


def parse_batch_results(results: List[Dict[str, Any]], conversation_id: str, chunks: List[Dict[str, Any]], metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Parse OpenAI Batch API results and combine embeddings + extractions

    Args:
        results: Raw results from OpenAI Batch API
        conversation_id: Conversation identifier
        chunks: Original chunks
        metadata: Conversation metadata

    Returns:
        List of records ready for Pinecone upsert
    """
    # Organize results by custom_id
    embeddings_map = {}
    extractions_map = {}

    for result in results:
        custom_id = result.get("custom_id", "")

        if custom_id.startswith("embed_"):
            chunk_id = custom_id.replace("embed_", "")
            response_body = result.get("response", {}).get("body", {})
            embedding = response_body.get("data", [{}])[0].get("embedding", [])
            embeddings_map[chunk_id] = embedding

        elif custom_id.startswith("extract_"):
            chunk_id = custom_id.replace("extract_", "")
            response_body = result.get("response", {}).get("body", {})
            content = response_body.get("choices", [{}])[0].get("message", {}).get("content", "{}")
            try:
                extracted_data = json.loads(content)
            except json.JSONDecodeError:
                extracted_data = {
                    "intents": [],
                    "entities": [],
                    "sentiment": 0.0,
                    "action_items": [],
                    "compliance_flags": []
                }
            extractions_map[chunk_id] = extracted_data

    # Combine into Pinecone records
    pinecone_records = []

    for chunk in chunks:
        chunk_id = f"{conversation_id}_chunk_{chunk['index']}"

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
                "conversation_id": conversation_id,
                "chunk_index": chunk["index"],
                "text": chunk["text"],
                "token_count": chunk["token_count"],
                # Merge conversation metadata
                **metadata,
                # Add extracted data
                "intents": extracted.get("intents", []),
                "entities": json.dumps(extracted.get("entities", [])),  # Store as JSON string
                "sentiment": extracted.get("sentiment", 0.0),
                "action_items": extracted.get("action_items", []),
                "compliance_flags": extracted.get("compliance_flags", [])
            }
        }

        pinecone_records.append(record)

    return pinecone_records
