#!/usr/bin/env python3
"""
Test script to submit a minimal extraction batch
This helps diagnose if the issue is with our batch format
"""
import os
import json
import tempfile
from openai import OpenAI

client = OpenAI()

# Create a minimal extraction request (same format as our production code)
extraction_request = {
    "custom_id": "test_extract_001",
    "method": "POST",
    "url": "/v1/chat/completions",
    "body": {
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "system",
                "content": "You are a conversation analysis assistant. Extract structured data from conversations and return only valid JSON. Your response must be valid JSON only, no other text."
            },
            {
                "role": "user",
                "content": """Extract from this conversation chunk:
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
Customer: Hi, I'd like to cancel my subscription.
Agent: I can help with that. Can I ask why you're canceling?
Customer: It's too expensive for what I'm getting.
Agent: I understand. Let me check if we have any discount options available for you."""
            }
        ],
        "temperature": 0.3
    }
}

print("Creating test batch with single extraction request...")
print(f"Request format:\n{json.dumps(extraction_request, indent=2)}\n")

# Create JSONL file
with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
    f.write(json.dumps(extraction_request) + '\n')
    temp_file = f.name

try:
    # Upload file
    print("Uploading batch file...")
    with open(temp_file, 'rb') as f:
        batch_file = client.files.create(file=f, purpose="batch")

    print(f"File uploaded: {batch_file.id}")

    # Create batch
    print("Creating batch job...")
    batch = client.batches.create(
        input_file_id=batch_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h"
    )

    print(f"\n{'='*70}")
    print(f"Batch created successfully!")
    print(f"{'='*70}")
    print(f"Batch ID: {batch.id}")
    print(f"Status: {batch.status}")
    print(f"Endpoint: {batch.endpoint}")
    print(f"Created at: {batch.created_at}")
    print(f"\nMonitor at: https://platform.openai.com/batches/{batch.id}")
    print(f"\nCheck status with:")
    print(f"  curl https://api.openai.com/v1/batches/{batch.id} \\")
    print(f"    -H \"Authorization: Bearer $OPENAI_API_KEY\"")

finally:
    os.unlink(temp_file)
