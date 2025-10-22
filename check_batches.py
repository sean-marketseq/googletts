#!/usr/bin/env python3
"""
Quick script to check OpenAI batch status
Usage: python check_batches.py
Set OPENAI_API_KEY environment variable first
"""
import os
from openai import OpenAI

client = OpenAI()

batch_ids = [
    "batch_68f8dfd41ed8819097eee460d5abeada",
    "batch_68f8dfcb363081908cb53d1b0a10f947"
]

for batch_id in batch_ids:
    print(f"\n{'='*70}")
    print(f"Batch: {batch_id}")
    print('='*70)

    batch = client.batches.retrieve(batch_id)

    print(f"Status: {batch.status}")
    print(f"Created: {batch.created_at}")
    print(f"Endpoint: {batch.endpoint}")
    print(f"Request counts:")
    print(f"  Total: {batch.request_counts.total}")
    print(f"  Completed: {batch.request_counts.completed}")
    print(f"  Failed: {batch.request_counts.failed}")

    if batch.status == "failed":
        print(f"\nFailed at: {batch.failed_at}")
        print(f"Errors: {batch.errors}")

    if batch.error_file_id:
        print(f"\nHas error file: {batch.error_file_id}")
        print("Downloading errors...")
        error_content = client.files.content(batch.error_file_id)
        print(error_content.text[:1000])  # First 1000 chars

    if batch.status == "in_progress":
        progress = (batch.request_counts.completed / batch.request_counts.total * 100)
        print(f"\nProgress: {progress:.1f}%")
        print(f"Time elapsed: Check created_at timestamp above")
