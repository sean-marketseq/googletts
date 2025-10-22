#!/usr/bin/env python3
"""
Monitor OpenAI batch progress in real-time
Usage: python monitor_batches.py batch_id1 batch_id2 ...
"""
import sys
import time
from openai import OpenAI
from datetime import datetime

if len(sys.argv) < 2:
    print("Usage: python monitor_batches.py <batch_id1> [batch_id2] ...")
    sys.exit(1)

client = OpenAI()
batch_ids = sys.argv[1:]

print(f"Monitoring {len(batch_ids)} batch(es)...")
print("Press Ctrl+C to stop\n")

try:
    while True:
        for batch_id in batch_ids:
            try:
                batch = client.batches.retrieve(batch_id)

                # Calculate progress
                total = batch.request_counts.total
                completed = batch.request_counts.completed
                failed = batch.request_counts.failed
                progress_pct = (completed / total * 100) if total > 0 else 0

                # Time info
                now = datetime.now()
                created = datetime.fromtimestamp(batch.created_at)
                elapsed = (now - created).total_seconds() / 60  # minutes

                print(f"[{now.strftime('%H:%M:%S')}] {batch_id[-8:]}")
                print(f"  Status: {batch.status}")
                print(f"  Progress: {completed}/{total} ({progress_pct:.1f}%)")
                print(f"  Failed: {failed}")
                print(f"  Elapsed: {elapsed:.1f} min")

                if batch.status == "completed":
                    print(f"  ✅ COMPLETED!")
                elif batch.status == "failed":
                    print(f"  ❌ FAILED: {batch.errors}")
                elif batch.status == "validating":
                    print(f"  ⏳ Still validating...")
                elif batch.status == "in_progress":
                    if completed > 0:
                        # Estimate time remaining
                        time_per_request = elapsed / completed
                        remaining = total - completed
                        eta_min = time_per_request * remaining
                        print(f"  ⚙️ Processing... ETA: {eta_min:.1f} min")
                    else:
                        print(f"  ⚙️ Processing started...")

                print()

            except Exception as e:
                print(f"Error checking {batch_id}: {e}\n")

        print("-" * 70)
        time.sleep(30)  # Check every 30 seconds

except KeyboardInterrupt:
    print("\nStopped monitoring.")
