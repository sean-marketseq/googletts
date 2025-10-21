"""
OpenAI Batch API wrapper for conversation processing
"""
import os
from typing import List, Dict, Any, BinaryIO
import json
import tempfile
from openai import OpenAI


class OpenAIBatchClient:
    """Wrapper for OpenAI Batch API operations"""

    def __init__(self, api_key: str = None):
        """
        Initialize OpenAI client

        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
        """
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

    def submit_batch(self, requests: List[Dict[str, Any]]) -> str:
        """
        Submit a batch job to OpenAI

        Args:
            requests: List of batch request objects in OpenAI format

        Returns:
            batch_id: Unique identifier for the batch job
        """
        # Create a temporary JSONL file with the requests
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            for request in requests:
                f.write(json.dumps(request) + '\n')
            temp_file_path = f.name

        try:
            # Upload the file to OpenAI
            with open(temp_file_path, 'rb') as file_obj:
                batch_input_file = self.client.files.create(
                    file=file_obj,
                    purpose="batch"
                )

            # Create the batch job
            batch = self.client.batches.create(
                input_file_id=batch_input_file.id,
                endpoint="/v1/embeddings",  # This is set generically; the actual endpoint is in each request
                completion_window="24h"
            )

            print(f"Batch submitted: {batch.id}")
            return batch.id

        finally:
            # Clean up temp file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)

    def check_batch(self, batch_id: str) -> Dict[str, Any]:
        """
        Check the status of a batch job

        Args:
            batch_id: The batch job identifier

        Returns:
            Dictionary with status information
        """
        batch = self.client.batches.retrieve(batch_id)

        return {
            "id": batch.id,
            "status": batch.status,
            "created_at": batch.created_at,
            "completed_at": batch.completed_at,
            "failed_at": batch.failed_at,
            "expired_at": batch.expired_at,
            "request_counts": {
                "total": batch.request_counts.total,
                "completed": batch.request_counts.completed,
                "failed": batch.request_counts.failed
            },
            "output_file_id": batch.output_file_id,
            "error_file_id": batch.error_file_id
        }

    def get_results(self, batch_id: str) -> List[Dict[str, Any]]:
        """
        Download and parse batch results

        Args:
            batch_id: The batch job identifier

        Returns:
            List of result objects
        """
        batch = self.client.batches.retrieve(batch_id)

        if batch.status != "completed":
            raise ValueError(f"Batch {batch_id} is not completed (status: {batch.status})")

        if not batch.output_file_id:
            raise ValueError(f"Batch {batch_id} has no output file")

        # Download the output file
        file_response = self.client.files.content(batch.output_file_id)

        # Parse JSONL results
        results = []
        for line in file_response.text.strip().split('\n'):
            if line:
                results.append(json.loads(line))

        return results

    def create_embedding(self, text: str, model: str = "text-embedding-3-small") -> List[float]:
        """
        Create a single embedding (for query processing)

        Args:
            text: Text to embed
            model: Embedding model to use

        Returns:
            Embedding vector
        """
        response = self.client.embeddings.create(
            model=model,
            input=text
        )
        return response.data[0].embedding

    def synthesize_answer(self, query: str, context_chunks: List[Dict[str, Any]], model: str = "gpt-4o-mini") -> str:
        """
        Use GPT to synthesize an answer from retrieved chunks

        Args:
            query: User's question
            context_chunks: List of relevant conversation chunks
            model: Model to use for synthesis

        Returns:
            Synthesized answer
        """
        # Format context
        context_text = "\n\n---\n\n".join([
            f"Conversation {chunk.get('conversation_id', 'unknown')} (Date: {chunk.get('date', 'unknown')}):\n{chunk.get('text', '')}"
            for chunk in context_chunks
        ])

        system_prompt = """You are a conversation insights assistant. Your job is to answer questions about customer conversations based on the provided context.

Guidelines:
- Only use information from the provided conversation chunks
- Be specific and cite which conversation(s) you're referencing
- If you can't answer based on the context, say so
- Highlight patterns, trends, or important insights
- Keep your answer concise but informative"""

        user_prompt = f"""Based on the following conversation excerpts, please answer this question:

Question: {query}

Context:
{context_text}

Please provide a clear, well-structured answer."""

        response = self.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.5,
            max_tokens=1000
        )

        return response.choices[0].message.content

    def transcribe_audio(self, audio_file: BinaryIO, filename: str) -> str:
        """
        Transcribe audio file using OpenAI Whisper API

        Args:
            audio_file: Binary file object of the audio
            filename: Original filename (used for format detection)

        Returns:
            Transcribed text
        """
        print(f"Transcribing audio file: {filename}")

        response = self.client.audio.transcriptions.create(
            model="whisper-1",
            file=(filename, audio_file),
            response_format="text"
        )

        print(f"Transcription complete: {len(response)} characters")
        return response
