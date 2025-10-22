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

    def submit_batch(self, requests: List[Dict[str, Any]], endpoint: str = "/v1/embeddings") -> str:
        """
        Submit a batch job to OpenAI

        Args:
            requests: List of batch request objects in OpenAI format
            endpoint: The API endpoint for this batch (all requests must use same endpoint)

        Returns:
            batch_id: Unique identifier for the batch job
        """
        # Create a temporary JSONL file with the requests
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            for i, request in enumerate(requests):
                jsonl_line = json.dumps(request)
                f.write(jsonl_line + '\n')
                # Debug: Print first request for each batch
                if i == 0:
                    print(f"[DEBUG] First request in batch ({endpoint}):")
                    print(f"[DEBUG] {jsonl_line[:500]}...")  # Print first 500 chars
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
                endpoint=endpoint,
                completion_window="24h"
            )

            print(f"Batch submitted ({endpoint}): {batch.id}")
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
            Dictionary with status information (all values are JSON-serializable)
        """
        batch = self.client.batches.retrieve(batch_id)

        return {
            "id": batch.id,
            "status": batch.status,
            "created_at": int(batch.created_at) if batch.created_at else None,
            "completed_at": int(batch.completed_at) if batch.completed_at else None,
            "failed_at": int(batch.failed_at) if batch.failed_at else None,
            "expired_at": int(batch.expired_at) if batch.expired_at else None,
            "request_counts": {
                "total": int(batch.request_counts.total),
                "completed": int(batch.request_counts.completed),
                "failed": int(batch.request_counts.failed)
            },
            "output_file_id": batch.output_file_id,
            "error_file_id": batch.error_file_id
        }

    def get_batch_errors(self, batch_id: str) -> List[Dict[str, Any]]:
        """
        Download and parse batch errors if any exist

        Args:
            batch_id: The batch job identifier

        Returns:
            List of error objects or empty list if no errors
        """
        batch = self.client.batches.retrieve(batch_id)

        if not batch.error_file_id:
            return []

        try:
            # Download the error file
            error_response = self.client.files.content(batch.error_file_id)

            # Parse JSONL errors
            errors = []
            for line in error_response.text.strip().split('\n'):
                if line:
                    errors.append(json.loads(line))

            return errors
        except Exception as e:
            print(f"Error retrieving batch errors: {e}")
            return []

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
            Embedding vector (1024 dimensions)
        """
        response = self.client.embeddings.create(
            model=model,
            input=text,
            dimensions=1024
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

    def transcribe_audio_with_diarization(self, audio_file: BinaryIO, filename: str) -> Dict[str, Any]:
        """
        Transcribe audio with speaker diarization using Whisper

        Args:
            audio_file: Binary file object of the audio
            filename: Original filename

        Returns:
            {
                "full_transcript": "...",
                "segments": [
                    {"start": 0.0, "end": 3.5, "speaker": "SPEAKER_0", "text": "Hello..."},
                    {"start": 3.5, "end": 8.2, "speaker": "SPEAKER_1", "text": "Hi..."}
                ]
            }
        """
        print(f"Transcribing with diarization: {filename}")

        # Request verbose JSON format with timestamps
        response = self.client.audio.transcriptions.create(
            model="whisper-1",
            file=(filename, audio_file),
            response_format="verbose_json",
            timestamp_granularities=["segment"]
        )

        # Get full transcript
        full_transcript = response.text

        # Extract segments with timestamps
        segments = []
        if hasattr(response, 'segments') and response.segments:
            for seg in response.segments:
                segments.append({
                    "start": getattr(seg, 'start', 0.0),
                    "end": getattr(seg, 'end', 0.0),
                    "text": getattr(seg, 'text', "")
                })

        # Apply simple speaker splitting heuristic
        labeled_segments = self._split_speakers_heuristic(segments)

        print(f"Diarization complete: {len(labeled_segments)} segments")
        return {
            "full_transcript": full_transcript,
            "segments": labeled_segments
        }

    def _split_speakers_heuristic(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Simple speaker splitting based on pauses and turn-taking

        Args:
            segments: List of segments with start, end, text

        Returns:
            Segments with speaker labels added
        """
        if not segments:
            return []

        labeled_segments = []
        current_speaker = "SPEAKER_0"

        for i, segment in enumerate(segments):
            # Detect speaker change based on pause duration
            if i > 0:
                pause = segment["start"] - segments[i-1]["end"]
                # If pause > 1 second, assume speaker change
                if pause > 1.0:
                    current_speaker = "SPEAKER_1" if current_speaker == "SPEAKER_0" else "SPEAKER_0"

            labeled_segments.append({
                "start": segment["start"],
                "end": segment["end"],
                "text": segment["text"],
                "speaker": current_speaker
            })

        return labeled_segments

    def label_speakers_as_agent_caller(
        self,
        segments: List[Dict[str, Any]]
    ) -> tuple:
        """
        Attempt to identify AGENT vs CALLER using heuristics

        Args:
            segments: List of segments with speaker labels

        Returns:
            Tuple of (speaker_labels_dict, confidence_flag)
            - High confidence: ({"SPEAKER_0": "AGENT", "SPEAKER_1": "CALLER"}, True)
            - Low confidence: ({"SPEAKER_0": "SPEAKER_A", "SPEAKER_1": "SPEAKER_B"}, False)
        """
        if not segments:
            return {"SPEAKER_0": "SPEAKER_A", "SPEAKER_1": "SPEAKER_B"}, False

        # Get first speaker
        first_speaker = segments[0]["speaker"]

        # Aggregate text by speaker
        speaker_0_text = " ".join([s["text"] for s in segments if s["speaker"] == "SPEAKER_0"]).lower()
        speaker_1_text = " ".join([s["text"] for s in segments if s["speaker"] == "SPEAKER_1"]).lower()

        # Agent greeting phrases
        greeting_phrases = [
            "thank you for calling",
            "thanks for calling",
            "how may i help",
            "how can i help",
            "how may i assist",
            "customer service",
            "technical support",
            "welcome to"
        ]

        # Check if either speaker uses agent greetings
        speaker_0_is_agent = any(phrase in speaker_0_text for phrase in greeting_phrases)
        speaker_1_is_agent = any(phrase in speaker_1_text for phrase in greeting_phrases)

        # High confidence scenarios
        if first_speaker == "SPEAKER_0" and speaker_0_is_agent:
            print("High confidence: SPEAKER_0 is AGENT (speaks first + uses greetings)")
            return {"SPEAKER_0": "AGENT", "SPEAKER_1": "CALLER"}, True

        elif first_speaker == "SPEAKER_1" and speaker_1_is_agent:
            print("High confidence: SPEAKER_1 is AGENT (speaks first + uses greetings)")
            return {"SPEAKER_1": "AGENT", "SPEAKER_0": "CALLER"}, True

        elif speaker_0_is_agent and not speaker_1_is_agent:
            print("Medium confidence: SPEAKER_0 is AGENT (uses greetings)")
            return {"SPEAKER_0": "AGENT", "SPEAKER_1": "CALLER"}, True

        elif speaker_1_is_agent and not speaker_0_is_agent:
            print("Medium confidence: SPEAKER_1 is AGENT (uses greetings)")
            return {"SPEAKER_1": "AGENT", "SPEAKER_0": "CALLER"}, True

        else:
            # Low confidence - use generic labels
            print("Low confidence: Unable to determine agent/caller, using SPEAKER_A/SPEAKER_B")
            return {"SPEAKER_0": "SPEAKER_A", "SPEAKER_1": "SPEAKER_B"}, False
