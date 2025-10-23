"""
Hume.ai Batch API wrapper for emotion prosody analysis
"""
import os
import json
import requests
from typing import BinaryIO, Dict, Any, List
import io


class HumeClient:
    """Wrapper for Hume.ai Batch API operations"""

    def __init__(self, api_key: str = None, secret_key: str = None):
        """
        Initialize Hume client

        Args:
            api_key: Hume API key (defaults to HUME_API_KEY env var)
            secret_key: Hume secret key (defaults to HUME_SECRET_KEY env var)
        """
        self.api_key = api_key or os.getenv("HUME_API_KEY")
        self.secret_key = secret_key or os.getenv("HUME_SECRET_KEY")
        self.base_url = "https://api.hume.ai/v0/batch"

    def submit_audio(self, audio_file: BinaryIO, filename: str, callback_url: str) -> str:
        """
        Submit audio to Hume Batch API for prosody analysis

        Args:
            audio_file: Binary file object of the audio
            filename: Original filename
            callback_url: Webhook URL for completion callback

        Returns:
            job_id: Unique identifier for the batch job
        """
        print(f"Submitting to Hume: {filename}")

        # Prepare multipart form data
        files = {
            'file': (filename, audio_file, 'audio/mpeg')
        }

        # JSON configuration for prosody model
        json_config = {
            "models": {
                "prosody": {}
            },
            "callback_url": callback_url
        }

        data = {
            'json': json.dumps(json_config)
        }

        try:
            response = requests.post(
                f"{self.base_url}/jobs",
                headers={
                    "X-Hume-Api-Key": self.api_key
                },
                files=files,
                data=data,
                timeout=60
            )

            response.raise_for_status()
            result = response.json()
            job_id = result.get("job_id")

            print(f"Hume job submitted: {job_id}")
            return job_id

        except Exception as e:
            print(f"Error submitting to Hume: {e}")
            raise

    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """
        Check Hume job status

        Args:
            job_id: The batch job identifier

        Returns:
            Dictionary with status information
        """
        try:
            response = requests.get(
                f"{self.base_url}/jobs/{job_id}",
                headers={
                    "X-Hume-Api-Key": self.api_key
                },
                timeout=30
            )

            response.raise_for_status()
            return response.json()

        except Exception as e:
            print(f"Error checking Hume job status: {e}")
            raise

    def get_predictions(self, job_id: str) -> Dict[str, Any]:
        """
        Fetch prosody predictions from completed job

        Args:
            job_id: The batch job identifier

        Returns:
            Dictionary with predictions including timestamps and emotion scores
        """
        try:
            response = requests.get(
                f"{self.base_url}/jobs/{job_id}/predictions",
                headers={
                    "X-Hume-Api-Key": self.api_key
                },
                timeout=60
            )

            response.raise_for_status()
            predictions = response.json()

            print(f"Retrieved predictions for job: {job_id}")
            return predictions

        except Exception as e:
            print(f"Error fetching Hume predictions: {e}")
            raise

    def align_emotions_with_speakers(
        self,
        hume_predictions: Dict[str, Any],
        whisper_segments: List[Dict[str, Any]],
        speaker_labels: Dict[str, str]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Map Hume emotion timestamps to speakers (CALLER vs AGENT or SPEAKER_A vs SPEAKER_B)

        Args:
            hume_predictions: Hume prosody data with timestamps
            whisper_segments: List of diarization segments with speaker labels
            speaker_labels: Mapping of SPEAKER_0/1 to AGENT/CALLER or SPEAKER_A/B

        Returns:
            {
                "speaker_a_timeline": [{"time": 1.5, "emotions": {"Joy": 0.45, ...}}],
                "speaker_b_timeline": [...]
            }
        """
        print("Aligning Hume emotions with speakers...")

        speaker_a_timeline = []
        speaker_b_timeline = []

        # Extract prosody predictions from Hume response
        # Structure varies, adapt to actual Hume API response
        prosody_predictions = []

        try:
            # DEBUG: Print raw Hume predictions structure
            print(f"DEBUG: Hume predictions type: {type(hume_predictions)}")
            if isinstance(hume_predictions, dict):
                print(f"DEBUG: Hume predictions keys: {hume_predictions.keys()}")
            elif isinstance(hume_predictions, list) and len(hume_predictions) > 0:
                print(f"DEBUG: Hume predictions list length: {len(hume_predictions)}")
                print(f"DEBUG: First item keys: {hume_predictions[0].keys() if isinstance(hume_predictions[0], dict) else 'not a dict'}")

            # Hume API structure: navigate to prosody predictions
            if isinstance(hume_predictions, list):
                for result in hume_predictions:
                    # Check for batch API structure with grouped_predictions
                    if "results" in result and "predictions" in result["results"]:
                        for pred_group in result["results"]["predictions"]:
                            if "models" in pred_group and "prosody" in pred_group["models"]:
                                grouped = pred_group["models"]["prosody"]["grouped_predictions"]
                                # Extract predictions from each grouped prediction
                                for group in grouped:
                                    if "predictions" in group:
                                        prosody_predictions.extend(group["predictions"])
                                    else:
                                        # Fallback: use the group itself if no nested predictions
                                        prosody_predictions.append(group)
                    # Check for direct predictions array structure (what we're actually getting)
                    elif "predictions" in result:
                        print(f"DEBUG: Found 'predictions' array with {len(result['predictions'])} items")
                        prosody_predictions.extend(result["predictions"])
            elif "predictions" in hume_predictions:
                prosody_predictions = hume_predictions["predictions"]
            else:
                # Try direct access
                prosody_predictions = hume_predictions

            print(f"Found {len(prosody_predictions)} prosody predictions")

        except Exception as e:
            print(f"Error parsing Hume predictions structure: {e}")
            # Return empty timelines if parsing fails
            return {
                "speaker_a_timeline": [],
                "speaker_b_timeline": []
            }

        # Align each Hume prediction window with speaker
        for idx, pred in enumerate(prosody_predictions):
            try:
                # DEBUG: Print first prediction structure to understand format
                if idx == 0:
                    print(f"DEBUG: First prediction structure: {pred}")
                    print(f"DEBUG: Prediction keys: {pred.keys()}")

                # Get time window
                time_info = pred.get("time", {})
                pred_start = time_info.get("begin", 0.0)
                pred_end = time_info.get("end", 0.0)
                pred_mid = (pred_start + pred_end) / 2

                # Get emotion scores
                emotions = {}
                if "emotions" in pred:
                    for emotion_data in pred["emotions"]:
                        emotion_name = emotion_data.get("name")
                        emotion_score = emotion_data.get("score", 0.0)
                        emotions[emotion_name] = emotion_score
                    if idx == 0:
                        print(f"DEBUG: Extracted {len(emotions)} emotions from first prediction")
                        print(f"DEBUG: Sample emotions: {list(emotions.items())[:3]}")
                else:
                    print(f"DEBUG: No 'emotions' key in prediction {idx}, keys: {pred.keys()}")

                # Find overlapping whisper segment to determine speaker
                matched_speaker = None
                for segment in whisper_segments:
                    seg_start = segment.get("start", 0.0)
                    seg_end = segment.get("end", 0.0)

                    # Check if prediction midpoint falls within segment
                    if seg_start <= pred_mid <= seg_end:
                        raw_speaker = segment.get("speaker", "SPEAKER_0")
                        matched_speaker = speaker_labels.get(raw_speaker, "SPEAKER_A")
                        break

                # If no match, assign to first speaker by default
                if not matched_speaker:
                    matched_speaker = list(speaker_labels.values())[0] if speaker_labels else "SPEAKER_A"

                emotion_entry = {
                    "time": pred_mid,
                    "emotions": emotions
                }

                # Route to appropriate timeline
                if matched_speaker in ["CALLER", "SPEAKER_A"]:
                    speaker_a_timeline.append(emotion_entry)
                else:
                    speaker_b_timeline.append(emotion_entry)

            except Exception as e:
                print(f"Error processing prediction: {e}")
                continue

        print(f"Speaker A timeline: {len(speaker_a_timeline)} entries")
        print(f"Speaker B timeline: {len(speaker_b_timeline)} entries")

        return {
            "speaker_a_timeline": speaker_a_timeline,
            "speaker_b_timeline": speaker_b_timeline
        }

    def compute_speaker_emotion_features(self, timeline: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compute 96-d emotion vector for ONE speaker (48 means + 48 maxes)

        Args:
            timeline: List of emotion entries with timestamps

        Returns:
            {
                "emotion_vector": [0.45, 0.12, ...],  # 96-d
                "top_5_emotions": ["Joy", "Frustration", ...],
                "mean_valence": 0.62,
                "peak_emotion": {"emotion": "Frustration", "time": 45.2, "score": 0.89}
            }
        """
        if not timeline or len(timeline) == 0:
            return {
                "emotion_vector": [0.0] * 96,
                "top_5_emotions": [],
                "mean_valence": 0.0,
                "peak_emotion": None
            }

        # Get all emotion names in consistent order
        first_emotions = timeline[0].get("emotions", {})
        emotion_names = sorted(first_emotions.keys())

        print(f"Computing features from {len(timeline)} timeline entries, {len(emotion_names)} emotions")

        # If no emotions in timeline, return zero vector
        if len(emotion_names) == 0:
            print("Warning: Timeline has no emotions, returning zero vector")
            return {
                "emotion_vector": [0.0] * 96,
                "top_5_emotions": [],
                "mean_valence": 0.0,
                "peak_emotion": None
            }

        # Compute means
        means = []
        for emotion in emotion_names:
            values = [entry["emotions"].get(emotion, 0.0) for entry in timeline]
            mean_val = sum(values) / len(values) if values else 0.0
            means.append(mean_val)

        # Compute maxes
        maxes = []
        for emotion in emotion_names:
            values = [entry["emotions"].get(emotion, 0.0) for entry in timeline]
            max_val = max(values) if values else 0.0
            maxes.append(max_val)

        # Create 96-d vector (48 means + 48 maxes)
        emotion_vector = means + maxes

        # Top 5 emotions by mean intensity
        emotion_scores = list(zip(emotion_names, means))
        emotion_scores.sort(key=lambda x: x[1], reverse=True)
        top_5 = [name for name, score in emotion_scores[:5]]

        # Find peak emotion across all timeline
        all_peaks = []
        for entry in timeline:
            for emotion_name, score in entry["emotions"].items():
                all_peaks.append({
                    "emotion": emotion_name,
                    "time": entry["time"],
                    "score": score
                })

        peak = max(all_peaks, key=lambda x: x["score"]) if all_peaks else None

        # Calculate valence (simplified: Joy - Sadness)
        mean_valence = 0.0
        if "Joy" in emotion_names and "Sadness" in emotion_names:
            joy_idx = emotion_names.index("Joy")
            sadness_idx = emotion_names.index("Sadness")
            mean_valence = means[joy_idx] - means[sadness_idx]

        return {
            "emotion_vector": emotion_vector,
            "top_5_emotions": top_5,
            "mean_valence": mean_valence,
            "peak_emotion": peak
        }

    def create_emotion_query_vector(self, primary_emotions: List[str], emotion_weight: float = 0.8) -> List[float]:
        """
        Create a synthetic 192-d emotion vector for querying based on desired emotions

        Args:
            primary_emotions: List of emotion names to emphasize (e.g., ["Anger", "Frustration"])
            emotion_weight: Score to assign to primary emotions (0.0-1.0)

        Returns:
            192-d vector (96-d per speaker) suitable for querying emotion index
        """
        # Hume AI's 48 emotion categories in alphabetical order
        hume_emotions = [
            "Admiration", "Adoration", "Aesthetic Appreciation", "Amusement", "Anger", "Anxiety",
            "Awe", "Awkwardness", "Boredom", "Calmness", "Concentration", "Confusion",
            "Contemplation", "Contempt", "Contentment", "Craving", "Desire", "Determination",
            "Disappointment", "Disgust", "Distress", "Doubt", "Ecstasy", "Embarrassment",
            "Empathic Pain", "Entrancement", "Envy", "Excitement", "Fear", "Guilt",
            "Horror", "Interest", "Joy", "Love", "Nostalgia", "Pain", "Pride",
            "Realization", "Relief", "Romance", "Sadness", "Satisfaction", "Shame",
            "Surprise (negative)", "Surprise (positive)", "Sympathy", "Tiredness", "Triumph"
        ]

        # Create base vector (all zeros)
        means = [0.0] * 48
        maxes = [0.0] * 48

        # Set high scores for primary emotions
        for emotion in primary_emotions:
            if emotion in hume_emotions:
                idx = hume_emotions.index(emotion)
                means[idx] = emotion_weight
                maxes[idx] = emotion_weight
            else:
                print(f"Warning: Emotion '{emotion}' not in Hume emotion list")

        # Create 96-d vector for one speaker (means + maxes)
        speaker_vector = means + maxes

        # Create 192-d vector (duplicate for both speakers)
        # This queries for calls where EITHER speaker shows these emotions
        combined_vector = speaker_vector + speaker_vector

        return combined_vector

    def compute_combined_features(
        self,
        speaker_a_features: Dict[str, Any],
        speaker_b_features: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Combine both speakers into 192-d vector

        Args:
            speaker_a_features: Features for speaker A
            speaker_b_features: Features for speaker B

        Returns:
            {
                "combined_vector": [...],  # 192-d (96 speaker_a + 96 speaker_b)
                "speaker_a": {...},
                "speaker_b": {...}
            }
        """
        # Concatenate 96-d vectors
        combined_vector = (
            speaker_a_features["emotion_vector"] +
            speaker_b_features["emotion_vector"]
        )

        return {
            "combined_vector": combined_vector,
            "speaker_a": speaker_a_features,
            "speaker_b": speaker_b_features
        }
