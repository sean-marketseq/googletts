"""
Pinecone vector database operations for conversation storage and retrieval
"""
import os
from typing import List, Dict, Any, Optional
from pinecone import Pinecone, ServerlessSpec


class PineconeClient:
    """Wrapper for Pinecone vector database operations"""

    def __init__(self, api_key: str = None, environment: str = None, index_name: str = None):
        """
        Initialize Pinecone client

        Args:
            api_key: Pinecone API key (defaults to PINECONE_API_KEY env var)
            environment: Pinecone environment (defaults to PINECONE_ENVIRONMENT env var)
            index_name: Pinecone index name (defaults to PINECONE_INDEX env var)
        """
        self.api_key = api_key or os.getenv("PINECONE_API_KEY")
        self.environment = environment or os.getenv("PINECONE_ENVIRONMENT", "us-east-1-aws")
        self.index_name = index_name or os.getenv("PINECONE_INDEX", "conversations")

        # Initialize Pinecone
        self.pc = Pinecone(api_key=self.api_key)
        self.index = None

    def setup_index(self, dimension: int = 1536, metric: str = "cosine"):
        """
        Create or connect to Pinecone index

        Args:
            dimension: Embedding dimension (1536 for text-embedding-3-small)
            metric: Distance metric (cosine, euclidean, dotproduct)
        """
        # Check if index exists
        existing_indexes = self.pc.list_indexes()

        if self.index_name not in [idx.name for idx in existing_indexes]:
            print(f"Creating new Pinecone index: {self.index_name}")

            # Create serverless index
            self.pc.create_index(
                name=self.index_name,
                dimension=dimension,
                metric=metric,
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
            print(f"Index {self.index_name} created successfully")
        else:
            print(f"Using existing index: {self.index_name}")

        # Connect to the index
        self.index = self.pc.Index(self.index_name)

    def upsert_conversation(self, records: List[Dict[str, Any]], namespace: str = "conversations"):
        """
        Insert or update conversation chunks in Pinecone

        Args:
            records: List of records with format:
                     {"id": str, "values": List[float], "metadata": Dict}
            namespace: Pinecone namespace for organization
        """
        if not self.index:
            raise ValueError("Index not initialized. Call setup_index() first.")

        # Pinecone recommends batching upserts for better performance
        batch_size = 100
        for i in range(0, len(records), batch_size):
            batch = records[i:i + batch_size]
            self.index.upsert(vectors=batch, namespace=namespace)
            print(f"Upserted {len(batch)} vectors to Pinecone")

        print(f"Total vectors upserted: {len(records)}")

    def query(
        self,
        embedding: List[float],
        top_k: int = 50,
        filter: Optional[Dict[str, Any]] = None,
        namespace: str = "conversations",
        include_metadata: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Query Pinecone for similar conversation chunks

        Args:
            embedding: Query embedding vector
            top_k: Number of results to return
            filter: Metadata filter (e.g., {"account_id": "acct_123"})
            namespace: Pinecone namespace
            include_metadata: Whether to include metadata in results

        Returns:
            List of matches with scores and metadata
        """
        if not self.index:
            raise ValueError("Index not initialized. Call setup_index() first.")

        # Query the index
        results = self.index.query(
            vector=embedding,
            top_k=top_k,
            filter=filter,
            namespace=namespace,
            include_metadata=include_metadata
        )

        # Format results
        matches = []
        for match in results.matches:
            matches.append({
                "id": match.id,
                "score": match.score,
                "metadata": match.metadata if include_metadata else None
            })

        return matches

    def get_all_conversations(self, namespace: str = "conversations") -> List[Dict[str, Any]]:
        """
        Get a list of all unique conversations stored in Pinecone

        Note: This is a simplified implementation. For production, consider
        maintaining a separate metadata store for conversation listings.

        Args:
            namespace: Pinecone namespace

        Returns:
            List of conversations with metadata
        """
        if not self.index:
            raise ValueError("Index not initialized. Call setup_index() first.")

        # Get index stats
        stats = self.index.describe_index_stats()
        print(f"Index stats: {stats}")

        # For this POC, we'll use a simple query approach
        # In production, you'd maintain a separate conversations metadata store

        # Query with a dummy vector to get some results
        # This is not ideal but works for POC with small datasets
        dummy_vector = [0.0] * 1536  # Zero vector

        results = self.index.query(
            vector=dummy_vector,
            top_k=1000,  # Adjust based on your needs
            namespace=namespace,
            include_metadata=True,
            filter=None
        )

        # Extract unique conversations
        conversations = {}
        for match in results.matches:
            metadata = match.metadata
            conv_id = metadata.get("conversation_id")

            if conv_id and conv_id not in conversations:
                conversations[conv_id] = {
                    "id": conv_id,
                    "date": metadata.get("date"),
                    "account_id": metadata.get("account_id"),
                    "agent_id": metadata.get("agent_id"),
                    "sentiment": metadata.get("sentiment"),
                    "intents": metadata.get("intents", [])
                }

        return list(conversations.values())

    def delete_conversation(self, conversation_id: str, namespace: str = "conversations"):
        """
        Delete all chunks for a specific conversation

        Args:
            conversation_id: Conversation identifier
            namespace: Pinecone namespace
        """
        if not self.index:
            raise ValueError("Index not initialized. Call setup_index() first.")

        # Delete by filter (if supported) or by IDs
        # For this POC, we'll delete by ID prefix pattern
        # In production, you'd track chunk IDs separately

        # Use the filter to delete all chunks for this conversation
        self.index.delete(
            filter={"conversation_id": conversation_id},
            namespace=namespace
        )
        print(f"Deleted conversation: {conversation_id}")

    def purge_all_data(self, namespace: str = "conversations"):
        """
        Delete ALL data from a namespace (use with caution!)

        Args:
            namespace: Pinecone namespace to purge
        """
        if not self.index:
            raise ValueError("Index not initialized. Call setup_index() first.")

        print(f"⚠️  PURGING ALL DATA from namespace: {namespace}")

        # Delete all vectors in the namespace
        self.index.delete(delete_all=True, namespace=namespace)

        print(f"✓ All data purged from namespace: {namespace}")

