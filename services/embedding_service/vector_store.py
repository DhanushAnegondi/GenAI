import pinecone
from typing import List, Dict, Any, Optional
import numpy as np
from dataclasses import dataclass

@dataclass
class VectorSearchResult:
    id: str
    score: float
    metadata: Dict[str, Any]
    text: str

class VectorStore:
    def __init__(self, index_name: str = "knowledge-base"):
        pinecone.init(
            api_key=os.getenv("PINECONE_API_KEY"),
            environment=os.getenv("PINECONE_ENV")
        )
        
        # Create index if doesn't exist
        if index_name not in pinecone.list_indexes():
            pinecone.create_index(
                name=index_name,
                dimension=1536,  # OpenAI ada-002 dimension
                metric="cosine",
                pod_type="p1"
            )
        
        self.index = pinecone.Index(index_name)
    
    async def upsert_embeddings(
        self,
        embeddings: List[np.ndarray],
        chunks: List[Dict[str, Any]],
        namespace: Optional[str] = None
    ):
        """
        Store embeddings with metadata in vector database
        """
        vectors = []
        
        for embedding, chunk in zip(embeddings, chunks):
            vectors.append({
                "id": chunk["chunk_id"],
                "values": embedding.tolist(),
                "metadata": {
                    "document_id": chunk["document_id"],
                    "chunk_index": chunk["chunk_index"],
                    "text": chunk["text"][:1000],  # Store first 1000 chars
                    **chunk.get("metadata", {})
                }
            })
        
        # Batch upsert
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            self.index.upsert(vectors=batch, namespace=namespace)
    
    async def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        filter: Optional[Dict] = None,
        namespace: Optional[str] = None
    ) -> List[VectorSearchResult]:
        """
        Search for similar vectors
        """
        results = self.index.query(
            vector=query_embedding.tolist(),
            top_k=top_k,
            filter=filter,
            namespace=namespace,
            include_metadata=True
        )
        
        return [
            VectorSearchResult(
                id=match["id"],
                score=match["score"],
                metadata=match["metadata"],
                text=match["metadata"].get("text", "")
            )
            for match in results["matches"]
        ]
