from typing import List, Optional, Dict, Any
import redis.asyncio as redis
import json
import hashlib
from ..embedding_service.generator import EmbeddingGenerator
from ..embedding_service.vector_store import VectorStore, VectorSearchResult

class SemanticSearcher:
    def __init__(
        self,
        vector_store: VectorStore,
        embedding_generator: EmbeddingGenerator,
        redis_client: redis.Redis
    ):
        self.vector_store = vector_store
        self.embedding_generator = embedding_generator
        self.cache = redis_client
        self.cache_ttl = 3600  # 1 hour
    
    async def search(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict] = None,
        use_cache: bool = True
    ) -> List[VectorSearchResult]:
        """
        Perform semantic search with caching
        """
        # Check cache
        if use_cache:
            cache_key = self._get_cache_key(query, top_k, filters)
            cached_result = await self.cache.get(cache_key)
            
            if cached_result:
                return self._deserialize_results(cached_result)
        
        # Generate query embedding
        query_embedding = await self.embedding_generator.generate_embeddings([query])
        
        # Search vector store
        results = await self.vector_store.search(
            query_embedding=query_embedding[0],
            top_k=top_k,
            filter=filters
        )
        
        # Re-rank results
        results = await self._rerank_results(query, results)
        
        # Cache results
        if use_cache:
            await self.cache.setex(
                cache_key,
                self.cache_ttl,
                self._serialize_results(results)
            )
        
        return results
    
    async def _rerank_results(
        self,
        query: str,
        results: List[VectorSearchResult]
    ) -> List[VectorSearchResult]:
        """
        Re-rank results using cross-encoder or custom logic
        """
        # Implement cross-encoder reranking
        # For now, using score-based ranking
        return sorted(results, key=lambda x: x.score, reverse=True)
    
    def _get_cache_key(
        self,
        query: str,
        top_k: int,
        filters: Optional[Dict]
    ) -> str:
        """
        Generate cache key for query
        """
        key_data = f"{query}_{top_k}_{json.dumps(filters or {})}"
        return f"search:{hashlib.md5(key_data.encode()).hexdigest()}"
    
    def _serialize_results(self, results: List[VectorSearchResult]) -> str:
        return json.dumps([
            {
                "id": r.id,
                "score": r.score,
                "metadata": r.metadata,
                "text": r.text
            }
            for r in results
        ])
    
    def _deserialize_results(self, data: str) -> List[VectorSearchResult]:
        results_data = json.loads(data)
        return [
            VectorSearchResult(**r)
            for r in results_data
        ]
