import openai
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Optional
import asyncio
from concurrent.futures import ThreadPoolExecutor

class EmbeddingGenerator:
    def __init__(self, model_type: str = "openai"):
        self.model_type = model_type
        
        if model_type == "openai":
            openai.api_key = os.getenv("OPENAI_API_KEY")
            self.model = "text-embedding-ada-002"
        else:
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        self.executor = ThreadPoolExecutor(max_workers=10)
    
    async def generate_embeddings(
        self,
        texts: List[str],
        batch_size: int = 100
    ) -> List[np.ndarray]:
        """
        Generate embeddings for text chunks
        """
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            if self.model_type == "openai":
                batch_embeddings = await self._generate_openai_embeddings(batch)
            else:
                batch_embeddings = await self._generate_local_embeddings(batch)
            
            embeddings.extend(batch_embeddings)
        
        return embeddings
    
    async def _generate_openai_embeddings(
        self,
        texts: List[str]
    ) -> List[np.ndarray]:
        """
        Generate embeddings using OpenAI API
        """
        loop = asyncio.get_event_loop()
        
        def get_embedding(text):
            response = openai.Embedding.create(
                input=text,
                model=self.model
            )
            return np.array(response['data'][0]['embedding'])
        
        tasks = [
            loop.run_in_executor(self.executor, get_embedding, text)
            for text in texts
        ]
        
        return await asyncio.gather(*tasks)
    
    async def _generate_local_embeddings(
        self,
        texts: List[str]
    ) -> List[np.ndarray]:
        """
        Generate embeddings using local model
        """
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            self.executor,
            self.model.encode,
            texts
        )
        return embeddings
