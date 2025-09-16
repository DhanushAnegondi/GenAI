from typing import List, Dict, Any, Optional
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import OpenAI

class ContextBuilder:
    def __init__(self):
        self.llm = OpenAI(temperature=0.3)
        self.summary_prompt = PromptTemplate(
            input_variables=["chunks", "query"],
            template="""
            Given the following retrieved chunks and user query,
            create a comprehensive context that answers the query.
            
            Query: {query}
            
            Retrieved Chunks:
            {chunks}
            
            Comprehensive Answer:
            """
        )
        self.summary_chain = LLMChain(
            llm=self.llm,
            prompt=self.summary_prompt
        )
    
    async def build_context(
        self,
        query: str,
        search_results: List[Dict[str, Any]],
        context_type: str = "comprehensive"
    ) -> Dict[str, Any]:
        """
        Build rich context from search results
        """
        if context_type == "comprehensive":
            return await self._build_comprehensive_context(query, search_results)
        elif context_type == "summary":
            return await self._build_summary_context(query, search_results)
        else:
            return await self._build_raw_context(search_results)
    
    async def _build_comprehensive_context(
        self,
        query: str,
        search_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Build comprehensive context with summaries and metadata
        """
        # Extract text chunks
        chunks_text = "\n\n".join([
            f"[Chunk {i+1}]: {result['text']}"
            for i, result in enumerate(search_results)
        ])
        
        # Generate summary
        summary = await self.summary_chain.arun(
            query=query,
            chunks=chunks_text
        )
        
        # Build context
        context = {
            "query": query,
            "summary": summary,
            "sources": [
                {
                    "document_id": result.get("metadata", {}).get("document_id"),
                    "chunk_id": result.get("id"),
                    "relevance_score": result.get("score"),
                    "text": result.get("text")
                }
                for result in search_results
            ],
            "metadata": {
                "total_sources": len(search_results),
                "context_type": "comprehensive"
            }
        }
        
        return context
