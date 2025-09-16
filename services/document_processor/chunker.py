from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Dict, Any
import hashlib

class IntelligentChunker:
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: List[str] = None
    ):
        self.separators = separators or ["\n\n", "\n", " ", ""]
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=self.separators,
            length_function=len,
        )
    
    def chunk_document(self, text: str, doc_id: str) -> List[Dict[str, Any]]:
        """
        Split document into semantic chunks with metadata
        """
        chunks = self.splitter.split_text(text)
        
        chunked_data = []
        for idx, chunk in enumerate(chunks):
            chunk_id = hashlib.md5(
                f"{doc_id}_{idx}_{chunk[:50]}".encode()
            ).hexdigest()
            
            chunked_data.append({
                "chunk_id": chunk_id,
                "document_id": doc_id,
                "chunk_index": idx,
                "text": chunk,
                "chunk_size": len(chunk),
                "metadata": {
                    "position": idx,
                    "total_chunks": len(chunks)
                }
            })
        
        return chunked_data
    
    def adaptive_chunking(self, text: str, doc_type: str) -> List[str]:
        """
        Adaptive chunking based on document type
        """
        if doc_type == "code":
            return self._chunk_code(text)
        elif doc_type == "legal":
            return self._chunk_legal(text)
        elif doc_type == "scientific":
            return self._chunk_scientific(text)
        else:
            return self.chunk_document(text, "generic")
    
    def _chunk_code(self, text: str) -> List[str]:
        # Custom logic for code files
        separators = ["\nclass ", "\ndef ", "\n\n", "\n"]
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=100,
            separators=separators
        )
        return splitter.split_text(text)
