from fastapi import APIRouter, UploadFile, File, BackgroundTasks, HTTPException
from typing import List
import uuid
import boto3
from ..services.document_processor import DocumentProcessor
from ..models.document import DocumentUploadResponse

router = APIRouter()
s3_client = boto3.client('s3')
processor = DocumentProcessor()

@router.post("/document", response_model=DocumentUploadResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    metadata: Optional[dict] = None
):
    """
    Upload and process a document for knowledge extraction
    """
    try:
        # Generate unique document ID
        doc_id = str(uuid.uuid4())
        
        # Upload to S3
        s3_key = f"documents/{doc_id}/{file.filename}"
        s3_client.upload_fileobj(
            file.file,
            "genai-knowledge-bucket",
            s3_key
        )
        
        # Queue background processing
        background_tasks.add_task(
            processor.process_document,
            doc_id=doc_id,
            s3_key=s3_key,
            metadata=metadata
        )
        
        return DocumentUploadResponse(
            document_id=doc_id,
            status="processing",
            message="Document uploaded successfully"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/batch", response_model=List[DocumentUploadResponse])
async def upload_batch(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...)
):
    """
    Batch upload multiple documents
    """
    responses = []
    for file in files:
        response = await upload_document(background_tasks, file)
        responses.append(response)
    return responses
