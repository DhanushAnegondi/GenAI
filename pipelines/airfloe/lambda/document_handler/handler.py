import json
import boto3
import os
from typing import Dict, Any

s3_client = boto3.client('s3')
sqs_client = boto3.client('sqs')

def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    AWS Lambda handler for document processing
    """
    try:
        # Parse S3 event
        for record in event['Records']:
            bucket = record['s3']['bucket']['name']
            key = record['s3']['object']['key']
            
            # Download document
            response = s3_client.get_object(Bucket=bucket, Key=key)
            document_content = response['Body'].read()
            
            # Process document
            processed_data = process_document(document_content, key)
            
            # Send to SQS for embedding generation
            message = {
                'document_id': processed_data['document_id'],
                'chunks': processed_data['chunks'],
                'metadata': processed_data['metadata']
            }
            
            sqs_client.send_message(
                QueueUrl=os.environ['EMBEDDING_QUEUE_URL'],
                MessageBody=json.dumps(message)
            )
            
            # Move to processed folder
            copy_source = {'Bucket': bucket, 'Key': key}
            new_key = key.replace('pending/', 'processed/')
            s3_client.copy_object(
                CopySource=copy_source,
                Bucket=bucket,
                Key=new_key
            )
            s3_client.delete_object(Bucket=bucket, Key=key)
        
        return {
            'statusCode': 200,
            'body': json.dumps('Documents processed successfully')
        }
        
    except Exception as e:
        print(f"Error processing document: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps(f'Error: {str(e)}')
        }

def process_document(content: bytes, key: str) -> Dict[str, Any]:
    """
    Process document content
    """
    # Implementation of document processing logic
    # This would include parsing, chunking, etc.
    return {
        'document_id': key.split('/')[-1],
        'chunks': [],  # Processed chunks
        'metadata': {}  # Document metadata
    }
