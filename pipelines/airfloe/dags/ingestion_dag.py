from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.aws.operators.lambda_function import LambdaInvokeFunctionOperator
from airflow.providers.snowflake.operators.snowflake import SnowflakeOperator

default_args = {
    'owner': 'data-team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5)
}

dag = DAG(
    'document_ingestion_pipeline',
    default_args=default_args,
    description='Automated document ingestion and processing',
    schedule_interval='*/30 * * * *',  # Every 30 minutes
    catchup=False
)

def scan_s3_for_documents(**context):
    """
    Scan S3 bucket for new documents
    """
    import boto3
    s3 = boto3.client('s3')
    
    # Get list of unprocessed documents
    response = s3.list_objects_v2(
        Bucket='genai-knowledge-bucket',
        Prefix='documents/pending/'
    )
    
    documents = []
    if 'Contents' in response:
        documents = [obj['Key'] for obj in response['Contents']]
    
    # Push to XCom for next task
    return documents

def trigger_processing(**context):
    """
    Trigger Lambda for each document
    """
    documents = context['task_instance'].xcom_pull(task_ids='scan_documents')
    
    for doc_key in documents:
        # Trigger Lambda function
        lambda_payload = {
            'document_key': doc_key,
            'processing_type': 'full'
        }
        # Lambda invocation handled by next operator

scan_task = PythonOperator(
    task_id='scan_documents',
    python_callable=scan_s3_for_documents,
    dag=dag
)

process_task = LambdaInvokeFunctionOperator(
    task_id='process_documents',
    function_name='document-processor',
    payload='{{ ti.xcom_pull(task_ids="scan_documents") }}',
    aws_conn_id='aws_default',
    dag=dag
)

update_analytics = SnowflakeOperator(
    task_id='update_snowflake',
    sql="""
        INSERT INTO document_processing_metrics
        SELECT 
            current_timestamp() as processed_at,
            '{{ ds }}' as batch_date,
            count(*) as documents_processed
        FROM staging_documents
        WHERE batch_id = '{{ run_id }}'
    """,
    snowflake_conn_id='snowflake_default',
    dag=dag
)

scan_task >> process_task >> update_analytics
