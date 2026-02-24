import logging
from google.cloud import storage
from pathlib import Path

logger = logging.getLogger(__name__)


def upload_to_gcs(bucket_name: str, local_path: Path, gcs_destination: str) -> str:
    """
    Uploads a local file to a Google Cloud Storage bucket and returns the gs:// URI.
    """
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(gcs_destination)

        blob.upload_from_filename(str(local_path))
        return f"gs://{bucket_name}/{gcs_destination}"
    except Exception as e:
        raise RuntimeError(f"Failed to upload to GCS: {e}") from e


def delete_from_gcs(bucket_name: str, gcs_path: str):
    """
    Deletes the file from Google Cloud Storage after we are done transpiling.
    """
    try:
        # Remove the gs://bucket/ part to get just the blob name
        blob_name = gcs_path.replace(f"gs://{bucket_name}/", "")

        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)

        if blob.exists():
            blob.delete()
    except Exception as e:
        # A deletion error shouldn't crash the pipeline, just warn
        logger.warning(f"Failed to cleanup GCS file {gcs_path}: {e}")
