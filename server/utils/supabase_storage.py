"""Supabase Storage utility for file management.

This module provides functions to upload, download, and manage files
in Supabase Storage instead of storing them in the database.
"""

import os
import logging
from typing import Optional, Tuple, Dict, Any
from datetime import datetime
import httpx

logger = logging.getLogger(__name__)

# Supabase configuration from environment
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY", "")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY", "")

# Storage bucket names
BUCKET_FILES = "files"
BUCKET_MODELS = "models"


def get_supabase_headers(use_service_key: bool = True) -> Dict[str, str]:
    """Get headers for Supabase API requests."""
    key = SUPABASE_SERVICE_KEY if use_service_key else SUPABASE_ANON_KEY
    return {
        "Authorization": f"Bearer {key}",
        "apikey": key,
    }


def get_storage_url() -> str:
    """Get the Supabase Storage API URL."""
    return f"{SUPABASE_URL}/storage/v1"


async def ensure_bucket_exists(bucket_name: str) -> bool:
    """Ensure a storage bucket exists, create if not.
    
    Args:
        bucket_name: Name of the bucket to create
        
    Returns:
        True if bucket exists or was created successfully
    """
    if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
        logger.warning("Supabase credentials not configured, skipping bucket creation")
        return False
    
    try:
        async with httpx.AsyncClient() as client:
            # Check if bucket exists
            response = await client.get(
                f"{get_storage_url()}/bucket/{bucket_name}",
                headers=get_supabase_headers()
            )
            
            if response.status_code == 200:
                logger.info(f"Bucket '{bucket_name}' already exists")
                return True
            
            # Create bucket if it doesn't exist
            response = await client.post(
                f"{get_storage_url()}/bucket",
                headers=get_supabase_headers(),
                json={
                    "id": bucket_name,
                    "name": bucket_name,
                    "public": False,
                    "file_size_limit": 209715200,  # 200MB
                    "allowed_mime_types": None  # Allow all types
                }
            )
            
            if response.status_code in (200, 201):
                logger.info(f"Created bucket '{bucket_name}'")
                return True
            else:
                logger.error(f"Failed to create bucket: {response.text}")
                return False
                
    except Exception as e:
        logger.error(f"Error ensuring bucket exists: {e}")
        return False


async def upload_file(
    bucket: str,
    path: str,
    content: bytes,
    content_type: str = "application/octet-stream",
    upsert: bool = True
) -> Tuple[bool, Optional[str]]:
    """Upload a file to Supabase Storage.
    
    Args:
        bucket: Bucket name (e.g., 'files', 'models')
        path: Path within the bucket (e.g., 'user_123/file_456.csv')
        content: File content as bytes
        content_type: MIME type of the file
        upsert: If True, overwrite existing file
        
    Returns:
        Tuple of (success, storage_path or error_message)
    """
    if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
        logger.warning("Supabase credentials not configured, falling back to DB storage")
        return False, "Supabase not configured"
    
    try:
        headers = get_supabase_headers()
        headers["Content-Type"] = content_type
        if upsert:
            headers["x-upsert"] = "true"
        
        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.post(
                f"{get_storage_url()}/object/{bucket}/{path}",
                headers=headers,
                content=content
            )
            
            if response.status_code in (200, 201):
                storage_path = f"{bucket}/{path}"
                logger.info(f"Uploaded file to storage: {storage_path} ({len(content)} bytes)")
                return True, storage_path
            else:
                error_msg = f"Upload failed: {response.status_code} - {response.text}"
                logger.error(error_msg)
                return False, error_msg
                
    except Exception as e:
        error_msg = f"Upload error: {str(e)}"
        logger.error(error_msg)
        return False, error_msg


async def download_file(bucket: str, path: str) -> Tuple[bool, Optional[bytes]]:
    """Download a file from Supabase Storage.
    
    Args:
        bucket: Bucket name
        path: Path within the bucket
        
    Returns:
        Tuple of (success, file_content or None)
    """
    if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
        return False, None
    
    try:
        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.get(
                f"{get_storage_url()}/object/{bucket}/{path}",
                headers=get_supabase_headers()
            )
            
            if response.status_code == 200:
                return True, response.content
            else:
                logger.error(f"Download failed: {response.status_code}")
                return False, None
                
    except Exception as e:
        logger.error(f"Download error: {e}")
        return False, None


async def delete_file(bucket: str, path: str) -> bool:
    """Delete a file from Supabase Storage.
    
    Args:
        bucket: Bucket name
        path: Path within the bucket
        
    Returns:
        True if deleted successfully
    """
    if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
        return False
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.delete(
                f"{get_storage_url()}/object/{bucket}/{path}",
                headers=get_supabase_headers()
            )
            
            if response.status_code in (200, 204):
                logger.info(f"Deleted file from storage: {bucket}/{path}")
                return True
            else:
                logger.error(f"Delete failed: {response.status_code}")
                return False
                
    except Exception as e:
        logger.error(f"Delete error: {e}")
        return False


def get_signed_url(bucket: str, path: str, expires_in: int = 3600) -> Optional[str]:
    """Get a signed URL for temporary file access.
    
    Args:
        bucket: Bucket name
        path: Path within the bucket
        expires_in: URL expiration time in seconds (default 1 hour)
        
    Returns:
        Signed URL or None if failed
    """
    if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
        return None
    
    try:
        import httpx
        
        with httpx.Client() as client:
            response = client.post(
                f"{get_storage_url()}/object/sign/{bucket}/{path}",
                headers=get_supabase_headers(),
                json={"expiresIn": expires_in}
            )
            
            if response.status_code == 200:
                data = response.json()
                return f"{SUPABASE_URL}/storage/v1{data['signedURL']}"
            else:
                logger.error(f"Failed to get signed URL: {response.status_code}")
                return None
                
    except Exception as e:
        logger.error(f"Signed URL error: {e}")
        return None


def generate_storage_path(user_id: int, file_id: int, filename: str) -> str:
    """Generate a storage path for a file.
    
    Args:
        user_id: User ID
        file_id: File ID
        filename: Original filename
        
    Returns:
        Storage path like 'user_123/file_456/original_name.csv'
    """
    # Sanitize filename
    safe_filename = "".join(c for c in filename if c.isalnum() or c in "._-")
    return f"user_{user_id}/file_{file_id}/{safe_filename}"


def generate_model_storage_path(user_id: int, model_id: int, model_name: str) -> str:
    """Generate a storage path for a trained model.
    
    Args:
        user_id: User ID
        model_id: Model ID
        model_name: Model name
        
    Returns:
        Storage path like 'user_123/model_456/model_name.pt'
    """
    safe_name = "".join(c for c in model_name if c.isalnum() or c in "._-")
    return f"user_{user_id}/model_{model_id}/{safe_name}.pt"


def is_storage_configured() -> bool:
    """Check if Supabase Storage is properly configured."""
    return bool(SUPABASE_URL and SUPABASE_SERVICE_KEY)


# Synchronous versions for use in non-async contexts
def upload_file_sync(
    bucket: str,
    path: str,
    content: bytes,
    content_type: str = "application/octet-stream",
    upsert: bool = True
) -> Tuple[bool, Optional[str]]:
    """Synchronous version of upload_file."""
    if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
        return False, "Supabase not configured"
    
    try:
        headers = get_supabase_headers()
        headers["Content-Type"] = content_type
        if upsert:
            headers["x-upsert"] = "true"
        
        with httpx.Client(timeout=300.0) as client:
            response = client.post(
                f"{get_storage_url()}/object/{bucket}/{path}",
                headers=headers,
                content=content
            )
            
            if response.status_code in (200, 201):
                storage_path = f"{bucket}/{path}"
                logger.info(f"Uploaded file to storage: {storage_path}")
                return True, storage_path
            else:
                return False, f"Upload failed: {response.status_code}"
                
    except Exception as e:
        return False, str(e)


def download_file_sync(bucket: str, path: str) -> Tuple[bool, Optional[bytes]]:
    """Synchronous version of download_file."""
    if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
        return False, None
    
    try:
        with httpx.Client(timeout=300.0) as client:
            response = client.get(
                f"{get_storage_url()}/object/{bucket}/{path}",
                headers=get_supabase_headers()
            )
            
            if response.status_code == 200:
                return True, response.content
            else:
                return False, None
                
    except Exception as e:
        logger.error(f"Download error: {e}")
        return False, None
