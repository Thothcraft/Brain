"""Folder management endpoints for file organization."""

from datetime import datetime
from typing import Dict, Any, Optional, List
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session
from pydantic import BaseModel

from server.db import get_db
from server.auth import get_current_user
from server.db import User, File
from server.utils.logging_utils import log_request_start, log_response, log_error
from server.utils.error_handler import (
    APIError, handle_api_error, not_found_error, validation_error,
    ErrorCode
)

router = APIRouter(prefix="/folders", tags=["folders"])

# Folder database model (add to your models.py)
class Folder(BaseModel):
    id: Optional[int] = None
    name: str
    parent_id: Optional[int] = None
    user_id: int
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    path: Optional[str] = None  # Computed path like "/folder/subfolder"

class FolderCreate(BaseModel):
    name: str
    parent_id: Optional[int] = None

class FolderUpdate(BaseModel):
    name: Optional[str] = None
    parent_id: Optional[int] = None

class FolderResponse(BaseModel):
    id: int
    name: str
    parent_id: Optional[int]
    path: str
    created_at: datetime
    updated_at: datetime
    file_count: int
    subfolder_count: int
    size_bytes: int

class FileMoveRequest(BaseModel):
    file_id: int
    folder_id: Optional[int] = None  # None means root

def build_folder_path(db: Session, folder_id: int) -> str:
    """Build full path for a folder."""
    path_parts = []
    current_id = folder_id
    
    # This would use a Folder model - for now, simulate with folder data
    # In production, you'd have a proper Folder table in the database
    folder_cache = {}  # In production, use proper caching
    
    while current_id is not None:
        if current_id in folder_cache:
            folder = folder_cache[current_id]
        else:
            # Query folder from database
            # folder = db.query(Folder).filter(Folder.id == current_id).first()
            # For now, simulate
            folder = {"name": f"folder_{current_id}", "parent_id": None}
            folder_cache[current_id] = folder
        
        path_parts.append(folder["name"])
        current_id = folder["parent_id"]
    
    return "/" + "/".join(reversed(path_parts))

@router.post("/", response_model=FolderResponse)
async def create_folder(
    folder: FolderCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> FolderResponse:
    """Create a new folder.
    
    Args:
        folder: Folder creation data
        
    Returns:
        Created folder information
    """
    try:
        log_request_start("/folders/", "POST", None, None, current_user.userId)
        
        # Validate folder name
        if not folder.name or folder.name.strip() == "":
            raise handle_api_error(validation_error("Folder name cannot be empty"))
        
        # Check for invalid characters
        invalid_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|']
        if any(char in folder.name for char in invalid_chars):
            raise handle_api_error(validation_error(
                f"Folder name contains invalid characters: {', '.join(invalid_chars)}"
            ))
        
        # In production, create folder in database
        # new_folder = Folder(
        #     name=folder.name,
        #     parent_id=folder.parent_id,
        #     user_id=current_user.userId,
        #     created_at=datetime.utcnow(),
        #     updated_at=datetime.utcnow()
        # )
        # db.add(new_folder)
        # db.commit()
        # db.refresh(new_folder)
        
        # For now, simulate folder creation
        folder_id = 1  # Simulated ID
        path = build_folder_path(db, folder_id) if folder.parent_id else f"/{folder.name}"
        
        log_response(201, f"Folder created: {folder.name}", "/folders/")
        
        return FolderResponse(
            id=folder_id,
            name=folder.name,
            parent_id=folder.parent_id,
            path=path,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            file_count=0,
            subfolder_count=0,
            size_bytes=0
        )
        
    except APIError as ae:
        log_error(f"APIError creating folder: {ae.message}")
        raise handle_api_error(ae)
    except Exception as e:
        log_error(f"Error creating folder: {str(e)}")
        raise handle_api_error(APIError(
            ErrorCode.SYSTEM_1701,
            "Failed to create folder",
            status.HTTP_500_INTERNAL_SERVER_ERROR
        ))

@router.get("/", response_model=List[FolderResponse])
async def list_folders(
    parent_id: Optional[int] = Query(None, description="Filter by parent folder"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> List[FolderResponse]:
    """List folders for the current user.
    
    Args:
        parent_id: Optional parent folder ID to filter by
        
    Returns:
        List of folders
    """
    try:
        log_request_start("/folders/", "GET", None, None, current_user.userId)
        
        # In production, query from database
        # query = db.query(Folder).filter(Folder.user_id == current_user.userId)
        # if parent_id is not None:
        #     query = query.filter(Folder.parent_id == parent_id)
        # folders = query.order_by(Folder.name).all()
        
        # For now, simulate folders
        folders = []
        if parent_id is None:
            # Root level folders
            folders = [
                {
                    "id": 1,
                    "name": "IMU Data",
                    "parent_id": None,
                    "created_at": datetime.utcnow(),
                    "updated_at": datetime.utcnow(),
                    "file_count": 15,
                    "subfolder_count": 2,
                    "size_bytes": 1024000
                },
                {
                    "id": 2,
                    "name": "CSI Measurements",
                    "parent_id": None,
                    "created_at": datetime.utcnow(),
                    "updated_at": datetime.utcnow(),
                    "file_count": 8,
                    "subfolder_count": 0,
                    "size_bytes": 512000
                }
            ]
        
        folder_responses = []
        for folder in folders:
            path = build_folder_path(db, folder["id"]) if folder["parent_id"] else f"/{folder['name']}"
            folder_responses.append(FolderResponse(
                id=folder["id"],
                name=folder["name"],
                parent_id=folder["parent_id"],
                path=path,
                created_at=folder["created_at"],
                updated_at=folder["updated_at"],
                file_count=folder["file_count"],
                subfolder_count=folder["subfolder_count"],
                size_bytes=folder["size_bytes"]
            ))
        
        log_response(200, f"Retrieved {len(folder_responses)} folders", "/folders/")
        
        return folder_responses
        
    except Exception as e:
        log_error(f"Error listing folders: {str(e)}")
        raise handle_api_error(APIError(
            ErrorCode.SYSTEM_1701,
            "Failed to list folders",
            status.HTTP_500_INTERNAL_SERVER_ERROR
        ))

@router.get("/{folder_id}", response_model=FolderResponse)
async def get_folder(
    folder_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> FolderResponse:
    """Get folder details.
    
    Args:
        folder_id: Folder ID
        
    Returns:
        Folder details
    """
    try:
        log_request_start(f"/folders/{folder_id}", "GET", None, None, current_user.userId)
        
        # In production, query from database
        # folder = db.query(Folder).filter(
        #     Folder.id == folder_id,
        #     Folder.user_id == current_user.userId
        # ).first()
        
        # if not folder:
        #     raise handle_api_error(not_found_error("Folder", str(folder_id)))
        
        # For now, simulate folder
        if folder_id == 1:
            folder_data = {
                "id": 1,
                "name": "IMU Data",
                "parent_id": None,
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow(),
                "file_count": 15,
                "subfolder_count": 2,
                "size_bytes": 1024000
            }
        else:
            raise handle_api_error(not_found_error("Folder", str(folder_id)))
        
        path = build_folder_path(db, folder_id) if folder_data["parent_id"] else f"/{folder_data['name']}"
        
        return FolderResponse(
            id=folder_data["id"],
            name=folder_data["name"],
            parent_id=folder_data["parent_id"],
            path=path,
            created_at=folder_data["created_at"],
            updated_at=folder_data["updated_at"],
            file_count=folder_data["file_count"],
            subfolder_count=folder_data["subfolder_count"],
            size_bytes=folder_data["size_bytes"]
        )
        
    except APIError as ae:
        log_error(f"APIError getting folder: {ae.message}")
        raise handle_api_error(ae)
    except Exception as e:
        log_error(f"Error getting folder: {str(e)}")
        raise handle_api_error(APIError(
            ErrorCode.SYSTEM_1701,
            "Failed to get folder",
            status.HTTP_500_INTERNAL_SERVER_ERROR
        ))

@router.put("/{folder_id}", response_model=FolderResponse)
async def update_folder(
    folder_id: int,
    folder_update: FolderUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> FolderResponse:
    """Update folder details.
    
    Args:
        folder_id: Folder ID
        folder_update: Updated folder data
        
    Returns:
        Updated folder information
    """
    try:
        log_request_start(f"/folders/{folder_id}", "PUT", None, None, current_user.userId)
        
        # In production, update in database
        # folder = db.query(Folder).filter(
        #     Folder.id == folder_id,
        #     Folder.user_id == current_user.userId
        # ).first()
        
        # if not folder:
        #     raise handle_api_error(not_found_error("Folder", str(folder_id)))
        
        # if folder_update.name:
        #     folder.name = folder_update.name
        # if folder_update.parent_id is not None:
        #     folder.parent_id = folder_update.parent_id
        # folder.updated_at = datetime.utcnow()
        # db.commit()
        # db.refresh(folder)
        
        # For now, simulate update
        folder_data = {
            "id": folder_id,
            "name": folder_update.name or "Updated Folder",
            "parent_id": folder_update.parent_id,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
            "file_count": 15,
            "subfolder_count": 2,
            "size_bytes": 1024000
        }
        
        path = build_folder_path(db, folder_id) if folder_data["parent_id"] else f"/{folder_data['name']}"
        
        log_response(200, f"Folder updated: {folder_data['name']}", f"/folders/{folder_id}")
        
        return FolderResponse(
            id=folder_data["id"],
            name=folder_data["name"],
            parent_id=folder_data["parent_id"],
            path=path,
            created_at=folder_data["created_at"],
            updated_at=folder_data["updated_at"],
            file_count=folder_data["file_count"],
            subfolder_count=folder_data["subfolder_count"],
            size_bytes=folder_data["size_bytes"]
        )
        
    except APIError as ae:
        log_error(f"APIError updating folder: {ae.message}")
        raise handle_api_error(ae)
    except Exception as e:
        log_error(f"Error updating folder: {str(e)}")
        raise handle_api_error(APIError(
            ErrorCode.SYSTEM_1701,
            "Failed to update folder",
            status.HTTP_500_INTERNAL_SERVER_ERROR
        ))

@router.delete("/{folder_id}")
async def delete_folder(
    folder_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """Delete a folder.
    
    Args:
        folder_id: Folder ID
        
    Returns:
        Deletion confirmation
    """
    try:
        log_request_start(f"/folders/{folder_id}", "DELETE", None, None, current_user.userId)
        
        # In production, check if folder exists and belongs to user
        # folder = db.query(Folder).filter(
        #     Folder.id == folder_id,
        #     Folder.user_id == current_user.userId
        # ).first()
        
        # if not folder:
        #     raise handle_api_error(not_found_error("Folder", str(folder_id)))
        
        # Check if folder is empty
        # file_count = db.query(File).filter(File.folder_id == folder_id).count()
        # subfolder_count = db.query(Folder).filter(Folder.parent_id == folder_id).count()
        
        # if file_count > 0 or subfolder_count > 0:
        #     raise handle_api_error(APIError(
        #         ErrorCode.RESOURCE_IN_USE,
        #         "Cannot delete folder: it contains files or subfolders",
        #         status.HTTP_400_BAD_REQUEST
        #     ))
        
        # Delete folder
        # db.delete(folder)
        # db.commit()
        
        log_response(200, f"Folder deleted: {folder_id}", f"/folders/{folder_id}")
        
        return {
            "success": True,
            "message": "Folder deleted successfully",
            "folder_id": folder_id
        }
        
    except APIError as ae:
        log_error(f"APIError deleting folder: {ae.message}")
        raise handle_api_error(ae)
    except Exception as e:
        log_error(f"Error deleting folder: {str(e)}")
        raise handle_api_error(APIError(
            ErrorCode.SYSTEM_1701,
            "Failed to delete folder",
            status.HTTP_500_INTERNAL_SERVER_ERROR
        ))

@router.post("/move-file", response_model=Dict[str, Any])
async def move_file_to_folder(
    request: FileMoveRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """Move a file to a folder.
    
    Args:
        request: File move request
        
    Returns:
        Move confirmation
    """
    try:
        log_request_start("/folders/move-file", "POST", None, None, current_user.userId)
        
        # Get file
        file_record = db.query(File).filter(
            File.fileId == request.file_id,
            File.userId == current_user.userId
        ).first()
        
        if not file_record:
            raise handle_api_error(not_found_error("File", str(request.file_id)))
        
        # Validate folder if specified
        if request.folder_id is not None:
            # In production, check if folder exists and belongs to user
            # folder = db.query(Folder).filter(
            #     Folder.id == request.folder_id,
            #     Folder.user_id == current_user.userId
            # ).first()
            
            # if not folder:
            #     raise handle_api_error(not_found_error("Folder", str(request.folder_id)))
            pass
        
        # Update file folder (add folder_id column to File model)
        # file_record.folder_id = request.folder_id
        # file_record.updated_at = datetime.utcnow()
        # db.commit()
        
        # For now, simulate move
        folder_name = "root" if request.folder_id is None else f"folder_{request.folder_id}"
        
        log_response(200, f"File {request.file_id} moved to {folder_name}", "/folders/move-file")
        
        return {
            "success": True,
            "message": f"File moved to {folder_name}",
            "file_id": request.file_id,
            "folder_id": request.folder_id
        }
        
    except APIError as ae:
        log_error(f"APIError moving file: {ae.message}")
        raise handle_api_error(ae)
    except Exception as e:
        log_error(f"Error moving file: {str(e)}")
        raise handle_api_error(APIError(
            ErrorCode.SYSTEM_1701,
            "Failed to move file",
            status.HTTP_500_INTERNAL_SERVER_ERROR
        ))
