"""Curriculum and Education Management Endpoints.

This module handles:
- Course content delivery
- Student progress tracking
- Lab management
- Educational resources
"""

from fastapi import APIRouter, HTTPException, Query, File, UploadFile
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from datetime import datetime, timedelta
from enum import Enum
import uuid

# Import shared models
from .models import StandardResponse

router = APIRouter(tags=["curriculum"])

# ============================================================================
# ENUMS
# ============================================================================

class ModuleType(str, Enum):
    """Types of curriculum modules."""
    LESSON = "lesson"
    LAB = "lab"
    PROJECT = "project"
    QUIZ = "quiz"
    ASSIGNMENT = "assignment"

class ContentType(str, Enum):
    """Types of educational content."""
    VIDEO = "video"
    PDF = "pdf"
    NOTEBOOK = "notebook"
    SLIDES = "slides"
    CODE = "code"
    INTERACTIVE = "interactive"

class DifficultyLevel(str, Enum):
    """Course difficulty levels."""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"

class ProgressStatus(str, Enum):
    """Student progress status."""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    REVIEWED = "reviewed"

# ============================================================================
# MODELS
# ============================================================================

class CurriculumModule(BaseModel):
    """Educational module definition."""
    module_id: str = Field(..., description="Unique module identifier")
    title: str = Field(..., description="Module title")
    description: str = Field(..., description="Module description")
    type: ModuleType = Field(..., description="Module type")
    difficulty: DifficultyLevel = Field(..., description="Difficulty level")
    duration_minutes: int = Field(..., description="Estimated duration")
    prerequisites: List[str] = Field([], description="Prerequisite module IDs")
    learning_objectives: List[str] = Field(..., description="Learning objectives")
    content_items: List[Dict[str, Any]] = Field(..., description="Content items")
    tags: List[str] = Field([], description="Topic tags")
    order: int = Field(..., description="Order in curriculum")

class ContentItem(BaseModel):
    """Individual content item within a module."""
    item_id: str = Field(..., description="Content item ID")
    title: str = Field(..., description="Content title")
    type: ContentType = Field(..., description="Content type")
    url: Optional[str] = Field(None, description="Content URL")
    file_path: Optional[str] = Field(None, description="File path for downloads")
    duration_minutes: Optional[int] = Field(None, description="Duration for videos")
    interactive_config: Optional[Dict[str, Any]] = Field(None, description="Config for interactive content")

class StudentProgress(BaseModel):
    """Student progress tracking."""
    student_id: str = Field(..., description="Student identifier")
    module_id: str = Field(..., description="Module identifier")
    status: ProgressStatus = Field(..., description="Progress status")
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    time_spent_minutes: int = Field(0, description="Total time spent")
    score: Optional[float] = Field(None, description="Quiz/assignment score")
    attempts: int = Field(0, description="Number of attempts")
    notes: Optional[str] = Field(None, description="Student notes")
    bookmarks: List[str] = Field([], description="Bookmarked sections")

class LabSubmission(BaseModel):
    """Lab work submission."""
    submission_id: str = Field(..., description="Submission ID")
    student_id: str = Field(..., description="Student ID")
    lab_id: str = Field(..., description="Lab module ID")
    code: Optional[str] = Field(None, description="Submitted code")
    notebook_path: Optional[str] = Field(None, description="Jupyter notebook path")
    output: Optional[str] = Field(None, description="Execution output")
    device_data: Optional[Dict[str, Any]] = Field(None, description="Thoth device data")
    submitted_at: datetime = Field(default_factory=datetime.now)
    grade: Optional[float] = Field(None, description="Lab grade")
    feedback: Optional[str] = Field(None, description="Instructor feedback")

class Course(BaseModel):
    """Complete course structure."""
    course_id: str = Field(..., description="Course identifier")
    title: str = Field(..., description="Course title")
    description: str = Field(..., description="Course description")
    instructor: str = Field(..., description="Instructor name")
    difficulty: DifficultyLevel = Field(..., description="Overall difficulty")
    duration_hours: int = Field(..., description="Total course duration")
    modules: List[str] = Field(..., description="Module IDs in order")
    enrolled_students: int = Field(0, description="Number of enrolled students")
    rating: float = Field(0.0, description="Course rating")
    tags: List[str] = Field([], description="Course tags")
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

# ============================================================================
# IN-MEMORY STORAGE (Replace with database in production)
# ============================================================================

# Curriculum modules
curriculum_modules: Dict[str, CurriculumModule] = {}

# Courses
courses: Dict[str, Course] = {}

# Student progress
student_progress: Dict[str, List[StudentProgress]] = {}

# Lab submissions
lab_submissions: Dict[str, List[LabSubmission]] = {}

# Initialize with sample curriculum
def init_sample_curriculum():
    """Initialize sample curriculum for Thoth education."""
    
    # Module 1: Introduction to Thoth
    module1 = CurriculumModule(
        module_id="mod_001",
        title="Introduction to Thoth Device",
        description="Learn about the Thoth IoT device, Raspberry Pi, Sense HAT, and PiSugar",
        type=ModuleType.LESSON,
        difficulty=DifficultyLevel.BEGINNER,
        duration_minutes=45,
        prerequisites=[],
        learning_objectives=[
            "Understand Thoth device architecture",
            "Learn about Sense HAT sensors",
            "Configure PiSugar power management"
        ],
        content_items=[
            {
                "type": "video",
                "title": "Thoth Device Overview",
                "url": "https://youtube.com/watch?v=example1",
                "duration_minutes": 15
            },
            {
                "type": "pdf",
                "title": "Hardware Specifications",
                "file_path": "/content/thoth_specs.pdf"
            },
            {
                "type": "interactive",
                "title": "3D Device Model",
                "url": "https://poly.cam/capture/ABE69FEA-A1DF-4CC5-BC65-CF1DB40BFEE8/embed"
            }
        ],
        tags=["iot", "hardware", "raspberry-pi"],
        order=1
    )
    
    # Module 2: WiFi Setup
    module2 = CurriculumModule(
        module_id="mod_002",
        title="WiFi Configuration and Network Setup",
        description="Configure WiFi, understand captive portals, and network management",
        type=ModuleType.LAB,
        difficulty=DifficultyLevel.BEGINNER,
        duration_minutes=60,
        prerequisites=["mod_001"],
        learning_objectives=[
            "Configure WiFi via captive portal",
            "Understand network security",
            "Troubleshoot connectivity issues"
        ],
        content_items=[
            {
                "type": "video",
                "title": "WiFi Setup Tutorial",
                "url": "https://youtube.com/watch?v=example2",
                "duration_minutes": 10
            },
            {
                "type": "notebook",
                "title": "Network Configuration Lab",
                "file_path": "/content/wifi_lab.ipynb"
            }
        ],
        tags=["networking", "wifi", "security"],
        order=2
    )
    
    # Module 3: Sensor Data Collection
    module3 = CurriculumModule(
        module_id="mod_003",
        title="Collecting Sensor Data",
        description="Read and analyze data from Sense HAT sensors",
        type=ModuleType.LAB,
        difficulty=DifficultyLevel.INTERMEDIATE,
        duration_minutes=90,
        prerequisites=["mod_002"],
        learning_objectives=[
            "Read temperature, humidity, pressure data",
            "Understand IMU and motion sensors",
            "Visualize sensor data in real-time"
        ],
        content_items=[
            {
                "type": "notebook",
                "title": "Sensor Data Collection",
                "file_path": "/content/sensor_lab.ipynb"
            },
            {
                "type": "code",
                "title": "Python Sensor Scripts",
                "file_path": "/content/sensor_scripts.py"
            }
        ],
        tags=["sensors", "data-collection", "python"],
        order=3
    )
    
    # Module 4: Basic AI/ML
    module4 = CurriculumModule(
        module_id="mod_004",
        title="Introduction to AI/ML on Edge Devices",
        description="Train and deploy basic ML models on Thoth device",
        type=ModuleType.LESSON,
        difficulty=DifficultyLevel.INTERMEDIATE,
        duration_minutes=120,
        prerequisites=["mod_003"],
        learning_objectives=[
            "Understand edge AI concepts",
            "Train simple models with sensor data",
            "Deploy models to Thoth device"
        ],
        content_items=[
            {
                "type": "video",
                "title": "Edge AI Fundamentals",
                "url": "https://youtube.com/watch?v=example4",
                "duration_minutes": 30
            },
            {
                "type": "notebook",
                "title": "Training Your First Model",
                "file_path": "/content/ml_basics.ipynb"
            }
        ],
        tags=["ai", "machine-learning", "edge-computing"],
        order=4
    )
    
    # Module 5: Federated Learning
    module5 = CurriculumModule(
        module_id="mod_005",
        title="Federated Learning with Multiple Devices",
        description="Implement privacy-preserving collaborative learning",
        type=ModuleType.PROJECT,
        difficulty=DifficultyLevel.ADVANCED,
        duration_minutes=180,
        prerequisites=["mod_004"],
        learning_objectives=[
            "Understand federated learning principles",
            "Implement Flower framework",
            "Apply differential privacy"
        ],
        content_items=[
            {
                "type": "pdf",
                "title": "Federated Learning Theory",
                "file_path": "/content/federated_theory.pdf"
            },
            {
                "type": "notebook",
                "title": "Federated Learning Implementation",
                "file_path": "/content/federated_lab.ipynb"
            }
        ],
        tags=["federated-learning", "privacy", "distributed-ai"],
        order=5
    )
    
    # Module 6: Capstone Project
    module6 = CurriculumModule(
        module_id="mod_006",
        title="Capstone: Smart Home Monitor",
        description="Build a complete smart home monitoring system",
        type=ModuleType.PROJECT,
        difficulty=DifficultyLevel.ADVANCED,
        duration_minutes=300,
        prerequisites=["mod_005"],
        learning_objectives=[
            "Design end-to-end IoT solution",
            "Integrate all learned concepts",
            "Deploy production-ready system"
        ],
        content_items=[
            {
                "type": "pdf",
                "title": "Project Requirements",
                "file_path": "/content/capstone_requirements.pdf"
            },
            {
                "type": "notebook",
                "title": "Project Starter Code",
                "file_path": "/content/capstone_starter.ipynb"
            }
        ],
        tags=["project", "iot", "smart-home"],
        order=6
    )
    
    # Store modules
    for module in [module1, module2, module3, module4, module5, module6]:
        curriculum_modules[module.module_id] = module
    
    # Create course
    course = Course(
        course_id="course_001",
        title="AI & IoT with Thoth",
        description="Complete course on building AI-powered IoT applications with Thoth device",
        instructor="Thoth Team",
        difficulty=DifficultyLevel.INTERMEDIATE,
        duration_hours=15,
        modules=["mod_001", "mod_002", "mod_003", "mod_004", "mod_005", "mod_006"],
        enrolled_students=0,
        rating=4.8,
        tags=["iot", "ai", "edge-computing", "raspberry-pi"]
    )
    courses[course.course_id] = course

# Initialize curriculum on module load
init_sample_curriculum()

# ============================================================================
# ENDPOINTS
# ============================================================================

@router.get("/curriculum", response_model=Dict[str, Any])
async def get_curriculum(
    course_id: Optional[str] = Query(None, description="Specific course ID"),
    module_type: Optional[ModuleType] = Query(None, description="Filter by module type"),
    difficulty: Optional[DifficultyLevel] = Query(None, description="Filter by difficulty"),
    tags: Optional[str] = Query(None, description="Comma-separated tags")
):
    """Fetch curriculum modules and course content.
    
    Returns structured curriculum with videos, PDFs, notebooks, and interactive content.
    """
    try:
        if course_id:
            if course_id not in courses:
                raise HTTPException(status_code=404, detail=f"Course {course_id} not found")
            
            course = courses[course_id]
            
            # Get all modules for the course
            course_modules = []
            for module_id in course.modules:
                if module_id in curriculum_modules:
                    module = curriculum_modules[module_id]
                    
                    # Apply filters
                    if module_type and module.type != module_type:
                        continue
                    if difficulty and module.difficulty != difficulty:
                        continue
                    if tags:
                        tag_list = tags.split(",")
                        if not any(tag in module.tags for tag in tag_list):
                            continue
                    
                    course_modules.append(module.model_dump())
            
            return {
                "success": True,
                "course": course.model_dump(),
                "modules": course_modules,
                "total_modules": len(course_modules)
            }
        else:
            # Return all modules with filters
            filtered_modules = []
            
            for module_id, module in curriculum_modules.items():
                # Apply filters
                if module_type and module.type != module_type:
                    continue
                if difficulty and module.difficulty != difficulty:
                    continue
                if tags:
                    tag_list = tags.split(",")
                    if not any(tag in module.tags for tag in tag_list):
                        continue
                
                filtered_modules.append(module.model_dump())
            
            # Sort by order
            filtered_modules.sort(key=lambda x: x["order"])
            
            return {
                "success": True,
                "modules": filtered_modules,
                "total_modules": len(filtered_modules),
                "courses": [c.model_dump() for c in courses.values()]
            }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch curriculum: {str(e)}")

@router.get("/curriculum/{module_id}", response_model=CurriculumModule)
async def get_module_details(module_id: str):
    """Get detailed information about a specific curriculum module."""
    try:
        if module_id not in curriculum_modules:
            raise HTTPException(status_code=404, detail=f"Module {module_id} not found")
        
        return curriculum_modules[module_id]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch module: {str(e)}")

@router.post("/curriculum/progress", response_model=StandardResponse)
async def update_progress(progress: StudentProgress):
    """Update student progress for a curriculum module.
    
    Tracks completion status, time spent, scores, and learning analytics.
    """
    try:
        # Validate module exists
        if progress.module_id not in curriculum_modules:
            raise HTTPException(status_code=404, detail=f"Module {progress.module_id} not found")
        
        # Initialize student progress list if needed
        if progress.student_id not in student_progress:
            student_progress[progress.student_id] = []
        
        # Check if progress already exists for this module
        existing = None
        for p in student_progress[progress.student_id]:
            if p.module_id == progress.module_id:
                existing = p
                break
        
        if existing:
            # Update existing progress
            existing.status = progress.status
            existing.time_spent_minutes += progress.time_spent_minutes
            if progress.completed_at:
                existing.completed_at = progress.completed_at
            if progress.score is not None:
                existing.score = progress.score
            existing.attempts = progress.attempts
            if progress.notes:
                existing.notes = progress.notes
            existing.bookmarks = progress.bookmarks
        else:
            # Add new progress record
            if not progress.started_at:
                progress.started_at = datetime.now()
            student_progress[progress.student_id].append(progress)
        
        # Check if module completion unlocks new modules
        unlocked_modules = []
        if progress.status == ProgressStatus.COMPLETED:
            for module_id, module in curriculum_modules.items():
                if progress.module_id in module.prerequisites:
                    # Check if all prerequisites are met
                    all_met = all(
                        any(p.module_id == prereq and p.status == ProgressStatus.COMPLETED 
                            for p in student_progress.get(progress.student_id, []))
                        for prereq in module.prerequisites
                    )
                    if all_met:
                        unlocked_modules.append(module_id)
        
        return StandardResponse(
            success=True,
            message=f"Progress updated for module {progress.module_id}",
            data={
                "student_id": progress.student_id,
                "module_id": progress.module_id,
                "status": progress.status,
                "score": progress.score,
                "unlocked_modules": unlocked_modules
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update progress: {str(e)}")

@router.get("/curriculum/progress/{student_id}", response_model=Dict[str, Any])
async def get_student_progress(
    student_id: str,
    course_id: Optional[str] = Query(None, description="Filter by course")
):
    """Get student's learning progress across all modules."""
    try:
        if student_id not in student_progress:
            return {
                "success": True,
                "student_id": student_id,
                "progress": [],
                "overall_completion": 0,
                "total_time_spent": 0
            }
        
        progress_list = student_progress[student_id]
        
        # Filter by course if specified
        if course_id and course_id in courses:
            course = courses[course_id]
            progress_list = [p for p in progress_list if p.module_id in course.modules]
        
        # Calculate statistics
        total_modules = len(curriculum_modules)
        completed_modules = sum(1 for p in progress_list if p.status == ProgressStatus.COMPLETED)
        total_time = sum(p.time_spent_minutes for p in progress_list)
        
        # Get detailed progress
        detailed_progress = []
        for p in progress_list:
            module = curriculum_modules.get(p.module_id)
            if module:
                detailed_progress.append({
                    "module_id": p.module_id,
                    "module_title": module.title,
                    "module_type": module.type,
                    "status": p.status,
                    "score": p.score,
                    "time_spent": p.time_spent_minutes,
                    "started_at": p.started_at.isoformat() if p.started_at else None,
                    "completed_at": p.completed_at.isoformat() if p.completed_at else None
                })
        
        return {
            "success": True,
            "student_id": student_id,
            "progress": detailed_progress,
            "overall_completion": (completed_modules / total_modules * 100) if total_modules > 0 else 0,
            "completed_modules": completed_modules,
            "total_modules": total_modules,
            "total_time_spent": total_time,
            "average_score": sum(p.score for p in progress_list if p.score) / len([p for p in progress_list if p.score]) if any(p.score for p in progress_list) else None
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get progress: {str(e)}")

@router.post("/curriculum/lab/submit", response_model=StandardResponse)
async def submit_lab(submission: LabSubmission):
    """Submit lab work including code, notebooks, and device data."""
    try:
        # Validate lab module exists
        if submission.lab_id not in curriculum_modules:
            raise HTTPException(status_code=404, detail=f"Lab {submission.lab_id} not found")
        
        module = curriculum_modules[submission.lab_id]
        if module.type != ModuleType.LAB:
            raise ValueError(f"Module {submission.lab_id} is not a lab")
        
        # Generate submission ID if not provided
        if not submission.submission_id:
            submission.submission_id = str(uuid.uuid4())
        
        # Store submission
        if submission.student_id not in lab_submissions:
            lab_submissions[submission.student_id] = []
        
        lab_submissions[submission.student_id].append(submission)
        
        # Auto-grade if possible (mock grading)
        if submission.code or submission.notebook_path:
            submission.grade = 85.0 + random.uniform(-10, 10)  # Mock grade
        
        # Update progress
        progress = StudentProgress(
            student_id=submission.student_id,
            module_id=submission.lab_id,
            status=ProgressStatus.COMPLETED,
            score=submission.grade,
            completed_at=datetime.now(),
            time_spent_minutes=module.duration_minutes
        )
        
        # Store progress
        if submission.student_id not in student_progress:
            student_progress[submission.student_id] = []
        student_progress[submission.student_id].append(progress)
        
        return StandardResponse(
            success=True,
            message=f"Lab submission received for {submission.lab_id}",
            data={
                "submission_id": submission.submission_id,
                "lab_id": submission.lab_id,
                "grade": submission.grade,
                "status": "submitted"
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to submit lab: {str(e)}")

@router.get("/leaderboard", response_model=Dict[str, Any])
async def get_leaderboard(
    course_id: Optional[str] = Query(None, description="Filter by course"),
    limit: int = Query(10, ge=1, le=100, description="Number of top students")
):
    """Get student leaderboard based on progress and scores."""
    try:
        leaderboard = []
        
        for student_id, progress_list in student_progress.items():
            # Filter by course if specified
            if course_id and course_id in courses:
                course = courses[course_id]
                progress_list = [p for p in progress_list if p.module_id in course.modules]
            
            if not progress_list:
                continue
            
            # Calculate metrics
            completed = sum(1 for p in progress_list if p.status == ProgressStatus.COMPLETED)
            avg_score = sum(p.score for p in progress_list if p.score) / len([p for p in progress_list if p.score]) if any(p.score for p in progress_list) else 0
            total_time = sum(p.time_spent_minutes for p in progress_list)
            
            leaderboard.append({
                "student_id": student_id,
                "completed_modules": completed,
                "average_score": avg_score,
                "total_time_minutes": total_time,
                "rank_score": completed * 100 + avg_score  # Composite score
            })
        
        # Sort by rank score
        leaderboard.sort(key=lambda x: x["rank_score"], reverse=True)
        
        # Add ranks
        for i, entry in enumerate(leaderboard[:limit], 1):
            entry["rank"] = i
        
        return {
            "success": True,
            "leaderboard": leaderboard[:limit],
            "total_students": len(leaderboard)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get leaderboard: {str(e)}")

# Import for random
import random
