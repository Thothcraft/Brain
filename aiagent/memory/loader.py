#!/usr/bin/env python3

"""
Memory Loading Module

This module handles loading memory data from storage files.
It provides functions to safely load memory data and handle errors.
:noindex:
"""

import json
import logging
import os
from datetime import datetime
from typing import Any, Dict

# Import memory file paths from package __init__
import aiagent.memory as memory


def load_memory(memory_type: str) -> Dict[str, Any]:
    """Load memory data from a JSON file.

    Args:
        memory_type (str): Type of memory to load. Must be either 'short-term' or 'long-term'.

    Returns:
        dict: Dictionary containing the memory data

    Example:
        >>> memory = load_memory("short-term")
        >>> if "conversations" in memory:
        ...     print(f"Loaded {len(memory['conversations'])} conversations")

    :noindex:
    """
    logging.info(f"[load_memory] Requested memory_type={memory_type}")
    # Log the file path being used
    if memory_type == "short-term":
        logging.info(f"[load_memory] Using file: {memory.SHORT_TERM_MEMORY_FILE}")
    elif memory_type == "long-term":
        logging.info(f"[load_memory] Using file: {memory.LONG_TERM_MEMORY_FILE}")
    if memory_type == "short-term":
        filepath = memory.SHORT_TERM_MEMORY_FILE
    elif memory_type == "long-term":
        filepath = memory.LONG_TERM_MEMORY_FILE
    else:
        raise ValueError(f"Invalid memory type: {memory_type}")
        
    # Check if we're in a serverless environment
    if os.environ.get("VERCEL"):
        # If we're trying to load from a file in the temporary directory but it doesn't exist,
        # try checking if there's an original file in the /var/task directory
        tmp_path = filepath
        if tmp_path.startswith("/tmp/") and not os.path.exists(tmp_path):
            # Try to find an original file in /var/task
            original_path = tmp_path.replace("/tmp/", "/var/task/")
            if os.path.exists(original_path) and os.path.isfile(original_path):
                logging.info(f"[load_memory] Found original file at {original_path}, will load from there")
                filepath = original_path
    
    try:
        logging.info(f"[load_memory] Loading from filepath={filepath}")
        with open(filepath, 'r', encoding='utf-8') as f:
            memory_data = json.load(f)
        logging.info(f"[load_memory] Loaded data from {filepath} (keys: {list(memory_data.keys())})")
        logging.info(f"Successfully loaded {memory_type} memory")
        return memory_data
    except FileNotFoundError:
        # If file doesn't exist, create it with default empty structure
        logging.warning(f"{memory_type} memory file not found: {filepath}")
        if memory_type == "short-term":
            default_memory = {
                "conversations": [],
                "active_url": {},
                "test": None,
                "preferences": {}
            }
        else:  # long-term
            default_memory = {
                "user_profile": {},
                "preferences": {},
                "long_term_goals": {},
                "last_updated": datetime.now().isoformat(),
            }
        save_memory(default_memory, memory_type)
        return default_memory
        
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON in {memory_type} memory: {str(e)}")
        raise
    except Exception as e:
        logging.error(f"Error loading {memory_type} memory: {str(e)}")
        raise
