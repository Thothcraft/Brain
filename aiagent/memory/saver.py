#!/usr/bin/env python3

"""
Memory Saving Module

This module handles saving memory data to storage files.
It provides functions to safely persist memory data and handle errors.
"""

import json
import logging
import os
from typing import Any, Dict

import aiagent.memory as memory

def save_memory(memory: Dict[str, Any], filepath: str) -> bool:
    """Save memory (short-term or long-term) to a JSON file.

    Ensures proper formatting and error handling when saving memory data.

    Args:
        memory (Dict[str, Any]): Memory data to save.
        filepath (str): Path to the file where memory should be saved.

    Returns:
        bool: True if save was successful, False otherwise

    Example:
        >>> memory_data = {"conversations": [{"query": "Hello", "response": "Hi there!"}]}
        >>> success = save_memory(memory_data, "short-term")
        >>> print(f"Memory saved successfully: {success}")

    :noindex:
    """
    # Check if we're in a serverless environment with read-only filesystem
    if os.environ.get("VERCEL"):
        # Replace any filepath in the read-only /var/task directory with a path in /tmp
        if filepath.startswith("/var/task/"):
            original_path = filepath
            filepath = filepath.replace("/var/task/", "/tmp/")
            logging.info(f"[save_memory] In serverless environment, redirecting save from {original_path} to {filepath}")
    
    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # Write the memory data
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(memory, f, indent=2)
        logging.info(f"[save_memory] Successfully saved memory to {filepath}")
        return True
    except OSError as e:
        logging.error(f"Error saving memory to {filepath}: {e}")
        # If in Vercel and still getting errors, provide a more specific message
        if os.environ.get("VERCEL"):
            logging.warning("This may be due to serverless filesystem restrictions")
        return False
            