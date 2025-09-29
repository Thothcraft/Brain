#!/usr/bin/env python3

"""
Memory Management Module

This module provides functions for managing AI agent memory.
It handles both short-term and long-term memory storage and retrieval.

Example usage:
    >>> from aiagent.memory import update_short_term_memory, get_memory_context
    >>> update_short_term_memory("What's the weather?", "Sunny", "Weather query")
    >>> context = get_memory_context()
"""

import os
import logging
from pathlib import Path

# Get the module's directory
MODULE_DIR = Path(__file__).parent
AI_AGENT_DIR = Path(__file__).parent.parent

# Check if we're in a serverless environment (like Vercel)
IN_SERVERLESS = os.environ.get("VERCEL") is not None

# Use /tmp for Vercel environments since it's the only writable directory
if IN_SERVERLESS:
    CLIENT_DIR = "/tmp/aiagent"
    logging.info(f"Using temporary directory for serverless environment: {CLIENT_DIR}")
else:
    CLIENT_DIR = AI_AGENT_DIR

DATA_DIR = os.path.join(CLIENT_DIR, "data")

REFERENCES_DIR = os.path.join(DATA_DIR,  "references")

# Create necessary directories if they don't exist
if IN_SERVERLESS:
    # In serverless environments, always create directories in /tmp which is writable
    try:
        os.makedirs(DATA_DIR, exist_ok=True)
        os.makedirs(REFERENCES_DIR, exist_ok=True)
        logging.info(f"Created directory structure in /tmp: {DATA_DIR} and {REFERENCES_DIR}")
    except Exception as e:
        logging.error(f"Failed to create directories in /tmp: {e}")
else:
    # In regular environments, try to create directories but handle failures gracefully
    try:
        os.makedirs(DATA_DIR, exist_ok=True)
        os.makedirs(REFERENCES_DIR, exist_ok=True)
        logging.info(f"Created directory structure: {DATA_DIR} and {REFERENCES_DIR}")
    except Exception as e:
        logging.warning(f"Could not create directory structure: {e}")


from aiagent.memory.memory_manager import BaseMemoryManager, ShortTermMemoryManager, LongTermMemoryManager


__all__ = [
    "SHORT_TERM_MEMORY_FILE",
    "LONG_TERM_MEMORY_FILE",
    "BaseMemoryManager",
    "ShortTermMemoryManager",
    "LongTermMemoryManager",
]
