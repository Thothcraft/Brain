#!/usr/bin/env python3

"""
Handler package for AI agent.
This file marks the handler directory as a Python package.
"""
__version__ = "0.1.0"

import os
from pathlib import Path

# Get the module's directory
MODULE_DIR = Path(__file__).parent
AI_AGENT_DIR = Path(__file__).parent.parent

# Ensure memory directory exists
import aiagent.memory as memory
os.makedirs(memory.DATA_DIR, exist_ok=True)

# Import public functions and variables for module-level access
from aiagent.handler.cli import main
from aiagent.handler.query import ask_ai

__all__ = [
    "main",
    "ask_ai",
]
    