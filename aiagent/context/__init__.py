#!/usr/bin/env python3

"""
Context Processing Package

This package handles reading and processing reference materials and webpage content for the AI Agent.
It provides functions to extract, process, and store context that the AI can use to provide more
informed responses.

Key components:
- reference.py: Handles reading and processing reference materials
- page.py: Manages webpage content processing and extraction
- extractor.py: Contains HTML extraction utilities
"""

from aiagent.context.extractor import extract_fields_from_page
from aiagent.context.page import save_page_content
from aiagent.context.reference import read_references

__all__ = [
    "read_references",
    "REFERENCES_DIR",
    "save_page_content",
    "extract_fields_from_page"
]
