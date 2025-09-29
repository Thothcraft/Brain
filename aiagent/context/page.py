#!/usr/bin/env python3

"""
Webpage Content Processing Module

This module handles the processing of webpage content for the AI Agent.
It provides functions to save, retrieve, and analyze webpage content.
"""

import json
import logging
import os
from typing import Any, Dict, Optional

from aiagent.context.extractor import extract_fields_from_page

import aiagent.memory as memory


def save_page_content(
    html_content: Optional[str] = None
) -> bool:
    """Save webpage content to the references directory.

    This function stores HTML content of a webpage for context.
    It also extracts structured data and saves it separately.
    This file provides context to the AI for answering queries about the current page.

    Args:
        
        html_content (str, optional): The HTML content of the webpage for structured extraction.

    Returns:
        bool: True if operation was successful, False otherwise

    Example:
        >>> success = save_page_content("This is page text", "<html><body>This is page text</body></html>")
        >>> print(f"Page content saved: {success}")
        Page content saved: True
    """
    REFERENCES_DIR = memory.REFERENCES_DIR
    # Create references directory if it doesn't exist
    if not os.path.exists(REFERENCES_DIR):
        logging.info(f"Creating references directory: {REFERENCES_DIR}")
        try:
            os.makedirs(REFERENCES_DIR)
        except OSError as e:
            logging.error(f"Failed to create references directory: {e}")
            return False

    
    html_filepath = os.path.join(REFERENCES_DIR, "current_page_html.html")

    success = True


    # Save the current page HTML content if available
    if html_content:
        try:
            with open(html_filepath, "w", encoding="utf-8") as f:
                f.write(html_content)
            logging.info(
                f"Saved page HTML content ({len(html_content)} chars) to {html_filepath}"
            )

            # Also extract structured data from HTML and save
            structured_data = extract_fields_from_page(html_content)
            if structured_data:
                struct_filepath = os.path.join(
                    REFERENCES_DIR, "structured_page_data.json"
                )
                with open(struct_filepath, "w", encoding="utf-8") as f:
                    json.dump(structured_data, f, indent=2)
                logging.info(f"Saved structured page data to {struct_filepath}")
        except Exception as e:
            logging.error(f"Error saving page HTML content: {e}")
            success = False

    # If no content provided at all, clean up potential stale files
    if not content and not html_content:
        try:
            for filepath in [text_filepath, html_filepath]:
                if os.path.exists(filepath):
                    os.remove(filepath)
                    logging.info(
                        f"Removed potentially stale page content file: {filepath}"
                    )
            struct_filepath = os.path.join(REFERENCES_DIR, "structured_page_data.json")
            if os.path.exists(struct_filepath):
                os.remove(struct_filepath)
                logging.info(
                    f"Removed potentially stale structured data file: {struct_filepath}"
                )
        except Exception as e:
            logging.error(f"Error removing stale content files: {e}")
            success = False

    return success


