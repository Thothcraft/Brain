"""
Cryptographic utility functions.

This module provides cryptographic helper functions used throughout the application.
"""

import hashlib
from typing import Union

def compute_sha256(data: Union[str, bytes]) -> str:
    """Compute SHA-256 hash of the input data.
    
    Args:
        data: Input data as string or bytes
        
    Returns:
        str: Hexadecimal string representation of the SHA-256 hash
    """
    if isinstance(data, str):
        data = data.encode('utf-8')
    
    return hashlib.sha256(data).hexdigest()
