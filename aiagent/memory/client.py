
import json
import os
import logging
from typing import Optional

import aiagent.memory as memory


def update_client(client_dir: Optional[str] = None) -> None:
    logging.info(f"[update_client] Called with client_dir={client_dir}")
    """Update the client directory for memory management."""
    
    if client_dir:
        memory.CLIENT_DIR = client_dir
        memory.DATA_DIR = os.path.join(client_dir, "data")
        memory.REFERENCES_DIR = os.path.join(client_dir, "references")
        memory.LONG_TERM_MEMORY_FILE = os.path.join(memory.DATA_DIR, "long_term_memory.json")
        memory.SHORT_TERM_MEMORY_FILE = os.path.join(memory.DATA_DIR, "short_term_memory.json")
        memory.CONTEXT_FILE = os.path.join(memory.DATA_DIR, "context.json")
        logging.info(f"[update_client] Paths set: CLIENT_DIR={memory.CLIENT_DIR}, DATA_DIR={memory.DATA_DIR}, LONG_TERM_MEMORY_FILE={memory.LONG_TERM_MEMORY_FILE}, SHORT_TERM_MEMORY_FILE={memory.SHORT_TERM_MEMORY_FILE}, CONTEXT_FILE={memory.CONTEXT_FILE}")

    # make sure the directory exists
    os.makedirs(memory.CLIENT_DIR, exist_ok=True)
    os.makedirs(memory.DATA_DIR, exist_ok=True)
    os.makedirs(memory.REFERENCES_DIR, exist_ok=True)

    # make sure long-term and short-term memory files exist
    if not os.path.exists(memory.LONG_TERM_MEMORY_FILE):
        with open(memory.LONG_TERM_MEMORY_FILE, 'w') as f:
            json.dump({}, f)
    if not os.path.exists(memory.SHORT_TERM_MEMORY_FILE):
        with open(memory.SHORT_TERM_MEMORY_FILE, 'w') as f:
            json.dump({}, f)

    # make sure context file exists
    if not os.path.exists(memory.CONTEXT_FILE):
        with open(memory.CONTEXT_FILE, 'w') as f:
            json.dump({}, f)

 