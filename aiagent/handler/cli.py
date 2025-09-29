#!/usr/bin/env python3
"""
AI Agent Command Line Interface

This module provides a professional CLI for the AI agent, allowing users to interact with the agent from the terminal.

Usage examples:
    python -m aiagent.cli "What is the capital of France?"
    python -m aiagent.cli --file webpage.txt "Summarize this webpage"
    python -m aiagent.cli --html page.html "Analyze this HTML page"
    python -m aiagent.cli --json "Tell me about my profile"
"""
import argparse
import logging
import sys
import os
from typing import Optional

from aiagent.handler.query import ask_ai
import json
from aiagent.memory.memory_manager import LongTermMemoryManager, ShortTermMemoryManager
from aiagent.context import read_references, save_page_content

MODULE_PATH = os.path.dirname(os.path.abspath(__file__))
BOT_PATH = os.path.dirname(MODULE_PATH)
DATA_PATH = os.path.join(BOT_PATH, "data")

def setup_logging(verbose: bool = False) -> None:
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(asctime)s - %(levelname)s - %(message)s")

def read_file_content(filepath: str) -> Optional[str]:
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        logging.error(f"Error reading file {filepath}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(
        description="AI Agent Command Line Interface",
        epilog="Example: python -m aiagent.cli --json 'Tell me about my profile'"
    )
    parser.add_argument("query", help="The query to send to the AI")
    parser.add_argument("--file", help="Path to a file containing webpage or text content for context")
    parser.add_argument("--json", action="store_true", help="Output in JSON format")
    parser.add_argument("--html", action="store_true", help="Output in HTML format")
    parser.add_argument("--max-tokens", type=int, default=1024, help="Maximum tokens in response")
    parser.add_argument("--temperature", type=float, default=0.7, help="Response randomness (0.0-1.0)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--output", nargs="?", const=True, help="Path to save the output to a file. If used without a filename, saves to data/context.json")
    args = parser.parse_args()

    setup_logging(args.verbose)

    def format_output(query, response, as_json=False):
        if as_json:
            return json.dumps({"query": query, "response": response}, indent=2, ensure_ascii=False)
        return response

    aux_data = {}
    if args.file:
        # cli --file $TEST_FILE "what can you see in this python file?"
        print("received files")
        print(args.file) 
        references = read_references(file_paths=[args.file])
        aux_data[args.file] = references.get(args.file)
        
    
    try:
        response = ask_ai(
            query=args.query,
            aux_data=aux_data,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        output = format_output(args.query, response, args.json)
        output_path = None
        if args.output:
            if args.output is True:
                output_path = os.path.join(DATA_PATH, "context.json")
            else:
                output_path = args.output
        if output_path:
            try:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(output)
                logging.info(f"Output saved to {output_path}")
            except Exception as e:
                logging.error(f"Error writing to output file: {e}")
                print(output)
        else:
            print(output)
        return 0
    except Exception as e:
        logging.error(f"Error in AI agent: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
