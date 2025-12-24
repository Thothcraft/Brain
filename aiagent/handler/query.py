#!/usr/bin/env python3

"""
AI Core Module

This module handles the core AI functionality, including:
- Communication with the OPENAI API
- Query generation with context
- Response processing
"""

import logging
import os
import sys
from typing import Any, Dict, List, Optional
import json
from dotenv import load_dotenv
from openai import OpenAI



# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

_MODULE_PATH = os.path.dirname(os.path.abspath(__file__))
_BOT_PATH = os.path.dirname(_MODULE_PATH) # Should be aiagent directory
DEFAULT_DATA_PATH = os.path.join(_BOT_PATH, "data")

from aiagent.memory.memory_manager import BaseMemoryManager, LongTermMemoryManager, ShortTermMemoryManager
from aiagent.context.reference import read_references
from aiagent.functions.registry import FunctionsRegistry


def query_openai(
    query: str,
    long_term_memory: LongTermMemoryManager,
    short_term_memory: ShortTermMemoryManager,
    max_tokens: int = 1024,
    temperature: float = 0.7,
    aux_data: Optional[Dict[str, Any]] = None
) -> str:
    """Send a query to OPENAI's model with context.

    Includes context from:
    - User profile and preferences stored in long_term_memory.json (if provided).
    - Content of files found in the 'references' directory.
    - The conversation will be summarized and stored in short_term_memory.json.

    Args:
        query: The user's question
        long_term_memory: User profile and preferences information
        max_tokens: Maximum tokens in response (default: 1024)
        temperature: Response randomness (0-1, default: 0.7)

    Returns:
        str: The AI's response, or an error message if the query fails.
    # """
    # try:

    # Initialize the Functions Registry
    tools = FunctionsRegistry()
    
    # Log registered functions for debugging
    logging.info(f"[AI Agent] Registered functions: {tools.get_registry_contents()}")

    # Get the callable functions from the registry
    function_map = tools.get_function_callable()
    
    # Initialize OpenAI client
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    if not client.api_key:
        return "Error: No OpenAI API key found. Set OPENAI_API_KEY in .env"

    # Build context messages from long-term memory
    messages = []

    # json to string
    long_term_memory_content = long_term_memory.get_memory_content()


    
    past_conversations_data = short_term_memory.get("conversations")
    past_conversations = json.dumps(past_conversations_data if past_conversations_data is not None else [])
    
    messages.append(
        {"role": "system", "content": "You are a personal assistant answering questions about Gad's portfolio."},
    )    

    messages.append(
        {"role": "system", "content": f"Here are the current user details: {long_term_memory_content}\n\nPast Conversations: {past_conversations}\n"}
    )

    # Add additional context if provided
    if aux_data and aux_data.get("context"):
        context_str = json.dumps(aux_data["context"]) if isinstance(aux_data["context"], dict) else str(aux_data["context"])
        messages.append(
            {"role": "system", "content": f"Additional context for this query: {context_str}"}
        )
        
    # Add user query
    messages.append({"role": "user", "content": query})

    #logging.info(f"Context messages: {messages}")
    
    # Make current user id globally accessible to function tools
    if aux_data and aux_data.get("current_user_id") is not None:
        globals()["CURRENT_USER_ID"] = aux_data["current_user_id"]

    # Query OPENAI
    model_name = os.getenv("MODEL_NAME")
    response = client.chat.completions.create(
        model=model_name, 
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        tools=tools.mapped_functions(),
        tool_choice="auto"
    )

    response_message = response.choices[0].message
    
    # Log if tool calls were requested
    logging.info(f"[AI Agent] Initial response tool_calls: {response_message.tool_calls}")
    logging.info(f"[AI Agent] Initial response content: {response_message.content}")

    # Iterate until model returns a final message without further tool calls
    max_tool_iterations = 5
    iteration = 0
    while True:
        tool_calls = response_message.tool_calls
        if not tool_calls or iteration >= max_tool_iterations:
            break
        iteration += 1
        # Append the assistant message that triggered the tool call so that the
        # subsequent `tool` role messages are valid according to the OpenAI
        # chat format. Omitting this was causing the 400 error:
        # "messages with role 'tool' must be a response to a preceeding message with 'tool_calls'."
        messages.append({
            "role": "assistant",
            # The assistant content is typically empty when the model decides to
            # call a tool.
            "content": response_message.content or "",
            # We must pass the full tool_calls array exactly as we received it.
            "tool_calls": [tc.model_dump(exclude_none=True) for tc in tool_calls],
        })

        for tool_call in tool_calls:
            function_name = tool_call.function.name
            raw_args = tool_call.function.arguments  # may be JSON string per OpenAI spec
            
            logging.info(f"[AI Agent] Executing tool: {function_name} with args: {raw_args}")

            # Ensure we have a dict before passing via **kwargs
            try:
                parsed_args = json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
            except Exception as e:
                logging.error(f"Failed to parse arguments for {function_name}: {e}. Raw: {raw_args}")
                parsed_args = {}

            # Call through registry helper first (allows additional wrappers)
            function_result = tools.resolve_function(function_name, parsed_args)
            logging.info(f"[AI Agent] Tool result: {function_result}")

            # If the concrete callable is present, execute directly as well
            if function_name in function_map:
                try:
                    function_result = function_map[function_name](**parsed_args)
                except Exception as e:
                    logging.error(f"Error while executing {function_name}: {e}")

            # Serialise result for message content
            serialised_result = (
                function_result if isinstance(function_result, str) else json.dumps(function_result, default=str)
            )

            messages.append({
                "tool_call_id": tool_call.id,
                "role": "tool",
                "name": function_name,
                "content": serialised_result,
            })
        response = client.chat.completions.create(
            model=model_name, 
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            tools=tools.mapped_functions(),
            tool_choice="auto"
        )
        response_message = response.choices[0].message
        
    # Extract and return the AI's response
    if response_message:
        ai_content = response.choices[0].message.content
        finish_reason = response.choices[0].finish_reason

        if ai_content is not None:
            #logging.info("Received response content from OPENAI.")
            return ai_content
        else:
            # Content is None, log finish_reason and the message object
            logging.error(
                f"OPENAI response error: Message content is None. Finish reason: '{finish_reason}'. Message: {response.choices[0].message.model_dump_json()}"
            )
            return f"Error querying AI: OPENAI response message content is None. Finish reason: {finish_reason}."
    else:
        # Message object itself is None or no choices
        #logging.error(f"OPENAI response error: No choices or message object found. Full response: {response.model_dump_json()}")
        return "Error querying AI: No choices or message object returned from OPENAI."

    # except Exception as e:
    #     #logging.error(f"Error querying OPENAI many info)
    #     return f"Error querying AI: {str(e)} - {e} - {e.__str__()}"


def summarize_conversation(query: str, response: str) -> str:
    """Create a brief summary of the conversation.

    Uses the same AI model to generate a concise summary for memory storage.

    Args:
        query: The user's question
        response: The AI's answer

    Returns:
        str: A brief summary of the conversation
    """
    # Initialize client and check API key here
    try:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            logging.error(
                "OPENAI API key not found. Please set the OPENAI_API_KEY environment variable."
            )
            return "Summary not available - OPENAI API key not found."
        client = OpenAI(api_key=api_key)
    except Exception as e:
        #logging.exception(f"Failed to initialize OPENAI client: {e}")
        return f"Summary not available - Failed to initialize OPENAI client: {e}"

    try:
        # Create a prompt to summarize the conversation
        summary_prompt = f"""Summarize the following conversation in a single short paragraph (max 50 words):
        
        User: {query}
        AI: {response}
        
        Summary:"""

        # Make the API call with a low max_tokens to ensure brevity
        completion = client.chat.completions.create(
            model="gpt-4.1-nano-2025-04-14", 
            messages=[
                {"role": "system", "content": "You are a personal assistant. The user is asking you a question. Answer briefly and concisely. If one sentence is enough, answer with one sentence. "},
                {"role": "user", "content": summary_prompt}, 
                ],
        )

        # Extract and clean up the summary
        summary = completion.choices[0].message.content

        # Remove any prefix the model might add like "Summary:" or "Here's a summary:"
        for prefix in ["Summary:", "Here's a summary:", "Here is a summary:"]:
            if summary.startswith(prefix):
                summary = summary[len(prefix) :].strip()

        #logging.info(f"Generated conversation summary ({len(summary)} chars)")
        return summary

    except Exception as e:
        #logging.error(f"Error generating conversation summary: {e}")
        return "Error generating summary."


def update_memory(query: str, response: str, memory: BaseMemoryManager):
    # Initialize client and check API key here
    
    try:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            logging.error(
                "OPENAI API key not found. Please set the OPENAI_API_KEY environment variable."
            )
            return "Summary not available - OPENAI API key not found."
        client = OpenAI(api_key=api_key)
    except Exception as e:
        #logging.exception(f"Failed to initialize OPENAI client: {e}")
        return f"Summary not available - Failed to initialize OPENAI client: {e}"
    try:
        updated = False 
        # Create a prompt to summarize the conversation
        summary_prompt = f"""Given the following query, response, and memory, suggest a field to be added or updated in the memory. 
        Return only the field name and value. 
        Ex. field_name: value 
        if multiple fields are suggested, return them in a list separated by commas. 
        Ex. field_name: value, field_name: value 
        if no field is suggested, return None: None 
        follow the format exactly as shown above (without quotes or brackets (no square brackets or curly brackets) or any other characters)
        :
        User: {query}
        AI: {response}
        Memory: {memory._memory_content}
        """

        # Make the API call with a low max_tokens to ensure brevity
        completion = client.chat.completions.create(
            model="gpt-4.1-nano-2025-04-14",
            messages=[
                {"role": "system", "content": "You are a personal assistant. The user is asking you a question. Answer briefly and concisely. If one sentence is enough, answer with one sentence. "},
                {"role": "user", "content": summary_prompt},
                ],
        )

        # Extract and clean up the summary
        suggested_updates = completion.choices[0].message.content
        
        #logging.info(f"Suggested updates: {suggested_updates}")
        # parse the suggested updates
        updates = suggested_updates.split(",")

        # update the memory
        for update in updates:
            update = update.strip()
            if update.startswith("None"):
                continue
            key, value = update.split(":")
            
            memory.set(key.strip(), value.strip())
            updated = True

        
        #logging.info(f"Memory updated: {memory._memory_content}")
        return updated
    except Exception as e:
        #logging.exception(f"Failed to update memory: {e}")
        return False


def ask_ai(
    query: str,
    max_tokens: int = 1024,
    temperature: float = 0.7,
    client_dir: Optional[str] = None,
    aux_data: Optional[Dict[str, Any]] = None,
    update_memory: bool = True,
) -> str:
    """
    Query the AI with a given query and optional auxiliary data.

    Args:
        query: The user's question
        aux_data: Optional auxiliary data to include in the context
        max_tokens: Maximum tokens in response
        temperature: Response randomness (0.0-1.0)
        update_memory: Whether to update memory with the conversation

    Returns:
        str: The AI's response
    """
    
    # Determine memory file paths
    if client_dir:
        base_memory_path = client_dir
        #logging.info(f"Using client-provided directory for memory: {base_memory_path}")
    else:
        base_memory_path = DEFAULT_DATA_PATH
        #logging.info(f"Using default data directory for memory: {base_memory_path}")
        os.makedirs(base_memory_path, exist_ok=True) # Ensure default data directory exists

    long_term_memory_file = os.path.join(base_memory_path, "long_term_memory.json")
    short_term_memory_file = os.path.join(base_memory_path, "short_term_memory.json")

    # Instantiate memory managers with correct file paths
    long_term_manager = LongTermMemoryManager(memory_file=long_term_memory_file)
    short_term_manager = ShortTermMemoryManager(memory_file=short_term_memory_file)

    # Read all reference files
    try:
        references = read_references()
    except Exception as e:
        #logging.error(f"Error loading references: {e}")
        references = {}
        
    # Check if we're running in a serverless environment
    if os.environ.get("VERCEL") and not references:
        #logging.info("Running in Vercel environment, providing minimal context due to filesystem limitations")
        # Add a synthetic reference to explain the limitations
        references = {
            "system_info.txt": "This is a limited deployment running on serverless infrastructure. " +
                            "Some features requiring filesystem access may be restricted."
        }

    # Query the AI with context
    response = query_openai(
        query=query,
        long_term_memory=long_term_manager,
        short_term_memory=short_term_manager,
        aux_data=aux_data,
        references=references,
        max_tokens=max_tokens,
        temperature=temperature,
    )

    # Only create summary and update memory if query was successful
    if not response.startswith("Error:") and update_memory:
        summary = summarize_conversation(query, response)
        # update short term memory: add to the conversations list
        conversations_data = short_term_manager.get("conversations") 
        conversations = conversations_data if conversations_data is not None else []
        short_term_manager.set("conversations", conversations + [{"query": query, "response": response, "summary": summary}])
        
    else:
        logging.warning("Skipping memory update due to query error")

    return response



