import json
import os
import logging
from typing import Dict, List, Optional, Any
    

class BaseMemoryManager:
    """
    Abstract base class for memory management.
    
    """

    def __init__(self, memory_content: Optional[Dict] = None):
        self._memory_content = memory_content if memory_content is not None else {}

    def load(self) -> Dict:
        return self._memory_content

    def save(self, memory_content: Optional[Dict] = None) -> None:
        if memory_content is not None:
            self._memory_content = memory_content

    @property
    def memory_type(self) -> str:
        raise NotImplementedError

    def set(self, key: str, value: Any) -> None:
        """Set a specific field in memory."""
        self._memory_content[key] = value

    def get(self, key: str) -> Optional[Any]:
        """Get a specific field from memory."""
        return self._memory_content.get(key, None)

class ShortTermMemoryManager(BaseMemoryManager):
    """Manages short-term memory operations.
    
    Short-term memory stores recent user interactions, current context,
    active URLs, and conversation history. This memory is more volatile
    and focuses on the immediate context of user interactions.
    
    Attributes:
        memory_content (Dict): The content of the short-term memory.

    """
    
    def __init__(self, memory_content: Optional[Dict] = None):
        super().__init__(memory_content=memory_content)


    @property
    def memory_type(self) -> str:
        return "short-term"

    def update_active_url(self, url: str, title: str) -> None:
        """Update the active URL in short-term memory."""
        # memory = self.load() # No need to load explicitly, get/set handles it or it's loaded in init
        active_url_data = self._memory_content.get('active_url', {})
        active_url_data['url'] = url
        active_url_data['title'] = title
        active_url_data['timestamp'] = self._get_timestamp()
        self._memory_content['active_url'] = active_url_data
        self.save()

    def _get_timestamp(self) -> str:
        from datetime import datetime
        return datetime.utcnow().isoformat()

    def get_recent_conversations(self, limit: int = 10) -> List[Dict]:
        # memory = self.load() # No need to load explicitly
        conversations = self._memory_content.get('conversations', [])
        return conversations[-limit:]

class LongTermMemoryManager(BaseMemoryManager):
    def get_content(self) -> Dict:
        """Get the entire memory content."""
        if not self._memory_content :
            self._memory_content = self.load()
        return self._memory_content
        
    def get_memory_content(self) -> Dict:
        """Alias for get_content to maintain backward compatibility.
        
        Returns:
            Dict: The entire memory content
        """
        return self.get_content()
    """Manages long-term memory operations.
    
    Long-term memory stores persistent user preferences, profile information,
    learning history, and important insights that should be remembered across
    multiple sessions. This memory provides continuity in the user experience.
    
    Attributes:
        memory_content (Dict): The content of the long-term memory.
        
    """
    
    def __init__(self, memory_content: Optional[Dict] = None):
        super().__init__(memory_content=memory_content)


    @property
    def memory_type(self) -> str:
        return "long-term"

    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime
        return datetime.utcnow().isoformat()
        

if __name__ == "__main__":
    # Define base path for memory files for the example
    # This assumes the script is run from somewhere that aiagent/data exists relative to it
    # For actual CLI/server usage, these paths might be configured differently.
    module_path = os.path.dirname(os.path.abspath(__file__))
    bot_path = os.path.dirname(module_path) # aiagent directory
    data_path = os.path.join(bot_path, "data") 
    os.makedirs(data_path, exist_ok=True)

    short_term_file = os.path.join(data_path, "short_term_memory_example.json")
    long_term_file = os.path.join(data_path, "long_term_memory_example.json")

    # load short-term memory
    short_term_memory = ShortTermMemoryManager(memory_file=short_term_file)
    print(f"Initial short_term_memory: {short_term_memory._memory_content}")

    # update active url
    short_term_memory.update_active_url("https://www.example.com", "Example")
    print(f"Short_term_memory after update_active_url: {short_term_memory._memory_content}")

    # set a field
    short_term_memory.set("test", "value")
    print(f"Short_term_memory after set: {short_term_memory._memory_content}")
    print(f"Get 'test' from short_term_memory: {short_term_memory.get('test')}")

    # Test LongTermMemoryManager
    long_term_memory = LongTermMemoryManager(memory_file=long_term_file)
    long_term_memory.set("user_preference", "dark_mode")
    print(f"Long_term_memory after set: {long_term_memory._memory_content}")
    print(f"Get 'user_preference' from long_term_memory: {long_term_memory.get('user_preference')}")

    # Clean up example files
    if os.path.exists(short_term_file):
        os.remove(short_term_file)
    if os.path.exists(long_term_file):
        os.remove(long_term_file)
