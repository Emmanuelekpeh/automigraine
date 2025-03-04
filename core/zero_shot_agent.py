"""
Zero-shot agent implementation for AutoMigraine
Implements a simple but effective agent that can use tools directly without reasoning steps
"""

import json
import re
import asyncio
from typing import Dict, List, Any, Callable, Optional, Union, TypeVar, Awaitable

# Type definitions
T = TypeVar('T')
SyncOrAsyncCallable = Union[Callable[[str], T], Callable[[str], Awaitable[T]]]

class Tool:
    """Tool definition for zero-shot agent"""
    
    def __init__(self, name: str, description: str, func: SyncOrAsyncCallable):
        """
        Initialize a tool
        
        Args:
            name: Name of the tool
            description: Description of what the tool does and when to use it
            func: Function to call when tool is used (can be sync or async)
        """
        self.name = name
        self.description = description
        self.func = func
        
    async def execute(self, query: str) -> str:
        """Execute the tool with the provided query"""
        if asyncio.iscoroutinefunction(self.func):
            return await self.func(query)
        else:
            return self.func(query)
            
    def to_dict(self) -> Dict[str, str]:
        """Convert tool to dictionary representation"""
        return {
            "name": self.name,
            "description": self.description
        }

class ZeroShotAgent:
    """
    Zero-shot agent implementation
    
    A straightforward agent that can use tools directly based on the task without
    explicit reasoning steps, making it efficient for simple to moderately complex tasks.
    """
    
    def __init__(
        self, 
        llm_connector: Any,
        tools: Dict[str, Union[Callable, Tool]] = None,
        system_prompt: Optional[str] = None,
        max_iterations: int = 5,
        verbose: bool = False
    ):
        """
        Initialize the zero-shot agent
        
        Args:
            llm_connector: Connector to the language model
            tools: Dictionary of tools (name -> callable or Tool object)
            system_prompt: Custom system prompt (defaults to a standard prompt if None)
            max_iterations: Maximum number of iterations for tool use
            verbose: Whether to print verbose output
        """
        self.llm = llm_connector
        self.max_iterations = max_iterations
        self.verbose = verbose
        
        # Process tools
        self.tools: Dict[str, Tool] = {}
        if tools:
            for name, tool in tools.items():
                if isinstance(tool, Tool):
                    self.tools[name] = tool
                else:
                    # Create Tool object from callable
                    self.tools[name] = Tool(
                        name=name,
                        description=f"Tool to {name.replace('_', ' ')}",
                        func=tool
                    )
        
        # Set up system prompt
        self.system_prompt = system_prompt or self._get_default_system_prompt()
        
    def _get_default_system_prompt(self) -> str:
        """Get the default system prompt for the agent"""
        tools_desc = "\n".join([f"- {tool.name}: {tool.description}" for tool in self.tools.values()])
        
        return f"""You are a helpful AI assistant that can use tools to answer questions.
        
Available tools:
{tools_desc}

To use a tool, respond with the following format:
```tool
{{"name": "tool_name", "query": "input to the tool"}}
