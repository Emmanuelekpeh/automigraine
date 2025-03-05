#!/usr/bin/env python
"""
Basic test script for AutoMigraine components
"""

import os
import json
import asyncio
from pathlib import Path

# Import the components you've implemented
from core.multi_provider import MultiProviderConnector
from core.git_database import GitDatabase
from core.zero_shot_agent import ZeroShotAgent, Tool

async def test_git_database():
    """Test the Git database implementation"""
    print("\n=== Testing GitDatabase ===")
    
    # Initialize database
    db = GitDatabase()
    
    # Test saving data
    test_data = {
        "test_key": "test_value",
        "nested": {
            "data": [1, 2, 3]
        }
    }
    
    task_id = "test_task_001"
    file_path = db.save_result(task_id, test_data)
    print(f"Saved data to: {file_path}")
    
    # Test retrieving data
    loaded_data = db.get_result(task_id)
    print(f"Retrieved data: {json.dumps(loaded_data, indent=2)}")
    
    # Test configuration
    config_path = db.save_config("test_config", {"setting1": "value1", "setting2": 42})
    print(f"Saved config to: {config_path}")
    
    config_data = db.get_config("test_config")
    print(f"Retrieved config: {json.dumps(config_data, indent=2)}")
    
    return True

async def test_provider(api_key_env, provider_name="together", model_id="mistralai/mistral-7b-instruct"):
    """Test the AI provider connector"""
    print(f"\n=== Testing MultiProviderConnector with {provider_name} ===")
    
    # Get API key
    api_key = os.environ.get(api_key_env)
    if not api_key:
        print(f"⚠️ Missing {api_key_env} environment variable, skipping provider test")
        return False
    
    # Initialize connector
    connector = MultiProviderConnector({
        "provider": provider_name,
        "model_id": model_id,
        "api_key": api_key
    })
    
    try:
        # Simple generation test
        messages = [
            {"role": "user", "content": "Write a short poem about AI"},
        ]
        
        print("Generating response...")
        response = await connector.generate(messages)
        print(f"Response received:\n{response}")
        return True
    except Exception as e:
        print(f"❌ Error testing provider: {e}")
        return False

async def test_zero_shot_agent():
    """Test the zero-shot agent implementation"""
    print("\n=== Testing ZeroShotAgent ===")
    
    # Get API key (try multiple providers in order of preference)
    api_key = None
    provider = None
    model_id = None
    
    provider_configs = [
        ("TOGETHER_API_KEY", "together", "mistralai/mistral-7b-instruct"),
        ("GROQ_API_KEY", "groq", "llama2-70b-4096"),
        ("HF_API_KEY", "huggingface", "mistralai/mistral-7b-instruct")
    ]
    
    for env_var, prov, model in provider_configs:
        if os.environ.get(env_var):
            api_key = os.environ.get(env_var)
            provider = prov
            model_id = model
            break
    
    if not api_key:
        print("⚠️ No API keys found, skipping agent test")
        return False
    
    # Initialize connector
    connector = MultiProviderConnector({
        "provider": provider,
        "model_id": model_id,
        "api_key": api_key
    })
    
    # Define simple tools
    tools = {
        "get_weather": Tool(
            name="get_weather",
            description="Get weather for a location",
            func=lambda location: f"Weather in {location}: Sunny, 72°F"
        ),
        "calculate": Tool(
            name="calculate",
            description="Calculate a mathematical expression",
            func=lambda expr: str(eval(expr))
        )
    }
    
    # Initialize agent
    agent = ZeroShotAgent(
        llm_connector=connector,
        tools=tools,
        verbose=True
    )
    
    # Test the agent
    task = "What's the weather in New York? Also, what is 25 * 4?"
    print(f"Running agent with task: {task}")
    
    try:
        result = await agent.run(task)
        print(f"\nFinal response: {result['response']}")
        print(f"Used {result['iterations']} iterations")
        return True
    except Exception as e:
        print(f"❌ Error testing agent: {e}")
        return False

async def main():
    """Run all tests"""
    print("Starting basic tests for AutoMigraine components...")
    
    # Test database
    db_success = await test_git_database()
    
    # Test at least one provider
    provider_success = False
    for env_var, provider, model in [
        ("TOGETHER_API_KEY", "together", "mistralai/mistral-7b-instruct"),
        ("GROQ_API_KEY", "groq", "llama2-70b-4096"),
        ("HF_API_KEY", "huggingface", "mistralai/mistral-7b-instruct")
    ]:
        if await test_provider(env_var, provider, model):
            provider_success = True
            break
    
    # Test agent
    agent_success = await test_zero_shot_agent()
    
    # Print summary
    print("\n=== Test Summary ===")
    print(f"GitDatabase: {'✅ Passed' if db_success else '❌ Failed'}")
    print(f"MultiProvider: {'✅ Passed' if provider_success else '❌ Failed'}")
    print(f"ZeroShotAgent: {'✅ Passed' if agent_success else '❌ Failed'}")

if __name__ == "__main__":
    asyncio.run(main())
