"""
Multi-provider connector for AutoMigraine
Allows switching between different AI API providers
"""

import os
import json
import aiohttp
import asyncio
from typing import Dict, Any, Optional, List

class MultiProviderConnector:
    """Connector that supports multiple AI API providers"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the connector with configuration"""
        self.provider = config.get('provider', 'together').lower()
        self.model_id = config.get('model_id')
        self.api_key = config.get('api_key') or os.environ.get(f'{self.provider.upper()}_API_KEY')
        
        if not self.api_key:
            raise ValueError(f"API key for {self.provider} is required")
        
        if not self.model_id:
            raise ValueError("Model ID is required")
        
        # Provider-specific endpoints
        self.endpoints = {
            'together': 'https://api.together.xyz/v1/completions',
            'groq': 'https://api.groq.com/openai/v1/chat/completions',
            'huggingface': f'https://api-inference.huggingface.co/models/{self.model_id}',
            'cohere': 'https://api.cohere.ai/v1/generate',
            'openrouter': 'https://openrouter.ai/api/v1/chat/completions'
        }
        
        # Default parameters
        self.default_params = {
            'temperature': 0.7,
            'max_tokens': 1024,
        }
    
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate text using the configured provider"""
        if self.provider == 'together':
            return await self._generate_together(prompt, **kwargs)
        elif self.provider == 'groq':
            return await self._generate_groq(prompt, **kwargs)
        elif self.provider == 'huggingface':
            return await self._generate_huggingface(prompt, **kwargs)
        elif self.provider == 'cohere':
            return await self._generate_cohere(prompt, **kwargs)
        elif self.provider == 'openrouter':
            return await self._generate_openrouter(prompt, **kwargs)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
    
    async def _generate_together(self, prompt: str, **kwargs) -> str:
        """Generate text using Together AI API"""
        params = self.default_params.copy()
        params.update(kwargs)
        
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        
        data = {
            'model': self.model_id,
            'prompt': prompt,
            'max_tokens': params.get('max_tokens', 1024),
            'temperature': params.get('temperature', 0.7),
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(self.endpoints['together'], 
                                   headers=headers,
                                   json=data) as response:
                if response.status == 200:
                    result = await response.json()
                    return result['choices'][0]['text']
                else:
                    error_text = await response.text()
                    raise Exception(f"Together API Error ({response.status}): {error_text}")
    
    async def _generate_groq(self, prompt: str, **kwargs) -> str:
        """Generate text using Groq API"""
        params = self.default_params.copy()
        params.update(kwargs)
        
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        
        data = {
            'model': self.model_id,
            'messages': [{'role': 'user', 'content': prompt}],
            'max_tokens': params.get('max_tokens', 1024),
            'temperature': params.get('temperature', 0.7),
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(self.endpoints['groq'], 
                                   headers=headers,
                                   json=data) as response:
                if response.status == 200:
                    result = await response.json()
                    return result['choices'][0]['message']['content']
                else:
                    error_text = await response.text()
                    raise Exception(f"Groq API Error ({response.status}): {error_text}")
    
    async def _generate_huggingface(self, prompt: str, **kwargs) -> str:
        """Generate text using Hugging Face Inference API"""
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        
        data = {
            'inputs': prompt,
            'parameters': {
                'max_new_tokens': kwargs.get('max_tokens', 1024),
                'temperature': kwargs.get('temperature', 0.7),
            }
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(self.endpoints['huggingface'], 
                                   headers=headers,
                                   json=data) as response:
                if response.status == 200:
                    result = await response.json()
                    # Different models return results in different formats
                    if isinstance(result, list) and len(result) > 0:
                        return result[0].get('generated_text', '')
                    elif isinstance(result, dict):
                        return result.get('generated_text', '')
                    else:
                        return str(result)
                else:
                    error_text = await response.text()
                    raise Exception(f"Hugging Face API Error ({response.status}): {error_text}")
    
    async def _generate_cohere(self, prompt: str, **kwargs) -> str:
        """Generate text using Cohere API"""
        params = self.default_params.copy()
        params.update(kwargs)
        
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        
        data = {
            'model': self.model_id,
            'prompt': prompt,
            'max_tokens': params.get('max_tokens', 1024),
            'temperature': params.get('temperature', 0.7),
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(self.endpoints['cohere'], 
                                   headers=headers,
                                   json=data) as response:
                if response.status == 200:
                    result = await response.json()
                    return result.get('generations', [{}])[0].get('text', '')
                else:
                    error_text = await response.text()
                    raise Exception(f"Cohere API Error ({response.status}): {error_text}")
    
    async def _generate_openrouter(self, prompt: str, **kwargs) -> str:
        """Generate text using OpenRouter API"""
        params = self.default_params.copy()
        params.update(kwargs)
        
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        
        data = {
            'model': self.model_id,
            'messages': [{'role': 'user', 'content': prompt}],
            'max_tokens': params.get('max_tokens', 1024),
            'temperature': params.get('temperature', 0.7),
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(self.endpoints['openrouter'], 
                                   headers=headers,
                                   json=data) as response:
                if response.status == 200:
                    result = await response.json()
                    return result['choices'][0]['message']['content']
                else:
                    error_text = await response.text()
                    raise Exception(f"OpenRouter API Error ({response.status}): {error_text}")
