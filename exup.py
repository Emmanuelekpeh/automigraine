#!/usr/bin/env python3
"""
Enhanced Dual Model System Setup
Author: Emmanuel Ekpeh
Date: 2025-03-03 20:02:48

Optimizes model selection from Hugging Face's most advanced offerings.
"""

import os
import sys
import yaml
import argparse
import requests
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import json
import time
import shutil

# Curated selection of top models on Hugging Face by role
HF_MODELS = {
    "manager": {  # Models good for planning and orchestration
        "small": "google/gemma-7b-it",
        "medium": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "large": "THUDM/cogvlm-chat-hf"
    },
    "developer1": {  # Primary code generation models
        "small": "deepseek-ai/deepseek-coder-6.7b-instruct",
        "medium": "WizardLM/WizardCoder-Python-34B-V1.0",
        "large": "Phind/Phind-CodeLlama-34B-v2"
    },
    "developer2": {  # Secondary/specialized code generation models
        "small": "bigcode/starcoder2-3b",
        "medium": "bigcode/starcoder2-15b",
        "large": "deepseek-ai/deepseek-coder-33b-instruct"
    },
    "moderator": {  # Lightweight models suitable for moderation
        "small": "microsoft/phi-2",
        "medium": "google/gemma-2b-it",
        "large": "mistralai/Mistral-7B-Instruct-v0.2"
    }
}

def find_local_models(base_dir: str = ".") -> Dict[str, List[str]]:
    """Find local GGUF models in the filesystem"""
    results = {
        "mistral": [],
        "codellama": []
    }
    
    # Walk directory tree looking for .gguf files
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".gguf"):
                path = os.path.join(root, file)
                
                # Classify models
                if "mistral" in file.lower():
                    results["mistral"].append(path)
                elif "codellama" in file.lower() or ("llama" in file.lower() and "code" in file.lower()):
                    results["codellama"].append(path)
    
    return results

def test_huggingface_connection(api_key: str) -> Tuple[bool, str]:
    """Test connection to Hugging Face API"""
    try:
        headers = {"Authorization": f"Bearer {api_key}"}
        response = requests.get(
            "https://huggingface.co/api/whoami", 
            headers=headers,
            timeout=10
        )
        
        if response.status_code == 200:
            user_data = response.json()
            return True, f"Connected as {user_data.get('name', 'authenticated user')}"
        else:
            return False, f"API Error ({response.status_code}): {response.text}"
    except Exception as e:
        return False, f"Connection error: {str(e)}"

def create_config(
    config_path: str, 
    use_huggingface: bool = False, 
    api_key: Optional[str] = None,
    manager_size: str = "medium", 
    dev1_size: str = "medium",
    dev2_size: str = "medium",
    mod_size: str = "small",
    use_local_moderator: bool = False
) -> Dict[str, Any]:
    """Create configuration file with model settings"""
    # Start with base configuration
    config = {
        "system": {
            "version": "1.0.0",
            "results_dir": "results",
            "max_iterations": 3,
            "use_moderator": True,  # Always use moderator with our new setup
            "logging": {
                "enabled": True,
                "level": "INFO",
                "file": "logs/system.log"
            },
            "debug_mode": True  # Enable visibility into model processes
        },
        "paths": {},
        "models": {},
        "ui": {
            "theme": "default",
            "port": 7860,
            "share": False
        }
    }
    
    # Create directories
    os.makedirs("results", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    local_models = find_local_models()
    
    if use_huggingface:
        # Configure Hugging Face models
        if not api_key:
            print("\n⚠️ Warning: No API key provided for Hugging Face. Severe usage limitations may apply.")
        
        # Validate model sizes
        valid_sizes = ["small", "medium", "large"]
        for size_name, size_value in [
            ("manager", manager_size),
            ("dev1", dev1_size),
            ("dev2", dev2_size),
            ("mod", mod_size)
        ]:
            if size_value not in valid_sizes:
                print(f"Invalid {size_name} size: {size_value}. Using medium.")
                if size_name == "manager":
                    manager_size = "medium"
                elif size_name == "dev1":
                    dev1_size = "medium"
                elif size_name == "dev2":
                    dev2_size = "medium"
                elif size_name == "mod":
                    mod_size = "small"  # Default to small for moderator
        
        # Setup manager model
        manager_model_id = HF_MODELS["manager"][manager_size]
        config["models"]["manager"] = {
            "type": "huggingface",
            "model_id": manager_model_id,
            "api_key": api_key or "",
            "temperature": 0.7,
            "max_tokens": 2048,
            "streaming": True,
            "description": f"Manager Model ({manager_size} - {manager_model_id})"
        }
        
        # Setup primary developer model
        dev1_model_id = HF_MODELS["developer1"][dev1_size]
        config["models"]["developer1"] = {
            "type": "huggingface",
            "model_id": dev1_model_id,
            "api_key": api_key or "",
            "temperature": 0.2,  # Lower temperature for more deterministic code
            "max_tokens": 4096,
            "streaming": True,
            "description": f"Primary Developer Model ({dev1_size} - {dev1_model_id})"
        }
        
        # Setup secondary developer model
        dev2_model_id = HF_MODELS["developer2"][dev2_size]
        config["models"]["developer2"] = {
            "type": "huggingface",
            "model_id": dev2_model_id,
            "api_key": api_key or "",
            "temperature": 0.3,  # Slightly higher for creative solutions
            "max_tokens": 4096,
            "streaming": True,
            "description": f"Secondary Developer Model ({dev2_size} - {dev2_model_id})"
        }
        
        # Setup moderator model - either local or HuggingFace
        if use_local_moderator:
            # Try to find a suitable local model
            mistral_available = len(local_models["mistral"]) > 0
            codellama_available = len(local_models["codellama"]) > 0
            
            if mistral_available:
                # Use local Mistral for moderation
                mistral_path = local_models["mistral"][0]
                if len(local_models["mistral"]) > 1:
                    print("\nMultiple Mistral models found. Using the first one for moderation:")
                    print(f"  → {mistral_path}")
                
                config["paths"]["moderator"] = mistral_path
                config["models"]["moderator"] = {
                    "type": "local",
                    "n_ctx": 2048,  # Smaller context to save memory
                    "n_threads": min(4, os.cpu_count() or 2),  # Fewer threads
                    "n_gpu_layers": 0,  # CPU only for moderator
                    "temperature": 0.5,
                    "max_tokens": 1024,  # Shorter responses for moderation
                    "streaming": True,
                    "description": f"Moderator (Local Mistral - {os.path.basename(mistral_path)})"
                }
                print(f"\n✅ Using local Mistral model as moderator: {mistral_path}")
            
            elif codellama_available:
                # Use local CodeLlama for moderation
                codellama_path = local_models["codellama"][0]
                if len(local_models["codellama"]) > 1:
                    print("\nMultiple CodeLlama models found. Using the first one for moderation:")
                    print(f"  → {codellama_path}")
                
                config["paths"]["moderator"] = codellama_path
                config["models"]["moderator"] = {
                    "type": "local",
                    "n_ctx": 2048,  # Smaller context to save memory
                    "n_threads": min(4, os.cpu_count() or 2),  # Fewer threads
                    "n_gpu_layers": 0,  # CPU only for moderator
                    "temperature": 0.5,
                    "max_tokens": 1024,  # Shorter responses for moderation
                    "streaming": True,
                    "description": f"Moderator (Local CodeLlama - {os.path.basename(codellama_path)})"
                }
                print(f"\n✅ Using local CodeLlama model as moderator: {codellama_path}")
            
            else:
                # No local models available, fall back to HuggingFace moderator
                print("\n⚠️ No local models found for moderation. Falling back to Hugging Face moderator.")
                use_local_moderator = False
        
        if not use_local_moderator:
            # Use Hugging Face for moderation
            mod_model_id = HF_MODELS["moderator"][mod_size]
            config["models"]["moderator"] = {
                "type": "huggingface",
                "model_id": mod_model_id,
                "api_key": api_key or "",
                "temperature": 0.5,
                "max_tokens": 1024,
                "streaming": True,
                "description": f"Moderator ({mod_size} - {mod_model_id})"
            }
            print(f"\n✅ Using Hugging Face model as moderator: {mod_model_id}")
        
        # Output configuration summary
        print("\n🔄 Configured models:")
        print(f"  - Manager: {manager_model_id} ({manager_size})")
        print(f"  - Primary Developer: {dev1_model_id} ({dev1_size})")
        print(f"  - Secondary Developer: {dev2_model_id} ({dev2_size})")
        print(f"  - Moderator: {config['models']['moderator']['description']}")
        
        # Test connection if API key provided
        if api_key:
            print("\nTesting connection to Hugging Face... ", end="", flush=True)
            success, message = test_huggingface_connection(api_key)
            if success:
                print(f"✅ Success! {message}")
            else:
                print(f"❌ Failed: {message}")
                print("Configuration will be saved but may not work correctly.")
    else:
        # Configure local models only - falling back to original setup
        print("\n⚠️ Using local models only (limited capability)")
        
        # Handle Mistral model selection for manager
        if local_models["mistral"]:
            if len(local_models["mistral"]) == 1:
                mistral_path = local_models["mistral"][0]
            else:
                print("\nMultiple Mistral models found. Please select one for manager role:")
                for i, path in enumerate(local_models["mistral"]):
                    print(f"  [{i+1}] {path}")
                choice = int(input("Enter number: ")) - 1
                mistral_path = local_models["mistral"][choice]
            
            config["paths"]["manager"] = mistral_path
            config["models"]["manager"] = {
                "type": "local",
                "n_ctx": 4096,
                "n_threads": os.cpu_count() or 4,
                "n_gpu_layers": 0,  # CPU only
                "temperature": 0.7,
                "max_tokens": 2048,
                "streaming": True,
                "description": f"Manager (Local Mistral - {os.path.basename(mistral_path)})"
            }
            print(f"\n✅ Using local Mistral model as manager: {mistral_path}")
        else:
            print("\n❌ No local Mistral models found for manager role")
        
        # Handle CodeLlama model selection for developer
        if local_models["codellama"]:
            if len(local_models["codellama"]) == 1:
                codellama_path = local_models["codellama"][0]
            else:
                print("\nMultiple CodeLlama models found. Please select one for developer role:")
                for i, path in enumerate(local_models["codellama"]):
                    print(f"  [{i+1}] {path}")
                choice = int(input("Enter number: ")) - 1
                codellama_path = local_models["codellama"][choice]
            
            config["paths"]["developer1"] = codellama_path
            config["models"]["developer1"] = {
                "type": "local",
                "n_ctx": 8192,
                "n_threads": os.cpu_count() or 4,
                "n_gpu_layers": 0,  # CPU only
                "temperature": 0.2,
                "max_tokens": 4096,
                "streaming": True,
                "description": f"Developer (Local CodeLlama - {os.path.basename(codellama_path)})"
            }
            print(f"\n✅ Using local CodeLlama model as developer: {codellama_path}")
            
            # Use the same model for developer2 with different settings
            config["paths"]["developer2"] = codellama_path
            config["models"]["developer2"] = {
                "type": "local",
                "n_ctx": 8192,
                "n_threads": os.cpu_count() or 4,
                "n_gpu_layers": 0,  # CPU only
                "temperature": 0.4,  # Higher temp for variety
                "max_tokens": 4096,
                "streaming": True,
                "description": f"Secondary Developer (Local CodeLlama - {os.path.basename(codellama_path)})"
            }
            
            # Also use Mistral as moderator if available
            if "manager" in config["paths"]:
                config["paths"]["moderator"] = config["paths"]["manager"]
                config["models"]["moderator"] = {
                    "type": "local",
                    "n_ctx": 2048,
                    "n_threads": min(2, os.cpu_count() or 1),
                    "n_gpu_layers": 0,
                    "temperature": 0.5,
                    "max_tokens": 1024,
                    "streaming": True,
                    "description": "Moderator (Local Mistral)"
                }
        else:
            print("\n❌ No local CodeLlama models found")
    
    # Write configuration to file
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"\n📝 Configuration saved to {config_path}")
    return config
# SECOND HALF OF SETUP_MODELS.PY - FIXED FOR PATH HANDLING
# Current Date: 2025-03-03 20:07:29
# User: Emmanuelekpeh

def create_model_connector(filepath: str):
    """Create the model connector implementation for multi-model setup"""
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, "w") as f:
        f.write('''#!/usr/bin/env python3
"""
Enhanced Model Connector: Unified interface for multi-model system
Supports Hugging Face, local models, and advanced caching
"""

import os
import sys
import gc
import json
import time
import asyncio
import logging
from typing import Dict, List, Any, Union, Optional, AsyncGenerator
from pathlib import Path
import traceback

try:
    import aiohttp
except ImportError:
    print("Warning: aiohttp not installed. Remote models will not work.")
    print("Install with: pip install aiohttp")

try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False
    print("Warning: llama-cpp-python not installed. Local models will not be available.")

class ModelConnector:
    """Unified interface for interacting with different model backends"""
    
    def __init__(self, model_name: str, config: Dict[str, Any], logger=None):
        """Initialize model connector"""
        self.model_name = model_name
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # Extract model configuration
        self.model_config = config.get("models", {}).get(model_name, {})
        self.model_type = self.model_config.get("type", "local")
        
        # Validate configuration
        if not self.model_config:
            raise ValueError(f"No configuration found for model {model_name}")
            
        # Local model instance (for local models)
        self.model_instance = None
        
        # Initialize based on model type
        if self.model_type == "local":
            self._setup_local_model()
        elif self.model_type == "huggingface":
            self._setup_huggingface()
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        self.logger.info(f"Initialized {self.model_name} connector ({self.model_type})")
    
    def _setup_local_model(self):
        """Setup local model parameters"""
        if not LLAMA_CPP_AVAILABLE:
            raise ImportError("llama_cpp is required for local models")
        
        # Get model path from config
        self.model_path = self.config.get("paths", {}).get(self.model_name)
        if not self.model_path:
            raise ValueError(f"No path configured for model {self.model_name}")
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        # Set parameters for model loading
        self.n_ctx = self.model_config.get("n_ctx", 4096)
        self.n_threads = self.model_config.get("n_threads", os.cpu_count() or 4)
        self.n_gpu_layers = self.model_config.get("n_gpu_layers", 0)
    
    def _setup_huggingface(self):
        """Setup Hugging Face API connection"""
        self.api_key = self.model_config.get("api_key")
        self.model_id = self.model_config.get("model_id")
        
        if not self.model_id:
            raise ValueError(f"No model_id provided for {self.model_name}")
        
        # Construct API URL
        self.api_url = f"https://api-inference.huggingface.co/models/{self.model_id}"
        self.logger.info(f"API URL for {self.model_name}: {self.api_url}")
    
    def _load_local_model(self):
        """Load local model if not already loaded"""
        if self.model_instance is None:
            self.logger.info(f"Loading model {self.model_name} from {self.model_path}")
            
            start_time = time.time()
            try:
                self.model_instance = Llama(
                    model_path=self.model_path,
                    n_ctx=self.n_ctx,
                    n_threads=self.n_threads,
                    n_gpu_layers=self.n_gpu_layers,
                    verbose=False
                )
                load_time = time.time() - start_time
                self.logger.info(f"Model {self.model_name} loaded in {load_time:.2f}s")
            except Exception as e:
                self.logger.error(f"Failed to load model {self.model_name}: {str(e)}")
                raise
    
    async def generate(self, 
                      prompt: str, 
                      max_tokens: int = None,
                      temperature: float = None, 
                      stop: List[str] = None) -> str:
        """Generate text from the model (non-streaming)"""
        # Use config values if not provided
        max_tokens = max_tokens or self.model_config.get("max_tokens", 2048)
        temperature = temperature or self.model_config.get("temperature", 0.7)
        stop = stop or ["</s>", "[/INST]"]
        
        try:
            if self.model_type == "local":
                return await self._generate_local(prompt, max_tokens, temperature, stop)
            elif self.model_type == "huggingface":
                return await self._generate_huggingface(prompt, max_tokens, temperature)
            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")
        except Exception as e:
            self.logger.error(f"Error generating with {self.model_name}: {str(e)}")
            self.logger.error(traceback.format_exc())
            return f"Error: {str(e)}"
    
    async def generate_stream(self, 
                             prompt: str, 
                             max_tokens: int = None,
                             temperature: float = None, 
                             stop: List[str] = None) -> AsyncGenerator[str, None]:
        """Stream text generation from the model"""
        # Use config values if not provided
        max_tokens = max_tokens or self.model_config.get("max_tokens", 2048)
        temperature = temperature or self.model_config.get("temperature", 0.7)
        stop = stop or ["</s>", "[/INST]"]
        
        try:
            if self.model_type == "local":
                async for token in self._generate_local_stream(prompt, max_tokens, temperature, stop):
                    yield token
            elif self.model_type == "huggingface":
                async for token in self._generate_huggingface_stream(prompt, max_tokens, temperature):
                    yield token
            else:
                yield f"Unsupported model type: {self.model_type}"
        except Exception as e:
            self.logger.error(f"Error streaming with {self.model_name}: {str(e)}")
            yield f"Error: {str(e)}"
    
    async def _generate_local(self, prompt: str, max_tokens: int, temperature: float, stop: List[str]) -> str:
        """Generate text using local model"""
        loop = asyncio.get_event_loop()
        
        try:
            self._load_local_model()
            
            # Run in thread pool to avoid blocking
            result = await loop.run_in_executor(
                None,
                lambda: self.model_instance.create_completion(
                    prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stop=stop,
                )
            )
            
            return result["choices"][0]["text"]
        except Exception as e:
            self.logger.error(f"Local model generation error: {str(e)}")
            raise
    
    async def _generate_local_stream(self, prompt: str, max_tokens: int, 
                                   temperature: float, stop: List[str]) -> AsyncGenerator[str, None]:
        """Stream text generation using local model"""
        try:
            self._load_local_model()
            
            # Create generator that yields from local model stream
            for token_data in self.model_instance.create_completion(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=stop,
                stream=True,
            ):
                token_text = token_data["choices"][0]["text"]
                if token_text:
                    yield token_text
                    await asyncio.sleep(0)  # Yield to event loop
        except Exception as e:
            self.logger.error(f"Local model streaming error: {str(e)}")
            raise
    
    async def _generate_huggingface(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """Generate text using Hugging Face API"""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
            
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": max_tokens,
                "temperature": temperature,
                "return_full_text": False
            }
        }
        
        async with aiohttp.ClientSession() as session:
            try:
                self.logger.info(f"Sending request to HF API: {self.api_url}")
                start_time = time.time()
                
                async with session.post(
                    self.api_url, 
                    headers=headers, 
                    json=payload, 
                    timeout=120  # Longer timeout for larger models
                ) as response:
                    duration = time.time() - start_time
                    
                    if response.status != 200:
                        error_text = await response.text()
                        self.logger.error(f"HF API error ({response.status}): {error_text}")
                        raise RuntimeError(f"API error ({response.status}): {error_text}")
                    
                    result = await response.json()
                    self.logger.info(f"HF API response received in {duration:.2f}s")
                    
                    # Parse response based on format
                    if isinstance(result, list) and len(result) > 0:
                        return result[0].get("generated_text", "")
                    elif isinstance(result, dict):
                        return result.get("generated_text", "")
                    else:
                        return str(result)
                        
            except aiohttp.ClientError as e:
                self.logger.error(f"HF API connection error: {str(e)}")
                raise
    
    async def _generate_huggingface_stream(self, 
                                         prompt: str, 
                                         max_tokens: int, 
                                         temperature: float) -> AsyncGenerator[str, None]:
        """Stream text from Hugging Face API (simulated streaming)"""
        # HF doesn't support true streaming, so we'll simulate it
        full_text = await self._generate_huggingface(prompt, max_tokens, temperature)
        
        # Break into smaller chunks to simulate streaming
        chunk_size = max(1, min(5, len(full_text) // 20))
        for i in range(0, len(full_text), chunk_size):
            yield full_text[i:i+chunk_size]
            await asyncio.sleep(0.02)  # Small delay for UI updates
''')

def create_init_file(core_dir: str):
    """Create __init__.py file in core directory"""
    # BUGFIX: First make sure the directory exists
    os.makedirs(core_dir, exist_ok=True)
    
    # BUGFIX: Use os.path.join properly to avoid nested path issue
    init_path = os.path.join(core_dir, "__init__.py")
    
    with open(init_path, "w") as f:
        f.write("""# Core module for Enhanced Dual Model System
# Author: Emmanuel Ekpeh
# Date: 2025-03-03 20:07:29

# Import core components
try:
    from .model_connector import ModelConnector
    from .dual_model_core import EnhancedModelSystem
except ImportError as e:
    print(f"Error importing core components: {e}")
""")
    print(f"Created init file: {init_path}")

def create_requirements_file():
    """Create requirements.txt file"""
    with open("requirements.txt", "w") as f:
        f.write("""# Enhanced Dual Model System Requirements
# Date: 2025-03-03 20:07:29

# Core dependencies
pyyaml>=6.0
aiohttp>=3.8.4
requests>=2.28.0
tqdm>=4.64.1
colorama>=0.4.6

# Local model support (optional)
llama-cpp-python>=0.2.0; platform_system != "Windows" or platform_machine != "ARM64"
llama-cpp-python-cuda>=0.2.0; platform_system == "Linux" and platform_machine == "x86_64"

# UI support (optional)
gradio>=3.50.0

# Telegram bot support (optional)
python-telegram-bot>=13.7
""")
    print("Created requirements.txt")

def create_dual_model_core_stub(filepath: str):
    """Create a stub dual_model_core.py file if it doesn't exist"""
    # BUGFIX: First make sure the directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Check if file already exists to avoid overwriting
    if os.path.exists(filepath):
        print(f"File {filepath} already exists, not overwriting.")
        return
        
    # Create the file with minimal implementation
    with open(filepath, "w") as f:
        f.write('''#!/usr/bin/env python3
"""
Enhanced Multi-Developer Model System
Integrates multiple models with different specialized roles
"""

import os
import sys
import time
import yaml
import json
import asyncio
import logging
from typing import Dict, List, Any, Optional, Callable, Tuple
from datetime import datetime
from pathlib import Path

# Import model connector
from core.model_connector import ModelConnector

class EnhancedModelSystem:
    """
    Enhanced model system with multiple specialized roles
    - Manager: Handles planning and coordination
    - Primary Developer: Main code implementation
    - Secondary Developer: Alternative implementations and specialized code
    - Moderator: Monitors process and ensures quality
    """
    
    def __init__(self, 
                 config_path: str = "config.yaml",
                 logger: Optional[logging.Logger] = None,
                 callbacks: Dict[str, Callable] = None):
        """Initialize the model system"""
        self.logger = logger or self._setup_logger()
        
        # Load configuration
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        
        # Set up callbacks
        self.callbacks = callbacks or {}
        
        # Initialize models
        self.models = {}
        self._setup_models()
        
        # System state
        self.is_running = False
        self.current_request = None
        self.results_dir = Path(self.config.get("system", {}).get("results_dir", "results"))
        self.results_dir.mkdir(exist_ok=True)
        
        self.logger.info("Enhanced Model System initialized")
    
    def _setup_logger(self) -> logging.Logger:
        """Set up logging"""
        logger = logging.getLogger("enhanced_model_system")
        logger.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        return logger
    
    def _setup_models(self):
        """Initialize all model connections"""
        model_roles = ["manager", "developer1", "developer2", "moderator"]
        
        for role in model_roles:
            if role in self.config.get("models", {}):
                self.logger.info(f"Initializing {role} model...")
                try:
                    # Create model connector
                    connector = ModelConnector(role, self.config, self.logger)
                    self.models[role] = connector
                    self.logger.info(f"Successfully initialized {role} model")
                except Exception as e:
                    self.logger.error(f"Failed to initialize {role} model: {str(e)}")
                    self.logger.error(f"System will continue without {role} model")
    
    async def run_development_process(self, user_request: str) -> Dict[str, Any]:
        """Run the development process using all models"""
        # This is a stub implementation - will be expanded in later versions
        self.logger.info(f"Processing request: {user_request[:50]}...")
        
        results = {
            "status": "completed",
            "timestamp": datetime.now().isoformat(),
            "request": user_request,
            "files": {}
        }
        
        if "manager" in self.models:
            prompt = f"Create a solution for this request: {user_request}"
            
            # Use callback if provided
            if "manager" in self.callbacks:
                full_response = ""
                async for token in self.models["manager"].generate_stream(prompt, max_tokens=4096):
                    full_response += token
                    self.callbacks["manager"](token)
                results["solution"] = full_response
            else:
                results["solution"] = await self.models["manager"].generate(prompt, max_tokens=4096)
        
        # Return results
        return results
''')
    print(f"Created stub file: {filepath}")

def create_main_script():
    """Create a simple main script for running the system"""
    with open("main.py", "w") as f:
        f.write('''#!/usr/bin/env python3
"""
Enhanced Dual Model System - Main Entry Point
Author: Emmanuel Ekpeh
Date: 2025-03-03 20:07:29
"""

import os
import sys
import yaml
import json
import asyncio
import argparse
from pathlib import Path
import logging
from typing import Dict, Any, Optional

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("main")

# Import core modules
try:
    from core.dual_model_core import EnhancedModelSystem
except ImportError as e:
    logger.error(f"Failed to import core modules: {e}")
    logger.error("Make sure setup_models.py has been run first")
    sys.exit(1)

async def run_system(config_path: str, request: str = None):
    """Run the enhanced model system"""
    logger.info(f"Starting Enhanced Dual Model System with config: {config_path}")
    
    # Initialize system with simple console logging callbacks
    def console_callback(role: str):
        def callback(token: str):
            print(token, end="", flush=True)
        return callback
    
    # Set up callbacks for different roles
    callbacks = {
        'manager': console_callback('Manager'),
        'developer1': console_callback('Developer1'),
        'developer2': console_callback('Developer2'),
        'moderator': console_callback('Moderator'),
        'status': lambda status, data: logger.info(f"Status update: {status} {data}")
    }
    
    # Create system
    system = EnhancedModelSystem(config_path, callbacks=callbacks)
    
    # If no request provided but file exists, read it
    if not request and os.path.exists('request.txt'):
        with open('request.txt', 'r') as f:
            request = f.read().strip()
    
    # If still no request, ask user
    if not request:
        print("\\n" + "="*60)
        print("Enter your development request (Ctrl+D or empty line to submit):")
        print("="*60)
        lines = []
        while True:
            try:
                line = input()
                if not line and lines:
                    break
                lines.append(line)
            except EOFError:
                break
        request = "\\n".join(lines)
    
    if not request.strip():
        logger.error("No request provided. Exiting.")
        return
    
    # Save request for reference
    with open('request.txt', 'w') as f:
        f.write(request)
    
    print("\\n" + "="*60)
    print(f"Processing request: {request[:100]}{'...' if len(request) > 100 else ''}")
    print("="*60 + "\\n")
    
    # Run development process
    try:
        results = await system.run_development_process(request)
        
        # Print summary of results
        print("\\n" + "="*60)
        print("DEVELOPMENT COMPLETE")
        print("="*60)
        
        if "summary" in results:
            print("\\nSUMMARY:")
            print(results["summary"])
        
        if "files" in results:
            print("\\nGENERATED FILES:")
            for filename in results["files"].keys():
                print(f"  - {filename}")
            
            # Print location of saved files
            if "output_dir" in results:
                print(f"\\nFiles saved to: {results['output_dir']}")
        
        if "duration" in results:
            duration = results["duration"]
            print(f"\\nTotal processing time: {duration:.2f} seconds")
        
        return results
    except Exception as e:
        logger.error(f"Error running system: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Enhanced Dual Model System")
    parser.add_argument("--config", default="config.yaml", help="Path to configuration file")
    parser.add_argument("--request", help="Development request (optional)")
    args = parser.parse_args()
    
    # Check if config exists
    if not os.path.exists(args.config):
        logger.error(f"Configuration file not found: {args.config}")
        logger.error("Please run setup_models.py first")
        sys.exit(1)
    
    # Run system
    asyncio.run(run_system(args.config, args.request))

if __name__ == "__main__":
    main()
''')
    print("Created main.py script")

def verify_setup():
    """Check if all required components are present"""
    print("🔍 Verifying setup components...")
    issues = []
    
    # Check for required directories
    for directory in ["core", "logs", "results"]:
        if not os.path.isdir(directory):
            print(f"Creating missing directory: {directory}")
            os.makedirs(directory, exist_ok=True)
    
    # BUGFIX: Use direct function calls instead of references
    # Check for required files in core
    if not os.path.exists("core/__init__.py"):
        print("Creating core/__init__.py")
        create_init_file("core")
    
    if not os.path.exists("core/model_connector.py"):
        print("Creating core/model_connector.py")
        create_model_connector("core/model_connector.py")
    
    if not os.path.exists("core/dual_model_core.py"):
        print("Creating core/dual_model_core.py")
        create_dual_model_core_stub("core/dual_model_core.py")
    
    # Check for other required files
    if not os.path.exists("requirements.txt"):
        create_requirements_file()
    
    if not os.path.exists("main.py"):
        create_main_script()
    
    print("✅ Setup verification complete")
    return len(issues) == 0

def main():
    """Main function for model setup"""
    print("\n🚀 Enhanced Dual Model System - Setup")
    print("====================================")
    print(f"Date: 2025-03-03 20:07:29  |  User: Emmanuelekpeh")
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Set up models for the Enhanced Dual Model System")
    parser.add_argument("--config", default="config.yaml", help="Path to save configuration")
    parser.add_argument("--huggingface", action="store_true", help="Use Hugging Face models instead of local models")
    parser.add_argument("--api-key", help="Hugging Face API key")
    parser.add_argument("--manager-size", default="medium", choices=["small", "medium", "large"], 
                        help="Size of manager model (Hugging Face only)")
    parser.add_argument("--dev1-size", default="medium", choices=["small", "medium", "large"], 
                        help="Size of primary developer model (Hugging Face only)")
    parser.add_argument("--dev2-size", default="medium", choices=["small", "medium", "large"], 
                        help="Size of secondary developer model (Hugging Face only)")
    parser.add_argument("--mod-size", default="small", choices=["small", "medium", "large"], 
                        help="Size of moderator model (Hugging Face only)")
    parser.add_argument("--local-moderator", action="store_true", 
                        help="Use local model for moderation (if available)")
    parser.add_argument("--force", action="store_true", 
                        help="Force overwrite existing configuration")
    args = parser.parse_args()
    
    # Check if config already exists and not forcing
    if os.path.exists(args.config) and not args.force:
        print(f"\n⚠️ Configuration file {args.config} already exists!")
        choice = input("Overwrite? (y/N): ").strip().lower()
        if choice != 'y':
            print("Setup canceled. Use --force to overwrite without prompting.")
            return
    
    # Create directory structure and verify setup first
    # BUGFIX: Do this before any file operations
    verify_setup()
    
    # Handle Hugging Face option
    if args.huggingface:
        # If API key isn't provided, prompt for it
        api_key = args.api_key
        if not api_key:
            print("\n🔑 Using Hugging Face models requires an API key for best performance.")
            print("You can get one at https://huggingface.co/settings/tokens")
            api_key = input("Enter your Hugging Face API key (or press Enter to continue without one): ").strip()
        
        # Show available model tiers
        print("\n📊 Available Model Tiers:")
        print("Manager Models:")
        for size, model in HF_MODELS["manager"].items():
            print(f"  - {size.capitalize()}: {model}")
        print("\nDeveloper Models (Primary):")
        for size, model in HF_MODELS["developer1"].items():
            print(f"  - {size.capitalize()}: {model}")
        print("\nDeveloper Models (Secondary):")
        for size, model in HF_MODELS["developer2"].items():
            print(f"  - {size.capitalize()}: {model}")
        print("\nModerator Models:")
        for size, model in HF_MODELS["moderator"].items():
            print(f"  - {size.capitalize()}: {model}")
            
        print(f"\nSelected: Manager={args.manager_size}, Dev1={args.dev1_size}, Dev2={args.dev2_size}, Mod={args.mod_size}")
        
        if args.local_moderator:
            print("Using local model for moderation if available")
    
    # Create configuration
    config = create_config(
        args.config, 
        args.huggingface, 
        api_key if args.huggingface else None,
        args.manager_size,
        args.dev1_size,
        args.dev2_size,
        args.mod_size,
        args.local_moderator
    )
    
    # Install requirements
    print("\n📦 Installing required packages...")
    try:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Packages installed successfully")
    except Exception as e:
        print(f"⚠️ Error installing packages: {str(e)}")
        print("Please install requirements manually: pip install -r requirements.txt")
    
    # Done!
    print("\n✨ Setup complete! Your system is ready to run.")
    print(f"Configuration saved to: {args.config}")
    print("\n📋 Next Steps:")
    print("1. Run the system:   python main.py")
    print("2. For help:         python main.py --help")
    
    # Create an example request file
    with open("request_example.txt", "w") as f:
        f.write("""Create a simple weather app that:
1. Fetches weather data from OpenWeatherMap API
2. Displays current conditions and 5-day forecast
3. Allows setting favorite locations
4. Has a clean, responsive UI

Use Python with Flask for the backend and React for the frontend.
""")
    print("3. Example request:  See request_example.txt")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nSetup canceled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error during setup: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)