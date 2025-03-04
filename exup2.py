# Enhanced Dual Model System with Together AI - Free Models Only (Part 1 of 2)

```python name=setup_together.py
#!/usr/bin/env python3
"""
Enhanced Dual Model System Setup with Together AI (Free Models)
Author: Emmanuel Ekpeh
Date: 2025-03-04 05:31:29

Optimizes model selection from Together AI's free model offerings.
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

# Curated selection of FREE models on Together AI
# These models are available without paid credits
TOGETHER_FREE_MODELS = {
    "manager": {  # Models good for planning and orchestration
        "small": "mistralai/Mistral-7B-Instruct-v0.1",
        "medium": "mistralai/Mistral-7B-Instruct-v0.2",
        "large": "NousResearch/Nous-Hermes-2-Vision-7B"
    },
    "developer1": {  # Primary code generation models
        "small": "codellama/CodeLlama-7b-Instruct-hf",  
        "medium": "open-orca/mistral-7b-openorca", 
        "large": "codellama/CodeLlama-13b-Instruct-hf"
    },
    "developer2": {  # Secondary/specialized code generation models
        "small": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "medium": "Qwen/Qwen1.5-7B-Chat", 
        "large": "mistralai/Mixtral-8x7B-Instruct-v0.1"
    },
    "moderator": {  # Lightweight models suitable for moderation
        "small": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "medium": "mistralai/Mistral-7B-Instruct-v0.1",
        "large": "NousResearch/Nous-Hermes-2-Mixtral-8x7B-SFT"
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

def test_together_connection(api_key: str) -> Tuple[bool, str]:
    """Test connection to Together AI API with free models"""
    try:
        url = "https://api.together.xyz/v1/models"
        headers = {
            "accept": "application/json",
            "authorization": f"Bearer {api_key}"
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            # Parse to identify free models
            models_data = response.json()
            all_models = models_data.get("data", [])
            
            # Filter to free models only (pricing.type = "INPUT_OUTPUT" and mode.input_cost = 0)
            free_models = []
            for model in all_models:
                pricing = model.get("pricing", {})
                # Check if the model is free by looking at input/output costs
                if pricing.get("input", {}).get("cost", 1) == 0 and pricing.get("output", {}).get("cost", 1) == 0:
                    free_models.append(model.get("id"))
            
            free_count = len(free_models)
            preview = free_models[:3] if free_models else []
            
            if free_count > 0:
                return True, f"Connected successfully. {free_count} free models available. Preview: {', '.join(preview)}"
            else:
                return True, f"Connected successfully, but no completely free models found. You may have some free credits."
        else:
            return False, f"API Error ({response.status_code}): {response.text}"
    except Exception as e:
        return False, f"Connection error: {str(e)}"

def create_config(
    config_path: str, 
    use_together: bool = True, 
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
            "version": "1.0.0-together-free",
            "results_dir": "results",
            "max_iterations": 3,
            "use_moderator": True,
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
    
    if use_together:
        # Configure Together AI models
        if not api_key:
            print("\n⚠️ Warning: No API key provided for Together AI. Models won't work without valid API key.")
        
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
        manager_model_id = TOGETHER_FREE_MODELS["manager"][manager_size]
        config["models"]["manager"] = {
            "type": "together",
            "model_id": manager_model_id,
            "api_key": api_key or "",
            "temperature": 0.7,
            "max_tokens": 2048,
            "streaming": True,
            "system_prompt": "You are a planning expert that specializes in software architecture and task management.",
            "description": f"Manager Model ({manager_size} - {manager_model_id})"
        }
        
        # Setup primary developer model
        dev1_model_id = TOGETHER_FREE_MODELS["developer1"][dev1_size]
        config["models"]["developer1"] = {
            "type": "together",
            "model_id": dev1_model_id,
            "api_key": api_key or "",
            "temperature": 0.2,  # Lower temperature for more deterministic code
            "max_tokens": 4096,
            "streaming": True,
            "system_prompt": "You are an expert developer focused on writing clean, efficient, and well-documented code.",
            "description": f"Primary Developer Model ({dev1_size} - {dev1_model_id})"
        }
        
        # Setup secondary developer model
        dev2_model_id = TOGETHER_FREE_MODELS["developer2"][dev2_size]
        config["models"]["developer2"] = {
            "type": "together",
            "model_id": dev2_model_id,
            "api_key": api_key or "",
            "temperature": 0.3,  # Slightly higher for creative solutions
            "max_tokens": 4096,
            "streaming": True,
            "system_prompt": "You are a creative developer specializing in alternative implementations and optimizations.",
            "description": f"Secondary Developer Model ({dev2_size} - {dev2_model_id})"
        }
        
        # Setup moderator model - either local or Together
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
                # No local models available, fall back to Together moderator
                print("\n⚠️ No local models found for moderation. Falling back to Together AI moderator.")
                use_local_moderator = False
        
        if not use_local_moderator:
            # Use Together for moderation
            mod_model_id = TOGETHER_FREE_MODELS["moderator"][mod_size]
            config["models"]["moderator"] = {
                "type": "together",
                "model_id": mod_model_id,
                "api_key": api_key or "",
                "temperature": 0.5,
                "max_tokens": 1024,
                "system_prompt": "You are a quality controller who evaluates code for correctness, efficiency, and adherence to best practices.",
                "streaming": True,
                "description": f"Moderator ({mod_size} - {mod_model_id})"
            }
            print(f"\n✅ Using Together AI model as moderator: {mod_model_id}")
        
        # Output configuration summary
        print("\n🔄 Configured models (FREE tier):")
        print(f"  - Manager: {manager_model_id} ({manager_size})")
        print(f"  - Primary Developer: {dev1_model_id} ({dev1_size})")
        print(f"  - Secondary Developer: {dev2_model_id} ({dev2_size})")
        print(f"  - Moderator: {config['models']['moderator']['description']}")
        
        # Test connection if API key provided
        if api_key:
            print("\nTesting connection to Together AI... ", end="", flush=True)
            success, message = test_together_connection(api_key)
            if success:
                print(f"✅ Success! {message}")
            else:
                print(f"❌ Failed: {message}")
                print("Configuration will be saved but may not work correctly.")
    else:
        # Configure local models only
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

def create_model_connector(filepath: str):
    """Create the model connector implementation for Together AI"""
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, "w") as f:
        f.write('''#!/usr/bin/env python3
"""
Enhanced Model Connector: Unified interface for multi-model system
Supports Together AI, local models, and advanced caching
Author: Emmanuel Ekpeh
Date: 2025-03-04 05:34:30
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
        elif self.model_type == "together":
            self._setup_together()
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
    
    def _setup_together(self):
        """Setup Together AI API connection"""
        self.api_key = self.model_config.get("api_key")
        self.model_id = self.model_config.get("model_id")
        
        if not self.model_id:
            raise ValueError(f"No model_id provided for {self.model_name}")
        if not self.api_key:
            raise ValueError(f"No API key provided for {self.model_name}")
        
        # Together AI API endpoint for chat completions
        self.api_url = "https://api.together.xyz/v1/chat/completions"
        self.logger.info(f"Using Together AI model: {self.model_id}")
    
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
            elif self.model_type == "together":
                return await self._generate_together(prompt, max_tokens, temperature, stop)
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
            elif self.model_type == "together":
                async for token in self._generate_together_stream(prompt, max_tokens, temperature, stop):
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
# Date: 2025-03-04 05:34:30

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
# Date: 2025-03-04 05:34:30

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
Author: Emmanuel Ekpeh
Date: 2025-03-04 05:34:30
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
        
        # Also add file handler if specified in config
        log_file = self.config.get("system", {}).get("logging", {}).get("file")
        if log_file:
            try:
                # Make sure directory exists
                os.makedirs(os.path.dirname(log_file), exist_ok=True)
                
                file_handler = logging.FileHandler(log_file)
                file_handler.setLevel(logging.INFO)
                file_handler.setFormatter(formatter)
                logger.addHandler(file_handler)
                logger.info(f"Logging to file: {log_file}")
            except Exception as e:
                print(f"Error setting up file logging: {str(e)}")
        
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
                    
                    # Log initialization details
                    model_type = self.config["models"][role]["type"]
                    model_id = self.config["models"][role].get("model_id", "(local)")
                    self.logger.info(f"Successfully initialized {role} model ({model_type}: {model_id})")
                    
                    # Test connection if it's a Together model and debug_mode is enabled
                    if model_type == "together" and self.config.get("system", {}).get("debug_mode", False):
                        asyncio.create_task(self._test_model_connection(role, connector))
                        
                except Exception as e:
                    self.logger.error(f"Failed to initialize {role} model: {str(e)}")
                    self.logger.error(traceback.format_exc())
                    self.logger.warning(f"System will continue without {role} model")
    
    async def _test_model_connection(self, role: str, model: ModelConnector):
        """Test connection to a model with a simple query"""
        try:
            self.logger.info(f"Testing connection to {role} model...")
            test_prompt = "Respond with a single word: 'connected'"
            response = await model.generate(test_prompt, max_tokens=10)
            if "connect" in response.lower():
                self.logger.info(f"Connection test successful for {role} model")
            else:
                self.logger.warning(f"Connection test for {role} model returned unexpected response: {response}")
        except Exception as e:
            self.logger.error(f"Connection test failed for {role} model: {str(e)}")
    
    async def run_development_process(self, user_request: str) -> Dict[str, Any]:
        """Run the development process using all models"""
        self.logger.info(f"Processing request: {user_request[:100]}...")
        self.current_request = user_request
        self.is_running = True
        
        start_time = time.time()
        
        # Create a results object
        results = {
            "status": "in_progress",
            "timestamp": datetime.now().isoformat(),
            "request": user_request,
            "files": {},
            "iterations": [],
            "output_dir": str(self.results_dir),
            "together_models": {}  # Track which models are Together AI models
        }
        
        # Record which models are Together AI
        for role, model in self.models.items():
            if model.model_type == "together":
                results["together_models"][role] = model.model_id
        
        # Create a unique directory for this request
        request_hash = str(abs(hash(user_request + datetime.now().isoformat())))[:10]
        output_dir = self.results_dir / f"request_{request_hash}"
        output_dir.mkdir(exist_ok=True)
        results["output_dir"] = str(output_dir)
        
        # Call status callback if provided
        if "status" in self.callbacks:
            self.callbacks["status"]("started", {"request_hash": request_hash, "output_dir": str(output_dir)})
        
        try:
            # Step 1: Manager model creates a plan
            if "status" in self.callbacks:
                self.callbacks["status"]("planning", {"phase": "starting"})
                
            plan = await self._run_manager_planning(user_request)
            results["plan"] = plan
            
            if "status" in self.callbacks:
                self.callbacks["status"]("planning", {"phase": "completed"})
            
            # Step 2: Execute development iterations
            max_iterations = self.config.get("system", {}).get("max_iterations", 3)
            
            for i in range(max_iterations):
                if "status" in self.callbacks:
                    self.callbacks["status"]("iteration", {"current": i+1, "max": max_iterations})
                    
                self.logger.info(f"Starting iteration {i+1}/{max_iterations}")
                
                # Execute iteration
                iteration_results = await self._run_development_iteration(
                    user_request, 
                    plan, 
                    i+1, 
                    max_iterations,
                    results.get("files", {}),  # Pass all previous files
                    results.get("iterations", [])  # Pass previous iterations
                )
                results["iterations"].append(iteration_results)
                
                # Store files from this iteration
                for filename, content in iteration_results.get("files", {}).items():
                    results["files"][filename] = content
                    
                    # Save to disk
                    file_path = output_dir / filename
                    os.makedirs(os.path.dirname(file_path), exist_ok=True)
                    with open(file_path, "w", encoding="utf-8") as f:
                        f.write(content)
                
                # Check if we need to continue iterations
                if iteration_results.get("completed", False):
                    self.logger.info(f"Development completed in iteration {i+1}")
                    break
            
            # Step 3: Final review with moderator if available
            if "moderator" in self.models and self.config.get("system", {}).get("use_moderator", True):
                if "status" in self.callbacks:
                    self.callbacks["status"]("review", {"phase": "starting"})
                    
                review = await self._run_moderator_review(user_request, results["files"])
                results["review"] = review
                
                if "status" in self.callbacks:
                    self.callbacks["status"]("review", {"phase": "completed"})
            
            # Generate summary
            if "status" in self.callbacks:
                self.callbacks["status"]("summary", {"phase": "starting"})
                
            results["summary"] = await self._generate_summary(user_request, results)
            results["status"] = "completed"
            results["duration"] = time.time() - start_time
            
            if "status" in self.callbacks:
                self.callbacks["status"]("summary", {"phase": "completed", "duration": results["duration"]})
            
            self.logger.info(f"Development process completed in {results['duration']:.2f} seconds")
            
        except Exception as e:
            self.logger.error(f"Error in development process: {str(e)}")
            self.logger.error(traceback.format_exc())
            results["status"] = "error"
            results["error"] = str(e)
            results["traceback"] = traceback.format_exc()
            
            if "status" in self.callbacks:
                self.callbacks["status"]("error", {"error": str(e)})
        
        finally:
            self.is_running = False
            
            # Save final results
            try:
                with open(output_dir / "results.json", "w", encoding="utf-8") as f:
                    # Use custom serialization to handle non-serializable items
                    json_str = json.dumps(
                        results, 
                        default=lambda o: str(o) if not isinstance(o, (dict, list, str, int, float, bool, type(None))) else o, 
                        indent=2
                    )
                    f.write(json_str)
            except Exception as e:
                self.logger.error(f"Error saving results: {str(e)}")
            
            if "status" in self.callbacks:
                self.callbacks["status"]("completed", {"output_dir": str(output_dir)})
                
        return results
    
    async def _run_manager_planning(self, user_request: str) -> str:
        """Run the manager model to create a development plan"""
        if "manager" not in self.models:
            self.logger.warning("No manager model available, skipping planning")
            return "No planning performed (manager model not available)"
            
        self.logger.info("Starting planning phase with manager model")
        
        # Get model type info for better prompt crafting
        model_type = self.models["manager"].model_type
        is_together = model_type == "together"
        
        # Construct planning prompt
        planning_prompt = f"""You are a project manager and architect for a software development team.
Create a detailed plan for implementing the following request:

---
{user_request}
---

Format your response as:
1. OVERVIEW: Brief summary of the request and goals
2. REQUIREMENTS: List of clear requirements derived from the request
3. ARCHITECTURE: Overall system architecture and key components 
4. DEVELOPMENT PLAN: Step by step development plan with:
   - File structure
   - Implementation sequence
   - Key algorithms and data structures
5. TESTING APPROACH: How the solution should be tested
6. POTENTIAL CHALLENGES: Any issues that might arise during implementation

Be specific and detailed but focus on what's essential to guide the development process.
"""

        # Use callback if provided
        if "manager" in self.callbacks:
            full_response = ""
            async for token in self.models["manager"].generate_stream(planning_prompt):
                full_response += token
                self.callbacks["manager"](token)
            return full_response
        else:
            # Without callback, get full response at once
            return await self.models["manager"].generate(planning_prompt)
    
    async def _run_development_iteration(
        self, 
        user_request: str, 
        plan: str, 
        iteration: int, 
        max_iterations: int,
        previous_files: Dict[str, str] = None,
        previous_iterations: List[Dict] = None
    ) -> Dict[str, Any]:
        """Run one iteration of the development process"""
        self.logger.info(f"Starting development iteration {iteration}/{max_iterations}")
        
        iteration_results = {
            "iteration": iteration,
            "files": {},
            "completed": False,
            "feedback": {}
        }
        
        previous_files = previous_files or {}
        previous_iterations = previous_iterations or []
        
        # Choose which developer model to use for this iteration
        # Alternate between dev1 and dev2 to get different perspectives
        developer_role = "developer1" if iteration % 2 == 1 else "developer2"
        
        # Fall back to the other developer if the chosen one is unavailable
        if developer_role not in self.models:
            developer_role = "developer2" if developer_role == "developer1" else "developer1"
            
        if developer_role not in self.models:
            self.logger.error("No developer models available, cannot continue")
            iteration_results["error"] = "No developer models available"
            return iteration_results
        
        # Get model type - Together or local
        model_type = self.models[developer_role].model_type
        is_together = model_type == "together"
            
        # Construct context from previously generated files
        previous_files_context = ""
        if previous_files:
            previous_files_context = "PREVIOUSLY GENERATED FILES:\n"
            for filename, content in previous_files.items():
                # For longer files, include only first part and last part
                content_preview = content
                if len(content) > 1500:
                    content_preview = content[:700] + "\n...(truncated)...\n" + content[-700:]
                
                previous_files_context += f"\n--- {filename} ---\n{content_preview}\n"
        
        # Extract feedback from previous iterations if any
        feedback_context = ""
        if len(previous_iterations) > 0 and iteration > 1:
            last_iteration = previous_iterations[-1]
            if "review" in last_iteration:
                feedback_context = f"\nFEEDBACK FROM PREVIOUS ITERATION:\n{last_iteration['review']}\n"
        
        # Construct development prompt
        progress_info = f"Iteration {iteration} of {max_iterations}"
        
        development_prompt = f"""You are an expert software developer implementing a system based on the following request and plan.
This is {progress_info}.

REQUEST:
{user_request}

DEVELOPMENT PLAN:
{plan}

{previous_files_context}
{feedback_context}

Your task is to implement the core files needed for this system. For each file:
1. Begin with "FILE: filename.ext" (proper extension for the language)
2. Follow with the complete implementation of that file
3. End with "END OF FILE"

Implement multiple files to create a working solution. Focus on:
- Write clean, well-commented, and complete code
- Include all necessary imports
- Ensure proper error handling
- Follow best practices for the language/framework
- Make files that work together as a complete system

After all files, briefly state whether the implementation is COMPLETE or needs another iteration,
and explain what aspects still need work if any.
"""

        # Get developer response
        self.logger.info(f"Running {developer_role} for implementation")
        
        # Use callback if provided
        developer_response = ""
        if developer_role in self.callbacks:
            async for token in self.models[developer_role].generate_stream(development_prompt):
                developer_response += token
                self.callbacks[developer_role](token)
        else:
            # Without callback, get full response at once
            developer_response = await self.models[developer_role].generate(development_prompt)

        # Process response to extract files
        files = self._extract_files_from_response(developer_response)
        iteration_results["files"] = files
        
        # Determine if implementation is complete
        completion_status = self._check_completion_status(developer_response)
        iteration_results["completed"] = completion_status["completed"]
        iteration_results["completion_notes"] = completion_status["notes"]
        
        # Add the raw response for debugging
        if self.config.get("system", {}).get("debug_mode", False):
            iteration_results["raw_response"] = developer_response[:10000]  # Limit for storage
        
        return iteration_results
    
    def _extract_files_from_response(self, response: str) -> Dict[str, str]:
        """Extract file contents from the model response"""
        files = {}
        
        # Look for file patterns like "FILE: filename.ext" ... "END OF FILE"
        parts = response.split("FILE: ")
        
        for part in parts[1:]:  # Skip the first part (before any FILE: marker)
            try:
                                # Extract filename
                filename_end = part.find("\n")
                if filename_end == -1:
                    continue
                    
                filename = part[:filename_end].strip()
                remaining_content = part[filename_end+1:]
                
                # Extract content (until END OF FILE or next FILE:)
                end_marker = "END OF FILE"
                content_end = remaining_content.find(end_marker)
                
                if content_end == -1:
                    # Try alternative end markers in case the model used different format
                    alternative_markers = ["END FILE", "```", "FILE:", "</FILE>", "ENDFILE"]
                    for marker in alternative_markers:
                        content_end = remaining_content.find(marker)
                        if content_end != -1:
                            end_marker = marker
                            break
                
                if content_end == -1:
                    # If no end marker found, use next triple-backtick section as possible boundary
                    triple_backtick = remaining_content.find("```")
                    if triple_backtick != -1 and triple_backtick > 10:  # Ensure some content exists
                        content_end = triple_backtick
                
                if content_end == -1:
                    # If still no end marker found, take everything until the next file or end of response
                    content = remaining_content.strip()
                else:
                    content = remaining_content[:content_end].strip()
                
                # Clean up the content - remove code block markers that might be included
                content = self._clean_code_content(content, filename)
                
                # Add to files dictionary - handle nested directories
                if "/" in filename or "\\" in filename:
                    # Ensure the file's directory structure is normalized
                    filename = filename.replace("\\", "/")
                
                files[filename] = content
                self.logger.debug(f"Extracted file: {filename} ({len(content)} bytes)")
                
            except Exception as e:
                self.logger.error(f"Error parsing file from response: {str(e)}")
                continue
                
        return files
    
    def _clean_code_content(self, content: str, filename: str) -> str:
        """Clean up code content by removing markdown code block markers"""
        # Remove leading and trailing whitespace
        content = content.strip()
        
        # Remove markdown code block markers
        if content.startswith("```"):
            # Find language identifier if any (```python, ```js, etc.)
            first_newline = content.find("\n")
            if first_newline != -1:
                # Skip the first line containing the code block start
                content = content[first_newline+1:]
        
        # Remove trailing code block end if present
        if content.endswith("```"):
            content = content[:content.rfind("```")].strip()
            
        # Check for language-specific implementations and ensure proper syntax
        ext = filename.split(".")[-1].lower() if "." in filename else ""
        
        # For Python files, ensure consistent indentation
        if ext == "py":
            # Attempt to fix common indentation issues
            lines = content.split("\n")
            has_tabs = any("\t" in line for line in lines)
            
            if has_tabs:
                # Convert tabs to spaces (4 spaces per tab)
                lines = [line.replace("\t", "    ") for line in lines]
                content = "\n".join(lines)
        
        return content
    
    def _check_completion_status(self, response: str) -> Dict[str, Any]:
        """Check if the implementation is marked as complete"""
        result = {
            "completed": False,
            "notes": ""
        }
        
        # Look for completion indicators in the last part of the response
        last_section = response[-1500:].lower()
        
        # Extract notes section after the last file
        notes_section = ""
        if "end of file" in response.lower():
            parts = response.lower().split("end of file")
            if len(parts) > 1:
                notes_section = parts[-1].strip()
        elif "```" in response:
            # Try to find completion notes after the last code block
            parts = response.split("```")
            if len(parts) > 1 and len(parts) % 2 == 1:  # Odd number means all code blocks are closed
                notes_section = parts[-1].strip()
                
        # Check for completion indicators
        completion_phrases = [
            "implementation is complete", 
            "implementation complete",
            "solution is complete",
            "solution complete",
            "no further iterations needed",
            "finished implementation",
            "fully implemented",
            "all requirements met"
        ]
        
        incomplete_phrases = [
            "another iteration",
            "need more",
            "not complete",
            "incomplete",
            "still need",
            "need to add",
            "missing functionality",
            "doesn't implement"
        ]
        
        # Check for completion phrases in the notes section and last part of response
        text_to_check = (notes_section + " " + last_section).lower()
        
        # First check for explicit completion indicators
        for phrase in completion_phrases:
            if phrase in text_to_check:
                result["completed"] = True
                result["notes"] = notes_section.strip()
                return result
        
        # Then check for explicit incompletion indicators
        for phrase in incomplete_phrases:
            if phrase in text_to_check:
                result["completed"] = False
                result["notes"] = notes_section.strip()
                return result
        
        # If no explicit indicators are found, try to infer from the notes
        if notes_section:
            if ("complet" in notes_section.lower() and 
                not any(neg in notes_section.lower() for neg in ["not complet", "incomplet"])):
                result["completed"] = True
                
        result["notes"] = notes_section.strip()
        return result
    
    async def _run_moderator_review(self, user_request: str, files: Dict[str, str]) -> str:
        """Run the moderator model to review the implementation"""
        if "moderator" not in self.models:
            self.logger.warning("No moderator model available, skipping review")
            return "No review performed (moderator model not available)"
            
        self.logger.info("Starting review phase with moderator model")
        
        # Check if the moderator is Together AI or local to adjust prompt
        is_together = self.models["moderator"].model_type == "together"
        
        # Construct context from generated files
        files_context = ""
        for filename, content in files.items():
            # For longer files, include only first part and summary
            if len(content) > 1500:
                files_context += f"\n\n--- {filename} ---\n{content[:700]}\n...(truncated, {len(content)} total chars)..."
                # Add a bit from the end too for context
                files_context += f"\n{content[-700:]}"
            else:
                files_context += f"\n\n--- {filename} ---\n{content}"
        
        # Construct review prompt with file length consideration
        prompt_files = files_context
        files_too_long = len(files_context) > 12000
        
        if files_too_long:
            # If files are too long, abbreviate them further
            prompt_files = ""
            for filename, content in files.items():
                file_ext = filename.split('.')[-1] if '.' in filename else ''
                file_size = len(content)
                
                # Show more of code files, less of data/config files
                if file_ext in ['py', 'js', 'ts', 'html', 'css', 'cpp', 'java']:
                    preview_size = 500
                else:
                    preview_size = 200
                    
                prompt_files += f"\n\n--- {filename} ({file_size} chars) ---\n"
                if file_size > preview_size * 2:
                    prompt_files += f"{content[:preview_size]}\n...(truncated)...\n{content[-preview_size:]}"
                else:
                    prompt_files += content
        
        # Construct review prompt
        review_prompt = f"""You are a senior software engineer reviewing code for quality and correctness.
Review the following implementation that was created based on this request:

REQUEST:
{user_request[:500]}{"..." if len(user_request) > 500 else ""}

IMPLEMENTATION FILES:
{prompt_files}

Provide a thorough code review covering:
1. CORRECTNESS: Does the code correctly implement the requirements?
2. QUALITY: Code structure, style, readability
3. SECURITY: Any security concerns
4. PERFORMANCE: Any performance issues
5. USABILITY: How easy will this be to use
6. IMPROVEMENTS: Specific suggestions for improvement

Be constructive and actionable in your feedback.
"""

        # Use callback if provided
        if "moderator" in self.callbacks:
            full_response = ""
            async for token in self.models["moderator"].generate_stream(review_prompt):
                full_response += token
                self.callbacks["moderator"](token)
            return full_response
        else:
            # Without callback, get full response at once
            return await self.models["moderator"].generate(review_prompt)
    
    async def _generate_summary(self, user_request: str, results: Dict[str, Any]) -> str:
        """Generate a summary of the development process"""
        # Use manager model for summary if available
        if "manager" not in self.models:
            return "Development process completed. See generated files for details."
            
        # Count files and their total size
        num_files = len(results.get("files", {}))
        total_size = sum(len(content) for content in results.get("files", {}).values())
        
        # Get the plan and review if available
        plan = results.get("plan", "No plan available")
        review = results.get("review", "No review available")
        if len(plan) > 1000:
            plan = plan[:500] + "..." + plan[-500:]
        if len(review) > 1000:
            review = review[:500] + "..." + review[-500:]
        
        # Get completion status
        iterations = results.get("iterations", [])
        completed = any(iteration.get("completed", False) for iteration in iterations)
        completion_status = "Completed" if completed else "Partial completion (max iterations reached)"
        
        # Structure of the files
        file_structure = "\n".join([f"- {filename}" for filename in results.get("files", {}).keys()])
        
        # Get Together AI model usage
        together_models = results.get("together_models", {})
        models_used = []
        for role, model_id in together_models.items():
            models_used.append(f"- {role}: {model_id}")
        
        models_info = "\n".join(models_used) if models_used else "No Together AI models used"
        
        summary_prompt = f"""You are a project manager summarizing the results of a development project.
Provide a concise executive summary of the development process and results.

REQUEST:
{user_request[:500]}{"..." if len(user_request) > 500 else ""}

STATISTICS:
- Number of files: {num_files}
- Total code size: {total_size} characters
- Completion status: {completion_status}
- Number of iterations: {len(iterations)}

TOGETHER AI MODELS USED:
{models_info}

FILE STRUCTURE:
{file_structure}

Your summary should cover:
1. OVERVIEW: A brief summary of what was developed
2. FEATURES: Key functionality implemented
3. ARCHITECTURE: High-level overview of the system design
4. USAGE: How to use the implemented system
5. LIMITATIONS: Any known limitations or issues

Keep your summary concise, technical, and informative.
"""

        # Use a lower token count for the summary to ensure it's focused
        return await self.models["manager"].generate(summary_prompt, max_tokens=1024)
                
