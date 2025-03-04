"""
GitHub-based database implementation for AutoMigraine
Stores and retrieves data from JSON files in the repository
"""

import os
import json
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Union

class GitDatabase:
    """A simple database using Git repository files"""
    
    def __init__(self, base_dir: str = "data"):
        """Initialize the database with the base directory"""
        self.base_dir = Path(base_dir)
        self.results_dir = self.base_dir / "results"
        self.queue_dir = self.base_dir / "queue"
        self.config_dir = self.base_dir / "config"
        
        # Create directories if they don't exist
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.queue_dir.mkdir(parents=True, exist_ok=True)
        self.config_dir.mkdir(parents=True, exist_ok=True)
    
    def save_result(self, task_id: str, data: Dict[str, Any]) -> str:
        """Save task results to the results directory"""
        # Add metadata
        if "timestamp" not in data:
            data["timestamp"] = datetime.utcnow().isoformat()
        if "id" not in data:
            data["id"] = task_id
            
        # Save to file
        file_path = self.results_dir / f"{task_id}.json"
        with open(file_path, "w") as f:
            json.dump(data, f, indent=2)
            
        return str(file_path)
    
    def get_result(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve task results by ID"""
        file_path = self.results_dir / f"{task_id}.json"
        
        if not file_path.exists():
            return None
            
        with open(file_path, "r") as f:
            return json.load(f)
    
    def list_results(self, limit: int = 10, offset: int = 0) -> List[Dict[str, Any]]:
        """List recent results with pagination"""
        # Get all JSON files in the results directory
        json_files = list(self.results_dir.glob("*.json"))
        
        # Sort by modification time (most recent first)
        json_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        # Apply pagination
        json_files = json_files[offset:offset + limit]
        
        # Load and return the data
        results = []
        for file_path in json_files:
            with open(file_path, "r") as f:
                results.append(json.load(f))
                
        return results
    
    def add_to_queue(self, task: Dict[str, Any]) -> str:
        """Add a task to the processing queue"""
        # Generate ID if not provided
        if "id" not in task:
            task["id"] = f"task_{int(datetime.utcnow().timestamp())}"
            
        # Add metadata
        if "timestamp" not in task:
            task["timestamp"] = datetime.utcnow().isoformat()
        if "status" not in task:
            task["status"] = "pending"
            
        # Save to file
        file_path = self.queue_dir / f"{task['id']}.json"
        with open(file_path, "w") as f:
            json.dump(task, f, indent=2)
            
        return task["id"]
    
    def get_queue_items(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get items in the queue, optionally filtered by status"""
        queue_items = []
        
        for file_path in self.queue_dir.glob("*.json"):
            with open(file_path, "r") as f:
                item = json.load(f)
                
            if status is None or item.get("status") == status:
                queue_items.append(item)
                
        return queue_items
    
    def update_queue_item(self, task_id: str, updates: Dict[str, Any]) -> bool:
        """Update a queue item with new values"""
        file_path = self.queue_dir / f"{task_id}.json"
        
        if not file_path.exists():
            return False
            
        # Load existing data
        with open(file_path, "r") as f:
            data = json.load(f)
            
        # Update with new values
        data.update(updates)
        
        # Save back to file
        with open(file_path, "w") as f:
            json.dump(data, f, indent=2)
            
        return True
    
    def move_to_archive(self, task_id: str) -> bool:
        """Move a processed queue item to the archive"""
        source_path = self.queue_dir / f"{task_id}.json"
        archive_dir = self.queue_dir / "archive"
        archive_dir.mkdir(exist_ok=True)
        target_path = archive_dir / f"{task_id}.json"
        
        if not source_path.exists():
            return False
            
        # Load data and update status
        with open(source_path, "r") as f:
            data = json.load(f)
            
        data["archived_at"] = datetime.utcnow().isoformat()
        
        # Save to archive
        with open(target_path, "w") as f:
            json.dump(data, f, indent=2)
            
        # Remove from queue
        os.remove(source_path)
        
        return True
    
    def save_config(self, name: str, config: Dict[str, Any], format: str = "json") -> str:
        """Save configuration data"""
        if format.lower() == "yaml":
            file_path = self.config_dir / f"{name}.yaml"
            with open(file_path, "w") as f:
                yaml.dump(config, f)
        else:
            file_path = self.config_dir / f"{name}.json"
            with open(file_path, "w") as f:
                json.dump(config, f, indent=2)
                
        return str(file_path)
    
    def get_config(self, name: str, format: str = "auto") -> Optional[Dict[str, Any]]:
        """Retrieve configuration data"""
        # Try to determine format if auto
        if format == "auto":
            yaml_path = self.config_dir / f"{name}.yaml"
            json_path = self.config_dir / f"{name}.json"
            
            if yaml_path.exists():
                file_path = yaml_path
                format = "yaml"
            elif json_path.exists():
                file_path = json_path
                format = "json"
            else:
                return None
        else:
            extension = "yaml" if format.lower() == "yaml" else "json"
            file_path = self.config_dir / f"{name}.{extension}"
            
        if not file_path.exists():
            return None
            
        # Load based on format
        with open(file_path, "r") as f:
            if format.lower() == "yaml":
                return yaml.safe_load(f)
            else:
                return json.load(f)
