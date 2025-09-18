import os
import zipfile
import json
import shutil
import aiohttp
import asyncio
from pathlib import Path
from typing import Dict, Any, Optional
import tempfile
import hashlib

from config import settings


class DynamicModelManager:
    """Manages automatic model downloading, extraction, and loading"""
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        
        # Mock central repository URL - replace with real URL
        self.central_repo_url = "https://api.synctalk2d.example.com/models"
        
        # Cache for model availability checks
        self.model_registry = {}
        
        # Track extracted models
        self.extracted_models = set()
        
        print(f"📁 Dynamic Model Manager initialized with directory: {self.models_dir.absolute()}")
    
    def get_model_path(self, model_name: str) -> Path:
        """Get the directory path for a specific model"""
        return self.models_dir / model_name
    
    def get_model_zip_path(self, model_name: str) -> Path:
        """Get the zip file path for a specific model"""
        return self.models_dir / f"{model_name}.zip"
    
    def is_model_extracted(self, model_name: str) -> bool:
        """Check if model is already extracted"""
        model_path = self.get_model_path(model_name)
        
        # Check if directory exists and has required files
        if not model_path.exists():
            return False
        
        # Check for essential files
        required_files = [
            "video.mp4",
            "aud_ave.npy", 
            "models/99.pth"  # Removed frame_index.json - not in extracted models
        ]
        
        for req_file in required_files:
            if not (model_path / req_file).exists():
                return False
        
        return True
    
    def is_model_zip_available(self, model_name: str) -> bool:
        """Check if model zip file exists locally"""
        return self.get_model_zip_path(model_name).exists()
    
    async def check_model_in_registry(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Check if model exists in central registry (mock implementation)"""
        
        # Mock registry - in real implementation, this would be an API call
        mock_registry = {
            "default_model": {
                "download_url": "https://api.synctalk2d.example.com/models/default_model.zip",
                "version": "1.0.0",
                "description": "Default SyncTalk2D model",
                "size_mb": 86.2,
                "checksum": "abc123def456"
            },
            "enhanced_model": {
                "download_url": "https://api.synctalk2d.example.com/models/enhanced_model.zip", 
                "version": "1.1.0",
                "description": "Enhanced quality model",
                "size_mb": 120.5,
                "checksum": "def456ghi789"
            }
        }
        
        # Simulate API delay
        await asyncio.sleep(0.1)
        
        return mock_registry.get(model_name)
    
    async def download_model(self, model_name: str, download_url: str) -> bool:
        """Download model from central repository"""
        
        zip_path = self.get_model_zip_path(model_name)
        
        try:
            print(f"📥 Downloading model '{model_name}' from {download_url}")
            
            # Mock download - in real implementation, use aiohttp
            # For now, copy from existing package if available
            mock_source = Path("D:/Projects/SyncTalk2D/result/optimized_package_v2.zip")
            
            if mock_source.exists():
                print(f"📋 Using mock source: {mock_source}")
                shutil.copy2(mock_source, zip_path)
                print(f"✅ Model '{model_name}' downloaded successfully")
                return True
            else:
                print(f"❌ Mock source not found: {mock_source}")
                return False
                
        except Exception as e:
            print(f"❌ Failed to download model '{model_name}': {e}")
            return False
    
    async def extract_model(self, model_name: str) -> bool:
        """Extract model zip file to model directory"""
        
        zip_path = self.get_model_zip_path(model_name)
        model_path = self.get_model_path(model_name)
        
        if not zip_path.exists():
            print(f"❌ Zip file not found: {zip_path}")
            return False
        
        try:
            print(f"📦 Extracting model '{model_name}' to {model_path}")
            
            # Remove existing directory if it exists
            if model_path.exists():
                shutil.rmtree(model_path)
            
            # Create model directory
            model_path.mkdir(parents=True, exist_ok=True)
            
            # Extract zip file
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(model_path)
            
            # Verify extraction
            if self.is_model_extracted(model_name):
                self.extracted_models.add(model_name)
                print(f"✅ Model '{model_name}' extracted successfully")
                return True
            else:
                print(f"❌ Model extraction verification failed for '{model_name}'")
                return False
                
        except Exception as e:
            print(f"❌ Failed to extract model '{model_name}': {e}")
            return False
    
    async def ensure_model_available(self, model_name: str) -> Dict[str, Any]:
        """Ensure model is available for loading (download and extract if needed)"""
        
        result = {
            "model_name": model_name,
            "success": False,
            "actions_taken": [],
            "model_path": None,
            "error": None
        }
        
        try:
            # Step 1: Check if already extracted
            if self.is_model_extracted(model_name):
                result["success"] = True
                result["model_path"] = os.path.normpath(str(self.get_model_path(model_name)))
                result["actions_taken"].append("model_already_extracted")
                return result
            
            # Step 2: Check if zip exists locally
            if not self.is_model_zip_available(model_name):
                result["actions_taken"].append("checking_registry")
                
                # Step 3: Check central registry
                registry_info = await self.check_model_in_registry(model_name)
                
                if not registry_info:
                    result["error"] = f"Model '{model_name}' not found in registry"
                    return result
                
                result["actions_taken"].append("found_in_registry")
                
                # Step 4: Download from registry
                download_success = await self.download_model(
                    model_name, 
                    registry_info["download_url"]
                )
                
                if not download_success:
                    result["error"] = f"Failed to download model '{model_name}'"
                    return result
                
                result["actions_taken"].append("downloaded")
            
            else:
                result["actions_taken"].append("zip_found_locally")
            
            # Step 5: Extract model
            extract_success = await self.extract_model(model_name)
            
            if not extract_success:
                result["error"] = f"Failed to extract model '{model_name}'"
                return result
            
            result["actions_taken"].append("extracted")
            result["success"] = True
            result["model_path"] = os.path.normpath(str(self.get_model_path(model_name)))
            
            return result
            
        except Exception as e:
            result["error"] = f"Unexpected error ensuring model availability: {e}"
            return result
    
    def list_local_models(self) -> Dict[str, Any]:
        """List all locally available models"""
        
        local_models = {
            "extracted": [],
            "zipped": [],
            "total": 0
        }
        
        # Check extracted models
        for item in self.models_dir.iterdir():
            if item.is_dir():
                model_name = item.name
                if self.is_model_extracted(model_name):
                    local_models["extracted"].append({
                        "name": model_name,
                        "path": str(item),
                        "status": "ready"
                    })
        
        # Check zip files
        for item in self.models_dir.iterdir():
            if item.is_file() and item.suffix == ".zip":
                model_name = item.stem
                if model_name not in [m["name"] for m in local_models["extracted"]]:
                    local_models["zipped"].append({
                        "name": model_name,
                        "path": str(item),
                        "status": "needs_extraction"
                    })
        
        local_models["total"] = len(local_models["extracted"]) + len(local_models["zipped"])
        
        return local_models
    
    async def cleanup_model(self, model_name: str, remove_zip: bool = False) -> Dict[str, Any]:
        """Clean up model files"""
        
        result = {
            "model_name": model_name,
            "removed_items": [],
            "errors": []
        }
        
        # Remove extracted directory
        model_path = self.get_model_path(model_name)
        if model_path.exists():
            try:
                shutil.rmtree(model_path)
                result["removed_items"].append(f"extracted_dir: {model_path}")
                
                if model_name in self.extracted_models:
                    self.extracted_models.remove(model_name)
                    
            except Exception as e:
                result["errors"].append(f"Failed to remove directory: {e}")
        
        # Remove zip file if requested
        if remove_zip:
            zip_path = self.get_model_zip_path(model_name)
            if zip_path.exists():
                try:
                    zip_path.unlink()
                    result["removed_items"].append(f"zip_file: {zip_path}")
                except Exception as e:
                    result["errors"].append(f"Failed to remove zip: {e}")
        
        return result


# Global dynamic model manager instance
dynamic_model_manager = DynamicModelManager()
