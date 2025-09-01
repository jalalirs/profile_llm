# JSON output

"""JSON output formatting and validation"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)

class OutputSchema(BaseModel):
    """Pydantic model for validating output JSON schema"""
    
    run_id: str
    suite: str
    timestamp: str
    duration: Optional[int] = None
    
    config: Dict[str, Any]
    results: Dict[str, Any] = {}
    system: Dict[str, Any] = {}
    profiling: Dict[str, Any] = {}
    
    status: str = "initialized"
    error: Optional[str] = None

class JSONOutputFormatter:
    """Handles formatting and validation of JSON output"""
    
    @staticmethod
    def format_experiment_result(experiment_result) -> Dict[str, Any]:
        """Format experiment result for JSON output"""
        return experiment_result.to_dict()
    
    @staticmethod
    def save_to_file(data: Dict[str, Any], filepath: str, pretty: bool = True) -> None:
        """Save data to JSON file"""
        try:
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            
            with open(filepath, 'w') as f:
                if pretty:
                    json.dump(data, f, indent=2, default=str)
                else:
                    json.dump(data, f, default=str)
            
            logger.info(f"Results saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save results to {filepath}: {str(e)}")
            raise
    
    @staticmethod
    def load_from_file(filepath: str) -> Dict[str, Any]:
        """Load data from JSON file"""
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load results from {filepath}: {str(e)}")
            raise

def validate_output_schema(data: Dict[str, Any]) -> bool:
    """Validate output data against expected schema"""
    try:
        OutputSchema(**data)
        return True
    except ValidationError as e:
        logger.error(f"Output validation failed: {str(e)}")
        return False