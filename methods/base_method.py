"""
Base abstract class for all GDPR analysis methods.

All methods (LLM, ReAct, RAG, Rule-based) 
must implement this interface for unified integration into predict.py
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional


class BaseMethod(ABC):
    """Abstract base class for GDPR compliance detection methods."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the method with optional configuration.
        
        Args:
            config: Method-specific configuration dictionary
        """
        self.config = config or {}
        self.initialize()
    
    @abstractmethod
    def initialize(self):
        """
        Initialize method-specific resources (models, tools, etc.).
        Called during __init__.
        """
        pass
    
    @abstractmethod
    def predict_file_level(self, file_path: str, code: str, **kwargs) -> List[int]:
        """
        Predict GDPR violations at file level.
        
        Args:
            file_path: Path to the file being analyzed
            code: Full file content
            **kwargs: Additional method-specific parameters
            
        Returns:
            List of violated GDPR article numbers (e.g., [6, 7, 32])
        """
        pass
    
    @abstractmethod
    def predict_module_level(self, file_path: str, module_name: str, 
                            code: str, **kwargs) -> List[int]:
        """
        Predict GDPR violations at module/class level.
        
        Args:
            file_path: Path to the file
            module_name: Name of the module/class
            code: Module/class content
            **kwargs: Additional method-specific parameters
            
        Returns:
            List of violated GDPR article numbers
        """
        pass
    
    @abstractmethod
    def predict_line_level(self, file_path: str, line_spans: str, 
                          code: str, description: str, **kwargs) -> List[int]:
        """
        Predict GDPR violations at line level.
        
        Args:
            file_path: Path to the file
            line_spans: Line range (e.g., "10-15" or "42")
            code: Code snippet for the specified lines
            description: Violation description
            **kwargs: Additional method-specific parameters
            
        Returns:
            List of violated GDPR article numbers
        """
        pass
    
    @abstractmethod
    def predict_snippet(self, snippet: str, snippet_path: str = "", **kwargs) -> List[int]:
        """
        Predict GDPR violations for a code snippet (Task 2).
        
        Args:
            snippet: Code snippet to analyze
            snippet_path: Path/identifier for the snippet
            **kwargs: Additional method-specific parameters
            
        Returns:
            List of violated GDPR article numbers
        """
        pass
    
    def cleanup(self):
        """
        Clean up method-specific resources.
        Override if cleanup is needed.
        """
        pass
    
    def get_name(self) -> str:
        """
        Get the method name for logging and identification.
        
        Returns:
            Method name string
        """
        return self.__class__.__name__

