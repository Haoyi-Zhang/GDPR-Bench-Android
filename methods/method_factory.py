"""
Factory for creating different GDPR analysis methods.

This factory provides a unified interface to instantiate any supported method.
"""

from typing import Dict, Any, Optional
from methods.base_method import BaseMethod

# Pre-import all methods to avoid concurrent import deadlocks
try:
    from methods.react_method import ReActMethod
    _REACT_LOADED = True
except ImportError:
    _REACT_LOADED = False

try:
    from methods.rag_method import RAGMethod
    _RAG_LOADED = True
except ImportError:
    _RAG_LOADED = False

try:
    from methods.formal_ast_method import FormalASTMethod
    _AST_LOADED = True
except ImportError:
    _AST_LOADED = False


class MethodFactory:
    """Factory class for creating GDPR analysis methods."""
    
    # Registry of available methods
    _methods = {}
    
    @classmethod
    def register(cls, method_name: str, method_class: type):
        """
        Register a new method class.
        
        Args:
            method_name: Name/identifier for the method
            method_class: The method class (must inherit from BaseMethod)
        """
        if not issubclass(method_class, BaseMethod):
            raise ValueError(f"Method class must inherit from BaseMethod")
        cls._methods[method_name.lower()] = method_class
    
    @classmethod
    def create(cls, method_name: str, config: Optional[Dict[str, Any]] = None) -> BaseMethod:
        """
        Create a method instance.
        
        Args:
            method_name: Name of the method to create
            config: Optional configuration dictionary
            
        Returns:
            Instance of the requested method
            
        Raises:
            ValueError: If method name is not recognized
        """
        method_name_lower = method_name.lower()
        
        # Handle method name variations
        method_name_map = {
            'react': 'react',
            'rag': 'rag',
            'ast': 'ast-based',
            'ast-based': 'ast-based',
        }
        
        # Map to standard name
        standard_name = method_name_map.get(method_name_lower, method_name_lower)
        
        # Try to lazy load the method if not registered
        if standard_name not in cls._methods:
            cls._lazy_load_method(standard_name)
        
        if standard_name not in cls._methods:
            available = ', '.join(cls._methods.keys())
            raise ValueError(
                f"Unknown method: {method_name}. "
                f"Available methods: {available}"
            )
        
        method_class = cls._methods[standard_name]
        return method_class(config)
    
    @classmethod
    def _lazy_load_method(cls, method_name: str):
        """Lazy load a method class when first requested (uses pre-imported classes)."""
        try:
            if method_name == 'react' and _REACT_LOADED:
                cls.register('react', ReActMethod)
            elif method_name == 'rag' and _RAG_LOADED:
                cls.register('rag', RAGMethod)
            elif method_name == 'ast-based' and _AST_LOADED:
                cls.register('ast-based', FormalASTMethod)
        except Exception as e:
            print(f"Warning: Could not register method '{method_name}': {e}")
    
    @classmethod
    def list_available_methods(cls) -> list:
        """
        List all available methods.
        
        Returns:
            List of method names
        """
        # Try to load all known methods
        for name in ['react', 'rag', 'ast-based']:
            if name not in cls._methods:
                cls._lazy_load_method(name)
        
        return sorted(cls._methods.keys())


def is_llm_method(method_name: str) -> bool:
    """
    Check if a method name represents an LLM-based method.
    
    Args:
        method_name: Name of the method
        
    Returns:
        True if it's an LLM method (gpt-4o, claude, etc.)
    """
    llm_keywords = [
        'gpt', 'claude', 'gemini', 'deepseek', 'qwen', 'o1',
        'llama', 'mistral', 'anthropic', 'openai'
    ]
    method_lower = method_name.lower()
    return any(keyword in method_lower for keyword in llm_keywords)

