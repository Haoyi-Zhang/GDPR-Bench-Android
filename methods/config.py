"""
Configuration for different GDPR analysis methods.
"""
import os

# Method configurations
METHOD_CONFIGS = {
    'react': {
        'model': 'gpt-4o',
        'api_base': os.environ.get('OPENAI_API_BASE', 'https://api.openai.com/v1'),
        'api_key': os.environ.get('OPENAI_API_KEY', ''),
        'max_iterations': 5,
        'temperature': 0.0,
        'timeout': 300,
    },
    'rag': {
        'model': 'gpt-4o',
        'api_base': os.environ.get('OPENAI_API_BASE', 'https://api.openai.com/v1'),
        'api_key': os.environ.get('OPENAI_API_KEY', ''),
        'embedding_model': 'text-embedding-ada-002',
        'embedding_api_base': os.environ.get('OPENAI_API_BASE', 'https://api.openai.com/v1'),
        'embedding_api_key': os.environ.get('OPENAI_EMBEDDING_API_KEY', os.environ.get('OPENAI_API_KEY', '')),
        'chunk_size': 1000,
        'chunk_overlap': 200,
        'retrieval_k': 5,
        'temperature': 0.0,
    },
    'ast-based': {
        'languages': ['java', 'kotlin', 'php', 'javascript', 'python'],
        'enable_data_flow': True,
        'enable_control_flow': True,
        'strict_mode': False,
    }
}


def get_method_config(method_name: str) -> dict:
    """
    Get configuration for a specific method.
    
    Args:
        method_name: Name of the method
        
    Returns:
        Configuration dictionary
    """
    return METHOD_CONFIGS.get(method_name.lower(), {})

