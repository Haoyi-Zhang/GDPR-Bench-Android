"""
Formal AST Method Adapter
Adapts FormalGDPRDetector to the BaseMethod interface

FIXED VERSION - Parameter names aligned with base class + Thread safety
"""

import threading
from typing import List, Dict, Any
from methods.base_method import BaseMethod
from methods.formal_gdpr_detector import FormalGDPRDetector


class FormalASTMethod(BaseMethod):
    """
    Formal AST-based GDPR compliance detection using first-order logic
    
    Key Features:
    - 36 formal rules based on predicate logic
    - 100% coverage of 28 GDPR articles
    - Formal proof generation
    - True data flow analysis
    - Thread-safe (using thread-local storage)
    - Multi-language support (Java, Kotlin, JavaScript, Python, C#, PHP)
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        # Use thread-local storage to avoid concurrency conflicts
        self._thread_local = threading.local()
    
    def get_name(self) -> str:
        return "Formal AST-based Analysis"
    
    def _get_detector(self) -> FormalGDPRDetector:
        """Get thread-local detector instance (thread-safe)"""
        if not hasattr(self._thread_local, 'detector'):
            self._thread_local.detector = FormalGDPRDetector()
        return self._thread_local.detector
    
    def initialize(self) -> bool:
        """Initialize the method"""
        return True
    
    def cleanup(self):
        """Clean up resources"""
        pass
    
    def predict_file_level(self, file_path: str, code: str, **kwargs) -> List[int]:
        """File-level GDPR violation detection"""
        detector = self._get_detector()
        return detector.analyze_code(code, file_path)
    
    def predict_module_level(self, file_path: str, module_name: str, code: str, **kwargs) -> List[int]:
        """Module-level GDPR violation detection"""
        detector = self._get_detector()
        return detector.analyze_code(code, file_path)
    
    def predict_line_level(self, file_path: str, line_spans: str, code: str, description: str = "", **kwargs) -> List[int]:
        """Line-level GDPR violation detection"""
        detector = self._get_detector()
        return detector.analyze_code(code, file_path)
    
    def predict_snippet(self, snippet: str, snippet_path: str = "", **kwargs) -> List[int]:
        """
        Code snippet GDPR violation detection (Task 2)
        
        Fixed: Parameter names changed from (code, code_path) to (snippet, snippet_path)
               to align with base class BaseMethod
        Fixed: Use thread-local detector to ensure concurrent safety
        """
        detector = self._get_detector()
        result = detector.analyze_code(snippet, snippet_path)
        return result if result != [0] else [0]
    
    def get_detailed_violations(self) -> List[Dict[str, Any]]:
        """Get detailed violation information"""
        detector = self._get_detector()
        return detector.get_violations()
    
    def generate_report(self) -> str:
        """Generate detection report"""
        detector = self._get_detector()
        return detector.generate_report()
    
    def generate_proof(self, violation: Dict[str, Any]) -> str:
        """Generate formal proof for a violation"""
        detector = self._get_detector()
        return detector.generate_proof(violation)


# Alias
ASTMethod = FormalASTMethod
