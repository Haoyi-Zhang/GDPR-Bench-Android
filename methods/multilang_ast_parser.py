"""
Multi-language AST Parser Module
Supports: Java, Python, JavaScript, Kotlin, C#, PHP
"""

import re
from typing import Optional, Dict, Any, List

# Try to import AST parsing libraries for each language
# Java
try:
    import javalang
    JAVA_AST_AVAILABLE = True
except ImportError:
    JAVA_AST_AVAILABLE = False

# Python
try:
    import ast as python_ast
    PYTHON_AST_AVAILABLE = True
except ImportError:
    PYTHON_AST_AVAILABLE = False

# JavaScript/TypeScript
try:
    import esprima
    JS_AST_AVAILABLE = True
except ImportError:
    JS_AST_AVAILABLE = False

# Tree-sitter (for Kotlin, C#, PHP)
try:
    from tree_sitter import Language, Parser
    TREE_SITTER_AVAILABLE = True
except ImportError:
    TREE_SITTER_AVAILABLE = False

# Kotlin
KOTLIN_AST_AVAILABLE = False
KOTLIN_LANGUAGE = None
if TREE_SITTER_AVAILABLE:
    try:
        from tree_sitter_kotlin import language
        KOTLIN_LANGUAGE = language
        KOTLIN_AST_AVAILABLE = True
    except ImportError:
        pass

# C#
CSHARP_AST_AVAILABLE = False
CSHARP_LANGUAGE = None
if TREE_SITTER_AVAILABLE:
    try:
        from tree_sitter_c_sharp import language
        CSHARP_LANGUAGE = language
        CSHARP_AST_AVAILABLE = True
    except ImportError:
        pass

# PHP
PHP_AST_AVAILABLE = False
PHP_LANGUAGE = None
if TREE_SITTER_AVAILABLE:
    try:
        from tree_sitter_php import language_php
        PHP_LANGUAGE = language_php
        PHP_AST_AVAILABLE = True
    except ImportError:
        pass


class MultiLangASTParser:
    """Multi-language AST Parser"""
    
    def __init__(self):
        """Initialize parser"""
        self.parsers = {}
        self._init_parsers()
    
    def _init_parsers(self):
        """Initialize parsers for each language"""
        # Tree-sitter parsers (New API - need Language wrapper)
        if TREE_SITTER_AVAILABLE:
            if KOTLIN_AST_AVAILABLE and KOTLIN_LANGUAGE:
                try:
                    kotlin_lang = Language(KOTLIN_LANGUAGE())
                    kotlin_parser = Parser(kotlin_lang)
                    self.parsers['kotlin'] = kotlin_parser
                except Exception as e:
                    print(f"Warning: Failed to init Kotlin parser: {e}")
            
            if CSHARP_AST_AVAILABLE and CSHARP_LANGUAGE:
                try:
                    csharp_lang = Language(CSHARP_LANGUAGE())
                    csharp_parser = Parser(csharp_lang)
                    self.parsers['csharp'] = csharp_parser
                except Exception as e:
                    print(f"Warning: Failed to init C# parser: {e}")
            
            if PHP_AST_AVAILABLE and PHP_LANGUAGE:
                try:
                    php_lang = Language(PHP_LANGUAGE())
                    php_parser = Parser(php_lang)
                    self.parsers['php'] = php_parser
                except Exception as e:
                    print(f"Warning: Failed to init PHP parser: {e}")
    
    def get_language_from_path(self, file_path: str) -> str:
        """Extract language type from file path"""
        if not file_path:
            return 'unknown'
        
        # Extract extension (remove line number part)
        if ':' in file_path:
            file_path = file_path.split(':')[0]
        
        ext = file_path.split('.')[-1].lower()
        
        lang_map = {
            'java': 'java',
            'py': 'python',
            'js': 'javascript',
            'ts': 'typescript',
            'jsx': 'javascript',
            'tsx': 'typescript',
            'kt': 'kotlin',
            'kts': 'kotlin',
            'cs': 'csharp',
            'php': 'php',
            'json': 'json',
            'xml': 'xml',
            'html': 'html',
        }
        
        return lang_map.get(ext, 'unknown')
    
    def parse(self, code: str, language: str) -> Optional[Any]:
        """Parse code into AST"""
        if language == 'java' and JAVA_AST_AVAILABLE:
            return self._parse_java(code)
        elif language == 'python' and PYTHON_AST_AVAILABLE:
            return self._parse_python(code)
        elif language in ['javascript', 'typescript'] and JS_AST_AVAILABLE:
            return self._parse_javascript(code)
        elif language == 'kotlin' and 'kotlin' in self.parsers:
            return self._parse_tree_sitter(code, 'kotlin')
        elif language == 'csharp' and 'csharp' in self.parsers:
            return self._parse_tree_sitter(code, 'csharp')
        elif language == 'php' and 'php' in self.parsers:
            return self._parse_tree_sitter(code, 'php')
        else:
            return None
    
    def _parse_java(self, code: str) -> Optional[Any]:
        """Parse Java code"""
        try:
            tree = javalang.parse.parse(code)
            return tree
        except Exception as e:
            return None
    
    def _parse_python(self, code: str) -> Optional[Any]:
        """Parse Python code"""
        try:
            tree = python_ast.parse(code)
            return tree
        except Exception as e:
            return None
    
    def _parse_javascript(self, code: str) -> Optional[Any]:
        """Parse JavaScript/TypeScript code"""
        try:
            tree = esprima.parseScript(code, {'loc': True, 'range': True, 'tolerant': True})
            return tree
        except Exception as e:
            try:
                # Try to parse as module
                tree = esprima.parseModule(code, {'loc': True, 'range': True, 'tolerant': True})
                return tree
            except:
                return None
    
    def _parse_tree_sitter(self, code: str, language: str) -> Optional[Any]:
        """Parse code using Tree-sitter"""
        try:
            parser = self.parsers.get(language)
            if parser:
                tree = parser.parse(bytes(code, "utf8"))
                return tree
            return None
        except Exception as e:
            return None
    
    def extract_api_calls(self, tree: Any, language: str) -> List[str]:
        """Extract API calls from AST"""
        if tree is None:
            return []
        
        if language == 'java':
            return self._extract_java_api_calls(tree)
        elif language == 'python':
            return self._extract_python_api_calls(tree)
        elif language in ['javascript', 'typescript']:
            return self._extract_js_api_calls(tree)
        elif language in ['kotlin', 'csharp', 'php']:
            return self._extract_tree_sitter_api_calls(tree, language)
        
        return []
    
    def _extract_java_api_calls(self, tree) -> List[str]:
        """Extract Java API calls"""
        api_calls = []
        try:
            for path, node in tree:
                if isinstance(node, javalang.tree.MethodInvocation):
                    api_calls.append(node.member)
        except:
            pass
        return api_calls
    
    def _extract_python_api_calls(self, tree) -> List[str]:
        """Extract Python API calls"""
        api_calls = []
        try:
            for node in python_ast.walk(tree):
                if isinstance(node, python_ast.Call):
                    if isinstance(node.func, python_ast.Name):
                        api_calls.append(node.func.id)
                    elif isinstance(node.func, python_ast.Attribute):
                        api_calls.append(node.func.attr)
        except:
            pass
        return api_calls
    
    def _extract_js_api_calls(self, tree) -> List[str]:
        """Extract JavaScript API calls"""
        api_calls = []
        try:
            def traverse(node):
                if isinstance(node, dict):
                    if node.get('type') == 'CallExpression':
                        callee = node.get('callee', {})
                        if callee.get('type') == 'Identifier':
                            api_calls.append(callee.get('name', ''))
                        elif callee.get('type') == 'MemberExpression':
                            prop = callee.get('property', {})
                            if prop.get('type') == 'Identifier':
                                api_calls.append(prop.get('name', ''))
                    
                    for key, value in node.items():
                        if isinstance(value, (dict, list)):
                            traverse(value)
                elif isinstance(node, list):
                    for item in node:
                        traverse(item)
            
            traverse(tree.toDict() if hasattr(tree, 'toDict') else tree)
        except:
            pass
        return api_calls
    
    def _extract_tree_sitter_api_calls(self, tree, language: str) -> List[str]:
        """Extract API calls from Tree-sitter AST"""
        api_calls = []
        try:
            def traverse(node):
                # Find function call nodes
                if node.type in ['call_expression', 'method_invocation', 'function_call']:
                    # Extract function name
                    for child in node.children:
                        if child.type in ['identifier', 'name', 'method_name']:
                            api_calls.append(child.text.decode('utf8'))
                
                # Recursively traverse child nodes
                for child in node.children:
                    traverse(child)
            
            traverse(tree.root_node)
        except:
            pass
        return api_calls
    
    def get_support_status(self) -> Dict[str, bool]:
        """Get support status for each language"""
        return {
            'java': JAVA_AST_AVAILABLE,
            'python': PYTHON_AST_AVAILABLE,
            'javascript': JS_AST_AVAILABLE,
            'typescript': JS_AST_AVAILABLE,
            'kotlin': KOTLIN_AST_AVAILABLE,
            'csharp': CSHARP_AST_AVAILABLE,
            'php': PHP_AST_AVAILABLE,
        }
    
    def get_fallback_patterns(self, language: str) -> Dict[str, List[str]]:
        """Get language-specific fallback regex patterns"""
        patterns = {
            'java': {
                'device_id': [r'getDeviceId\s*\(', r'getIMEI\s*\(', r'getSerialNumber\s*\('],
                'location': [r'getLocation\s*\(', r'getLatitude\s*\(', r'LocationManager'],
                'camera': [r'Camera\.open\s*\(', r'camera\.takePicture'],
            },
            'kotlin': {
                'device_id': [r'getDeviceId\s*\(', r'deviceId\s*=', r'Settings\.Secure\.getString.*ANDROID_ID'],
                'location': [r'getLocation\s*\(', r'location\s*=', r'FusedLocationProviderClient'],
                'camera': [r'Camera\.open\s*\(', r'camera\s*=.*Camera'],
            },
            'javascript': {
                'device_id': [r'navigator\.deviceId', r'getDeviceId\s*\(', r'device\.uuid'],
                'location': [r'navigator\.geolocation', r'getCurrentPosition\s*\(', r'coords\.latitude'],
                'camera': [r'getUserMedia\s*\(.*video', r'navigator\.camera'],
            },
            'csharp': {
                'device_id': [r'DeviceExtendedProperties', r'GetDeviceUniqueID\s*\(', r'DeviceId'],
                'location': [r'Geolocator', r'GetGeopositionAsync\s*\(', r'Position\.Coordinate'],
                'camera': [r'CameraCaptureUI', r'CapturePhotoAsync\s*\('],
            },
            'php': {
                'device_id': [r'\$_SERVER\[.*REMOTE_ADDR', r'get_user_ip\s*\('],
                'location': [r'geoip_', r'ip2location'],
                'file': [r'file_get_contents\s*\(', r'fopen\s*\(', r'\$_FILES'],
            },
        }
        
        return patterns.get(language, {})


# Global instance (thread-safe)
import threading

_parser_instance = None
_parser_lock = threading.Lock()

def get_parser() -> MultiLangASTParser:
    """Get parser singleton (thread-safe)"""
    global _parser_instance
    if _parser_instance is None:
        with _parser_lock:
            # Double-check locking
            if _parser_instance is None:
                _parser_instance = MultiLangASTParser()
    return _parser_instance

