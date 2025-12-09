"""
Code Extraction Utilities for Documentation Generation

This module provides tools to extract, format, and highlight code snippets
from Python source files for inclusion in academic papers and documentation.
"""

import ast
import inspect
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

try:
    from pygments import highlight
    from pygments.lexers import PythonLexer, get_lexer_by_name
    from pygments.formatters import TerminalFormatter, HtmlFormatter
    PYGMENTS_AVAILABLE = True
except ImportError:
    PYGMENTS_AVAILABLE = False
    print("Warning: pygments not available. Install with: pip install pygments")


@dataclass
class CodeSnippet:
    """Represents an extracted code snippet with metadata."""
    name: str
    code: str
    docstring: Optional[str]
    start_line: int
    end_line: int
    snippet_type: str  # 'class', 'function', 'method'
    parent_class: Optional[str] = None


class CodeExtractor:
    """
    Extract and format code snippets from Python source files.
    
    Features:
    - Extract specific classes, functions, or methods
    - Syntax highlighting with pygments
    - Format code examples for markdown
    - Generate simplified pseudocode from implementation
    - Extract docstrings and signatures
    
    Example:
        extractor = CodeExtractor()
        snippet = extractor.extract_class("spectral_gpt/wave_gpt.py", "WavePacketEmbedding")
        markdown = extractor.format_for_markdown(snippet, language="python")
    """
    
    def __init__(self):
        self.python_lexer = PythonLexer() if PYGMENTS_AVAILABLE else None
        
    def extract_class(
        self, 
        file_path: str, 
        class_name: str,
        include_methods: Optional[List[str]] = None
    ) -> Optional[CodeSnippet]:
        """
        Extract a class definition from a Python file.
        
        Args:
            file_path: Path to the Python source file
            class_name: Name of the class to extract
            include_methods: Optional list of method names to include (None = all)
            
        Returns:
            CodeSnippet object or None if not found
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
            
        with open(file_path, 'r', encoding='utf-8') as f:
            source = f.read()
            
        try:
            tree = ast.parse(source)
        except SyntaxError as e:
            print(f"Syntax error parsing {file_path}: {e}")
            return None
            
        # Find the class node
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                # Extract source lines
                lines = source.split('\n')
                start_line = node.lineno - 1
                end_line = node.end_lineno if hasattr(node, 'end_lineno') else len(lines)
                
                # If specific methods requested, filter the class
                if include_methods is not None:
                    code = self._extract_class_with_methods(
                        lines, node, include_methods
                    )
                else:
                    code = '\n'.join(lines[start_line:end_line])
                
                # Extract docstring
                docstring = ast.get_docstring(node)
                
                return CodeSnippet(
                    name=class_name,
                    code=code,
                    docstring=docstring,
                    start_line=start_line + 1,
                    end_line=end_line,
                    snippet_type='class'
                )
                
        return None
    
    def extract_function(
        self, 
        file_path: str, 
        function_name: str,
        parent_class: Optional[str] = None
    ) -> Optional[CodeSnippet]:
        """
        Extract a function or method definition from a Python file.
        
        Args:
            file_path: Path to the Python source file
            function_name: Name of the function to extract
            parent_class: If extracting a method, name of the parent class
            
        Returns:
            CodeSnippet object or None if not found
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
            
        with open(file_path, 'r', encoding='utf-8') as f:
            source = f.read()
            
        try:
            tree = ast.parse(source)
        except SyntaxError as e:
            print(f"Syntax error parsing {file_path}: {e}")
            return None
            
        lines = source.split('\n')
        
        # If parent_class specified, find method within class
        if parent_class:
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef) and node.name == parent_class:
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef) and item.name == function_name:
                            return self._create_function_snippet(
                                item, lines, function_name, parent_class, 'method'
                            )
        else:
            # Find top-level function
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name == function_name:
                    # Check if it's a top-level function (not inside a class)
                    if not any(isinstance(parent, ast.ClassDef) 
                              for parent in ast.walk(tree) 
                              if hasattr(parent, 'body') and node in getattr(parent, 'body', [])):
                        return self._create_function_snippet(
                            item, lines, function_name, None, 'function'
                        )
                        
        return None
    
    def _create_function_snippet(
        self,
        node: ast.FunctionDef,
        lines: List[str],
        name: str,
        parent_class: Optional[str],
        snippet_type: str
    ) -> CodeSnippet:
        """Helper to create a CodeSnippet from a function AST node."""
        start_line = node.lineno - 1
        end_line = node.end_lineno if hasattr(node, 'end_lineno') else len(lines)
        code = '\n'.join(lines[start_line:end_line])
        docstring = ast.get_docstring(node)
        
        return CodeSnippet(
            name=name,
            code=code,
            docstring=docstring,
            start_line=start_line + 1,
            end_line=end_line,
            snippet_type=snippet_type,
            parent_class=parent_class
        )
    
    def _extract_class_with_methods(
        self,
        lines: List[str],
        class_node: ast.ClassDef,
        method_names: List[str]
    ) -> str:
        """Extract a class with only specified methods."""
        result = []
        start_line = class_node.lineno - 1
        
        # Add class definition and docstring
        for node in class_node.body:
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
                # This is the docstring
                doc_start = node.lineno - 1
                doc_end = node.end_lineno if hasattr(node, 'end_lineno') else doc_start + 1
                result.extend(lines[start_line:doc_end])
                break
        else:
            # No docstring, just add class definition line
            result.append(lines[start_line])
            
        # Add __init__ if present (always include)
        for node in class_node.body:
            if isinstance(node, ast.FunctionDef) and node.name == '__init__':
                method_start = node.lineno - 1
                method_end = node.end_lineno if hasattr(node, 'end_lineno') else len(lines)
                result.append('')
                result.extend(lines[method_start:method_end])
                break
                
        # Add requested methods
        for node in class_node.body:
            if isinstance(node, ast.FunctionDef) and node.name in method_names:
                method_start = node.lineno - 1
                method_end = node.end_lineno if hasattr(node, 'end_lineno') else len(lines)
                result.append('')
                result.extend(lines[method_start:method_end])
                
        return '\n'.join(result)
    
    def format_for_markdown(
        self,
        snippet: CodeSnippet,
        language: str = "python",
        include_docstring: bool = True,
        max_lines: Optional[int] = None
    ) -> str:
        """
        Format a code snippet for inclusion in markdown documentation.
        
        Args:
            snippet: The code snippet to format
            language: Language identifier for syntax highlighting
            include_docstring: Whether to include the docstring as a comment
            max_lines: Maximum number of lines (truncate if longer)
            
        Returns:
            Formatted markdown string with code block
        """
        result = []
        
        # Add header comment with snippet info
        if snippet.parent_class:
            result.append(f"# {snippet.parent_class}.{snippet.name} ({snippet.snippet_type})")
        else:
            result.append(f"# {snippet.name} ({snippet.snippet_type})")
            
        if include_docstring and snippet.docstring:
            result.append(f"# {snippet.docstring.split(chr(10))[0]}")  # First line only
            
        result.append("")
        
        # Add code
        code_lines = snippet.code.split('\n')
        if max_lines and len(code_lines) > max_lines:
            code_lines = code_lines[:max_lines]
            code_lines.append("    # ... (truncated)")
            
        result.extend(code_lines)
        
        # Wrap in markdown code block
        markdown = f"```{language}\n" + '\n'.join(result) + "\n```"
        return markdown
    
    def generate_pseudocode(
        self,
        snippet: CodeSnippet,
        simplify: bool = True
    ) -> str:
        """
        Generate simplified pseudocode from a code snippet.
        
        Args:
            snippet: The code snippet to convert
            simplify: Whether to simplify implementation details
            
        Returns:
            Pseudocode as a string
        """
        lines = snippet.code.split('\n')
        pseudocode = []
        
        for line in lines:
            stripped = line.strip()
            
            # Skip empty lines and comments
            if not stripped or stripped.startswith('#'):
                continue
                
            # Simplify common patterns
            if simplify:
                # Replace torch operations with math notation
                line = re.sub(r'torch\.(\w+)', r'\1', line)
                line = re.sub(r'nn\.(\w+)', r'\1', line)
                line = re.sub(r'F\.(\w+)', r'\1', line)
                
                # Simplify type hints
                line = re.sub(r': \w+(\[.*?\])?', '', line)
                line = re.sub(r' -> \w+(\[.*?\])?', '', line)
                
                # Remove decorators except important ones
                if stripped.startswith('@') and 'property' not in stripped:
                    continue
                    
            pseudocode.append(line)
            
        return '\n'.join(pseudocode)
    
    def highlight_code(
        self,
        code: str,
        language: str = "python",
        formatter: str = "terminal"
    ) -> str:
        """
        Apply syntax highlighting to code.
        
        Args:
            code: The code to highlight
            language: Programming language
            formatter: Output format ('terminal', 'html')
            
        Returns:
            Highlighted code string
        """
        if not PYGMENTS_AVAILABLE:
            return code
            
        try:
            lexer = get_lexer_by_name(language)
            if formatter == "html":
                fmt = HtmlFormatter(style='monokai')
            else:
                fmt = TerminalFormatter()
                
            return highlight(code, lexer, fmt)
        except Exception as e:
            print(f"Error highlighting code: {e}")
            return code
    
    def extract_signature(
        self,
        snippet: CodeSnippet
    ) -> str:
        """
        Extract just the function/method signature.
        
        Args:
            snippet: The code snippet
            
        Returns:
            Function signature as a string
        """
        lines = snippet.code.split('\n')
        signature_lines = []
        
        for line in lines:
            signature_lines.append(line)
            if ':' in line and 'def ' in line:
                # Found the end of the signature
                break
                
        return '\n'.join(signature_lines)
    
    def extract_api_example(
        self,
        file_path: str,
        example_name: str = "main"
    ) -> Optional[str]:
        """
        Extract usage examples from a file (typically from __main__ block).
        
        Args:
            file_path: Path to the Python source file
            example_name: Name of the example function or 'main' for __main__ block
            
        Returns:
            Example code as a string or None if not found
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
            
        with open(file_path, 'r', encoding='utf-8') as f:
            source = f.read()
            
        # Look for if __name__ == "__main__": block
        if example_name == "main":
            match = re.search(
                r'if __name__ == ["\']__main__["\']\s*:(.*?)(?=\n\S|\Z)',
                source,
                re.DOTALL
            )
            if match:
                return match.group(1).strip()
                
        # Otherwise look for a function with the given name
        snippet = self.extract_function(str(file_path), example_name)
        return snippet.code if snippet else None
