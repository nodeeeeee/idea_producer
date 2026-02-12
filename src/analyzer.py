import ast
from pathlib import Path
from typing import List, Dict, Any, Optional
from pydantic import BaseModel

class Symbol(BaseModel):
    name: str
    type: str  # class, function, method
    line_number: int
    docstring: Optional[str] = None
    params: List[str] = []

class FileAnalysis(BaseModel):
    path: str
    classes: List[Symbol] = []
    functions: List[Symbol] = []
    imports: List[str] = []
    loc: int = 0

class RepoAnalyzer:
    def analyze_python_file(self, file_path: Path, rel_path: str) -> FileAnalysis:
        with open(file_path, "r", errors="ignore") as f:
            code = f.read()
        
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return FileAnalysis(path=rel_path, loc=len(code.splitlines()))

        analysis = FileAnalysis(path=rel_path, loc=len(code.splitlines()))
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                analysis.classes.append(Symbol(
                    name=node.name,
                    type="class",
                    line_number=node.lineno,
                    docstring=ast.get_docstring(node)
                ))
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # Check if it's a method (inside a class)
                # For simplicity, we'll just check all functions
                analysis.functions.append(Symbol(
                    name=node.name,
                    type="function" if isinstance(node, ast.FunctionDef) else "async_function",
                    line_number=node.lineno,
                    docstring=ast.get_docstring(node),
                    params=[arg.arg for arg in node.args.args]
                ))
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    analysis.imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    analysis.imports.append(node.module)

        return analysis

    def analyze_repo(self, repo_path: Path, manifest: Any) -> Dict[str, FileAnalysis]:
        results = {}
        for rel_path, entry in manifest.files.items():
            if entry.language == "python":
                abs_path = repo_path / rel_path
                results[rel_path] = self.analyze_python_file(abs_path, rel_path)
        return results
