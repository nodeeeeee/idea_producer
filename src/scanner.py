import os
import xxhash
import pathspec
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set, Optional
from .models import FileEntry, Manifest

class RepoScanner:
    def __init__(self, repo_path: str, ignore_file: str = ".idea-agent-ignore"):
        self.repo_path = Path(repo_path).resolve()
        self.ignore_file = ignore_file
        self.spec = self._load_ignore_spec()

    def _load_ignore_spec(self) -> pathspec.PathSpec:
        patterns = [".git/", ".idea/", "__pycache__/", "*.pyc", "node_modules/", ".idea-producer/"]
        
        # Load .gitignore if it exists
        gitignore_path = self.repo_path / ".gitignore"
        if gitignore_path.exists():
            with open(gitignore_path, "r") as f:
                patterns.extend(f.readlines())
        
        # Load agent-specific ignore file
        agent_ignore_path = self.repo_path / self.ignore_file
        if agent_ignore_path.exists():
            with open(agent_ignore_path, "r") as f:
                patterns.extend(f.readlines())
        
        return pathspec.PathSpec.from_lines("gitignore", patterns)

    def _get_file_hash(self, file_path: Path) -> str:
        hasher = xxhash.xxh64()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(65536), b""):
                    hasher.update(chunk)
        except (FileNotFoundError, OSError, PermissionError) as e:
            # Fallback to a hash of the path if file is unreadable but exists
            return xxhash.xxh64(str(file_path)).hexdigest()
        return hasher.hexdigest()

    def _guess_language(self, file_path: Path) -> Optional[str]:
        extension_map = {
            ".py": "python",
            ".js": "javascript",
            ".ts": "typescript",
            ".tsx": "typescript-react",
            ".jsx": "javascript-react",
            ".go": "go",
            ".rs": "rust",
            ".java": "java",
            ".cpp": "cpp",
            ".c": "c",
            ".h": "cpp",
            ".hpp": "cpp",
            ".cs": "csharp",
            ".rb": "ruby",
            ".php": "php",
            ".swift": "swift",
            ".kt": "kotlin",
            ".sh": "shell",
            ".md": "markdown",
            ".json": "json",
            ".yml": "yaml",
            ".yaml": "yaml",
            ".xml": "xml",
            ".html": "html",
            ".css": "css",
            ".sql": "sql",
        }
        return extension_map.get(file_path.suffix.lower())

    def scan(self) -> Manifest:
        file_entries: Dict[str, FileEntry] = {}
        
        for root, dirs, files in os.walk(self.repo_path):
            rel_root = os.path.relpath(root, self.repo_path)
            if rel_root == ".":
                rel_root = ""
            
            # Filter directories in-place for os.walk
            dirs[:] = [d for d in dirs if not self.spec.match_file(os.path.join(rel_root, d) + "/")]
            
            for file in files:
                rel_path = os.path.join(rel_root, file)
                if not self.spec.match_file(rel_path):
                    abs_path = self.repo_path / rel_path
                    try:
                        # Only process regular files, skip broken symlinks and special files
                        if not abs_path.is_file() or abs_path.is_symlink():
                            continue
                            
                        stats = abs_path.stat()
                        
                        file_entries[rel_path] = FileEntry(
                            path=rel_path,
                            size=stats.st_size,
                            hash=self._get_file_hash(abs_path),
                            last_modified=datetime.fromtimestamp(stats.st_mtime),
                            language=self._guess_language(abs_path)
                        )
                    except (FileNotFoundError, OSError, PermissionError) as e:
                        print(f"Skipping unreadable file {abs_path}: {e}")
        
        return Manifest(repo_path=str(self.repo_path), files=file_entries)

    def get_diff(self, old_manifest: Manifest, new_manifest: Manifest) -> Dict[str, List[str]]:
        diff = {
            "added": [],
            "modified": [],
            "removed": []
        }
        
        old_files = old_manifest.files
        new_files = new_manifest.files
        
        for path in new_files:
            if path not in old_files:
                diff["added"].append(path)
            elif old_files[path].hash != new_files[path].hash:
                diff["modified"].append(path)
        
        for path in old_files:
            if path not in new_files:
                diff["removed"].append(path)
                
        return diff
