from datetime import datetime
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field

class FileEntry(BaseModel):
    path: str
    size: int
    hash: str
    last_modified: datetime
    language: Optional[str] = None

class KnowledgePoint(BaseModel):
    category: str # e.g., Architecture, Domain, Security, Debt
    summary: str
    detailed_insight: str
    evidence: List[str] # file paths, symbols, or citations

class ProjectKnowledge(BaseModel):
    repo_name: str
    repo_path: str
    analyzed_at: datetime = Field(default_factory=datetime.now)
    executive_summary: str
    tech_stack: List[str]
    architecture_description: str
    component_map: Dict[str, str] # Name -> Responsibility
    key_findings: List[KnowledgePoint]
    research_context: List[str] # How it fits in the global tech landscape
    manifest_hash: str # To detect if re-analysis is needed

class Manifest(BaseModel):
    version: str = "1.0.0"
    repo_path: str
    scanned_at: datetime = Field(default_factory=datetime.now)
    files: Dict[str, FileEntry]  # path -> FileEntry

    def calculate_hash(self) -> str:
        import hashlib
        import json
        # Use only path and hash for strictly stable caching
        data = {k: v.hash for k, v in self.files.items()}
        return hashlib.md5(json.dumps(data, sort_keys=True).encode()).hexdigest()

class Citation(BaseModel):
    title: str
    url: str
    source: str
    snippet: Optional[str] = None
    credibility_note: Optional[str] = None

class Idea(BaseModel):
    title: str
    rationale: str  # High-level "Why"
    detailed_description: str # Comprehensive explanation
    research_backing: List[Citation]
    implementation_plan: str # Step-by-step technical plan
    feasibility: str
    risks_and_mitigations: str
    impact: str
    success_metrics: str # How to measure the outcome
    grounding_references: List[str] = []

class ResearchFinding(BaseModel):
    paper_title: str
    url: str
    key_idea: str
    relevance_to_repo: str

class IdeaReport(BaseModel):
    repo_path: str
    generated_at: datetime = Field(default_factory=datetime.now)
    ideas: List[Idea]
    research_findings: List[ResearchFinding] = []
    metrics: Dict[str, Any] = {}

class IdeaScore(BaseModel):
    novelty: float  # 0-5
    feasibility: float # 0-5
    impact: float # 0-5
    alignment: float # 0-5
    evidence_quality: float # 0-5
    repo_grounding: float # 0-5
    overall: float
    rationale: str

class EvaluationResult(BaseModel):
    idea_title: str
    score: IdeaScore
    evaluated_at: datetime = Field(default_factory=datetime.now)
