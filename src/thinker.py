from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime
from llama_index.core import Settings
from llama_index.core.program import LLMTextCompletionProgram
from .models import ProjectKnowledge, KnowledgePoint, Manifest
from .indexer import RepoIndexer
from .analyzer import RepoAnalyzer

class RepoThinker:
    def __init__(self, indexer: RepoIndexer, analyzer: RepoAnalyzer):
        self.indexer = indexer
        self.analyzer = analyzer

    def synthesize_knowledge(self, manifest: Manifest, repo_path: Path) -> ProjectKnowledge:
        # 1. Get raw symbol inventory
        raw_analysis = self.analyzer.analyze_repo(repo_path, manifest)
        
        # 2. Get high-level context from index
        retriever = self.indexer.get_retriever(similarity_top_k=10)
        query = "What is the high-level architecture, purpose, and key logic of this project?"
        nodes = retriever.retrieve(query)
        repo_context = "\n".join([n.node.get_content() for n in nodes])
        
        # 3. Use LLM to think through the repo and synthesize structured knowledge
        from llama_index.core.llms.mock import MockLLM
        if not isinstance(Settings.llm, MockLLM):
             knowledge = self._real_think(repo_context, raw_analysis, manifest)
        else:
             knowledge = self._mock_think(manifest)
             
        return knowledge

    def _real_think(self, repo_context: str, raw_analysis: Dict[str, Any], manifest: Manifest) -> ProjectKnowledge:
        manifest_hash = manifest.calculate_hash()
        
        prompt_template_str = (
            "You are a Senior Principal Software Architect. Your task is to analyze the provided "
            "repository context and symbol inventory to build a comprehensive, formal, and structured "
            "Knowledge Base for this project.\n\n"
            "This Knowledge Base will be used by other agents to understand the project deeply.\n\n"
            "REPO CONTEXT (From Search):\n{repo_context}\n\n"
            "SYMBOL INVENTORY (Sample):\n{symbol_sample}\n\n"
            "TASK:\n"
            "1. Provide a sharp Executive Summary.\n"
            "2. Identify the Tech Stack.\n"
            "3. Describe the Formal Architecture.\n"
            "4. Map logical components to their responsibilities.\n"
            "5. Extract 'Key Findings' - non-obvious insights about design patterns, technical debt, "
            "   domain-specific logic, or critical paths.\n"
            "6. Explain the 'Research Context' - how this repo relates to modern engineering trends.\n\n"
            "Ensure the output is highly detailed and formal. Output as JSON matching ProjectKnowledge schema."
        )

        # Truncate symbol inventory for prompt
        symbol_sample = {k: v.model_dump() for k, v in list(raw_analysis.items())[:15]}
        
        program = LLMTextCompletionProgram.from_defaults(
            output_cls=ProjectKnowledge,
            prompt_template_str=prompt_template_str,
            verbose=True
        )
        
        print("Thinking through the repository architecture...")
        
        knowledge = program(
            repo_context=repo_context[:8000], 
            symbol_sample=str(symbol_sample)[:4000]
        )
        
        # Use model_copy to ensure fields are updated correctly in Pydantic v2
        knowledge = knowledge.model_copy(update={
            "repo_path": str(manifest.repo_path),
            "manifest_hash": str(manifest_hash)
        })
        
        return knowledge

    def _mock_think(self, manifest: Manifest) -> ProjectKnowledge:
        manifest_hash = manifest.calculate_hash()
        
        return ProjectKnowledge(
            repo_name=Path(manifest.repo_path).name,
            repo_path=str(manifest.repo_path),
            executive_summary="This is a mock executive summary for the repository.",
            tech_stack=["Python", "LlamaIndex", "Pydantic"],
            architecture_description="A modular agentic system with scanning, indexing, and generation layers.",
            component_map={
                "Scanner": "Identifies files and handles ignore rules.",
                "Indexer": "Builds hybrid RAG search capabilities.",
                "Generator": "Synthesizes ideas based on repo and web research."
            },
            key_findings=[
                KnowledgePoint(
                    category="Architecture",
                    summary="Modular decoupling",
                    detailed_insight="The system uses a clear separation between data ingestion and idea synthesis.",
                    evidence=["src/scanner.py", "src/generator.py"]
                )
            ],
            research_context=["Aligned with autonomous RAG agent research."],
            manifest_hash=manifest_hash
        )

    @staticmethod
    def to_markdown(knowledge: ProjectKnowledge) -> str:
        nl = "\n"
        md = f"# Project Knowledge Base: {knowledge.repo_name}{nl}{nl}"
        md += f"**Analyzed at**: {knowledge.analyzed_at}{nl}{nl}"
        
        md += f"## Executive Summary{nl}{knowledge.executive_summary}{nl}{nl}"
        
        md += f"## Tech Stack{nl}"
        for tech in knowledge.tech_stack:
            md += f"- {tech}{nl}"
        md += nl
        
        md += f"## Architecture Description{nl}{knowledge.architecture_description}{nl}{nl}"
        
        md += f"## Component Map{nl}"
        for name, resp in knowledge.component_map.items():
            md += f"- **{name}**: {resp}{nl}"
        md += nl
        
        md += f"## Key Findings{nl}{nl}"
        for finding in knowledge.key_findings:
            md += f"### [{finding.category}] {finding.summary}{nl}"
            md += f"{finding.detailed_insight}{nl}{nl}"
            if finding.evidence:
                md += f"**Evidence**:{nl}"
                for e in finding.evidence:
                    md += f"- `{e}`{nl}"
                md += nl
        
        md += f"## Global Research Context{nl}"
        for ctx in knowledge.research_context:
            md += f"- {ctx}{nl}"
        
        return md