from typing import List, Dict, Any
from pathlib import Path
from llama_index.core import Settings
from llama_index.core.llms.mock import MockLLM
from .models import Idea, IdeaReport, Citation, Manifest, ProjectKnowledge
from .indexer import RepoIndexer
from .researcher import WebResearcher, ResearchResult

class IdeaGenerator:
    def __init__(self, indexer: RepoIndexer, researcher: WebResearcher):
        self.indexer = indexer
        self.researcher = researcher

    def generate_ideas(self, manifest: Manifest, knowledge: ProjectKnowledge, num_ideas: int = 5) -> IdeaReport:
        # 1. Use synthesized knowledge as the primary core context
        repo_context = self.get_context_from_knowledge(knowledge)
        
        # 2. Dynamically determine research topic based on deep knowledge
        research_topic = self._generate_research_topic(repo_context)
        print(f"Researching: {research_topic}...")
        
        research_results = self.researcher.perform_research(research_topic)
        
        # 3. Use LLM to generate ideas and research findings
        if isinstance(Settings.llm, MockLLM):
            report_data = self._mock_generate_report(repo_context, research_results, num_ideas)
        else:
            report_data = self._real_generate_report(repo_context, research_results, num_ideas)
        
        return IdeaReport(
            repo_path=manifest.repo_path,
            ideas=report_data["ideas"],
            research_findings=report_data["research_findings"],
            metrics={"num_research_results": len(research_results)}
        )

    def get_context_from_knowledge(self, knowledge: ProjectKnowledge) -> str:
        nl = "\n"
        context = f"PROJECT: {knowledge.repo_name}{nl}"
        context += f"SUMMARY: {knowledge.executive_summary}{nl}"
        context += f"ARCHITECTURE: {knowledge.architecture_description}{nl}"
        context += f"TECH STACK: {', '.join(knowledge.tech_stack)}{nl}"
        context += "KEY FINDINGS:\n"
        for f in knowledge.key_findings:
            context += f"- [{f.category}] {f.summary}: {f.detailed_insight}{nl}"
        return context

    def _generate_research_topic(self, repo_context: str) -> str:
        if isinstance(Settings.llm, MockLLM):
            return "agentic software research tools 2026"
            
        from llama_index.core.prompts import PromptTemplate
        
        prompt = PromptTemplate(
            "Analyze the following repository context and provide a single, high-leverage "
            "web research query (e.g. 'state of the art in X 2025') that would help "
            "generate innovative product ideas for this project.\n\n"
            "CONTEXT:\n{repo_context}\n\n"
            "QUERY:"
        )
        
        response = Settings.llm.complete(prompt.format(repo_context=repo_context[:4000]))
        return response.text.strip().strip('"')

    def _real_generate_report(self, repo_context: str, research: List[ResearchResult], num: int) -> Dict[str, Any]:
        from llama_index.core.program import LLMTextCompletionProgram
        from pydantic import BaseModel, Field
        from .models import ResearchFinding, Idea

        class ResearchAnalysis(BaseModel):
            research_findings: List[ResearchFinding]

        class SingleIdea(BaseModel):
            idea: Idea

        # STEP 1: RESEARCH ANALYSIS
        analysis_prompt = (
            "You are a Principal Agent Systems Architect. Analyze the provided web research.\n"
            "Extract the most relevant 'Key Idea' from each source and explain its specific "
            "'Relevance' to this repository's current architecture and future goals.\n\n"
            "REPO CONTEXT:\n{repo_context}\n\n"
            "WEB RESEARCH:\n{research_context}\n\n"
            "Output as JSON matching ResearchAnalysis schema."
        )
        
        research_context = "\n".join([f"- {r.title}: {r.snippet[:300]} ({r.url})" for r in research])
        
        analysis_program = LLMTextCompletionProgram.from_defaults(
            output_cls=ResearchAnalysis,
            prompt_template_str=analysis_prompt,
            verbose=True
        )
        
        print("Performing research analysis...")
        analysis_output = analysis_program(repo_context=repo_context[:5000], research_context=research_context)
        
        # STEP 2: ITERATIVE PROPOSAL SYNTHESIS
        ideas = []
        idea_prompt = (
            "You are a Lead Scientist. Based on the repo and research, generate a single formal engineering proposal.\n"
            "This is proposal {index} of {total}.\n\n"
            "REPO CONTEXT:\n{repo_context}\n\n"
            "RESEARCH FINDINGS:\n{findings}\n\n"
            "The proposal must be comprehensive (Rationale, Detailed Description, Implementation Plan, etc.) "
            "and formal. Avoid repeating previous ideas: {previous_titles}"
        )
        
        idea_program = LLMTextCompletionProgram.from_defaults(
            output_cls=SingleIdea,
            prompt_template_str=idea_prompt,
            verbose=True
        )
        
        findings_str = "\n".join([f"- {f.paper_title}: {f.key_idea}" for f in analysis_output.research_findings])
        
        for i in range(num):
            print(f"Generating formal proposal {i+1}/{num}...")
            previous_titles = ", ".join([id.title for id in ideas])
            res = idea_program(
                index=i+1, 
                total=num, 
                repo_context=repo_context[:5000], 
                findings=findings_str,
                previous_titles=previous_titles
            )
            ideas.append(res.idea)
            
        return {"ideas": ideas, "research_findings": analysis_output.research_findings}

    def _mock_generate_report(self, repo_context: str, research: List[ResearchResult], num: int) -> Dict[str, Any]:
        import random
        from .models import ResearchFinding
        
        # Generate mock research findings
        findings = [
            ResearchFinding(
                paper_title=res.title,
                url=res.url,
                key_idea="Agentic reasoning loops improve accuracy in coding tasks.",
                relevance_to_repo="Can be used to improve the IdeaGenerator's output quality."
            ) for res in research[:3]
        ]
        
        # Better repo name detection
        storage_path = Path(self.indexer.storage_dir)
        if storage_path.name == "index" and storage_path.parent.name == ".idea-producer":
            repo_name = storage_path.parent.parent.name
        else:
            repo_name = storage_path.parent.name

        ideas = [
            Idea(
                title=f"Advanced Agentic Tooling Idea {i+1} for {repo_name}",
                rationale="The repo currently lacks automated evaluation for agentic workflows.",
                detailed_description="This proposal introduces a comprehensive evaluation harness...",
                research_backing=[Citation(title=research[0].title, url=research[0].url, source=research[0].source)],
                implementation_plan="1. Define metrics. 2. Implement evaluator. 3. Integrate with CLI.",
                feasibility="High. Estimated 3 days.",
                risks_and_mitigations="Risk: API cost. Mitigation: Caching.",
                impact="Reduces hallucination rate by 20%.",
                success_metrics="Pass rate on SWE-bench.",
                grounding_references=["src/generator.py"]
            )
            for i in range(num)
        ]
        
        return {"ideas": ideas, "research_findings": findings}
