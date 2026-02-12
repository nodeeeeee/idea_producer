import httpx
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from abc import ABC, abstractmethod

class ResearchResult(BaseModel):
    title: str
    url: str
    snippet: str
    source: str
    date: Optional[str] = None
    credibility_note: Optional[str] = None

class SearchProvider(ABC):
    @abstractmethod
    def search(self, query: str, limit: int = 5) -> List[ResearchResult]:
        pass

class MockSearchProvider(SearchProvider):
    def search(self, query: str, limit: int = 5) -> List[ResearchResult]:
        real_papers = [
            ResearchResult(
                title="The AI Scientist: Towards Fully Automated Machine Learning",
                url="https://arxiv.org/abs/2408.06292",
                snippet="Introduces the first comprehensive system for fully automated scientific discovery, enabling LLM agents to generate hypotheses, write code, and execute experiments.",
                source="Sakana AI / arXiv",
                date="2024-08-12",
                credibility_note="Pioneering work in autonomous research agents."
            ),
            ResearchResult(
                title="SWE-agent: Agent-Computer Interfaces Enable LLMs to Solve Software Issues",
                url="https://arxiv.org/abs/2405.15793",
                snippet="Describes the SWE-agent system which uses a specialized Agent-Computer Interface (ACI) to allow LLMs to browse, edit, and execute code within repositories.",
                source="Princeton University / arXiv",
                date="2024-05-24",
                credibility_note="State-of-the-art performance on SWE-bench."
            ),
            ResearchResult(
                title="Agentless: Demystifying LLM-based Software Engineering Agents",
                url="https://arxiv.org/abs/2407.01489",
                snippet="A critical analysis showing that simple, non-agentic workflows (finding, localizing, and fixing) can outperform complex autonomous agents in software repair.",
                source="arXiv",
                date="2024-07-01",
                credibility_note="Highlights the importance of precision over complexity."
            ),
            ResearchResult(
                title="OpenDevin: An Open Platform for AI Software Developers",
                url="https://github.com/OpenDevin/OpenDevin",
                snippet="An open-source initiative to build autonomous AI agents capable of handling complex software engineering tasks in collaboration with humans.",
                source="OpenDevin / GitHub",
                date="2024-03-12",
                credibility_note="Leading community-driven agent platform."
            ),
            ResearchResult(
                title="CodePlan: Repository-level Planning with Large Language Models",
                url="https://arxiv.org/abs/2309.12499",
                snippet="Proposes a framework for repository-level multi-step planning, enabling agents to handle changes that span across multiple files and modules.",
                source="Microsoft Research / arXiv",
                date="2023-09-22",
                credibility_note="Foundation for complex repo-scale reasoning."
            )
        ]
        return real_papers[:limit]

class BraveSearchProvider(SearchProvider):
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.base_url = "https://api.search.brave.com/res/v1/web/search"

    def search(self, query: str, limit: int = 5) -> List[ResearchResult]:
        if not self.api_key:
            raise ValueError("Brave API Key is required")
        
        headers = {"Accept": "application/json", "X-Subscription-Token": self.api_key}
        params = {"q": query, "count": limit}
        
        response = httpx.get(self.base_url, headers=headers, params=params)
        response.raise_for_status()
        data = response.json()
        
        results = []
        for item in data.get("web", {}).get("results", []):
            results.append(ResearchResult(
                title=item.get("title", ""),
                url=item.get("url", ""),
                snippet=item.get("description", ""),
                source="Brave Search",
                credibility_note="Retrieved via Brave Search API"
            ))
        return results

class LLMSearchProvider(SearchProvider):
    def search(self, query: str, limit: int = 5) -> List[ResearchResult]:
        from llama_index.core.program import LLMTextCompletionProgram
        from llama_index.core import Settings
        from pydantic import BaseModel

        class SearchResults(BaseModel):
            results: List[ResearchResult]

        # Refined prompt to avoid "real-time" refusal
        prompt = (
            "You are a Senior Research Architect. Based on your internal knowledge of the state-of-the-art "
            "in software engineering and AI, identify {limit} highly relevant, authoritative research papers "
            "or technical methodologies that directly address the following project query.\n\n"
            "QUERY: {query}\n\n"
            "For each result, provide the exact title, a valid canonical URL (e.g. arXiv, GitHub, or ACM), "
            "a technical summary, the source, and the approximate publication year (2020-2024).\n"
            "Output as valid JSON matching the SearchResults schema."
        )

        try:
            program = LLMTextCompletionProgram.from_defaults(
                output_cls=SearchResults,
                prompt_template_str=prompt,
                verbose=True
            )
            
            print(f"Agent is analyzing internal research knowledge for: {query[:50]}...")
            output = program(query=query, limit=limit)
            return output.results
        except Exception as e:
            print(f"Warning: Knowledge search failed ({e}). Falling back to baseline authoritative papers.")
            # Fallback to the real papers defined in MockSearchProvider
            fallback = MockSearchProvider().search(query, limit)
            return fallback

class WebResearcher:
    def __init__(self, provider: SearchProvider):
        self.provider = provider

    def perform_research(self, topic: str, depth: int = 1) -> List[ResearchResult]:
        # In a real agent, we'd use an LLM to generate multiple queries
        # For now, we use the topic directly
        results = self.provider.search(topic, limit=5 * depth)
        return results
