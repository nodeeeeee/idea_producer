from typing import List, Dict, Any, Optional
from .models import Idea, IdeaScore, EvaluationResult
from .researcher import WebResearcher

class IdeaEvaluator:
    def __init__(self, researcher: Optional[WebResearcher] = None, weights: Optional[Dict[str, float]] = None):
        self.researcher = researcher
        self.weights = weights or {
            "novelty": 1.0,
            "feasibility": 1.0,
            "impact": 1.5,
            "alignment": 1.0,
            "evidence_quality": 0.5,
            "repo_grounding": 1.0
        }

    def evaluate_idea(self, idea: Idea) -> EvaluationResult:
        from llama_index.core.llms.mock import MockLLM
        from llama_index.core import Settings
        
        verification_research = []
        if self.researcher and not isinstance(Settings.llm, MockLLM):
            print(f"Verifying idea '{idea.title}' with targeted online research...")
            # Generate a specific verification query
            query = f"state of the art and competitors for: {idea.title}"
            verification_research = self.researcher.perform_research(query, depth=1)
        
        if isinstance(Settings.llm, MockLLM):
            score = self._mock_score_idea(idea)
        else:
            score = self._real_score_idea(idea, verification_research)
            
        return EvaluationResult(
            idea_title=idea.title,
            score=score
        )

    def _real_score_idea(self, idea: Idea, verification_research: List[Any] = []) -> IdeaScore:
        from llama_index.core.program import LLMTextCompletionProgram
        
        research_context = ""
        if verification_research:
            research_context = "\n\nVERIFICATION RESEARCH:\n" + "\n".join(
                [f"- {r.title}: {r.snippet}" for r in verification_research]
            )

        prompt_template_str = (
            "You are a Senior Technical Reviewer. Evaluate the following product/engineering idea.\n\n"
            "IDEA:\n{idea_json}{research_context}\n\n"
            "RUBRIC (Score each 0-5):\n"
            "- Novelty: Is this truly unique? Check against the verification research.\n"
            "- Feasibility: Time-to-implement and technical risk.\n"
            "- Impact: Expected measurable gains.\n"
            "- Alignment: Fit with repo purpose.\n"
            "- Evidence Quality: Authority and recency of research.\n"
            "- Repo Grounding: Quality of concrete code references.\n\n"
            "Provide scores and a formal, critical rationale. Output as JSON matching IdeaScore schema."
        )

        program = LLMTextCompletionProgram.from_defaults(
            output_cls=IdeaScore,
            prompt_template_str=prompt_template_str,
            verbose=True
        )
        
        print(f"Evaluating idea: {idea.title}...")
        score = program(
            idea_json=idea.model_dump_json(),
            research_context=research_context
        )
        
        # Recalculate overall based on weights if the LLM didn't (or to ensure consistency)
        total_weight = sum(self.weights.values())
        scores = {
            "novelty": score.novelty,
            "feasibility": score.feasibility,
            "impact": score.impact,
            "alignment": score.alignment,
            "evidence_quality": score.evidence_quality,
            "repo_grounding": score.repo_grounding
        }
        score.overall = sum(scores[k] * self.weights[k] for k in scores) / total_weight
        
        return score

    def _mock_score_idea(self, idea: Idea) -> IdeaScore:
        # Simulate scoring
        scores = {
            "novelty": 4.0,
            "feasibility": 3.5,
            "impact": 4.5,
            "alignment": 5.0,
            "evidence_quality": 4.0,
            "repo_grounding": 4.5
        }
        
        # Calculate weighted overall
        total_weight = sum(self.weights.values())
        weighted_sum = sum(scores[k] * self.weights[k] for k in scores)
        overall = weighted_sum / total_weight
        
        return IdeaScore(
            **scores,
            overall=overall,
            rationale="Idea is strongly aligned with repo architecture and grounded in recent research."
        )

    def evaluate_report(self, report: Any) -> List[EvaluationResult]:
        return [self.evaluate_idea(idea) for idea in report.ideas]
