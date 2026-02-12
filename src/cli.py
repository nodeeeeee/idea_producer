import typer
import json
from pathlib import Path
from datetime import datetime
from llama_index.core import Settings
from llama_index.core.embeddings import MockEmbedding
from llama_index.core.llms import MockLLM

from .scanner import RepoScanner
from .indexer import RepoIndexer
from .researcher import WebResearcher, MockSearchProvider
from .generator import IdeaGenerator
from .evaluator import IdeaEvaluator
from .observability import CostTracker, setup_structured_logging
from .thinker import RepoThinker
from .analyzer import RepoAnalyzer
from .models import ProjectKnowledge

from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
import os

app = typer.Typer(help="Idea-Generation Research Agent CLI")

# Detect project root (directory containing src/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_API_KEY = PROJECT_ROOT / "api_key" / "openai.txt"
DEFAULT_RUNS_DIR = PROJECT_ROOT / "runs"
DEFAULT_INDICES_DIR = PROJECT_ROOT / "data" / "indices"

def setup_mocks():
    # Placeholder for real configuration
    Settings.embed_model = MockEmbedding(embed_dim=1536)
    Settings.llm = MockLLM()

def setup_openai(api_key: str, model: str = "gpt-4o"):
    import httpx
    os.environ["OPENAI_API_KEY"] = api_key
    
    # Create a custom httpx client with a 10-minute timeout
    # This is the most reliable way to override the 60s default
    timeout = httpx.Timeout(600.0, connect=60.0)
    client = httpx.Client(timeout=timeout)
    
    Settings.llm = OpenAI(model=model, http_client=client, max_retries=3)
    Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small", http_client=client)

@app.command()
def generate(
    repo_path: str = typer.Argument(".", help="Path to the repository"),
    output_dir: str = typer.Option(None, help="Directory for output (defaults to repo/.idea-producer/runs)"),
    num_ideas: int = typer.Option(5, help="Number of ideas to generate"),
    mock: bool = typer.Option(True, help="Use mocked components"),
    skip_index: bool = typer.Option(False, help="Skip indexing if index already exists"),
    api_key_path: str = typer.Option(str(DEFAULT_API_KEY), help="Path to OpenAI API key"),
    model: str = typer.Option("gpt-4o", help="OpenAI model to use")
):
    """Run the full pipeline to generate and evaluate ideas."""
    if mock:
        setup_mocks()
    else:
        api_key_p = Path(api_key_path)
        if not api_key_p.exists():
            typer.echo(f"Error: API key file not found at {api_key_p}")
            raise typer.Exit(1)
        with open(api_key_p, "r") as f:
            api_key = f.read().strip()
        setup_openai(api_key, model=model)
        
    repo_path_obj = Path(repo_path).resolve()
    agent_dir = repo_path_obj / ".idea-producer"
    agent_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. Output Dir (inside target repo)
    if output_dir:
        run_dir = Path(output_dir)
    else:
        run_dir = agent_dir / "runs" / f"{timestamp}_full_run"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    setup_structured_logging(run_dir / "run.log")
    cost_tracker = CostTracker()
    
    typer.echo(f"Starting run for {repo_path_obj}...")
    
    # 1. Scan
    typer.echo("Scanning repository...")
    scanner = RepoScanner(str(repo_path_obj))
    manifest = scanner.scan()
    
    # 2. Index (inside target repo)
    storage_dir = agent_dir / "index"
    storage_dir.mkdir(parents=True, exist_ok=True)
    
    indexer = RepoIndexer(storage_dir=str(storage_dir))
    if skip_index:
        typer.echo("Skipping indexing as requested...")
        indexer.load_or_create()
    else:
        typer.echo("Checking/Updating repository index...")
        indexer.update_index(manifest, repo_path_obj)
        # Estimate embedding cost: manifest files content
        total_chars = sum(f.size for f in manifest.files.values() if f.language)
        cost_tracker.estimate_and_add(" " * total_chars, "", model="gpt-4-turbo")
    
    # 2.1. Think (Synthesize Knowledge)
    knowledge_path = agent_dir / "knowledge.md"
    knowledge_json_p = agent_dir / "knowledge.json"
    
    current_manifest_hash = manifest.calculate_hash()
    
    knowledge = None
    if knowledge_json_p.exists():
        try:
            with open(knowledge_json_p, "r") as f:
                content = f.read()
                old_knowledge = ProjectKnowledge.model_validate_json(content)
                if old_knowledge.manifest_hash == current_manifest_hash:
                    typer.echo("Repository unchanged. Loading existing Knowledge Base...")
                    knowledge = old_knowledge
        except Exception as e:
            typer.echo(f"Warning: Could not load existing knowledge base: {e}")

    if not knowledge:
        typer.echo("Analyzing repository architecture and building knowledge base...")
        analyzer = RepoAnalyzer()
        thinker = RepoThinker(indexer, analyzer)
        knowledge = thinker.synthesize_knowledge(manifest, repo_path_obj)
        
        # Save Knowledge Base inside target repo
        with open(knowledge_path, "w") as f:
            f.write(RepoThinker.to_markdown(knowledge))
        
        with open(knowledge_json_p, "w") as f:
            f.write(knowledge.model_dump_json(indent=2))
    
    # 3. Generate
    typer.echo("Generating comprehensive ideas...")
    if mock:
        provider = MockSearchProvider()
    else:
        from .researcher import LLMSearchProvider
        provider = LLMSearchProvider()
        
    researcher = WebResearcher(provider)
    generator = IdeaGenerator(indexer, researcher)
    report = generator.generate_ideas(manifest, knowledge, num_ideas=num_ideas)
    # Estimate generation cost: context + report
    context_size = len(generator.get_context_from_knowledge(knowledge))
    report_size = len(report.model_dump_json())
    cost_tracker.estimate_and_add(" " * context_size, " " * report_size, model="gpt-4-turbo")
    
    # 4. Evaluate
    typer.echo("Evaluating and verifying ideas with targeted research...")
    evaluator = IdeaEvaluator(researcher=researcher)
    eval_results = evaluator.evaluate_report(report)
    cost_tracker.add_usage(500, 100) # Simulated evaluation cost
    
    # 5. Save Report
    report_json = run_dir / "report.json"
    with open(report_json, "w") as f:
        f.write(report.model_dump_json(indent=2))
        
    eval_json = run_dir / "evaluation.json"
    with open(eval_json, "w") as f:
        json.dump([r.model_dump(mode='json') for r in eval_results], f, indent=2)
        
    # Save Metrics & Cost
    summary = cost_tracker.get_summary()
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(summary, f, indent=2)
        
    # Generate Markdown summary
    summary_path = run_dir / "report.md"
    nl = chr(10)
    with open(summary_path, "w") as f:
        f.write(f"# Idea Generation Report{nl}")
        f.write(f"Repo: {repo_path_obj.name}{nl}")
        f.write(f"Date: {timestamp}{nl}{nl}")
        f.write(f"## Cost Summary{nl}")
        f.write(f"- Total Cost: ${summary['total_cost_usd']:.4f}{nl}")
        f.write(f"- Tokens: {summary['token_usage']['input']} in / {summary['token_usage']['output']} out{nl}{nl}")
        
        f.write(f"## Research Base & Analyzed Papers{nl}")
        if report.research_findings:
            for finding in report.research_findings:
                f.write(f"### Paper: {finding.paper_title}{nl}")
                f.write(f"- **URL**: {finding.url}{nl}")
                f.write(f"- **Key Idea**: {finding.key_idea}{nl}")
                f.write(f"- **Relevance**: {finding.relevance_to_repo}{nl}{nl}")
        else:
            f.write(f"No specific research findings extracted.{nl}{nl}")
            
        f.write(f"---{nl}{nl}")
        f.write(f"## Proposed Ideas{nl}{nl}")
        for i, (idea, eval_res) in enumerate(zip(report.ideas, eval_results)):
            f.write(f"## PROPOSAL {i+1}: {idea.title}{nl}")
            f.write(f"**Overall Quality Score: {eval_res.score.overall:.2f}/5.00**{nl}{nl}")
            
            f.write(f"### 1. RATIONALE{nl}{idea.rationale}{nl}{nl}")
            
            f.write(f"### 2. DETAILED DESCRIPTION{nl}{idea.detailed_description}{nl}{nl}")
            
            f.write(f"### 3. RESEARCH BACKING & CITATIONS{nl}")
            for c in idea.research_backing:
                f.write(f"- **{c.title}** ({c.source}): {c.url}{nl}")
            f.write(f"{nl}")
            
            f.write(f"### 4. TECHNICAL IMPLEMENTATION PLAN{nl}{idea.implementation_plan}{nl}{nl}")
            
            f.write(f"### 5. FEASIBILITY ANALYSIS{nl}{idea.feasibility}{nl}{nl}")
            
            f.write(f"### 6. RISKS AND MITIGATIONS{nl}{idea.risks_and_mitigations}{nl}{nl}")
            
            f.write(f"### 7. EXPECTED IMPACT{nl}{idea.impact}{nl}{nl}")
            
            f.write(f"### 8. SUCCESS METRICS{nl}{idea.success_metrics}{nl}{nl}")
            
            f.write(f"### 9. REPOSITORY GROUNDING{nl}")
            for ref in idea.grounding_references:
                f.write(f"- `{ref}`{nl}")
            f.write(f"{nl}")
            
            f.write(f"---{nl}{nl}")

    typer.echo(f"Run completed successfully! Results in {run_dir}")
    typer.echo(f"Estimated Cost: ${summary['total_cost_usd']:.4f}")

@app.command()
def scan(repo_path: str = typer.Argument(".", help="Path to the repository")):
    """Scan the repository and print file count."""
    scanner = RepoScanner(repo_path)
    manifest = scanner.scan()
    typer.echo(f"Scanned {len(manifest.files)} files in {repo_path}")

if __name__ == "__main__":
    app()
