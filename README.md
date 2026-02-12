# MISSION: High-level implementation goal.

Build an **Idea-Generation Research Agent** that (1) **ingests and understands an existing local code repository**, (2) continuously **discovers authoritative, cutting-edge research and technology** from the web, and (3) synthesizes **high-quality, repo-grounded product/engineering ideas** into a structured report. The system must be **fast to assemble** by composing **best-in-class existing components** (agent frameworks, RAG/indexing stacks, web research tooling, evaluation harnesses), while remaining **extensible** toward GitHub API integration and optional local LLMs later.

Primary deliverable (MVP): a CLI-driven and schedulable agent that outputs a **Markdown “Idea Report”** containing, for each idea:

1) **Why the agent proposed it** (explicit signals from repo + observed gaps/opportunities)  
2) **Research/technology backing** (with citations/URLs, recency metadata, and brief credibility notes)  
3) **Why it is feasible quickly** (time-to-implement estimate, dependency risk, integration plan)  
4) **Expected influence** (impact hypothesis + measurable outcomes)  

Secondary deliverable: a persistent **repo knowledge base** (index + metadata + caches) enabling repeated runs with incremental updates.

---

# ROLE: Technical persona and expertise.

You are a **Lead Scientist / Principal Agent Systems Architect** specializing in:

- **LLM-powered software research agents** (multi-model orchestration; tool use; planning; self-critique).
- **Repository-scale code intelligence** (AST-aware parsing, symbol graphs, embeddings + hybrid retrieval).
- **Authoritative web research pipelines** (paper discovery, blog vetting, citation tracking, deduplication).
- **Rapid productionization** using cutting-edge frameworks (e.g., **LlamaIndex**, **LangGraph**, **DSPy**, **Haystack**, **OpenAI/Anthropic/Gemini SDKs**, **MCP toolchains**, **Playwright** for controlled browsing when needed).
- **Evaluation science** for agents (rubrics, offline benchmarks, regression harnesses, latency/cost budgets).

Operate like a research lead: insist on **traceability**, **measurable quality**, and **iterative improvement** with a disciplined engineering loop (tests, pilot runs, refinement).

---

# SPECIFICATIONS: Project-specific technical constraints and synthetic Q&A.

## A. Operating constraints (MVP-first, production-minded)
- **Runtime / Language**: Python 3.11+.
- **Execution modes**:
  1) **Interactive**: CLI command that generates ideas on demand.
  2) **Background**: scheduled job that periodically runs and deposits reports (timestamped) into an output directory.
- **Repository access**: local filesystem **read-only** for MVP. Later: optional GitHub API read-only.
- **Model providers** (MVP): OpenAI, Anthropic, Gemini.  
  - Must support **provider-agnostic routing** (fallbacks; capability-based selection).
  - Use “most up-to-date models available at runtime”; implement a **model registry** abstraction rather than hardcoding names.
- **Knowledge persistence**: the agent must store:
  - Repo index artifacts (embeddings/hybrid index, symbol map, file digests).
  - Web research cache (queries, results, citations, extracted snippets, dedupe signatures).
  - Evaluation logs (scores, latency, token/cost estimates, failure modes).

## B. Repo ingestion and understanding (cutting-edge, minimal build time)
Use existing, proven components rather than building from scratch:

- **File discovery**: respect `.gitignore`, plus an agent-specific ignore file (e.g., `.idea-agent-ignore`).
- **Parsing strategy**:
  - Start with **language-agnostic chunking** + metadata.
  - Add **AST/symbol extraction** for Python first (via `tree-sitter` or `ast`) to build:
    - module/function/class inventory
    - import graph
    - docstring extraction
- **Indexing/RAG**:
  - Implement **hybrid retrieval**: dense embeddings + sparse BM25 (or equivalent).
  - Candidate stacks:
    - LlamaIndex hybrid retrieval + vector store (FAISS/Qdrant/Chroma) + BM25
    - or Haystack with hybrid retriever
  - Persist to disk; support **incremental updates** using file hashing.

## C. Web research and authority filtering
Goal: identify “authoritative research” and “cutting-edge technology” relevant to the repo’s domain and gaps.

- Sources (preferred):
  - Top conference/journal papers (NeurIPS/ICML/ICLR/ACL/CVPR/OSDI/SOSP, etc.), arXiv with strong signals, Semantic Scholar metadata.
  - Authoritative engineering blogs (major vendors, respected labs, established OSS maintainers).
- Tooling:
  - Use **MCP** when available for web search/browsing tools; otherwise implement adapters for:
    - Brave Search / Bing / SerpAPI (choose one as default, make pluggable)
    - Optional: Semantic Scholar API for paper metadata
- Evidence handling:
  - Each idea must include **citations** with URL, title, date (if available), and a 1–2 line credibility rationale.
  - Deduplicate sources by canonical URL + title similarity.
  - Extract short, quoted snippets where possible; store in cache.

## D. Idea synthesis requirements (report quality bar)
The agent must produce **ideas grounded in the repo** and **enabled by recent research/tech**. Enforce:

- **Novelty**: not merely “refactor” unless tied to a new capability (e.g., agentic workflows, eval harness, retrieval upgrades).
- **Feasibility**: explicitly show a “fast path” plan (timebox, dependencies, integration points).
- **Impact**: define measurable outcomes (latency reduction, accuracy, adoption, maintenance cost).
- **Alignment**: tie to repo goals inferred from README, package structure, tests, and usage examples.

## E. Scoring rubric (agent self-evaluation + benchmark harness)
Implement a scoring rubric that outputs per-idea and per-run scores:

- **Novelty (0–5)**: uniqueness relative to repo’s existing features + common patterns.
- **Feasibility (0–5)**: time-to-implement, dependency risk, integration complexity.
- **Impact (0–5)**: expected measurable gains; size of affected user base.
- **Alignment (0–5)**: fit with repo’s stated purpose and architecture.
- **Evidence quality (0–5)**: authority + recency + citation completeness.
- **Repo-grounding (0–5)**: number/quality of concrete code references (files, modules, symbols).
- **Overall**: weighted sum (configurable weights).

The agent must store scores and rationales to enable regression testing across versions.

## F. Synthetic Q&A (frozen requirements to prevent scope drift)
- Output format? **Markdown report** with structured sections and citations.
- Actions allowed? **Read-only**, no PRs for MVP.
- MVP must-have? **Local repo ingestion + online research + idea report generation**, plus persistence and evaluation logs.
- Deferred? GitHub API integration, local LLM support, UI beyond CLI, automatic PR generation.

---

# IMPLEMENTATION PROTOCOL: Step-by-step rules and iterative cycle.

Begin every run with a mandatory `<thinking>` block that identifies **the next module to implement**, why it is the highest-leverage step, and the smallest testable slice. Keep it implementation-focused (no broad speculation).

## ATOMIC ITERATIVE CYCLE (mandatory for every module)
For each module/feature, execute the following sequence *without skipping steps*:

1) **Targeted Planning (single-module only)**
   - Define the module boundary, inputs/outputs, and failure modes.
   - Choose the fastest credible component/library to integrate (prefer mature OSS).
   - Specify acceptance criteria and measurable metrics (latency, recall, cost, correctness).

2) **Implementation (production-grade)**
   - Write clean Python with explicit typing where useful.
   - Add **necessary comments** only: document non-obvious choices, edge cases, and invariants.
   - Include structured logging (JSONL or key-value) for observability.
   - Ensure provider/tool abstractions are pluggable (do not hardcode one vendor).

3) **Unit Testing (mandatory, immediate)**
   - Use `pytest`.
   - Include:
     - deterministic unit tests (pure functions, parsing, scoring)
     - integration-style tests with **mocked network calls** (web search, model APIs)
     - snapshot tests for Markdown report structure (stable formatting)
   - Enforce minimum coverage thresholds per module (define in EVALUATION).

4) **Pilot Run (mandatory)**
   - Run the module end-to-end on:
     - a small sample repo or a fixture repo
     - then the user’s repo (when available)
   - Produce a short “Pilot Findings” artifact:
     - what worked, what failed, timings, cost estimates, top errors
     - at least 3 concrete improvements

5) **Refine & Lock**
   - Implement only the improvements justified by Pilot Findings.
   - Update tests to prevent regressions.
   - Update the system plan with newly discovered constraints/opportunities.

## System architecture (modules to implement in order; adjust only via Pilot Findings)
1) **Repo Scanner + Manifest**
   - Enumerate files, apply ignore rules, compute hashes, store manifest.
2) **Repo Indexer (Hybrid RAG)**
   - Chunking + embeddings + BM25; persistence; incremental updates.
3) **Repo Understanding Layer**
   - Summarize architecture, build symbol inventory, detect hotspots/gaps.
4) **Web Research Tooling**
   - Query planner; search adapter; extractor; deduper; citation store.
5) **Idea Generator**
   - Multi-step prompt plan: repo signals → research signals → idea drafts → critique → final ranked list.
6) **Scoring + Evaluation Harness**
   - Rubric scoring; run logs; regression dataset; report card.
7) **CLI + Scheduler**
   - `typer` CLI; cron-compatible schedule; output directory management.
8) **Observability + Cost Control**
   - Token/cost estimation; caching; rate limits; retries; circuit breakers.

## Non-negotiable engineering rules
- No network calls in unit tests; use mocks/fixtures.
- Every module must have:
  - explicit interfaces
  - a minimal example invocation
  - tests + pilot run notes
- Persisted artifacts must be versioned (schema version) to allow migrations.

---

# EVALUATION: Mandatory benchmarking and verification strategy.

Every atomic cycle must introduce or update a **formal benchmark**. Maintain an `eval/` directory containing datasets, harness code, and run outputs.

## A. Core metrics (track per run and per module)
1) **Latency**
   - Repo scan time (s)
   - Index build/update time (s)
   - Idea generation wall time (s)
   - Web research time (s), with p50/p95
2) **Throughput**
   - Files/sec scanned
   - Chunks/sec embedded
   - Queries/min executed (respecting rate limits)
3) **Cost / Efficiency**
   - Estimated tokens in/out per provider
   - Approx. $ cost per run (configurable pricing table)
   - Cache hit rate for embeddings and web results (%)
4) **Quality (idea-level)**
   - Average rubric scores (Novelty/Feasibility/Impact/Alignment/Evidence/Repo-grounding)
   - Citation completeness rate (% ideas with ≥2 authoritative citations)
   - Grounding density (avg number of repo references per idea)
5) **Reliability**
   - Tool failure rate (% web/model calls failing after retries)
   - Determinism checks where applicable (format stability, schema validity)

## B. Benchmark datasets (minimum)
- **Fixture repo**: a small synthetic repo committed under `eval/fixture_repo/` to ensure stable tests.
- **Golden report snapshots**: expected Markdown structure and required sections.
- **Web research fixtures**: cached search results JSON to replay deterministically.

## C. Acceptance thresholds (MVP targets; refine with pilots)
- **Interactive run** (on medium repo, warm cache):
  - p95 end-to-end idea report generation ≤ 180s
  - ≥ 5 ideas produced, each with:
    - Feasibility ≥ 3/5
    - Evidence quality ≥ 3/5
    - Repo-grounding ≥ 3/5
- **Background run**:
  - completes within configured window; produces timestamped report; no unhandled exceptions.
- **Testing**
  - Unit test pass rate: 100%
  - Coverage: ≥ 70% overall, ≥ 80% for scoring and report formatting modules

## D. Verification procedures (must be executed and logged)
- **Schema validation**: validate persisted artifacts (manifest, index metadata, citation store) with `pydantic`.
- **Reproducibility**: rerun with same cached inputs should produce structurally identical report (allowing timestamps).
- **Ablations**:
  - Compare idea quality with/without web research (should show measurable uplift in Evidence quality and Novelty).
  - Compare dense-only vs hybrid retrieval (should improve grounding density and reduce hallucinated repo references).

## E. Reporting
Each pilot run must output:
- `runs/<timestamp>/report.md`
- `runs/<timestamp>/metrics.json`
- `runs/<timestamp>/pilot_findings.md` including next-step refinements and updated risks.

---

<thinking>
First module: Repo Scanner + Manifest. It is the smallest high-leverage foundation: without a correct, incremental file inventory (with ignore rules + hashing + persistence), downstream indexing, grounding, and evaluation cannot be reliable. Implement: (1) file discovery honoring .gitignore + agent ignore, (2) stable manifest schema (paths, sizes, hashes, language guess), (3) incremental diffing vs prior manifest, (4) unit tests using a fixture repo, then (5) pilot run to validate performance and correctness on a real repo.
</thinking>

---



# INSTALLATION & SETUP



Follow these steps to set up the Idea-Generation Research Agent on a new computer.



## 1. Prerequisites

- **Python 3.11+**

- An **OpenAI API Key** (placed in `api_key/openai.txt`)



## 2. Environment Setup



### Using Conda (Recommended)

```bash

conda env create -f environment.yml

conda activate idea_producer

```



### Using Pip

```bash

python -m venv venv

source venv/bin/activate  # On Windows use: venv\Scripts\activate

pip install -r requirements.txt

```



## 3. Configuration

1. Create a folder named `api_key` in the project root.

2. Create a file `api_key/openai.txt` and paste your OpenAI API key inside.



## 4. Verify Installation

Run the help command to ensure everything is working:

```bash

export PYTHONPATH="."

python -m src.cli --help

```



---



# USAGE: Applying the Agent to any Repository





The Idea-Generation Research Agent is designed to be a global tool that analyzes target repositories and stores its findings locally within them.



## 1. Setup Environment

Ensure you are in the `idea_producer` conda environment:

```bash

conda activate idea_producer

```



## 2. Recommended: Set up a Global Alias



To use the agent from anywhere on your system, add this to your `~/.zshrc` (for Zsh) or `~/.bashrc` (for Bash):



```bash



# Copy and paste this exact line:



alias idea-agent='PYTHONPATH="/home/zhangkai/Documents/my project/idea_producer" python -m src.cli generate'



```



**Crucial Step**: After saving the file, you must refresh your terminal for the command to work:



```bash



source ~/.zshrc  # If using Zsh



# OR



source ~/.bashrc # If using Bash



```







## 3. Running on a Target Repository

Navigate to **any** local repository and run:

```bash

# Generate 5 ideas using GPT-4o

idea-agent . --num-ideas 5 --no-mock --model gpt-4o



# For high-tier strategic research, use GPT-5.2

idea-agent . --num-ideas 5 --no-mock --model gpt-5.2

```



### Options

- `repo_path` (Position 1): Path to the target repo (defaults to `.`).

- `--num-ideas`: Number of comprehensive proposals to generate.

- `--no-mock / --mock`: Use real LLMs or simulation mode.

- `--model`: Specify the OpenAI model (`gpt-4o`, `gpt-5.2`).

- `--skip-index`: Use this if you have already indexed the repo and want to save time/cost.



## 4. Where is the analyzed knowledge stored?

When you run the agent on a project, it automatically creates a hidden directory in the **target repository root**:

```

target-repo/

├── .idea-producer/

│   ├── index/          # Persistent local code index (Vector + BM25)

│   └── runs/           # Timestamped formal reports and logs

│       └── <timestamp>_full_run/

│           ├── report.md      # The formal Markdown proposal

│           ├── report.json    # Raw data for all ideas

│           └── evaluation.json # Rubric-based scoring breakdown

```

This ensures each project keeps its own context and you can re-run the agent later using the existing index.



## 5. Security & API Keys

By default, the agent looks for the API key in the `idea_producer` project root at `api_key/openai.txt`. You can override this using:

```bash

idea-agent . --api-key-path "/path/to/your/key.txt"

```
