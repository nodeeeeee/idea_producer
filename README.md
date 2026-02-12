# Idea-Generation Research Agent (MVP)

An advanced engineering and product research agent that deeply ingests local code repositories, synthesizes structured knowledge, and generates formal, grounded ideas backed by state-of-the-art research.

## Core Capabilities

- **Project Knowledge Base (PKB)**: Performs a deep architectural analysis of the target repository, extracting tech stacks, logical component maps, and non-obvious engineering insights.
- **Iterative Idea Synthesis**: Generates comprehensive, multi-section proposals (Rationale, Implementation Plan, Feasibility, Success Metrics) using a robust stepwise generation pipeline to prevent timeouts and ensure high-tier detail.
- **LLM-Powered Research**: Dynamically identifies project-relevant research topics and identifies real, authoritative papers (2023-2025) from the model's internal knowledge base.
- **Per-Idea Verification**: Automatically conducts targeted web research for each proposed idea during the evaluation phase to validate its novelty against the current technical landscape.
- **Multi-Repo Isolated Context**: Stores all analyzed knowledge, persistent indices (Hybrid Vector + BM25), and reports locally within the target repository's `.idea-producer/` directory.

---

## 🛠 INSTALLATION & SETUP

### 1. Prerequisites
- **Python 3.11+**
- An **OpenAI API Key** (placed in `api_key/openai.txt`)

### 2. Environment Setup

#### Using Conda (Recommended)
```bash
conda env create -f environment.yml
conda activate idea_producer
```

#### Using Pip
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Configuration
1. Create a folder named `api_key` in the project root.
2. Create a file `api_key/openai.txt` and paste your OpenAI API key inside.

---

## 🚀 USAGE: Applying the Agent to any Repository

The agent is designed to be a global tool. You can run it on any local repository by pointing to its path.

### 1. Set up a Global Alias
Add this to your `~/.zshrc` (for Zsh) or `~/.bashrc` (for Bash):
```bash
# Replace /path/to/idea_producer with the actual absolute path
alias idea-agent='PYTHONPATH="/path/to/idea_producer" python -m src.cli generate'
```
Then refresh your terminal: `source ~/.zshrc` or `source ~/.bashrc`.

### 2. Run on a Target Repository
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
- `--skip-index`: Skips the embedding phase if the index is already built (saves time/cost).

---

## 📁 Storage Structure

When run, the agent creates an isolated context inside the target repo:
```
target-repo/
├── .idea-producer/
│   ├── knowledge.md    # The Deep Synthesis Knowledge Base
│   ├── knowledge.json  # Machine-readable project insights
│   ├── index/          # Persistent Hybrid Search Index
│   └── runs/           # Timestamped formal reports and logs
│       └── <timestamp>_full_run/
│           ├── report.md      # The formal Markdown proposal
│           ├── evaluation.json # Rubric-based scoring breakdown
│           └── run.log        # Structured observability logs
```

---

## 🔬 System Architecture

1. **Scanner**: Recursive file discovery with `.gitignore` and `.idea-producer` awareness.
2. **Indexer**: Build/Update persistent hybrid RAG (FAISS + BM25) with incremental hashing.
3. **Thinker**: Deep architectural synthesis and Knowledge Base generation.
4. **Researcher**: Dynamic topic generation and LLM-driven paper discovery.
5. **Generator**: Iterative, stepwise proposal synthesis for maximum detail and stability.
6. **Evaluator**: Quantitative rubric-based scoring with per-idea verification research.
7. **Observability**: Real-time cost tracking and structured JSONL logging.