import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

class CostTracker:
    def __init__(self, budget_usd: float = 5.0):
        self.total_cost = 0.0
        self.budget = budget_usd
        self.token_usage = {"input": 0, "output": 0}

    def add_usage(self, input_tokens: int, output_tokens: int, model: str = "gpt-4-turbo"):
        # Modernized pricing (roughly OpenAI/Anthropic averages)
        prices = {
            "gpt-4-turbo": {"input": 0.01 / 1000, "output": 0.03 / 1000},
            "claude-3-opus": {"input": 0.015 / 1000, "output": 0.075 / 1000},
            "gemini-1.5-pro": {"input": 0.0035 / 1000, "output": 0.0105 / 1000},
            "mock": {"input": 0.0, "output": 0.0}
        }
        
        p = prices.get(model, prices["gpt-4-turbo"])
        cost = (input_tokens * p["input"]) + (output_tokens * p["output"])
        
        self.total_cost += cost
        self.token_usage["input"] += input_tokens
        self.token_usage["output"] += output_tokens
        
        if self.total_cost > self.budget:
            logging.warning(f"BUDGET EXCEEDED: {self.total_cost:.4f} > {self.budget}")

    def estimate_and_add(self, text_in: str, text_out: str, model: str = "gpt-4-turbo"):
        # Standard heuristic: 1 token ~= 4 characters for English
        input_tokens = len(text_in) // 4
        output_tokens = len(text_out) // 4
        self.add_usage(input_tokens, output_tokens, model)

    def get_summary(self) -> Dict[str, Any]:
        return {
            "total_cost_usd": self.total_cost,
            "token_usage": self.token_usage,
            "budget_usd": self.budget,
            "percent_of_budget": (self.total_cost / self.budget) * 100 if self.budget > 0 else 0
        }

def setup_structured_logging(log_file: Optional[Path] = None):
    # Setup simple JSON-ish logging
    logging.basicConfig(
        level=logging.INFO,
        format='{"timestamp": "%(asctime)s", "level": "%(levelname)s", "module": "%(module)s", "message": "%(message)s"}',
        datefmt='%Y-%m-%dT%H:%M:%S'
    )
    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setFormatter(logging.Formatter('{"timestamp": "%(asctime)s", "level": "%(levelname)s", "module": "%(module)s", "message": "%(message)s"}'))
        logging.getLogger().addHandler(fh)
