"""Quality suite: wrapper around EleutherAI's lm-evaluation-harness.

Install: pip install lm-eval

Exposes two presets:
    quick: consistency-adjacent quality checks (HellaSwag, ARC-Easy) - ~10 min on 0.5B
    full: MMLU, HellaSwag, ARC-Challenge, ARC-Easy, GSM8K, IFEval, TruthfulQA - hours on 8B

Each task already has a canonical accuracy metric reported by lm-eval. We keep the
raw lm-eval dict so report.py can diff runs per-task.
"""

from dataclasses import dataclass, asdict, field
from typing import List


QUICK_TASKS = ["hellaswag", "arc_easy"]
FULL_TASKS = ["hellaswag", "arc_easy", "arc_challenge", "mmlu", "gsm8k",
              "ifeval", "truthfulqa_mc2"]


@dataclass
class QualityResult:
    tasks: List[str] = field(default_factory=list)
    raw: dict = field(default_factory=dict)  # full lm-eval results JSON
    scores: dict = field(default_factory=dict)  # flat {task: primary_metric}

    def to_dict(self):
        return asdict(self)


def run(hf_backend, preset: str = "quick", limit: int = None) -> QualityResult:
    """Run lm-evaluation-harness against an HFBackend's wrapped model."""
    try:
        from lm_eval import simple_evaluate
        from lm_eval.models.huggingface import HFLM
    except ImportError as e:
        raise ImportError(
            "quality suite requires lm-eval. install with: "
            "pip install lm-eval"
        ) from e

    tasks = {"quick": QUICK_TASKS, "full": FULL_TASKS}[preset]

    lm = HFLM(pretrained=hf_backend.model, tokenizer=hf_backend.tokenizer,
              batch_size=1)
    raw = simple_evaluate(model=lm, tasks=tasks, limit=limit, log_samples=False)

    # Flatten primary metrics per task for quick reporting
    scores = {}
    for task, metrics in raw.get("results", {}).items():
        # Primary metric convention: acc if present, else acc_norm, else first numeric
        for key in ("acc,none", "acc_norm,none", "exact_match,none"):
            if key in metrics:
                scores[task] = metrics[key]
                break
        else:
            nums = [(k, v) for k, v in metrics.items() if isinstance(v, (int, float))]
            if nums:
                scores[task] = nums[0][1]

    return QualityResult(tasks=tasks, raw=raw, scores=scores)
