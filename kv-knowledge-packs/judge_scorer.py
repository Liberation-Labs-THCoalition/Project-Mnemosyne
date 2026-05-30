"""Claude Judge Scorer — uses Claude subagent for semantic evaluation.

Replaces keyword matching with a Claude API call that evaluates
whether responses correctly reference the knowledge graph.

For use in powered studies where DeepSeek v2 judging is too slow
or we need higher-quality discernment.

Can be called from any experiment script:
    from judge_scorer import judge_batch
    results = judge_batch(qa_pairs, graph_triples)
"""

import json
import logging
import os
import subprocess
from pathlib import Path

log = logging.getLogger("judge")


def format_judge_prompt(qa_pairs: list[dict], triples: list[dict]) -> str:
    """Build a prompt for batch judging."""
    # Select relevant triples
    triple_lines = []
    for t in triples[:50]:
        triple_lines.append(f"  {t['s']} —[{t.get('p', 'related')}]→ {t['o']}")
    triples_text = "\n".join(triple_lines)

    qa_lines = []
    for i, qa in enumerate(qa_pairs):
        qa_lines.append(
            f"{i+1}. Q: {qa['question']}\n"
            f"   Expected: {qa['expected']}\n"
            f"   Response: {qa['response']}"
        )
    qa_text = "\n".join(qa_lines)

    return f"""Judge whether each response correctly answers a question about this knowledge graph.

GRAPH TRIPLES:
{triples_text}

QUESTIONS AND RESPONSES:
{qa_text}

For each numbered item, respond with EXACTLY:
N. CORRECT|PARTIAL|WRONG — brief reason

CORRECT = identifies a valid graph relationship (even if worded differently)
PARTIAL = shows graph awareness but imprecise or incomplete
WRONG = hallucinated, refused, or irrelevant

Then on the last line: TOTALS: Xc Yp Zw"""


def judge_batch(qa_pairs: list[dict], triples: list[dict],
                condition_name: str = "") -> dict:
    """Judge a batch of Q/A pairs using Claude as subagent.

    Returns dict with per-item verdicts and aggregate scores.
    """
    prompt = format_judge_prompt(qa_pairs, triples)

    # Write prompt to temp file for subagent
    prompt_file = Path(f"/tmp/judge_prompt_{condition_name}.txt")
    prompt_file.write_text(prompt)

    log.info(f"Judging {len(qa_pairs)} pairs for {condition_name}")

    # For now, return the prompt — caller spawns the subagent
    return {
        "condition": condition_name,
        "n_pairs": len(qa_pairs),
        "prompt_file": str(prompt_file),
        "prompt": prompt,
    }


def parse_verdicts(judge_response: str, n_items: int) -> dict:
    """Parse judge response into structured verdicts."""
    import re

    verdicts = []
    for line in judge_response.strip().split("\n"):
        m = re.match(r'\d+\.\s*(CORRECT|PARTIAL|WRONG)\s*[—-]\s*(.*)', line, re.IGNORECASE)
        if m:
            verdicts.append({
                "verdict": m.group(1).upper(),
                "reason": m.group(2).strip(),
            })

    correct = sum(1 for v in verdicts if v["verdict"] == "CORRECT")
    partial = sum(1 for v in verdicts if v["verdict"] == "PARTIAL")
    wrong = sum(1 for v in verdicts if v["verdict"] == "WRONG")
    total = len(verdicts) or 1

    return {
        "verdicts": verdicts,
        "correct": correct,
        "partial": partial,
        "wrong": wrong,
        "score": (correct + 0.5 * partial) / total,
    }
