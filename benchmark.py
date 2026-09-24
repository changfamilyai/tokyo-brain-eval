#!/usr/bin/env python3
"""
Tokyo Brain Evaluation Harness — LongMemEval Benchmark Runner

Downloads the LongMemEval dataset from HuggingFace, stores conversations
into the memory system, then runs recall queries and scores accuracy.

Usage:
    export TOKYO_BRAIN_API_KEY=tb-your-key
    python benchmark.py
"""

import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone

try:
    from datasets import load_dataset
except ImportError:
    print("Please install: pip install datasets")
    sys.exit(1)

try:
    from tokyo_brain import Brain, TokyoBrainError
except ImportError:
    print("Please install: pip install tokyo-brain")
    sys.exit(1)

from config import (
    DATASET_NAME,
    MATCH_THRESHOLD,
    MEMORY_API_KEY,
    MEMORY_API_URL,
    RESULTS_DIR,
    TOP_K,
)

# Errors a Brain call can raise: API errors (TokyoBrainError), transport errors
# (requests.RequestException subclasses OSError) and malformed JSON (ValueError).
BRAIN_ERRORS = (TokyoBrainError, OSError, ValueError)


def load_longmemeval():
    """Load LongMemEval dataset from HuggingFace."""
    print(f"Loading {DATASET_NAME} from HuggingFace...")
    ds = load_dataset(DATASET_NAME, split="test")
    print(f"  Loaded {len(ds)} questions")
    return ds


def store_conversations(brain, dataset):
    """Store all source conversations into the memory system."""
    print("Storing conversations...")
    stored = set()
    count = 0

    for item in dataset:
        # Each item has source conversations that should be stored
        conversations = item.get("conversations", item.get("history", []))
        session_id = item.get("session_id", item.get("id", f"session_{count}"))

        if session_id in stored:
            continue
        stored.add(session_id)

        # Store each conversation turn
        if isinstance(conversations, list):
            for i, conv in enumerate(conversations):
                if isinstance(conv, dict):
                    text = conv.get("content", conv.get("text", str(conv)))
                    role = conv.get("role", "user")
                elif isinstance(conv, str):
                    text = conv
                    role = "user" if i % 2 == 0 else "assistant"
                else:
                    continue

                doc = f"{role}: {text}"
                try:
                    brain.store(
                        document=doc,
                        track="memories",
                        metadata={
                            "session_id": session_id,
                            "turn": i,
                            "role": role,
                            "source": "longmemeval",
                        },
                    )
                    count += 1
                except BRAIN_ERRORS as e:
                    print(f"  Store error: {e}")

        if count % 100 == 0 and count > 0:
            print(f"  Stored {count} turns...")

    print(f"  Total: {count} turns stored from {len(stored)} sessions")
    return count


def evaluate(brain, dataset):
    """Run recall queries and score accuracy."""
    print("Evaluating...")
    results_by_dim = defaultdict(lambda: {"correct": 0, "total": 0})
    total_correct = 0
    total_questions = 0

    for i, item in enumerate(dataset):
        question = item.get("question", item.get("query", ""))
        expected = item.get("answer", item.get("expected", ""))
        dimension = item.get("dimension", item.get("category", "unknown"))

        if not question or not expected:
            continue

        # Recall from memory system
        try:
            result = brain.recall(query=question, top_k=TOP_K)
            recalled_texts = [m.document for m in result.memories]
            recalled_combined = " ".join(recalled_texts).lower()
        except BRAIN_ERRORS as e:
            print(f"  Recall error on Q{i}: {e}")
            recalled_combined = ""

        # Score: check if expected answer appears in recalled context
        expected_lower = expected.lower().strip()
        is_correct = False

        # Exact substring match
        if expected_lower in recalled_combined:
            is_correct = True
        else:
            # Fuzzy: check if key phrases from expected appear
            key_phrases = [p.strip() for p in expected_lower.split(",") if len(p.strip()) > 3]
            if key_phrases:
                matches = sum(1 for p in key_phrases if p in recalled_combined)
                if matches / len(key_phrases) >= MATCH_THRESHOLD:
                    is_correct = True

        results_by_dim[dimension]["total"] += 1
        if is_correct:
            results_by_dim[dimension]["correct"] += 1
            total_correct += 1
        total_questions += 1

        if (i + 1) % 50 == 0:
            pct = total_correct / total_questions * 100 if total_questions else 0
            print(f"  Progress: {i+1}/{len(dataset)} ({pct:.1f}% so far)")

    return results_by_dim, total_correct, total_questions


def print_results(results_by_dim, total_correct, total_questions):
    """Print formatted results."""
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    for dim in sorted(results_by_dim.keys()):
        r = results_by_dim[dim]
        pct = r["correct"] / r["total"] * 100 if r["total"] else 0
        bar = "█" * int(pct / 2)
        print(f"  {dim:35s} {r['correct']:>3d}/{r['total']:<3d} ({pct:5.1f}%) {bar}")

    overall = total_correct / total_questions * 100 if total_questions else 0
    print(f"\n  {'OVERALL':35s} {total_correct:>3d}/{total_questions:<3d} ({overall:5.1f}%)")
    print("=" * 60)

    return overall


def save_results(results_by_dim, total_correct, total_questions, overall):
    """Save results to JSON file."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(RESULTS_DIR, f"longmemeval_{timestamp}.json")

    data = {
        "benchmark": "LongMemEval",
        "dataset": DATASET_NAME,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "api_url": MEMORY_API_URL,
        "top_k": TOP_K,
        "match_threshold": MATCH_THRESHOLD,
        "overall_score": overall,
        "total_correct": total_correct,
        "total_questions": total_questions,
        "dimensions": {
            dim: {
                "correct": r["correct"],
                "total": r["total"],
                "accuracy": r["correct"] / r["total"] * 100 if r["total"] else 0,
            }
            for dim, r in results_by_dim.items()
        },
    }

    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nResults saved to {filepath}")
    return filepath


# === MAIN ===
if __name__ == "__main__":
    if not MEMORY_API_KEY:
        print("Error: Set TOKYO_BRAIN_API_KEY environment variable")
        sys.exit(1)

    print("Tokyo Brain Evaluation Harness")
    print(f"  API: {MEMORY_API_URL}")
    print(f"  Dataset: {DATASET_NAME}")
    print(f"  Top-K: {TOP_K}")
    print()

    # Initialize
    brain = Brain(api_key=MEMORY_API_KEY, base_url=MEMORY_API_URL)

    # Health check
    try:
        health = brain.health()
        print(f"  Brain status: {health.status} (v{health.version})")
    except BRAIN_ERRORS as e:
        print(f"  Warning: Health check failed: {e}")

    # Load dataset
    dataset = load_longmemeval()

    # Phase 1: Store conversations
    t0 = time.time()
    store_count = store_conversations(brain, dataset)
    store_time = time.time() - t0
    print(f"  Store time: {store_time:.1f}s")

    # Wait for async indexing
    print("  Waiting 30s for async indexing...")
    time.sleep(30)

    # Phase 2: Evaluate
    t0 = time.time()
    results_by_dim, total_correct, total_questions = evaluate(brain, dataset)
    eval_time = time.time() - t0
    print(f"  Eval time: {eval_time:.1f}s")

    # Print & save
    overall = print_results(results_by_dim, total_correct, total_questions)
    save_results(results_by_dim, total_correct, total_questions, overall)

    print(f"\nDone. Overall: {overall:.1f}%")
