# Tokyo Brain Evaluation Harness

Reproducible benchmark runner for evaluating AI memory systems on [LongMemEval](https://huggingface.co/datasets/xiaowu0205/LongMemEval).

## What this does

1. Downloads the LongMemEval dataset (500 questions across 6 cognitive dimensions)
2. Stores synthetic multi-session conversations into the memory system
3. Runs recall queries and compares against ground truth
4. Reports per-dimension accuracy and overall score

## Quick Start

```bash
pip install tokyo-brain datasets

# Set your API key
export TOKYO_BRAIN_API_KEY=tb-your-key
export TOKYO_BRAIN_URL=https://onboarding.tokyobrain.ai

# Run the benchmark
python benchmark.py
```

## What it measures

| Dimension | Questions | What It Tests |
|-----------|-----------|---------------|
| Single-session preference | 30 | "What does this user prefer?" |
| Temporal reasoning | 133 | "When did X happen relative to Y?" |
| Knowledge update | 78 | "X changed from A to B — what's current?" |
| Multi-session | 133 | "Across conversations, what's consistent?" |
| Single-session user | 70 | "What did the user say about themselves?" |
| Single-session assistant | 56 | "What did the AI recommend?" |

## Our Results (April 2026)

These are from our internal reproduction runs using default configurations:

| System | Score |
|--------|-------|
| Tokyo Brain | 83.8% |
| Supermemory | 81.6% |
| Zep | 71.2% |
| Mem0 | 49.0% |

*We welcome independent reproduction and corrections.*

## Configuration

The API endpoint and key are read from environment variables (see `config.py`); do not hard-code your key in the file:

| Variable | Default | Purpose |
|----------|---------|---------|
| `TOKYO_BRAIN_URL` | `https://onboarding.tokyobrain.ai` | Memory system base URL |
| `TOKYO_BRAIN_API_KEY` | *(none, required)* | API key |

Benchmark settings (`TOP_K`, `MATCH_THRESHOLD`, `DATASET_NAME`, `RESULTS_DIR`) are edited directly in `config.py`. Results are written as JSON to `results/`.

## License

MIT
