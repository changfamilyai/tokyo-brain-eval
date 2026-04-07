"""Configuration for Tokyo Brain Evaluation Harness."""
import os

# Memory system to evaluate
MEMORY_API_URL = os.environ.get("TOKYO_BRAIN_URL", "https://onboarding.tokyobrain.ai")
MEMORY_API_KEY = os.environ.get("TOKYO_BRAIN_API_KEY", "")

# Benchmark settings
DATASET_NAME = "xiaowu0205/LongMemEval"
TOP_K = 15
MATCH_THRESHOLD = 0.5  # Minimum similarity to count as correct

# Output
RESULTS_DIR = "results"
