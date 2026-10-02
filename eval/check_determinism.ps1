# Runs the hard set twice with deterministic rewrites. Both runs must print identical numbers.
# First run regenerates rewrites once (temperature 0); the second is served from the cache.
$env:CC_DATA_DIR = ".cc_ast"
codecopilot eval eval\requests_gold.jsonl      --save p4-std-det
codecopilot eval eval\requests_gold_hard.jsonl --save p4-hard-det
codecopilot eval eval\requests_gold_hard.jsonl
