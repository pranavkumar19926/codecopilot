# Phase 3 comparison on the Phase 2 index (.cc_ast): {none, rerank, rewrite, rewrite+rerank} x {standard, hard} gold sets.
# Run from the project root with the venv active and Ollama running (rewriting calls qwen2.5-coder):
#     powershell -ExecutionPolicy Bypass -File .\eval\run_phase3.ps1
# First rewrite run makes one LLM call per question (~74 calls, roughly 10-20 min on CPU).
# Results are cached in .cache\, so the rewrite+rerank runs reuse them and take seconds.

$env:CC_DATA_DIR = ".cc_ast"
if (-not (Test-Path ".cc_ast\meta.json")) {
    Write-Host "No .cc_ast index found - building it (Phase 2 settings)..." -ForegroundColor Yellow
    codecopilot index ..\requests -c ast -x "tests/*" -x "docs/*"
}

foreach ($gold in @("requests_gold_hard", "requests_gold")) {
    $tag = if ($gold -eq "requests_gold_hard") { "hard" } else { "std" }
    Write-Host "`n=== $gold ===" -ForegroundColor Cyan
    codecopilot eval eval\$gold.jsonl --no-rerank --no-rewrite --save "p3-$tag-base"
    codecopilot eval eval\$gold.jsonl --rerank    --no-rewrite --save "p3-$tag-rerank"
    codecopilot eval eval\$gold.jsonl --no-rerank --rewrite    --save "p3-$tag-rewrite"
    codecopilot eval eval\$gold.jsonl --rerank    --rewrite    --save "p3-$tag-rewrite+rerank"
}

Write-Host "`n=== Phase 3 runs ===" -ForegroundColor Cyan
codecopilot report --gold hard
codecopilot report --gold standard
