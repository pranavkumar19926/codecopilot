# Phase 2 comparison: {fixed, ast} chunks x {dense, bm25, hybrid} retrieval = 6 eval runs.
# Run from the project root with the venv active:   .\eval\run_phase2.ps1
# Assumes the requests repo sits next to this project at commit 611c616 (see README).

$repo = "..\requests"
Remove-Item Env:CC_EXCLUDE_GLOBS -ErrorAction SilentlyContinue   # -x below sets excludes explicitly

Write-Host "`n=== Fixed 60-line chunks (Phase 1 chunking) ===" -ForegroundColor Cyan
$env:CC_DATA_DIR = ".cc_fixed"
codecopilot index $repo -c fixed -x "tests/*" -x "docs/*"
codecopilot eval eval\requests_gold.jsonl -m all --no-rerank --no-rewrite --save p2-fixed

Write-Host "`n=== AST chunks (Phase 2) ===" -ForegroundColor Cyan
$env:CC_DATA_DIR = ".cc_ast"
codecopilot index $repo -c ast -x "tests/*" -x "docs/*"
codecopilot eval eval\requests_gold.jsonl -m all --no-rerank --no-rewrite --save p2-ast

Write-Host "`n=== All saved runs ===" -ForegroundColor Cyan
codecopilot report

# Leave the terminal pointed at the Phase 2 index so `ask` / `search` use it.
$env:CC_DATA_DIR = ".cc_ast"
