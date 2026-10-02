# Phase 4: call graph + strict citations. Run from the project root, venv active, Ollama running:
#     powershell -ExecutionPolicy Bypass -File .\eval\run_phase4.ps1
# Part A (~3-5 min): re-index with the call graph, trace eval, regression check on the old gold sets.
# Part B (~20-30 min on CPU): answer-level citation eval, plain vs strict, 6 questions each. Cached after the first run.

$env:CC_DATA_DIR = ".cc_ast"

Write-Host "`n=== A1. Re-index requests (adds the call graph) ===" -ForegroundColor Cyan
codecopilot index ..\requests -c ast -x "tests/*" -x "docs/*"

Write-Host "`n=== A2. Trace questions: graph expansion vs same-size retrieval ===" -ForegroundColor Cyan
codecopilot eval eval\requests_gold_trace.jsonl --save p4-trace

Write-Host "`n=== A3. Regression check: Phase 3 numbers should not move ===" -ForegroundColor Cyan
codecopilot eval eval\requests_gold.jsonl      --save p4-std
codecopilot eval eval\requests_gold_hard.jsonl --save p4-hard

Write-Host "`n=== A4. Graph commands (demo) ===" -ForegroundColor Cyan
codecopilot callers get_auth_from_url
codecopilot impact should_strip_auth

Write-Host "`n=== B. Answer citation quality: plain (Phase 3) vs strict (Phase 4) ===" -ForegroundColor Cyan
codecopilot eval-answers eval\requests_gold.jsonl -n 6 --no-strict --save p4-answers-plain
codecopilot eval-answers eval\requests_gold.jsonl -n 6 --strict    --save p4-answers-strict

Write-Host "`n=== Results ===" -ForegroundColor Cyan
codecopilot report
