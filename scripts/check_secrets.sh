#!/usr/bin/env bash
set -euo pipefail

patterns='(sk_(live|test|restricted)_[A-Za-z0-9]{16,}|whsec_[A-Za-z0-9]{16,}|sk-proj-[A-Za-z0-9_-]{16,}|AKIA[0-9A-Z]{16})'
if git grep -nEI "$patterns" -- . ':!*.example' ':!.venv/**' ':!venv/**'; then
  echo 'Credential-shaped value found in a tracked source file.' >&2
  exit 1
fi
