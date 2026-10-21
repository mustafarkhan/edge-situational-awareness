#!/usr/bin/env bash
# Every check CI runs that does not need a GPU, ROS or a simulator. Run it before every push:
#
#   bash scripts/check.sh
#
# Needs: pip install ruff pytest actionlint-py shellcheck-py zizmor (plus the repo's own requirements).
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
fail=0
step() {  # name command...
  local name=$1; shift
  if "$@" > /tmp/esa_check.log 2>&1; then echo "ok    $name"; else echo "FAIL  $name"; cat /tmp/esa_check.log; fail=1; fi
}
step "ruff lint"            ruff check .
step "ruff format"          ruff format --check .
step "unit tests"           python -m pytest -q
step "README numbers"       python benchmarks/check_readme.py
step "workflow files"       actionlint .github/workflows/ci.yml .github/workflows/ros2.yml .github/workflows/sitl.yml
step "workflow security"    zizmor --offline .github/workflows
step "shell scripts"        bash -c 'shellcheck -S warning scripts/*.sh scripts/kaggle/*.sh ros2_ws/*.sh benchmarks/*.sh'
step "no stray files"       bash -c '[ -z "$(git status --porcelain --untracked-files=all | grep "^??")" ]'
exit $fail
