# Contributing

## Workflow

1. Branch from `main` (`feat/...`, `fix/...`, `docs/...`, `chore/...`). Nothing is pushed to `main` directly.
2. Before every push, run the local checks and fix anything they report:

   ```bash
   pip install ruff pytest actionlint-py shellcheck-py zizmor pre-commit
   pre-commit install          # fast checks on every commit
   bash scripts/check.sh       # everything CI checks that needs no GPU, ROS or simulator
   ```

3. Open a pull request. It is merged only when every workflow that runs on it is green (CI, and ROS 2 / SITL flight
   when their paths change). Review your own diff on GitHub before merging, as a stranger would read it.
4. Squash-merge, with one commit message saying what changed and why. Fix-ups stay inside the pull request.

## What goes in the repository

- Code, configs, documentation, and the **results files that README numbers are checked against**
  (`benchmarks/results/`, verified by `benchmarks/check_readme.py`).
- Small, freely licensed demo assets that let a stranger run the project without downloading a dataset
  (listed as exceptions in `.gitignore`).
- Never: datasets, training runs, logs, simulator state, credentials, or anything from an employer.

## Numbers

Every number in the README comes from a results file produced by a script in this repository, and
`benchmarks/check_readme.py` checks that they match. A new claim needs its script, its results file and a
`check_readme.py` entry in the same pull request. Results that have not been measured (for example on a Jetson
or in the field) are described as not yet measured.

## Dependencies and workflows

GitHub Actions are pinned to full commit SHAs (the version is in a comment), every workflow runs with read-only
repository access, and `zizmor` checks the workflow files in CI. Dependabot proposes weekly updates as pull requests;
they go through the same checks as any other change. Model files are checked against `models/SHA256SUMS`.

## Releases

Add `docs/releases/vX.Y.Z.md` (release notes) and date the changelog section in a pull request. After it is merged,
tag the merge commit `vX.Y.Z` and push the tag: the Release workflow verifies the model checksums and publishes the
release with the models and project videos attached.
