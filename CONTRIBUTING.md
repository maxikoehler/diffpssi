# Contributing

Thanks for your interest in contributing! This file explains how to set up a local
development environment, run tests and linters, and prepare pull requests. The
instructions are tailored to the project's structure (Python, `pyproject.toml`,
`tox.ini`, and MkDocs-based docs).

## 1. Fork & Branching

- Fork this repository to your GitHub account.
- Create a feature branch from `development` with a descriptive name, e.g.
	`feature/<short-description>` or `fix/<short-description>`.

Example:

```bash
# from your local clone of your fork
$ git checkout development
$ git pull upstream development   # if you have an upstream remote configured
$ git checkout -b feature/my-change
```

## 2. Setting up the development environment

We recommend using a virtual environment (venv). Check `pyproject.toml` for
the supported Python version. Further this project uses Poetry in the version 2.1.4 as package manager.

For installation see `INSTALLATION.md` in the developers version. Don't forget to install pre-commit hooks.

## 3. Running tests

Unit and integration tests live in the `tests/` directory.

Run tests directly with pytest:

```bash
$ pytest
```

Or use tox (if configured; `-r` for re-creating environments) also to check against multiple Python versions:

```bash
$ tox
```

If tests fail, include failing logs in the PR or open an issue for help.

## 4. Linting, formatting and type checking

Before committing, please run the following checks:

- Formatting: black
- Import sorting: isort
- Ensuring Docstrings: pydocstyle
<!-- - Static types: mypy
- Linting: pylint -->

Example commands:

```bash
$ black src/ tests/
$ isort src/ tests/
$ pydocstyle src/
```

## 5. Documentation and examples

Project documentation is built with MkDocs (see `mkdocs.yaml`). To preview docs locally:

```bash
mkdocs serve
```

Example scripts and notebooks are under `examples/`. Use the same virtual environment for running notebooks.

## 6. Commit & Pull Request workflow

- Keep commits small and focused.
- Explain the rationale in commit messages, not only what changed.
- Regularly rebase or merge from `development` to avoid large conflicts.

PR checklist (before opening a PR):

- [ ] Tests pass locally and in CI.
- [ ] Code is formatted with black and passes linters.
- [ ] Types are updated or mypy passes for changes affecting public APIs.
- [ ] PR description explains purpose and impact of changes.
- [ ] Update docs/examples if public interfaces change.

## 7. CI / Checks

The CI workflow (GitHub Actions / tox / other) typically runs tests, linting and builds.
If CI fails, inspect the logs and fix issues locally before pushing updates.

## 8. Pre-commit hooks (optional but recommended)

To automatically run formatting and checks before commits, install pre-commit:

```bash
pip install pre-commit
pre-commit install
pre-commit run --all-files
```

## 9. Release & changelog

Add a changelog entry for visible behavior or API changes. The repository contains
`CHANGELOG.md` — please keep it up to date.

## 10. Issues and support

- Open issues for bugs or larger feature proposals.
- Reference related issues in PRs (e.g. `Fixes #123`).

## 11. Questions

If in doubt, open an issue or comment on the PR to ask maintainers for guidance.
