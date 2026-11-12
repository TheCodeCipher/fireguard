# Contributing to FireGuard

## 1. Development Workflow

FireGuard follows a branch-based collaborative development workflow.

- Every task must have a GitHub Issue.
- Every Issue must have an assignee, role, phase and priority.
- Contributors must work on feature branches.
- Direct commits to main are prohibited.
- All changes must be submitted through Pull Requests.
- Only reviewed and approved contributions are integrated into main.

## 2. Branch Naming Convention

Use:

role-<number>/<component>/<task-name>

Examples:

role-1/data/wsts-loader
role-2/ml/unet-baseline
role-3/ml/convlstm
role-4/research/evaluation
role-5/backend/api-foundation
role-6/frontend/live-monitor

## 3. Commit Message Convention

Use conventional commit prefixes:

feat: New functionality
fix: Bug fixes
docs: Documentation
test: Tests
refactor: Code restructuring
chore: Configuration and maintenance

Example:

feat(backend): add health endpoint

## 4. Pull Request Requirements

Every PR must include:

- Linked GitHub Issue
- Summary of changes
- Implementation details
- Tests performed
- Test results
- Screenshots where applicable
- Known limitations

## 5. Data and Security Policy

- Never commit raw datasets.
- Never commit API keys or passwords.
- Never commit private user reports.
- Keep model checkpoints and generated artifacts outside Git.
- Respect original dataset licenses.

## 6. Review Process

The repository maintainer reviews:

- Acceptance criteria
- Correctness
- Tests
- Architecture compatibility
- Security
- Documentation

Only approved and tested changes are merged into main.
