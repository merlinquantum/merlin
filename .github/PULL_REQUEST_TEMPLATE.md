<!--
Thank you for contributing to MerLin!
Please fill out the sections below to help us review your PR efficiently.

NOTE: this repo is a public repository, therefore, do NOT paste Jira URLs. Use the Jira issue key only (e.g., PML-126).
The Jira–GitHub integration will link PRs/commits automatically when the key is present.
-->

<!-- Add the Jira issue key in the title -->
<!-- e.g., PML-126 Updating PR template -->

## Summary
<!-- What does this PR do? A clear, concise description. -->

## Related Jira ticket (required)
<!-- Use the Jira issue key ONLY (no URL), e.g.:
Related Jira: PML-126
-->
Related Jira:

## Context / Related Issues
<!-- Why do you do this PR ? Is it linked to a previous PR ? A clear, concise description. -->
<!-- e.g., Closes #123, Fixes #456 -->

## Type of change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Refactor / Cleanup
- [ ] Performance improvement
- [ ] CI / Build / Tooling
- [ ] Breaking change (requires migration notes)


## Proposed changes
<!-- Bulleted list of key changes. If API changes, list them explicitly. -->

## How to test / How to run
<!-- Describe test plan and steps for reviewers to validate and run your changes locally. Include datasets if relevant. -->

1. Command lines

```
Block of code
```

## Screenshots / Logs (optional)
<!-- Add images or paste relevant logs for UI/Doc changes or failures you fixed. -->


## Performance considerations (optional)
<!-- Note expected speed/memory impact and how you measured it. -->

## Documentation
- [ ] User docs updated (Sphinx)
- [ ] Examples / notebooks updated
- [ ] Docstrings updated

## Checklist
- [ ] PR title includes Jira issue key (e.g., PML-126)
- [ ] "Related Jira ticket" section includes the Jira issue key (no URL)
- [ ] Code formatted (ruff format)
- [ ] Lint passes (ruff)
- [ ] Static typing passes (mypy) if applicable
- [ ] Unit tests added/updated (pytest)
- [ ] Tests pass locally (pytest)
- [ ] Tests pass on GPU (pytest)
- [ ] Test coverage not decreased significantly
- [ ] Docs build locally if affected (sphinx)
- [ ] Dependencies updated (if needed) and pinned appropriately
- [ ] PR description explains what changed and how to validate it

<!-- Helpful local commands – run from repo root:

# Lint & format
ruff format && ruff check .

# Type check (if used)
mypy .

# Tests with coverage
pytest

# Build docs
pip install -e .[docs] && make -C docs html

-->