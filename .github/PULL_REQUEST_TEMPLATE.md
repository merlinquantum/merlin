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

## Related Issue
   <!-- For internal contributors: Use Jira key (e.g., Related Jira: PML-126)
        For external contributors: Link to GitHub issue if applicable -->

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
- [ ] Updated the API

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
- [ ] With this command: 

        > SPHINXOPTS="-W --keep-going -n" make -C docs clean html 

    the docs are built without any warning or errors.
- [ ] New public classes/methods/packages are added in the API following the methodology presented in other files.
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
pip install -r requirements-docs.txt && make -C docs html

-->
