---
description: Generate/expand tests based on the feature spec's Test plan. Updates the feature spec with exact test files added.
---

# Techjays /tests (Feature-First)

Feature slug: $ARGUMENTS

## Rules

- Must read `docs/features/<slug>.md` and implement tests from its Test plan.
- Update the feature spec with:
  - test files added/changed
  - edge cases covered
- Do not invent endpoints/models not in repo.

## Steps

1) Read feature spec Test plan.
2) Implement tests:
   - unit tests first
   - integration/e2e if supported
3) Run test command(s).
4) Update `docs/features/<slug>.md`:
   - add exact test file paths
   - record commands to run
5) Output: tests summary + results
