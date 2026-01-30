---
description: Create PR using gh CLI. PR description is generated from feature spec (objective, changes, test plan, rollout).
---

# Techjays /pr (Feature-First + gh)

Feature slug: $ARGUMENTS

## Rules

- Must be on `feature/<slug>`.
- Must run `/verify` (or run quality gates here).
- Must use gh CLI.

## Steps

1) Read `docs/features/<slug>.md`.
2) Run quality gates (from CLAUDE.md) or `gh pr checks` after pushing.
3) Generate PR title:
   - `feat: <human feature name>` or `fix: <...>`
4) Generate PR body from feature spec:
   - Objective
   - What changed
   - Impacted files/modules (high-level)
   - Data model impact
   - Test plan + commands
   - Rollout notes
5) Push branch:
   - `git push -u origin feature/<slug>`
6) Create PR using gh:
   - `gh pr create --title "<title>" --body "<body>"`
7) Update feature spec Status to `In Review`.

## Output

- PR URL (from gh output)
- Summary + how to test
