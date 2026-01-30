---
description: Implement feature per spec, keep diffs small, update spec sections as needed. Must run tests. Uses the current feature branch.
---

# Techjays /execute (Feature-First)

Feature: $ARGUMENTS

## Rules

- Must be on branch `feature/<slug>`.
- Must read and follow `docs/features/<slug>.md`.
- Any scope change requires updating the feature spec first.
- Add/adjust tests aligned to feature spec.
- Run relevant verification (fast tests often, full gates at end).

## Steps

1) Confirm branch name matches `feature/<slug>`.
2) Read feature spec and restate:
   - Objective
   - Acceptance criteria
   - Test plan
3) Implement in small increments.
4) If you discover extra impacted files/data models:
   - update feature spec "Code impact" / "Data model impact"
5) Run:
   - unit tests for affected modules (fast)
   - lint (if relevant)
6) At end: run full quality gates (from CLAUDE.md).
7) Set feature Status to `In Progress` (or `In Review` when PR created).

## Output

- Summary of changes
- Tests added + commands run
- Feature spec updates made
