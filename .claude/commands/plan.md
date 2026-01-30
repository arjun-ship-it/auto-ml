---
description: Plan work and update the feature spec with objective/scope/impacts/tests/rollout. No code edits.
---

# Techjays /plan (Feature-First)

Input: $ARGUMENTS

## Mandatory

- You must identify the Feature Spec file:
  - If provided: use it
  - If not: pick the closest existing or instruct to run `/feature-new <slug>` and proceed with best assumption.
- Planning must UPDATE the feature spec.

## Rules

- No code edits.
- You MAY edit docs (feature spec + feature index).
- Ask only if critically blocked.

## Steps

1) Read `CLAUDE.md` + any relevant `docs/ai/*`.
2) Open `docs/features/<slug>.md`.
3) Fill/update:
   - Objective
   - Scope (in/out)
   - Functional requirements
   - Data model impact
   - Integration impact
   - Code impact (files/modules likely to change)
   - Test plan (unit/integration/e2e + edge cases)
   - Rollout plan
4) Update feature Status to `Planned`.
5) Provide a concise plan output in chat:
   - steps
   - impacted files
   - test plan summary
   - risks

## Output

- The updated feature spec must be saved.
- Show a short "doc diff summary" (what sections updated).
