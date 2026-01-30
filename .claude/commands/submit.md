---
description: Final gate before merge. Ensures PR checks green, feature spec marked Done, and provides post-merge verification checklist.
---

# Techjays /submit (gh)

Feature slug or PR: $ARGUMENTS

## Rules

- Must use gh CLI for status
- Feature spec must be up to date
- No merge unless checks are green (or explicit override)

## Steps

1) Identify PR (slug -> `gh pr list --head feature/<slug>`).
2) Check CI:
   - `gh pr checks <PR>`
3) If failing:
   - fix and re-run
4) Confirm feature spec:
   - acceptance criteria met
   - tests listed are present
   - rollout notes complete
5) Set feature Status to `Done`.
6) Post-merge checklist:
   - smoke test endpoints/pages
   - monitor logs/metrics if applicable

## Output

- Merge-ready summary
- Post-merge checklist
