---
description: Fetch PR comments via gh, apply fixes, rerun tests, and respond with gh pr comment. Keeps feature spec current.
---

# Techjays /pr-comments (gh)

Input: $ARGUMENTS (PR number/url or feature slug)

## Rules

- Use gh CLI to read comments.
- Address in small commits.
- Update feature spec if scope/test plan changes.

## Steps

1) Identify PR:
   - If feature slug: find PR via `gh pr list --head feature/<slug>`
2) Fetch comments:
   - `gh pr view <PR> --comments`
3) Group comments and implement fixes.
4) Run targeted tests + full gates if needed.
5) Respond using:
   - `gh pr comment <PR> --body "<response>"`
6) If needed, request re-review.

## Output

- Changes made
- Tests run
- Comment response drafted + sent
