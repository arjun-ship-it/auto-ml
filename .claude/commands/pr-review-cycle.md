---
description: Create PR and iterate with Gemini code review until all comments are addressed, then merge.
---

# Techjays /pr-review-cycle

## Overview
When making code changes, follow this PR workflow to ensure code quality through automated Gemini review.

## Rules

1. **Create Feature Branch**: Use naming convention `feature/<short_meaningful_name>`
2. **Commit Changes**: Use conventional commit format (`feat:`, `fix:`, `chore:`, etc.)
3. **Create PR**: Use `gh pr create` with proper title and description
4. **Wait for Gemini Review**: Gemini Code Assist will automatically review the PR
5. **Address Comments**: Fix all issues raised by Gemini (high, medium, low priority)
6. **Push Fixes**: Commit and push fixes with descriptive message
7. **Repeat**: Wait for Gemini to re-review. Repeat steps 5-6 until no new comments
8. **Merge**: Once Gemini has no more comments, merge the PR

## Steps

1) Ensure you're on a feature branch:
   ```bash
   git checkout -b feature/<slug>
   ```

2) Stage and commit changes:
   ```bash
   git add <files>
   git commit -m "feat/fix: description"
   ```

3) Push branch and create PR:
   ```bash
   git push -u origin feature/<slug>
   gh pr create --title "feat/fix: title" --body "## Summary\n..."
   ```

4) Wait for Gemini review (typically 30-60 seconds):
   ```bash
   gh api repos/{owner}/{repo}/pulls/{pr_number}/reviews
   gh api repos/{owner}/{repo}/pulls/{pr_number}/comments
   ```

5) Address each Gemini comment:
   - Read the suggestion carefully
   - Apply the fix to the codebase
   - Verify build still passes: `npm run build`

6) Commit and push fixes:
   ```bash
   git add <fixed_files>
   git commit -m "fix: address Gemini review comments"
   git push
   ```

7) Check for new comments:
   ```bash
   gh api repos/{owner}/{repo}/pulls/{pr_number}/comments --jq '.[] | select(.commit_id == "<latest_commit>") | .body'
   ```

8) Once no new comments, merge:
   ```bash
   gh pr merge {pr_number} --squash --delete-branch
   ```

## Gemini Priority Levels

- **High Priority**: Must fix - potential bugs, security issues, breaking changes
- **Medium Priority**: Should fix - type safety, code quality, best practices
- **Low Priority**: Optional - style suggestions, minor improvements

## Output

- PR URL
- Summary of Gemini comments addressed
- Final merge confirmation
