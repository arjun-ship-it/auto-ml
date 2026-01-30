---
description: Onboard repo + set up feature-first docs workflow + ensure gh CLI usage conventions.
---

# Techjays /onboard (Feature-First)

I am the primary AI developer for this repo.
I will run onboarding sequentially and only stop for critical blockers.

## Step 1: Repo discovery & command detection

1) Read `package.json`/Makefile/go.mod/pyproject/sln/etc.
2) Determine exact commands: install, run, build, lint, test, typecheck, migrations.
3) Update `CLAUDE.md` with real commands + stack + frameworks + repo rules.

## Step 2: Setup docs structure (feature-first)

1) Ensure directories exist:
   - `docs/ai/`
   - `docs/features/`
2) Create `docs/features/README.md` if missing (feature index with a table).
3) Add "Feature-first workflow" section to `CLAUDE.md` if missing.

## Step 3: Documentation sanity (diff-first for README)

1) Compare README instructions vs discovered commands.
2) If changes needed: produce diff in `docs/ai/proposals/README.diff` and show it before saving.

## Step 4: Verify test harness

1) Run test command.
2) If no framework: propose minimal framework + add a sanity test.
3) Record how to test in `CLAUDE.md`.

## Step 5: Baseline tech intelligence docs

Create/update:
- `docs/ai/architecture.md`
- `docs/ai/codestructure.md`
- `docs/ai/datamodel.md`
- `docs/ai/integrations.md`
- `docs/ai/api-endpoints.md`
- `docs/ai/utilities.md`
- `docs/ai/technical-debt.md` (top 5 items)

## Step 6: GitHub CLI readiness

1) Verify `gh --version` works (or note missing).
2) Record GitHub workflow conventions in `CLAUDE.md`:
   - branch naming `feature/<slug>`
   - PR creation via `gh pr create`
   - PR checks via `gh pr checks`
   - PR comments via `gh pr comment`

## Final report

Output:
- real commands
- docs created
- tests/lint status
- top technical debt
- blockers
