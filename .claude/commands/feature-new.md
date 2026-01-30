---
description: Start a feature: create branch feature/<slug>, create docs/features/<slug>.md, and capture references (figma, screenshots, jira, etc.) from YAML or flags.
---

# Techjays /feature-new (Feature-First)

Input (two supported formats):

## Format A (Preferred): YAML block

Example:

```yaml
slug: user-ledger-export
title: User ledger export
jira: BUP-123
figma: https://figma.com/file/xxx
screenshots:
  - docs/assets/user-ledger-export/screen-1.png
  - docs/assets/user-ledger-export/screen-2.png
loom: https://loom.com/share/xxx
notes: |
  Any extra context.
```

## Format B: One-liner flags

Example:

```
user-ledger-export --title "User ledger export" --jira BUP-123 --figma https://... --screenshots a.png,b.png --loom https://... --notes "..."
```

---

## Rules (mandatory)

1) Branch name must be: `feature/<short_meaningful_name>`
2) Feature spec must be created at: `docs/features/<slug>.md`
3) Use the provided metadata to populate the Feature Spec "References" section.
4) Do not include secrets. Do not read `.env` files.

---

## Step 1: Parse $ARGUMENTS into a feature metadata object

Extract:
- slug (required)
- title (optional; default: derived from slug)
- jira (optional)
- figma (optional)
- screenshots (optional list; allow comma-separated in flags mode)
- loom (optional)
- notes (optional)

Validation:
- slug: lowercase, hyphen-separated, no spaces, no dates
- screenshots: must be repo-relative paths if provided

If any required field is missing (slug), stop and ask for it.

---

## Step 2: Ensure docs folders exist

- Ensure `docs/features/` exists
- Ensure `docs/assets/<slug>/` exists (create folder)

Guidance:
- If screenshots were provided and they are not under `docs/assets/<slug>/`,
  propose moving/copying them there and updating paths (do so if safe).

---

## Step 3: Create branch

- Create and checkout:
  - `git checkout -b feature/<slug>`

If already on a branch, confirm it matches `feature/<slug>`; if not, create the correct branch.

---

## Step 4: Create Feature Spec file

Create `docs/features/<slug>.md` using the org template and populate:
- Title from `title` (or derived)
- Slug = <slug>
- Status = Draft
- Owner = (leave blank if unknown)
- Created/Last updated = today
- References section filled from metadata:
  - Jira/Tracker
  - Figma
  - Screenshots (paths)
  - Loom
  - Notes (if provided)

---

## Step 5: Update feature index

If `docs/features/README.md` exists:
- Add a row/entry for the new feature (Status = Draft, link to spec)

Else:
- Create `docs/features/README.md` with a simple table and add the feature row.

---

## Step 6: Commit the initialization

- `git add docs/features/<slug>.md docs/features/README.md docs/assets/<slug>/`
- Commit message:
  - `chore(feature): initialize <slug> spec`

---

## Output (strict)

- Feature slug:
- Branch name:
- Spec path:
- References captured (jira/figma/screenshots/loom):
- Next recommended command:
  - `/plan <slug>`
