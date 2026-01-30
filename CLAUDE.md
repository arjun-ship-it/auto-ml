# AutoML Agent - Project Configuration

## Stack & Frameworks

### Backend
- **Framework:** FastAPI + Python 3.13
- **AI:** Google Gemini (google-genai)
- **Database:** SQLAlchemy + SQLite (async)
- **WebSocket:** Native FastAPI WebSocket support

### Frontend
- **Framework:** React 18 + TypeScript
- **Build Tool:** Vite
- **Styling:** Tailwind CSS
- **State:** Zustand (planned)

## Commands

| Action | Command |
|--------|---------|
| Install Backend | `cd backend && pip install -r requirements.txt` |
| Run Backend | `cd backend && uvicorn app.main:app --reload` |
| Install Frontend | `cd frontend && npm install` |
| Run Frontend | `cd frontend && npm run dev` |
| Build Frontend | `cd frontend && npm run build` |

## Project Structure

```
auto-ml/
├── backend/
│   ├── app/
│   │   ├── agent/        # Gemini agent (core, tools, prompts, memory)
│   │   ├── api/          # REST routes, WebSocket handlers
│   │   ├── data/         # Data connectors, validators
│   │   ├── db/           # Database models, migrations
│   │   ├── execution/    # Code sandbox, safety checks
│   │   ├── ml/           # ML pipeline (EDA, preprocessing, training, evaluation)
│   │   ├── config.py     # App settings
│   │   └── main.py       # FastAPI app entry
│   ├── requirements.txt
│   └── .env
├── frontend/
│   ├── src/
│   │   ├── components/   # Chat, DataPreview, Pipeline, Results
│   │   ├── services/     # API client, WebSocket
│   │   └── store/        # State management
│   └── package.json
└── docker-compose.yml
```

## Environment Variables

Backend `.env`:
```
GOOGLE_API_KEY=your-gemini-api-key
GEMINI_MODEL=gemini-2.0-flash
DEBUG=true
HOST=0.0.0.0
PORT=8000
DATABASE_URL=sqlite+aiosqlite:///./automl.db
```

## Feature-first Workflow (mandatory)

**Documentation First:** Every change must be tied to a Feature Spec located at `docs/features/<feature-slug>.md`.

**Planning:** The Planning phase must involve creating or updating the Feature Spec (defining objective, scope, impacted files, data model, test plan, and rollout).

**Execution:** Code generation and test creation must explicitly reference requirements defined in the Feature Spec.

**Branching Strategy:** Use the naming convention `feature/<short_meaningful_name>`.

**Git Operations:** Use the `gh` CLI for all GitHub operations (PR creation, comments, status checks).

## GitHub Workflow

- Branch naming: `feature/<slug>`
- PR creation: `gh pr create`
- PR checks: `gh pr checks`
- PR comments: `gh pr comment`

## PR Review Cycle (mandatory)

When making code changes, follow this workflow:

1. **Create Feature Branch**: `git checkout -b feature/<slug>`
2. **Commit Changes**: Use conventional commits (`feat:`, `fix:`, etc.)
3. **Create PR**: `gh pr create` with proper title and description
4. **Wait for Review**: Automated review checks
5. **Address Comments**: Fix all issues (high/medium/low priority)
6. **Push Fixes**: Commit and push with descriptive message
7. **Repeat**: Wait for re-review, repeat steps 5-6 until no new comments
8. **Merge**: Once approved, merge the PR

See `.claude/commands/pr-review-cycle.md` for detailed instructions.

## Rules

- Do not read `.env` files directly
- Use conventional commits (`feat:`, `fix:`, `chore:`, `docs:`, `refactor:`)
- All API calls go through the agent tools system
- Keep ML logic in `backend/app/ml/` modules
- Keep agent logic in `backend/app/agent/` modules
