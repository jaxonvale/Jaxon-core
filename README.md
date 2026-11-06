# Jaxon Core

The brain and memory system for **Jaxon Vale** — an AI CEO and partner to the founder.

Jaxon runs the business with you day to day: it remembers everything about the company, turns
conversations into tasks with owners and due dates, logs decisions with their rationale, tracks
goals/OKRs, KPIs, cash and runway, researches the market on the web, and briefs a team of
AI department heads (CFO, CMO, CTO, COO, CPO, Counsel) in parallel before making a call.

Built on Claude (`claude-opus-5-5`) via the Anthropic Python SDK.

## Quick start

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...

python -m jaxon chat            # talk to Jaxon in the terminal
python -m jaxon serve           # web app at http://127.0.0.1:5000
```

On first run Jaxon's company profile is empty, so it will ask you about the business and save
the answers. Tell it your cash balance and it will track burn and runway from there.

## Commands

| Command | What it does |
|---|---|
| `python -m jaxon chat [--session NAME]` | Interactive conversation (sessions persist) |
| `python -m jaxon ask "question"` | One-shot question |
| `python -m jaxon briefing` | Daily briefing: top priority, fires, today's top 3, decisions needed |
| `python -m jaxon review` | Weekly review: scorecard, risks, next week's plan (creates the tasks) |
| `python -m jaxon status` | Print the company snapshot (no API call) |
| `python -m jaxon serve [--host H --port P]` | Web app + JSON API |

Add `--db PATH` to any command to use a different database.

Run the briefing automatically every weekday morning with cron:

```cron
0 8 * * 1-5  cd /path/to/Jaxon-core && ANTHROPIC_API_KEY=... python -m jaxon briefing >> ~/.jaxon/briefings.log
```

## How it works

```
jaxon/
  agent.py        Jaxon (the CEO) + the tool-use loop
  departments.py  CFO / CMO / CTO / COO / CPO / Counsel sub-agents (read-only, run in parallel)
  tools.py        Tools: memory, company profile, tasks, decisions, goals, KPIs, finances
  memory.py       SQLite store (full-text search over memories, plus all company data and sessions)
  llm.py          Anthropic SDK wrapper (streaming, adaptive thinking, prompt caching, refusal fallback)
  routines.py     Daily briefing and weekly review
  server.py       Flask JSON API + web UI (jaxon/web/index.html)
  cli.py          Command line
```

- **Memory**: everything lives in one SQLite file (`~/.jaxon/jaxon.db` by default). Long-term
  memories are searchable with FTS5. Each turn, a fresh company snapshot (profile, active tasks,
  goals, latest KPIs, cash/runway, recent decisions) is attached to your message so Jaxon always
  works from current state.
- **Delegation**: the `delegate` tool briefs several department heads at once. Each one is its own
  Claude agent with read-only access to company data and web search, and returns a memo. Jaxon
  makes the final call.
- **Guardrails**: department heads can't change data. Jaxon is told to defer irreversible calls
  (contracts, hiring/firing, big spend, fundraising terms) to the founder.

## Configuration

| Env var | Default | Meaning |
|---|---|---|
| `ANTHROPIC_API_KEY` | — | Claude API key (required for chat) |
| `JAXON_MODEL` | `claude-opus-5-5` | Model |
| `JAXON_EFFORT` | `high` | CEO reasoning effort (`low`…`max`) |
| `JAXON_DEPT_EFFORT` | `medium` | Department-head effort |
| `JAXON_DB` | `~/.jaxon/jaxon.db` | Database path |
| `JAXON_WEB_SEARCH` | `1` | Set `0` to disable web search |
| `JAXON_FALLBACKS` | `1` | Server-side refusal fallback (Claude API only; set `0` on Bedrock/Vertex) |
| `JAXON_TOKEN` | — | If set, the web API requires `Authorization: Bearer <token>` |

## Web API

`POST /api/chat {message, session_id?}` → `{session_id, reply, tool_calls, usage}`.
Also: `/api/snapshot`, `/api/tasks` (GET/POST, PATCH `/api/tasks/<id>`), `/api/goals`,
`/api/decisions`, `/api/metrics[/<name>]`, `/api/finance`, `/api/memories?q=`,
`/api/sessions[/<id>]`, `POST /api/briefing`, `POST /api/review`, `/api/health`.

The server binds to localhost by default. If you expose it, set `JAXON_TOKEN`.

## Deploy on Railway

1. Railway → **New Project → Deploy from GitHub repo** → pick this repo.
2. Service **Settings → Source → Branch**: pick the branch with this code.
3. **Variables**: `ANTHROPIC_API_KEY`, `JAXON_TOKEN` (a long random password), `JAXON_DB=/data/jaxon.db`.
4. Right-click the service → **Attach Volume**, mount path `/data` (keeps memory across deploys).
5. **Settings → Networking → Generate Domain**. Open it, tap the key icon, paste your `JAXON_TOKEN`.

The server (`jaxon/wsgi.py`, run by gunicorn) refuses to start without `JAXON_TOKEN`.

## Tests

```bash
python -m pytest -q
```

The tests use a scripted fake Claude client, so they need no API key or network.
