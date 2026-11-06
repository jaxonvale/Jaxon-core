"""Explanations of logged decisions.

The explanation is produced AFTER the fact from the journal and setup log. It
is read-only: nothing here feeds back into signals, sizing or execution. An
LLM, when enabled, only rewrites the deterministic facts into prose; the
deterministic summary is always printed alongside it.
"""
from __future__ import annotations

import json
from pathlib import Path

SYSTEM = ("You explain decisions made by a deterministic, rules-based paper-trading engine. "
          "Use only the facts provided. Do not suggest trades, change rules, predict prices, or "
          "claim profitability. If a fact is missing, say it is missing. Keep it under 200 words.")


def load_run(out_dir: str | Path) -> tuple[list[dict], list[dict]]:
    out = Path(out_dir)
    setups = json.loads((out / "setups.json").read_text())
    journal = [json.loads(l) for l in (out / "journal.jsonl").read_text().splitlines() if l.strip()]
    return setups, journal


def facts(setup_id: str, setups: list[dict], journal: list[dict]) -> dict:
    s = next((x for x in setups if x["setup_id"] == setup_id), None)
    if s is None:
        raise KeyError(f"setup {setup_id} not found")
    related = [r for r in journal if setup_id in json.dumps(r) or
               (r.get("zone_id") == s["zone_id"] and r["kind"].startswith("zone"))]
    return {"setup": s, "journal": related}


def deterministic_text(f: dict) -> str:
    s = f["setup"]
    lines = [f"Setup {s['setup_id']} on {s['symbol']} (config {s.get('config_version')}).",
             f"4H bias at touch: {s['bias_at_touch']}; known 4H swings: {s['htf_swings']}.",
             f"Demand zone {s['zone_id']}: {s['zone_lower']} – {s['zone_upper']}, active from {s['zone_activated_at']}.",
             f"Touch at {s['touch_at']}; frozen break threshold {s['break_threshold']} "
             f"(5m swing high from {s['threshold_pivot_start']}, known at {s['threshold_pivot_available_at']})."]
    if s.get("confirm_at"):
        lines.append(f"Structure shift confirmed at {s['confirm_at']}. VWAP snapshot {s.get('vwap_snapshot')} "
                     f"({s.get('vwap_note')}).")
        lines.append(f"Proposed entry {s.get('proposed_entry')}, stop {s.get('stop')}, target {s.get('target')}, "
                     f"reward/risk {s.get('proposed_rr')}.")
    if s.get("actual_entry") is not None:
        lines.append(f"Filled {s.get('quantity')} @ {s['actual_entry']}; exit {s.get('exit_reason')} @ "
                     f"{s.get('exit_price')}; net {s.get('net_pnl')} ({s.get('r_multiple')} R); flags: "
                     f"{s.get('execution_flags') or 'none'}.")
    lines.append(f"Outcome: {s['outcome']} at {s.get('outcome_at')}.")
    lines.append(f"Data: {s.get('data_provenance')}.")
    return "\n".join(lines)


def llm_text(f: dict, model: str) -> str:
    """Optional prose rewrite via the Anthropic API. Raises if the SDK or credentials are missing."""
    import anthropic

    client = anthropic.Anthropic()
    resp = client.beta.messages.create(
        model=model,
        max_tokens=16000,
        output_config={"effort": "low"},
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        system=SYSTEM,
        messages=[{"role": "user", "content": "Explain this logged decision:\n" +
                   json.dumps(f, indent=2, default=str)}],
    )
    if resp.stop_reason == "refusal":
        return "[LLM declined to explain; use the deterministic summary above.]"
    return "".join(b.text for b in resp.content if b.type == "text").strip()


def explain(out_dir: str | Path, setup_id: str, use_llm: bool = False, model: str = "claude-opus-5-5") -> str:
    setups, journal = load_run(out_dir)
    f = facts(setup_id, setups, journal)
    text = "DETERMINISTIC RECORD\n" + deterministic_text(f)
    if use_llm:
        try:
            text += "\n\nLLM EXPLANATION (informational only; cannot change any decision)\n" + llm_text(f, model)
        except Exception as e:  # missing SDK, credentials or network: the record above still stands
            text += f"\n\n[LLM explanation unavailable: {type(e).__name__}: {e}]"
    return text
