"""Regression tests for the needs_input triage re-dispatch loop (t_7630e23d).

Board incident 2026-10-21 (card t_c9ec490a): a worker-blocked ``needs_input``
card escalated to ``triage`` by ``block_loop_detected`` was immediately picked
up again by the automatic triage sweeps (auto-decompose tick / ``specify
--all``), re-specified (destroying the body), promoted, re-claimed and
re-blocked — 690 loop detections, 692 worker spawns, 644 identical comments,
once per ~60 s tick, for a month.

Covered here:
* A loop-parked triage card is NOT re-specified, re-promoted or re-claimed on
  the next dispatch tick, while a genuine triage stub still is.
* ``unblock_task`` is the human release: it un-parks the card (and a same-kind
  re-block after release escalates straight back — recurrences survive).
* A ``specify`` rewrite records the overwritten title/body verbatim in the
  ``specified`` event payload (``previous``) — same for ``edit_task``.
* ``add_comment`` dedupes an identical comment against the immediately
  preceding one from the same author.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli import kanban_decompose as decomp
from hermes_cli import kanban_specify as spec


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


@pytest.fixture
def spawnable_profiles(monkeypatch: pytest.MonkeyPatch) -> None:
    """No real profiles exist in the isolated home; make every assignee
    spawnable so dispatch ticks genuinely claim + spawn."""
    from hermes_cli import profiles as profiles_mod

    monkeypatch.setattr(profiles_mod, "profile_exists", lambda name: True)


@pytest.fixture(autouse=True)
def stub_decomposer(monkeypatch: pytest.MonkeyPatch) -> None:
    """Canned aux-LLM reply for the decomposer: always "tighten" into a stub
    body (exactly what destroyed the production card's body each cycle)."""

    def fake_call_aux(verb, task_id, **kw):
        return json.dumps({
            "fanout": False,
            "rationale": "single unit of work",
            "title": "Re-specified stub",
            "body": "Stub spec referencing PRESERVE VERBATIM blocks (absent).",
        }), ""

    monkeypatch.setattr(decomp, "_call_aux", fake_call_aux)


def _fake_spawn_factory(spawns: list[str]):
    def fake_spawn(task, workspace, board=None):
        spawns.append(task.id)
        return 42

    return fake_spawn


def _drive_to_loop_park(conn, spawns: list[str], *, reason="awaiting owner decision"):
    """Create a ready card and drive it through two needs_input blocks (with a
    human unblock between, as recorded on the production card) so the second
    block hits the recurrence limit and routes it to triage."""
    tid = kb.create_task(conn, title="Record the owner's decision", assignee="worker")
    spawn = _fake_spawn_factory(spawns)

    kbd.dispatch_once(conn, spawn_fn=spawn)               # claim #1
    kb.add_comment(conn, tid, "worker", "Blocks absent; cannot proceed.")
    kb.block_task(conn, tid, reason=reason, kind="needs_input")
    assert kb.get_task(conn, tid).status == "blocked"

    kb.unblock_task(conn, tid)                            # the one human unblock
    kbd.dispatch_once(conn, spawn_fn=spawn)               # claim #2
    kb.block_task(conn, tid, reason=reason, kind="needs_input")
    assert kb.get_task(conn, tid).status == "triage"
    return tid


def _kinds(conn, tid) -> list[str]:
    return [e.kind for e in kb.list_events(conn, tid)]


# ---------------------------------------------------------------------------
# Defect 1 — detection must stop the re-dispatch
# ---------------------------------------------------------------------------


def test_loop_parked_triage_is_not_redispatched_on_next_tick(
    kanban_home, spawnable_profiles,
):
    """Acceptance: after loop detection the card is not re-claimed on the next
    tick. Drives the exact production cycle — dispatch tick, worker comment +
    needs_input block, auto-decompose sweep — with no wall-clock sleeps."""
    spawns: list[str] = []
    spawn = _fake_spawn_factory(spawns)

    with kbc.connect() as conn:
        tid = _drive_to_loop_park(conn, spawns)
        stub = kb.create_task(conn, title="genuine rough idea", assignee="worker", triage=True)

        assert spawns == [tid, tid]
        assert _kinds(conn, tid).count("claimed") == 2

        # --- the gateway auto-decompose sweep + the next dispatch tick ---
        swept = decomp.list_triage_ids()
        # `specify --all` sees the same filtered set.
        assert spec.list_triage_ids() == swept == [stub]
        for t in swept:
            decomp.decompose_task(t, author="auto-decomposer")
        kbd.dispatch_once(conn, spawn_fn=spawn)
        kb.recompute_ready(conn)

        # The parked card: still in triage, never re-specified/promoted/claimed.
        assert kb.get_task(conn, tid).status == "triage"
        kinds = _kinds(conn, tid)
        assert kinds.count("claimed") == 2, "loop-parked card was re-claimed"
        assert "specified" not in kinds, "loop-parked card was re-specified"
        assert "promoted" not in kinds, "loop-parked card was re-promoted"
        assert spawns.count(tid) == 2, "loop-parked card was re-spawned"

        # The genuine stub WAS swept, specified and promoted.
        assert stub in swept
        stub_kinds = _kinds(conn, stub)
        assert "specified" in stub_kinds
        # swept -> todo -> promoted -> (claimed by the same tick) ready/running
        assert kb.get_task(conn, stub).status in ("todo", "ready", "running")


def test_unblock_releases_loop_parked_triage(kanban_home, spawnable_profiles):
    """``unblock`` is the human release: the card becomes dispatchable again —
    and because ``block_recurrences`` survive, one more same-kind block
    escalates straight back to triage (the park cannot be beaten by retrying)."""
    spawns: list[str] = []
    with kbc.connect() as conn:
        tid = _drive_to_loop_park(conn, spawns)
        assert kb.is_loop_parked_triage(conn, tid) is True

        assert kb.unblock_task(conn, tid) is True
        assert kb.get_task(conn, tid).status == "ready"
        assert kb.is_loop_parked_triage(conn, tid) is False

        # Release makes it dispatchable again...
        kbd.dispatch_once(conn, spawn_fn=_fake_spawn_factory(spawns))
        assert kb.get_task(conn, tid).status == "running"
        assert spawns.count(tid) == 3

        # ...but the loop memory survives: one re-block re-parks it.
        kb.block_task(conn, tid, reason="awaiting owner decision", kind="needs_input")
        assert kb.get_task(conn, tid).status == "triage"
        assert kb.is_loop_parked_triage(conn, tid) is True


def test_sweepable_triage_ids_exclude_only_loop_parks(kanban_home):
    """kb-level contract both sweeps rely on: fresh triage cards are sweepable,
    loop-parked ones are not, and the predicate flips on unblock."""
    with kbc.connect() as conn:
        parked = kb.create_task(conn, title="parked", assignee="worker", triage=True)
        fresh = kb.create_task(conn, title="fresh", assignee="worker", triage=True)
        with kb.write_txn(conn):
            conn.execute(
                "INSERT INTO task_events (task_id, kind, created_at) VALUES (?, ?, 0)",
                (parked, "block_loop_detected"),
            )
        assert kb.sweepable_triage_ids(conn) == [fresh]
        assert kb.is_loop_parked_triage(conn, parked) is True
        assert kb.is_loop_parked_triage(conn, fresh) is False

        kb.unblock_task(conn, parked)
        # The release moves the card OUT of triage (to ready): it is no longer
        # parked, and the triage sweep naturally stops seeing it.
        assert kb.is_loop_parked_triage(conn, parked) is False
        assert kb.get_task(conn, parked).status == "ready"
        assert kb.sweepable_triage_ids(conn) == [fresh]


# ---------------------------------------------------------------------------
# Defect 2 — rewrites must be reversible from the event log
# ---------------------------------------------------------------------------


def test_specify_rewrite_preserves_previous_title_and_body(kanban_home):
    """Acceptance: a specify rewrite records the prior title/body in the event
    payload, so the destroyer of an implementation-grade body is diffable and
    reversible from task_events alone."""
    original_body = (
        "**Goal**\nComplete proposal.\n\n**Constraints**\n- five constraints\n"
        "- verification plan\n- PRESERVE VERBATIM blocks"
    )
    with kbc.connect() as conn:
        tid = kb.create_task(
            conn, title="Original title", body=original_body, assignee="worker", triage=True,
        )
        ok = kb.specify_triage_task(
            conn, tid, title="Re-specified stub", body="Stub body.", author="specifier",
        )
        assert ok is True

        task = kb.get_task(conn, tid)
        assert task.title == "Re-specified stub"
        assert task.body == "Stub body."

        events = [e for e in kb.list_events(conn, tid) if e.kind == "specified"]
        assert events, "no specified event"
        payload = events[-1].payload or {}
        assert payload.get("changed_fields") == ["title", "body"]
        assert payload.get("previous") == {"title": "Original title", "body": original_body}


def test_edit_records_previous_title_and_body(kanban_home):
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="Old title", body="Old body", assignee="worker")
        assert kb.edit_task(conn, tid, title="New title", body="New body") is True

        events = [e for e in kb.list_events(conn, tid) if e.kind == "edited"]
        assert events
        payload = events[-1].payload or {}
        assert payload.get("previous") == {"title": "Old title", "body": "Old body"}

        # A no-op rewrite records nothing to recover — there is nothing to undo.
        assert kb.edit_task(conn, tid, title="New title") is True
        payload2 = ([e for e in kb.list_events(conn, tid) if e.kind == "edited"][-1].payload or {})
        assert "previous" not in payload2


# ---------------------------------------------------------------------------
# Comment dedupe
# ---------------------------------------------------------------------------


def test_add_comment_dedupes_consecutive_identical(kanban_home):
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="t", assignee="worker")

        first = kb.add_comment(conn, tid, "worker", "same report")
        dup = kb.add_comment(conn, tid, "worker", "same report")
        # Stored bodies are stripped: trailing whitespace still dedupes.
        dup_ws = kb.add_comment(conn, tid, "worker", "same report\n")

        assert dup == first and dup_ws == first
        assert len(kb.list_comments(conn, tid)) == 1
        assert _kinds(conn, tid).count("commented") == 1

        # A different author posting the same body is NOT a duplicate.
        assert kb.add_comment(conn, tid, "reviewer", "same report") != first
        assert len(kb.list_comments(conn, tid)) == 2

        # Once a different comment intervenes, the same report is no longer
        # adjacent and is stored again.
        kb.add_comment(conn, tid, "worker", "a different report")
        again = kb.add_comment(conn, tid, "worker", "same report")
        assert again != first
        assert len(kb.list_comments(conn, tid)) == 4
