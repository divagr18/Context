"""Folded-state semantics over the append-only OpLog (PLAN.md section 3).

Locks contract C4: SUPERSEDE retains the prior value flagged superseded,
UPSERT-over-existing behaves as SUPERSEDE with the stored old value, DROP
removes the current view only, MARK UNRESOLVED / RESOLVE interact as
specified, unbound eids resolve to None, and fold is pure + idempotent.
"""

from __future__ import annotations

from track_a.parse_compact import OpLog

ENTITY_E1 = "UPSERT ENTITY E1 type=person name=Alice_Chen"
ENTITY_E2 = "UPSERT ENTITY E2 type=project name=Project_Kestrel"


def _log(*lines: str) -> OpLog:
    log = OpLog()
    failures = log.append(list(lines))
    assert failures == []
    return log


def test_superse_retains_prior_value_flagged() -> None:
    log = _log(
        ENTITY_E1,
        "UPSERT FACT E1.role = lead pos=1",
        "SUPERSEDE FACT E1.role : lead => senior pos=2",
    )
    state = log.fold()
    assert state.current == {("Alice_Chen", "role"): ("senior", 2)}
    assert state.history("Alice_Chen", "role") == [
        ("lead", 1, True),
        ("senior", 2, False),
    ]


def test_two_step_superse_chain_keeps_full_history() -> None:
    log = _log(
        ENTITY_E1,
        "UPSERT FACT E1.role = lead pos=1",
        "SUPERSEDE FACT E1.role : lead => senior pos=2",
        "SUPERSEDE FACT E1.role : senior => staff pos=3",
    )
    state = log.fold()
    assert state.current == {("Alice_Chen", "role"): ("staff", 3)}
    assert state.history("Alice_Chen", "role") == [
        ("lead", 1, True),
        ("senior", 2, True),
        ("staff", 3, False),
    ]


def test_upsert_over_existing_key_behaves_as_superse() -> None:
    log = _log(
        ENTITY_E1,
        "UPSERT FACT E1.role = lead pos=1",
        "UPSERT FACT E1.role = senior pos=2",
    )
    state = log.fold()
    assert state.current == {("Alice_Chen", "role"): ("senior", 2)}
    assert state.history("Alice_Chen", "role") == [
        ("lead", 1, True),
        ("senior", 2, False),
    ]


def test_drop_clears_current_but_history_remains() -> None:
    log = _log(
        ENTITY_E1,
        "UPSERT FACT E1.role = lead pos=1",
        "SUPERSEDE FACT E1.role : lead => senior pos=2",
        "SUPERSEDE FACT E1.role : senior => staff pos=3",
        "DROP FACT E1.role pos=4",
    )
    state = log.fold()
    assert ("Alice_Chen", "role") not in state.current
    assert state.history("Alice_Chen", "role") == [
        ("lead", 1, True),
        ("senior", 2, True),
        ("staff", 3, False),
    ]
    # A later UPSERT on the dropped key appends; the dropped entry stays unflagged.
    log.append(["UPSERT FACT E1.role = lead pos=5"])
    state = log.fold()
    assert state.current == {("Alice_Chen", "role"): ("lead", 5)}
    assert state.history("Alice_Chen", "role") == [
        ("lead", 1, True),
        ("senior", 2, True),
        ("staff", 3, False),
        ("lead", 5, False),
    ]


def test_resolve_after_mark_unresolved() -> None:
    log = _log(
        ENTITY_E1,
        "MARK UNRESOLVED E1.eta candidates=[soon@1, later@2]",
    )
    state = log.fold()
    assert state.unresolved == {("Alice_Chen", "eta"): (("soon", 1), ("later", 2))}
    assert ("Alice_Chen", "eta") not in state.current

    log.append(["RESOLVE E1.eta = soon pos=3"])
    state = log.fold()
    assert state.unresolved == {}
    assert state.current == {("Alice_Chen", "eta"): ("soon", 3)}
    assert state.history("Alice_Chen", "eta") == [("soon", 3, False)]


def test_unbound_eid_resolves_to_none_and_is_dropped() -> None:
    log = _log(
        "UPSERT FACT E9.a = v pos=1",
        "UPSERT REL E1 manages E9 pos=2",
        "UPSERT NEG E9.a pos=3",
        "MARK UNRESOLVED E9.a candidates=[x@1, y@2]",
    )
    state = log.fold()
    assert state.name_of("E9") is None
    assert state.name_of("E7") is None
    assert state.current == {}
    assert state.relations == set()
    assert state.negations == set()
    assert state.unresolved == {}


def test_entity_binding_anywhere_in_log_resolves_eid() -> None:
    # PLAN.md section 4: "the grader resolves eid->name via ENTITY records".
    # A FACT citing an eid with no ENTITY record anywhere = unbound = dropped;
    # once the ENTITY op exists in the log, the eid resolves.
    log = _log("UPSERT FACT E9.a = v pos=1")
    assert log.fold().current == {}
    log.append(["UPSERT ENTITY E9 type=person name=Zoe_Q"])
    state = log.fold()
    assert state.name_of("E9") == "Zoe_Q"
    assert state.current == {("Zoe_Q", "a"): ("v", 1)}


def test_relations_and_negations_lifecycle() -> None:
    log = _log(
        ENTITY_E1,
        ENTITY_E2,
        "UPSERT REL E1 manages E2 pos=1",
        "UPSERT NEG E1.remote pos=2",
    )
    state = log.fold()
    assert state.relations == {("Alice_Chen", "manages", "Project_Kestrel")}
    assert state.negations == {("Alice_Chen", "remote")}

    log.append(["DROP REL E1 manages E2 pos=3", "DROP NEG E1.remote pos=4"])
    state = log.fold()
    assert state.relations == set()
    assert state.negations == set()


def test_fold_is_pure_and_idempotent() -> None:
    log = _log(
        ENTITY_E1,
        "UPSERT FACT E1.role = lead pos=1",
        "SUPERSEDE FACT E1.role : lead => senior pos=2",
        "UPSERT REL E1 owns E2 pos=3",
        "MARK UNRESOLVED E1.eta candidates=[soon@1, later@2]",
    )
    ops_before = log.ops
    first = log.fold()
    second = log.fold()
    # Idempotent: identical views.
    assert first.current == second.current
    assert first.relations == second.relations
    assert first.negations == second.negations
    assert first.unresolved == second.unresolved
    assert first.history("Alice_Chen", "role") == second.history("Alice_Chen", "role")
    assert first.history("Alice_Chen", "role") is not second.history("Alice_Chen", "role")
    # Pure: the stored log is untouched by folding.
    assert len(log) == len(ops_before)
    assert log.ops == ops_before
    # Unknown keys yield empty history, never an error.
    assert first.history("Nobody", "nothing") == []
