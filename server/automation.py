"""Brain-side automation engine (§17).

Automation evaluation lives in **Brain**: it is the reasoning/orchestration
layer, so rules keep firing while control surfaces (Portal, apps) are
offline. Rules are declarative: ``{"when": {...}, "then": {...}}`` — no
executable code is stored or evaluated. The engine is edge-triggered: a
rule fires only on a false→true transition of its ``when`` clause, with
an optional per-rule cooldown. Evaluation runs against a context
snapshot — the automation never touches sensors or models directly.

``AutomationEngine`` is the pure evaluator; ``PersistentAutomation``
(below) is the production runtime that keeps rule state, executions and
action requests durable in the DB and dispatches to Thoth nodes.
"""

from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional


class AutomationEngine:
    """Evaluates declarative rules against a context snapshot."""

    def __init__(self,
                 actuate: Optional[Callable[[str, Dict[str, Any]], Dict[str, Any]]] = None):
        self._actuate = actuate or self._record_only
        self._rules: List[Dict[str, Any]] = []
        self._last_fired: Dict[str, float] = {}
        self._matched: Dict[str, bool] = {}   # edge-triggered: fire on false→true
        #: name of the rule currently firing — lets the actuate callable
        #: attribute executions/actions to the rule without a signature
        #: change (``actuate(actuator_id, command)``).
        self.current_rule_name: Optional[str] = None
        self.results: List[Dict[str, Any]] = []

    def add_rule(self, rule: Dict[str, Any]) -> Dict[str, Any]:
        if not rule.get("name") or not isinstance(rule.get("when"), dict) \
                or not isinstance(rule.get("then"), dict):
            raise ValueError("rule needs name, when{}, and then{}")
        self._rules.append(rule)
        return rule

    def remove_rule(self, name: str) -> bool:
        before = len(self._rules)
        self._rules = [r for r in self._rules if r.get("name") != name]
        return len(self._rules) < before

    def list_rules(self) -> List[Dict[str, Any]]:
        return [dict(r) for r in self._rules]

    @staticmethod
    def _record_only(actuator_id: str, command: Dict[str, Any]) -> Dict[str, Any]:
        return {"status": "queued", "actuator_id": actuator_id,
                "command": command}

    def _matches(self, when: Dict[str, Any],
                 states: List[Dict[str, Any]]) -> bool:
        key = when.get("key")
        entity = when.get("entity_id")
        for s in states:
            if s.get("key") != key:
                continue
            if entity and s.get("entity_id") != entity:
                continue
            if "equals" in when and s.get("value") != when["equals"]:
                continue
            if "min_confidence" in when and \
                    (s.get("confidence") or 0) < when["min_confidence"]:
                continue
            return True
        return False

    def evaluate(self, snapshot: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Fire rules whose ``when`` matches the snapshot's states."""
        states = snapshot.get("states") or []
        fired = []
        now = time.time()
        for rule in self._rules:
            name = rule["name"]
            cooldown = float(rule.get("cooldown_s") or 0)
            matched = self._matches(rule["when"], states)
            was = self._matched.get(name, False)
            self._matched[name] = matched
            # Edge-triggered: fire on the false→true transition only;
            # cooldown additionally rate-limits re-arms.
            if not matched or (was and now - self._last_fired.get(name, 0)
                               < cooldown):
                continue
            if was and cooldown <= 0:
                continue  # still matched, no re-fire without a transition
            action = rule["then"]
            command = {"operation": action.get("operation", ""),
                       "params": action.get("params") or {},
                       "device_id": action.get("device_id", ""),
                       "expires_in_s": action.get("expires_in_s")}
            self.current_rule_name = name
            try:
                result = self._actuate(action.get("actuator_id", ""),
                                       command)
            finally:
                self.current_rule_name = None
            record = {"rule": name, "action": command,
                      "result": result}
            self.results.append(record)
            fired.append(record)
            self._last_fired[name] = now
        return fired


# ---------------------------------------------------------------------------
# Persistent runtime — durable rule state + executions + action dispatch
# ---------------------------------------------------------------------------

class PersistentAutomation:
    """Production automation runtime for one user.

    Wraps :class:`AutomationEngine` with DB-backed state so edge-trigger
    latches and cooldown clocks survive restarts, every firing is
    recorded as an :class:`AutomationExecution`, and every action is a
    durable, idempotent :class:`ActionRequest` dispatched to the owning
    Thoth node.
    """

    def __init__(self, db, user_id: int):
        from .db import AutomationRule, AutomationRuleState
        self.db = db
        self.user_id = user_id
        self.engine = AutomationEngine(actuate=self._actuate)
        self._rule_rows: Dict[str, Any] = {}
        self._trigger: Dict[str, Any] = {}
        rows = db.query(AutomationRule).filter(
            AutomationRule.user_id == user_id,
            AutomationRule.enabled == True).all()  # noqa: E712
        for row in rows:
            spec = {"name": row.name,
                    "when": json.loads(row.when or "{}"),
                    "then": json.loads(row.then or "{}"),
                    "cooldown_s": row.cooldown_s or 0.0}
            try:
                self.engine.add_rule(spec)
            except ValueError:
                continue
            self._rule_rows[row.name] = row
            state = db.query(AutomationRuleState).filter(
                AutomationRuleState.rule_id == row.id).first()
            if state is not None:
                self.engine._matched[row.name] = bool(state.matched)
                if state.last_fired_at:
                    self.engine._last_fired[row.name] = \
                        state.last_fired_at.timestamp()

    def evaluate(self, snapshot: Dict[str, Any],
                 trigger: Optional[Dict[str, Any]] = None
                 ) -> List[Dict[str, Any]]:
        """Evaluate rules against the snapshot; persist everything."""
        self._trigger = dict(trigger or {})
        try:
            fired = self.engine.evaluate(snapshot)
        finally:
            self._trigger = {}
        self._persist_states()
        return fired

    # -- internals -------------------------------------------------------------
    def _actuate(self, actuator_id: str, command: Dict[str, Any]
                 ) -> Dict[str, Any]:
        """Create the execution + durable ActionRequest, then dispatch."""
        from .actuation import dispatch_action
        from .db import ActionRequest, AutomationExecution

        rule_name = self.engine.current_rule_name or ""
        rule_row = self._rule_rows.get(rule_name)
        execution = AutomationExecution(
            execution_id=uuid.uuid4().hex,
            rule_id=rule_row.id if rule_row else 0,
            user_id=self.user_id,
            rule_name=rule_name,
            trigger_state_id=self._trigger.get("state_id"),
            trigger_event_id=self._trigger.get("event_id"),
            trigger_snapshot=json.dumps(
                self._trigger.get("state") or {}))
        self.db.add(execution)

        expires_in = command.get("expires_in_s")
        action = ActionRequest(
            action_id=uuid.uuid4().hex,
            execution_id=execution.execution_id,
            user_id=self.user_id,
            device_id=str(command.get("device_id") or ""),
            actuator_id=actuator_id,
            operation=str(command.get("operation") or ""),
            params=json.dumps(command.get("params") or {}),
            origin=f"automation:{rule_name}",
            expires_at=(datetime.utcnow() + timedelta(seconds=float(expires_in))
                        if expires_in else None),
            correlation=json.dumps({
                "evidence_id": self._trigger.get("evidence_id"),
                "state_id": self._trigger.get("state_id"),
                "event_id": self._trigger.get("event_id"),
                "execution_id": execution.execution_id}))
        self.db.add(action)
        self.db.commit()

        dispatch_action(self.db, action)
        result = json.loads(action.result_json) if action.result_json else {
            "status": action.status}
        result = dict(result)
        result["action_id"] = action.action_id
        result["execution_id"] = execution.execution_id
        return result

    def _persist_states(self) -> None:
        from .db import AutomationRuleState
        for name, row in self._rule_rows.items():
            state = self.db.query(AutomationRuleState).filter(
                AutomationRuleState.rule_id == row.id).first()
            if state is None:
                state = AutomationRuleState(rule_id=row.id,
                                            user_id=self.user_id)
                self.db.add(state)
            state.matched = bool(self.engine._matched.get(name, False))
            fired_ts = self.engine._last_fired.get(name)
            state.last_fired_at = (datetime.utcfromtimestamp(fired_ts)
                                   if fired_ts else state.last_fired_at)
        self.db.commit()


def evaluate_user_rules(db, user_id: int,
                        trigger: Optional[Dict[str, Any]] = None
                        ) -> List[Dict[str, Any]]:
    """Load the user's context snapshot and evaluate all enabled rules.

    This is the event-driven entry point: called from the context
    state/event path so automation fires without any manual
    ``POST /v1/automation/evaluate`` — and while every control surface
    is offline.
    """
    from .db import ContextState as _CS
    rows = db.query(_CS).filter(_CS.user_id == user_id).all()
    snapshot = {"states": [r.to_dict() for r in rows]}
    return PersistentAutomation(db, user_id).evaluate(
        snapshot, trigger=trigger)


__all__ = ["AutomationEngine", "PersistentAutomation",
           "evaluate_user_rules"]
