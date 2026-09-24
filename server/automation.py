"""Automation rules — context-state triggers → actuator actions.

Automation evaluation lives in **Brain**: it is the reasoning/orchestration
layer, so rules keep firing while control surfaces (Portal, apps) are
offline. A rule fires when a context state matches ``when`` (key + optional
entity + expected value). Actions are dispatched through an injected
``actuate`` callable — the default records the action for later delivery.

Rules are declarative dicts — no executable code is stored::

    {"name": "desk-occupied-light",
     "when": {"key": "spatial.occupancy/v1", "entity_id": "space:lab",
              "equals": "occupied"},
     "then": {"actuator_id": "light-desk", "operation": "set",
              "params": {"on": true}},
     "cooldown_s": 60}
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional


class AutomationEngine:
    """Evaluates declarative rules against a context snapshot."""

    def __init__(self,
                 actuate: Optional[Callable[[str, Dict[str, Any]], Dict[str, Any]]] = None):
        self._actuate = actuate or self._record_only
        self._rules: List[Dict[str, Any]] = []
        self._last_fired: Dict[str, float] = {}
        self._matched: Dict[str, bool] = {}   # edge-triggered: fire on false→true
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
            then = rule["then"]
            result = self._actuate(
                str(then.get("actuator_id") or ""),
                {"operation": then.get("operation") or "set",
                 "params": then.get("params") or {}})
            record = {"rule": name, "fired_at": now, "result": result}
            self.results.append(record)
            fired.append(record)
            self._last_fired[name] = now
        return fired
