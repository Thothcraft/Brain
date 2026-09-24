"""End-to-end context loop — the architecture's core claim, exercised.

    observation source (fixture) → ModelRunner.infer() → InferenceResult
        → ContextEvidence → ContextState (transition → ContextEvent)
        → automation rule fires → ActionResult

"Context is all you need": the automation never touches the sensor or
the model — it reacts to the derived context state alone. Runs only when
whispy is importable (workspace layout or installed package).
"""

import time

import pytest

whispy = pytest.importorskip("whispy")

from whispy.contracts import (  # noqa: E402
    ContextEvidence, ContextEvent, ContextState, InferenceResult,
    ModelManifest, Prediction,
)
from whispy.devices.local import LocalDevice  # noqa: E402
from whispy.models import ModelRunner  # noqa: E402
from whispy.processors.base import Processor, ProcessorMeta  # noqa: E402
from whispy.sensors import FixtureDriver  # noqa: E402

from server.automation import AutomationEngine  # noqa: E402


class _OccupancyProcessor(Processor):
    """Fixture model: 'occupied' when the window has any samples."""

    def metadata(self):
        return ProcessorMeta(name="occ", processor_type="rule",
                             inputs=("radar",), task="occupancy")

    def predict(self, window):
        n = sum(len(v) for v in window.samples.values())
        return Prediction(
            label="occupied" if n else "empty",
            confidence=0.9 if n else 0.6,
            task="occupancy",
            metadata={"samples": n})


def _device():
    dev = LocalDevice(device_id="node-1",
                      drivers={"fixture": FixtureDriver()})
    dev.open({"fixture": {"sensor_type": "radar", "payloads": [[1.0]],
                          "sample_rate": 20}})
    return dev


def test_full_context_loop():
    dev = _device()
    try:
        handle = dev.source("radar-0")
        manifest = ModelManifest(
            id="occ-v1", name="occ-v1", processor="rule",
            version="1.0.0", format="whispy-model/v2",
            task="occupancy", lifecycle="windowed")
        runner = ModelRunner(_OccupancyProcessor(), {"radar": handle},
                             window_seconds=0.2, device_id="node-1",
                             manifest=manifest, runtime_id="rm-1")

        # 1. inference → canonical InferenceResult with trace
        result = runner.infer(warmup_s=0.3)
        assert isinstance(result, InferenceResult)
        assert result.status == "succeeded"
        assert result.prediction.label == "occupied"
        assert result.trace.model_id == "occ-v1"
        assert result.trace.execution_device == "node-1"
        assert result.trace.input_bindings["radar"] == handle.info.id
        assert result.trace.latency_ms is not None
        runner.stop()

        # 2. prediction → evidence (predictions are evidence, not truth)
        evidence = ContextEvidence(
            key="spatial.presence/v1",
            value=result.prediction.label,
            timestamp=time.time(),
            source_id=handle.info.id,
            device_id="node-1",
            model_id=result.trace.model_id,
            model_version=result.trace.model_version,
            confidence=result.prediction.confidence,
            execution_class=result.trace.execution_class,
            provenance={"runtime_id": result.trace.runtime_id,
                        "latency_ms": result.trace.latency_ms})
        ev = evidence.to_dict()
        assert ev["model_id"] == "occ-v1"

        # 3. evidence → derived state → transition event
        state = ContextState(
            key="spatial.occupancy/v1", value="occupied",
            entity_id="space:lab", confidence=ev["confidence"],
            since=ev["timestamp"], evidence_ids=[ev["id"]],
            estimator="test")
        event = ContextEvent(
            key="spatial.occupancy/v1", event_type="entered",
            entity_id="space:lab", value="occupied",
            timestamp=ev["timestamp"],
            provenance={"evidence": ev["id"]})
        assert event.event_type == "entered"

        # 4. context snapshot → Brain automation fires → ActionResult
        actions = []
        engine = AutomationEngine(actuate=lambda aid, cmd: (
            actions.append({"actuator": aid, "command": cmd}),
            {"status": "succeeded", "action_type": "light"})[1])
        engine.add_rule({
            "name": "lab-light-on",
            "when": {"key": "spatial.occupancy/v1",
                     "entity_id": "space:lab", "equals": "occupied"},
            "then": {"actuator_id": "light-lab", "operation": "set",
                     "params": {"on": True}}})
        fired = engine.evaluate({"states": [state.to_dict()]})
        assert len(fired) == 1
        assert fired[0]["result"]["status"] == "succeeded"
        assert actions[0]["actuator"] == "light-lab"
        assert actions[0]["command"]["params"]["on"] is True
    finally:
        dev.close()
