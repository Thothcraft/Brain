# Thothcraft Architecture Migration — Living Report

Governing principle: **Context is all you need.**

Started: 2026-09-24. Updated incrementally per phase.

## Repository baseline

| Repo | Branch | HEAD | Stack | Baseline result |
|---|---|---|---|---|
| gadm21/whispy | main | b4a743d0bd003d0e059fe0ced045dfc2ecbabad0 | Python ≥3.10, setuptools, entry-point plugins | `pytest tests` → **96 passed** (BASELINE_PASS; `test_runner_window_has_binding_map` is timing-flaky, passes on retry) |
| Thothcraft/thoth | main | 0bc483b78f5d563e3fab7bcf42dca6b04828fe5c | Python ≥3.10, Flask/Jinja, whispy dep | `pytest tests` → 48 passed, **1 failed** (`test_model_occupancy_publishes_only_explicit_binary_class` — stale `chunk_index` kwarg; BASELINE_FAILURE_EXISTING), `test_settings_and_occupancy.py` collection error (`fcntl` missing on Windows; CANNOT_RUN_ENVIRONMENT) |
| Thothcraft/Brain | main | bb6ba4ff684f6e2a15db1267a9c7aa82d5efe9b0 | FastAPI, SQLAlchemy, uvicorn | `pytest tests` → **55 passed** (BASELINE_PASS) |
| Thothcraft/ResearchPortal | main | 62935b2a734f64b4da6ff78f6eceaa377d5eeb6f | Next.js 15.5, React 19, TS 5.3, Supabase | `tsc --noEmit` PASS, `next build` PASS; `next lint` deprecated (CANNOT_RUN_ENVIRONMENT — no eslint CLI wired) |
| Thothcraft/thoth-app | main | e3e40f1aa1248117a4ff649c36c3636330ee6601 | Flutter 3.47.5 / Dart 3.13.4 (SDK at C:\flutter, not on PATH) | `flutter test` → **1 passed** (BASELINE_PASS) |
| Thothcraft/website | main | 0a14bbdc892cbc26e34b4e49cff7537a0086ede7 | Vite 6, React 19, R3F 9, TS 5.8 | `typecheck` PASS, `lint` PASS, `build` PASS (chunk-size warning only) |
| Research-Portal-app, EducationPortal, PCA, radar | — | not cloned locally | — | CANNOT_RUN_ENVIRONMENT (reference only) |

## Security/hygiene audit findings

- **thoth/.env** — committed to git with real secrets (BRAIN_AUTH_TOKEN, API_KEY, FLASK_SECRET_KEY, MQTT_PASSWORD). Action: untrack, add `.env.example`, **credential rotation required**.
- **ResearchPortal/.env.local** — committed despite `.gitignore` rule (force-added historically). Contains NEXT_PUBLIC_API_URL/WS_URL (public-scoped, low sensitivity). Action: untrack, add example.
- **Brain/.venv** — was committed; already untracked in HEAD commit bb6ba4f. Resolved.
- **website/dist/** — intentionally versioned (install scripts ship from dist). Retained.
- **thoth/src/backend/config.py** — `HOST` defaults to `0.0.0.0` (LAN-exposed by default), `SECRET_KEY` hardcoded fallback `thoth-dev-secret-key`, `AP_PASSWORD` default `thoth123`.
- **thoth/src/backend/app.py** — `CORS(app)` wide open, `socketio cors_allowed_origins="*"`, hardcoded demo users `admin/admin123` + `user/password123`, no `login_required` on routes.
- **thoth/src/backend/terminal_manager.py** — SSH provisioning via pexpect; must be admin-scoped (verified below).

## Phase log

### Phase 0 — Baseline + security (done)

- Baseline table above; all runnable suites executed.
- **thoth**: `.env` untracked + `.env.example`; `config.py` defaults to
  loopback (`THOTH_BIND_MODE=lan` opts in), `SECRET_KEY`/`AP_PASSWORD` are
  per-device persisted secrets; `app.py` CORS/SocketIO restricted to
  loopback origins (`THOTH_CORS_ORIGINS` allowlist); dead demo USERS removed.
- **ResearchPortal**: `.env.local` untracked + `.env.example`.
- **Fixed baseline failure**: `test_model_occupancy_publishes_only_explicit_binary_class`
  updated to renamed `second_index` kwarg + dual-post (binary + probability)
  HA publish behavior.

### Phase 1 — Canonical contracts in whispy (done)

- `whispy/contracts/schemas/*.schema.json` — 15 JSON Schemas are the single
  semantic source: observation, source-descriptor, device-descriptor,
  compute-capability, prediction, action-request/result, inference-
  request/trace/result, context-evidence/state/event, relationship,
  minute-manifest.
- `whispy/tools/generate_contracts.py` — emits TypeScript + Dart bindings
  from schemas; `validate` subcommand checks fixtures.
- `whispy/contracts/fixtures/*.json` — 15 shared cross-language test vectors;
  `tests/test_contract_fixtures.py` round-trips each through the Python
  dataclasses (112 whispy tests pass).

### Phase 2 — ObservationSource generalization (done)

- `Observation` (source_id, schema, quality/provenance, optional
  confidence/accuracy/spatial_reference/privacy_classification) with
  `from_sample`/`to_sample` projection to legacy `SensorSample`.
- `SourceDescriptor` = `SensorDescriptor` + `source_class` ("sensor"|
  "context") + `health`; `SourceHandle` = `SensorHandle` alias;
  `ObservationAdapter` = `SensorAdapter`; `ContextAdapter` for
  non-physical sources; `DeviceHandle.source()`/`sources()`.
- `AmbiguousSourceError`/`SourceNotFoundError`/`SourceUnavailableError`
  (KeyError subclasses — backward compatible) wired into LocalDevice and
  LanDevice resolution.

### Phase 3 — Model manifest v2 (done)

- `whispy-model/v2` in `ModelManifest`: id/version/task/lifecycle
  (streaming|windowed|batch)/execution classes/resources/privacy/
  config_schema. v1 + `thoth-model/v1` still accepted and normalized.
- `thoth/src/backend/model_runtime.py` delegates format constants to
  `whispy.contracts` and accepts v2 (constraints-wrapped inputs, outputs
  list, lifecycle→cadence mapping).

### Phase 4 — Canonical minute (done)

- `thoth-minute/v1` schema + `MinuteManifest`/`MinuteSourceData` contracts.
- `whispy/minutes.py` — `read_minute()` normalizes legacy
  `thoth-minute-manifest/v5–v7` (chunk_index→second_index, chunks→seconds,
  expected_chunks→expected_seconds) in memory; `write_minute_manifest()`
  for new writers; `iter_minute_dirs()`.
- `thoth/tools/{audit_minutes,migrate_minutes,verify_minute_migration}.py`
  — explicit, non-destructive, idempotent migration (writes `minute.json`
  alongside untouched legacy files). Fixture tests in
  `thoth/tests/test_minute_migration.py` (5 pass).

### Phase 5/6 — Multi-instance + actuators (already satisfied)

- Stable descriptor ids (hardware_id hash), ambiguity errors, actuator
  contracts + conformance were landed in the prior whispy migration.

### Phase 7/12 — Thoth v1 local API (done)

- `thoth/local_api/server.py` + `thoth/daemon/service.py`: `/api/v1/device`
  (with compute), `/health`, `/compute`, `/sources`, `/sources/{id}`,
  `/sources/{id}/observations`, `/actuators` (+actions), `/models`,
  `/model-deployments`, `/inference` (canonical InferenceResult+Trace),
  `/minutes`, `/minutes/{id}`, `/minutes/{id}/seconds/{s}`, `/privacy`,
  `/sync`. Token auth unchanged; legacy `/api/*` routes preserved.
- `whispy/compute.py` — `probe_compute()` (psutil + nvidia-smi best-effort,
  unknown metrics stay None) + `model_fits()` resource check.
- `ModelRunner.infer()` — canonical InferenceResult with full
  InferenceTrace (model id/version/artifact hash, runtime id, execution
  device/class, input bindings, input interval, latency, confidence).
- **Fix**: `ThreadingHTTPServer.server_bind` reverse-DNS (`getfqdn`)
  stalled ~20s on Windows — bypassed via `_FastBindHTTPServer`.
- `tests/test_v1_api.py` — 5 tests pass.

### Phase 9 — Hub scaffold (done)

- New `hub/` repo: `ContextCache` (TTL snapshot mirror of Brain's
  `/v1/context/snapshot`), `AutomationEngine` (declarative rules,
  edge-triggered on state transitions, cooldown re-arm), `DeviceRegistry`
  (Brain `/v1/devices` + lazy LAN health probes), `IntegrationRegistry`
  (declared source/sink providers), `Hub`+`HubServer` local API
  (`/api/v1/{status,devices,context,context/state,rules,integrations,
  context/refresh,tick}`, bearer-token, `_FastBindHTTPServer`).
- `tests/test_hub.py` (5) + `tests/test_context_loop.py` — the
  end-to-end claim: fixture source → `ModelRunner.infer()` →
  InferenceResult+Trace → ContextEvidence → ContextState → ContextEvent
  → automation fires → ActionResult. 6 tests pass.

### Phase 11 — Brain context model (done)

- `server/db.py`: `context_entity`, `context_relationship`,
  `context_evidence`, `context_state`, `context_event` — user-scoped,
  JSON payloads, validity windows, evidence links.
- `server/v1/context.py`: `/v1/context/{entities,relationships,evidence,
  state,events,snapshot}` — entity upsert, relationship end (history
  preserved), batch evidence ingest, state upsert emitting entered/
  exited/changed events, full snapshot. Tenant isolation tested.
- `run_migrations.py`: CREATE TABLE IF NOT EXISTS for all five.
- 61 Brain tests pass.

### Phase 13 — Client contract bindings (partial)

- `ResearchPortal/lib/contracts.generated.ts` + `thoth-app/lib/
  contracts.generated.dart` generated from canonical schemas; `tsc
  --noEmit` clean, `dart analyze` clean. Component wiring is follow-up.

### Remaining

- Wire generated contracts into ResearchPortal/thoth-app/website views.
- Demo scaffolds (workspace `demos/`) + live end-to-end verification.
- Brain WS topics for context updates (polling works today).
