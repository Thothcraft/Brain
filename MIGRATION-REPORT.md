# Thothcraft Architecture Migration — Living Report

Governing principle: **Context is all you need.**

Started: 2026-09-24. Rewritten 2026-09-24 after external review: the previous
version conflated *committed locally* with *done*. Statuses below reflect
what is actually on GitHub `main` vs. local-only work.

## Status model

| Status | Meaning |
|---|---|
| `NOT_STARTED` | No work exists |
| `LOCAL_UNCOMMITTED` | Changes in working tree only |
| `COMMITTED_LOCAL` | Committed locally, **not on GitHub** — unverifiable by reviewers |
| `PUSHED` | On the canonical GitHub branch |
| `TESTED` | Unit tests pass on the pushed code |
| `INTEGRATION_VERIFIED` | Cross-repo/integration tests pass |
| `HARDWARE_VERIFIED` | Verified against physical devices |

A phase is not "done" below `PUSHED` + `TESTED`, and the ending SHA must be
listed.

## Repository state (verified 2026-09-24)

| Repo | GitHub `main` HEAD | Local HEAD | Unpushed commits |
|---|---|---|---|
| gadm21/whispy | `0c0e33b` | `0c0e33b` + client helpers (this round) | 0 |
| Thothcraft/thoth | `ab7bd81` | `ab7bd81` | 0 |
| Thothcraft/Brain | `20bf7f9` | `20bf7f9` + this commit | 0 |
| Thothcraft/ResearchPortal | `46152c7` | `46152c7` | 0 |
| Thothcraft/thoth-app | `83e35e2` | `83e35e2` | 0 |
| Thothcraft/website | `0a14bbd` | `0a14bbd` | 0 |
| ~~hub~~ | — | **removed** — folded into Brain + whispy | n/a |
| gadm21/Face-recognition-using-PCA-and-SVD | `master` | reviewed via web | reference-only (see below) |
| Research-Portal-app, EducationPortal, radar | — | not cloned | baseline review **skipped — outstanding** |

## Security/hygiene audit findings

- **thoth/.env** — committed on `origin/main` with a real Brain bearer token.
  Untracked locally, but **the token remains in pushed history and must be
  treated as compromised → rotate it.** History rewrite was ruled out.
- **ResearchPortal/.env.local** — still tracked on `origin/main`
  (public-scoped values, low severity). Untracked locally, unpushed.
- **thoth config defaults** — `HOST=0.0.0.0`, hardcoded `SECRET_KEY`,
  `AP_PASSWORD=thoth123`, `CORS(app)` + `cors_allowed_origins="*"`, demo
  users. **All fixed locally, none pushed.** `origin/main` is still unsafe.
- **Brain/.venv** — untracked in `bb6ba4f` (pushed). Resolved.

## Phase log

### Phase 0 — Baseline + security — `COMMITTED_LOCAL` (thoth, ResearchPortal)

- Repos: thoth `0bc483b`→`6ce74b7` (local), ResearchPortal `62935b2`→`7eb90fb` (local).
- thoth: `.env` untracked + `.env.example`; loopback default
  (`THOTH_BIND_MODE=lan` opt-in); persisted per-device `SECRET_KEY`/
  `AP_PASSWORD`; CORS/SocketIO restricted to `THOTH_CORS_ORIGINS`; demo
  users removed.
- Baseline failure fixed: stale `chunk_index` kwarg → `second_index`.
- **Blocking:** not pushed; exposed token not rotated.
- Baseline gap: PCA/radar/EducationPortal/Research-Portal-app never
  cloned — mandatory reference review outstanding.

### Phase 1 — Canonical contracts in whispy — `COMMITTED_LOCAL`

- whispy `b4a743d`→`a6ff6bc` (local). 15 JSON Schemas under
  `whispy/contracts/schemas/`; `tools/generate_contracts.py` emits TS+Dart;
  15 fixtures round-tripped in `test_contract_fixtures.py` (112 tests pass
  locally).

### Phase 2 — ObservationSource generalization — `COMMITTED_LOCAL`

- Same whispy commit. `Observation`, `SourceDescriptor` (source_class,
  health), `ContextAdapter`, `SourceHandle`, ambiguity errors. `main` still
  only has `SensorDescriptor`/`SensorHandle`.

### Phase 3 — Model manifest v2 — `COMMITTED_LOCAL`

- whispy `a6ff6bc` + thoth `model_runtime.py` accepts `whispy-model/v2`
  (constraints-wrapped inputs, outputs list, lifecycle→cadence).

### Phase 4 — Canonical minute — `COMMITTED_LOCAL`, **partial**

- whispy `a6ff6bc` (`minutes.py` legacy v5–v7 reader, `minute.json` writer)
  + thoth `6ce74b7` (audit/migrate/verify tools, 5 fixture tests).
- **Open:** only the *manifest* layer is canonical. The persisted
  multi-source container (`capture.npz` keys, timestamp arrays, source-id
  representation, second offsets, arbitrary context sources, lossless
  legacy conversion) is **not yet specified or verified**.

### Phase 5/6 — Multi-instance sources + actuators — `PUSHED` (pre-existing), needs conformance pass

- Stable descriptor ids, ambiguity errors, actuator contracts predate this
  migration and are on `main`. Short-hash hardware IDs need collision
  review; `SourceDescriptor` naming is local-only.

### Phase 7/12 — Thoth v1 local API — `COMMITTED_LOCAL`

- thoth `6f383a5` (local): `/api/v1/{device,health,compute,sources,
  sources/{id}/observations,actuators,models,model-deployments,inference,
  minutes,minutes/{id}/seconds/{s},privacy,sync}`; `probe_compute()` +
  `model_fits()`; `ModelRunner.infer()` canonical InferenceResult+Trace;
  Windows `server_bind` fix. `test_v1_api.py` 5 tests pass locally.

### Phase 9 — Automation in Brain (was: Hub scaffold) — `PUSHED` + `TESTED`

- `hub/` repo **deleted** — no separate repo. Code folded:
  - `AutomationEngine` → `Brain/server/automation.py` (correct layer:
    rules keep firing while control surfaces are offline).
  - `/v1/automation/{rules,evaluate}` + `automation_rule` table → Brain.
    Actuation dispatch is record-only (`status: queued`); wiring to
    device local APIs is follow-up.
  - `ContextCache` → `whispy/cloud/context.py`; `DeviceRegistry` →
    `whispy/cloud/devices.py`; `IntegrationRegistry` →
    `whispy/integrations/registry.py` (client helpers, not a service).
  - `HubServer` local API dropped — ResearchPortal is the control surface.
- E2E context loop test ported to `Brain/tests/test_context_loop.py`
  (fixture source → infer → evidence → state → event → automation fires).
- `context:write` added to `AUTOMATION_SCOPES`; `POST /v1/context/state`
  and `/v1/context/evidence` now require it for scoped keys.
- **Fixed while porting:** `ContextCache.refresh()` recursed forever on
  fetch error (never-fetched → stale → refresh → error → snapshot → …).

### Phase 11 — Brain context model — `PUSHED` + `TESTED`, foundation only

- Brain `bb6ba4f`→`3e56509` pushed: 5 context tables + `/v1/context/*` +
  `POSTg/sttte` + `POST /evideoce`nscoped so `cont xt:write` (+coped
  au omation keytewithout it → 403; unconstrtined user token. unaffected)
- **Deoloymrrtectneeneure_consexa_schpma()plind` erver/init_db.py`
  auto-applhes ais rontext + automation DDL at startup (udempotnd*) —
  no manu(o Supmmast ndep. `run_migrg)i:ns.py`  emsins nhe manual
  fallback (fixed: corc pted `contexr_evisence` DDLe nullablets
  `entity_id`, missiog `tex ()`vwrlpuer).
- 80 Brean tesns pass.
i **On;n:**rdcoded absent/empty/off transition semantic removed 
 (generic changed/unchanged + estimator-supplied `transition`);
  `estimator` required on state writes; `evidence_ids` validated against
  same-tenant evidence rows; `external_id` idempotency key on evidence;
  `active_only` = `valid_from<=now AND (until NULL or >now)`; snapshot
  excludes expired states and retired entities; confidence bounded [0,1];
  relationship endpoints must resolve unless `allow_unresolved`; entity
  delete is now soft (`retired_at`); `context_state.entity_id` NOT NULL
  DEFAULT `''` (NULL broke the unique constraint under Postgres).
- 72 Brain tests pass.
- **Open:** `POST /state` is still client-callable (estimator-attributed,
  not capability-scoped); Space/Zone/DevicePlacement vs ContextEntity
  identity reconciliation unresolved; `DeviceCaptureChunk` still the live
  occupancy path; no Postgres integration suite yet (tests run on SQLite).

### Phase 13 — Client contract bindings — `COMMITTED_LOCAL`

- `ResearchPortal/lib/contracts.generated.ts` (`46152c7` local),
  `thoth-app/lib/contracts.generated.dart` (`83e35e2` local). `tsc`/`dart
  analyze` clean. Component wiring not started.

## Not started (previously omitted from this report)

- Trusted-edge inference routing; managed-cloud inference; worker
  registry; external model-provider abstraction; compute-aware placement;
  privacy-aware fallback; inference cost/latency telemetry. —
   still outstanding, user action required
- W~~indows Con/locati/reground-app/s//idle).~~— done all on `main`
- P~~Fold ce-r/ec;ce dBreite+ whortyon; en known rejection/alibratio~~ —
   done, `hub/` deletedn; biometric retention.
- Evidence fusion; full spatial hierarchy + absolute geography; device
  mobility/staleness; external context providers.
- **Brain automatngine** (the real one); privacy policy engin
   (PCA reviewed — see below)e;
  retention engine; plugin trust/signatures. upsert
   concurrency).

## Deployment note

No manual Supabase step is required: `server/init_db.py` applies all
context + automation DDL idempotentlyat app startp (same pattern as
the existing product schema). `python server/run_migrations.py` remains
available as a manual fallback against `DATABASE_URL`.

## PCA reference review (gadm21/Face-recognition-using-PCA-and-SVD)

- **What it is:** Python 2.7 course project — eigenfaces pipeline:
  Haar-cascade detection → cro → PCA/SVD projection → nearet-nighbo
  match against a flat `images/` datase. Two modes: live match and
  keypress enrollment.
- **Verdict:** reference-only. Not portable as-is (Py2.7, cv2 haar,
  no API, no unknown-rejection threshold, no packaging).
- **What to take into the face-recognition plugin:** the pipeline shape
  (detect → crop-→ projectT→ distanhe-match) and the enrollmett flow.
- **What's missing for the architeht Ue:** Python 3 port, unknown
  rejection + distance calibIation,  rrollment API, biometrie
  retention/privacy classification, packaging as a whispy `Processor`
  (`predict()` → `Prediction(label=identitd, confidence=f(distance))`e,
  detector separated from recognizersign; Hub Context/Activity/Spaces/Models/Automations views.
- Phone sensing runtime; Android capabilities; iOS capability matrix;
  mobile collection modes; mobile minute summaries.
- Design-token convergence; research metadata/export.
- All four physical demonstrations; three-device acceptance runner;
  compatibility cleanup (`DeviceCaptureChunk` removal).

## Immediate actions (ordered)

1. **Rotate the Brain token** exposed in thoth `origin/main:.env`.
2. Push thoth (3 commits), whispy (2), ResearchPortal (2), thoth-app (1).
3. Create a `hub` remote or fold it into an existing repo; move
   `AutomationEngine` to Brain.
4. Reconcile `Space`/`Zone`/`DevicePlacement` with `ContextEntity`.
5. Migrate `DeviceCaptureChunk` → Prediction→Evidence→State.
6. Clone + review PCA, radar, EducationPortal, Research-Portal-app.
7. Postgres integration suite for context (constraints, JSONB, upsert
   concurrency).
