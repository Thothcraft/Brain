# ThothCraft: Target System Architecture and Migration Plan

## Architecture Documentation

**Version:** 3.0  
**Last Updated:** September 2026  
**Architecture Status:** Target architecture approved for migration  
**Current Product Status:** Partially implemented; current repositories do not yet satisfy all target contracts or acceptance tests  
**Primary Validation Environment:** Windows development laptop + two Raspberry Pi devices on the same LAN

---

## 1. Purpose of This Document

This document defines the target architecture for the ThothCraft platform and, equally importantly, explains how the existing system will be migrated into that architecture.

It intentionally separates three concepts that were previously mixed together:

1. **Current implementation** — what exists now across `whispy`, `thoth`, `Brain`, `ResearchPortal`, `thoth-app`, and the website.
2. **Target architecture** — the architecture all repositories should converge toward.
3. **Acceptance state** — what must be tested before a capability can be described as implemented or production-ready.

The architecture is centered around four primary technical components:

- **Whispy** — the reusable Python sensing SDK for local and remote sensing.
- **Thoth** — the installable application that turns a computer or edge device into a managed sensing node.
- **Brain** — the cloud control plane, API, identity, storage, orchestration, and remote-access broker.
- **thothHUB** — the browser application used to manage devices, captures, models, data, and account services.

The existing **Flutter mobile app** remains part of the architecture as a mobile Brain client, but it must consume the same stable public contracts as Whispy and thothHUB instead of maintaining a separate interpretation of Brain responses.

---

## 2. Canonical Public Domains

The following domains are the permanent public interface of the platform:

```text
www.thothcraft.com       marketing, product information, public content
hub.thothcraft.com       thothHUB authenticated web application
api.thothcraft.com       Brain public API and device control plane
get.thothcraft.com       Thoth installers, packages, releases and update metadata
docs.thothcraft.com      Whispy, Thoth and public API developer documentation
```

Provider-specific deployment URLs such as Vercel preview URLs, Railway hostnames, raw storage URLs, or temporary infrastructure addresses are implementation details and must not appear in the public API contract, SDK defaults, documentation, purchase links, mobile deep links, or device configuration.

The canonical API should be versioned from the start:

```text
https://api.thothcraft.com/v1/...
```

Breaking API changes require a new version rather than silently changing the behavior expected by Whispy, Thoth, thothHUB, or the mobile app.

---

## 3. Executive Summary

ThothCraft is an edge-first Sensor-Model-Actuator platform.

The target system has a simple conceptual structure:

```text
                                THOTHCRAFT
                                     │
          ┌──────────────────────────┼──────────────────────────┐
          │                          │                          │
          ▼                          ▼                          ▼
   www.thothcraft.com        docs.thothcraft.com        get.thothcraft.com
      marketing/docs           developer docs            installers/releases
                                     │
                                     ▼
                           api.thothcraft.com
                                  BRAIN
                                     │
                 ┌───────────────────┼───────────────────┐
                 │                   │                   │
                 ▼                   ▼                   ▼
          hub.thothcraft.com      WHISPY              MOBILE APP
              thothHUB          remote Python          Flutter
                 │                   │                   │
                 └──────────────┬────┴───────────────┬──┘
                                │                    │
                                ▼                    │
                              BRAIN                  │
                                │                    │
                         secure device channel       │
                                │                    │
                                ▼                    │
                              THOTH                  │
                       installed node app            │
                                │                    │
                              WHISPY                  │
                      local sensing runtime          │
                                │                    │
                 ┌──────────────┼──────────────┐      │
                 ▼              ▼              ▼      │
              Radar            CSI          Camera/IMU│
```

The core rule is:

```text
Thoth depends on Whispy.
Whispy does not depend on Thoth.
Whispy remote mode depends only on the public Brain API.
thothHUB and the mobile app also depend on the same public Brain API.
```

This means the same Whispy library is useful in two places:

```text
ON A THOTH NODE
Thoth → Whispy → physical sensors

ON A RESEARCHER COMPUTER
Python → Whispy → Brain → Thoth → Whispy → physical sensors
```

The purpose of Brain is not to replace edge execution. Brain provides identity, ownership, authorization, device discovery, routing, synchronization, model deployment, storage, cloud jobs, subscriptions, and remote connectivity.

---

# Part I — Current System

## 4. Current Repository Landscape

The current project already contains most of the necessary capabilities, but the responsibilities overlap and several repositories implement different versions of the same contract.

### 4.1 `whispy`

Current role:

- reusable sensing SDK concepts;
- sensor abstractions;
- datasets and processors;
- actuator abstractions;
- local device access;
- a separate CLI/device daemon;
- partial model deployment and local prediction behavior.

Current problem:

The repository currently mixes the **reusable SDK** with behavior that belongs to the **installed Thoth application**. Its CLI daemon also does not yet implement the full documented sensor → real window → processor → persistent model → actuator runtime.

Target role:

```text
WHISPY = SDK/library only
```

The Python package identity should become consistently:

```bash
pip install whispy
```

```python
import whispy
```

Whispy remains capable of both local and remote sensing.

---

### 4.2 `thoth`

Current role:

- older Raspberry Pi-focused application;
- sensor collection and device management;
- Flask/local dashboard functionality;
- Brain communication;
- dedicated hardware behavior;
- setup and appliance-oriented workflows.

Current problem:

The repository contains useful device/application behavior, but it duplicates sensing/runtime responsibilities that now belong in Whispy. It is also more Pi-specific than the desired cross-platform product.

Target role:

```text
THOTH = the installed device application
```

It will contain:

- CLI;
- persistent background daemon/service;
- local authenticated API/IPC;
- local dashboard;
- capture orchestration;
- pairing and device identity;
- Brain connection;
- model/deployment lifecycle;
- configuration;
- update management;
- operating-system integration.

Sensor implementations and reusable processor/action logic should move behind Whispy APIs rather than be independently maintained in Thoth.

---

### 4.3 `Brain`

Current role:

- FastAPI backend;
- account authentication;
- devices;
- files/captures;
- datasets;
- model records and deployments;
- cloud training-related endpoints;
- subscriptions and Stripe integration;
- AI/research assistant behavior;
- admin functionality.

Current problem:

Brain contains much of the correct infrastructure, but some endpoints and clients disagree on contracts. There are legacy routes, incomplete deployment acknowledgements, storage lifecycle gaps, payment-state problems, and inconsistent client assumptions.

Target role:

```text
BRAIN = authoritative cloud control plane and versioned public API
```

Every remote client — Whispy, Thoth, thothHUB, and mobile — should consume the same versioned public contracts.

---

### 4.4 `ResearchPortal`

Current role:

- Next.js web application;
- authenticated fleet interface;
- device management;
- captures/data;
- models/deployments;
- training/processing pages;
- billing/settings;
- administrative interfaces.

Current problem:

The portal currently contains some behavior that effectively defines its own interpretation of backend contracts. Binary upload proxying and clean dependency installation also require repair.

Target role:

```text
ResearchPortal repository → deployed product name: thothHUB
```

thothHUB should be a browser client for Brain rather than a separate backend domain model.

---

### 4.5 `thoth-app`

Current role:

- existing **Flutter** mobile application;
- authentication;
- device/fleet views;
- live data views;
- Shop/Plans navigation;
- settings and partial device controls.

This application is not a planned React Native app. It already exists in Flutter and should remain Flutter unless a separate future decision changes that.

Current problems include:

- live-data cursor/response interpretation differs from Brain;
- offline devices are not consistently included;
- some advertised settings/features are placeholders;
- push-notification infrastructure is not complete;
- purchase/product navigation does not preserve intent.

Target role:

```text
MOBILE = thin Flutter client for the Brain v1 API
```

It should provide:

- fleet/device status;
- capture controls;
- recent predictions;
- notifications;
- lightweight actuator override where explicitly permitted;
- account/subscription views;
- deep links into thothHUB for complex research workflows.

The mobile app must not define a second live-data contract or duplicate sensing/business logic.

---

### 4.6 Website

Current role:

- product marketing;
- product/shop pages;
- plans;
- download links;
- public messaging.

Target role:

```text
www.thothcraft.com = public marketing/product surface only
```

Authenticated operational flows should move to `hub.thothcraft.com`.

---

## 5. Current Architectural Problems to Remove During Migration

The migration is not simply a repository rename. The following structural problems must be removed:

1. Two partially independent edge runtimes exist: the older dedicated Thoth application and the newer Whispy CLI daemon.
2. Some model deployments are acknowledged with incompatible runtime identifiers.
3. Some model/runtime paths treat processor types inconsistently.
4. Some current inference paths pass sensor availability instead of real sensor measurements.
5. Rule/action schemas differ between documentation, SDK, daemon, portal, and dedicated Thoth code.
6. Local SDK and local daemon contracts disagree on some fields and endpoints.
7. Brain has legacy and current APIs that are not always aligned.
8. Portal binary proxying is not safe for arbitrary binary model/data uploads.
9. Cloud training output and deployment contracts are not yet fully aligned.
10. Mobile live-data behavior differs from Brain's current response contract.
11. Local edge access is broader than the target trust model.
12. Public domains and documentation are inconsistent.
13. Commerce and subscription behavior require independent repair before unattended customer launch.

The target architecture is designed to eliminate these root causes rather than preserve parallel implementations.

---

# Part II — Target Architecture

## 6. Component Ownership Rules

| Capability | Owner | Consumers |
|---|---|---|
| Sensor drivers | Whispy | Thoth, local Python |
| Sensor discovery | Whispy | Thoth, local Python |
| Timestamped sensor samples | Whispy | Thoth, processors, remote stream |
| Windowing/synchronization | Whispy | processors, Thoth |
| Local datasets | Whispy | Python, Thoth |
| Rule processors | Whispy | Thoth, local Python |
| TorchScript processors | Whispy | Thoth, local Python |
| Actuator abstractions | Whispy | Thoth |
| Device application lifecycle | Thoth | user/OS |
| CLI | Thoth | user |
| Daemon/service | Thoth | OS, CLI |
| Local dashboard | Thoth | local browser |
| Capture orchestration | Thoth | CLI, hub, mobile |
| Device pairing | Thoth + Brain | user |
| Remote device channel | Thoth + Brain | Whispy/hub/mobile |
| User accounts | Brain | all remote clients |
| Device ownership | Brain | all remote clients |
| API keys/scopes | Brain | Whispy/automation |
| Cloud storage metadata | Brain | hub/mobile/Whispy |
| Model registry/deployment queue | Brain | Thoth/hub/Whispy |
| Cloud training orchestration | Brain | hub/Whispy |
| Billing/orders | Brain | hub/mobile/website |
| Browser UX | thothHUB | users |
| Mobile UX | Flutter mobile app | users |
| Marketing | website | public |
| Installation/bootstrap | get.thothcraft.com | Thoth users |
| Developer docs | docs.thothcraft.com | developers/researchers |

---

## 7. Whispy Architecture

Whispy is the programmable sensing layer of ThothCraft.

### 7.1 Whispy goals

Whispy must make physical sensing accessible through a consistent Python API whether the sensor is:

- attached to the same computer;
- attached to a local Thoth device;
- attached to a remote Thoth device over the Internet.

### 7.2 Target package structure

```text
whispy/
├── sensors/
│   ├── base.py
│   ├── radar/
│   ├── csi/
│   ├── camera/
│   ├── imu/
│   ├── environmental/
│   └── system/
│
├── streams/
├── synchronization/
├── windows/
├── datasets/
│
├── processors/
│   ├── base.py
│   ├── rules.py
│   ├── torchscript.py
│   └── fusion.py
│
├── actuators/
│   ├── base.py
│   ├── home_assistant.py
│   ├── device.py
│   └── webhook.py
│
├── devices/
│   ├── base.py
│   ├── local.py
│   └── remote.py
│
├── cloud/
│   ├── client.py
│   ├── auth.py
│   ├── devices.py
│   ├── captures.py
│   ├── datasets.py
│   ├── models.py
│   └── streaming.py
│
└── integrations/
```

### 7.3 Sensor contract

Each sensor exposes real samples, not availability booleans.

```python
sensor = device.sensor("radar")

async for sample in sensor.stream():
    print(sample.timestamp, sample.payload)
```

Minimum sample contract:

```text
SensorSample
├── device_id
├── sensor_id
├── sensor_type
├── timestamp
├── sequence
├── sample_rate
├── payload_type
├── payload
└── metadata
```

### 7.4 Local and remote parity

Local:

```python
import whispy

local = whispy.local()
radar = local.sensor("radar")
```

Remote:

```python
import whispy

client = whispy.Client()
pi = client.device("thoth-pi-a")
radar = pi.sensor("radar")
```

The developer should be able to apply the same downstream code to both.

### 7.5 Processor contract

Initial processor types:

```text
RuleProcessor
TorchScriptProcessor
FusionProcessor
```

Possible later additions:

```text
ONNXProcessor
TensorRTProcessor
ClassicalMLProcessor
```

Processors receive actual windows produced from Whispy streams.

### 7.6 Actuator contract

Actuators must return an explicit execution result:

```text
queued
executing
succeeded
failed
unsupported
```

No actuator may report success solely because its configuration parsed successfully.

Actions support:

- confidence threshold;
- debounce;
- delay;
- cooldown;
- timeout;
- retry policy;
- asynchronous execution.

---

## 8. Thoth Architecture

Thoth is the installable device application.

It is not synonymous with the CLI. The CLI is one interface into the running Thoth application.

### 8.1 Target structure

```text
thoth/
├── cli/
├── daemon/
├── ipc/
├── local_api/
├── dashboard/
├── pairing/
├── capture/
├── deployment/
├── models/
├── sync/
├── config/
├── updates/
├── platform/
│   ├── linux/
│   ├── raspberrypi/
│   ├── jetson/
│   ├── windows/
│   └── macos/
└── diagnostics/
```

### 8.2 Persistent daemon

The daemon owns hardware while Thoth is running.

```text
OS service
    │
    ▼
thoth daemon
    │
    ├── Whispy sensor registry
    ├── capture manager
    ├── processor registry
    ├── action dispatcher
    ├── Brain connection
    ├── local authenticated interface
    └── telemetry/health
```

### 8.3 CLI

Examples:

```bash
thoth status
thoth sensors
thoth capture start
thoth capture stop
thoth captures
thoth models
thoth models install ...
thoth pair
thoth config
thoth doctor
thoth update
```

Commands communicate with the persistent daemon instead of independently opening the same sensor hardware.

### 8.4 Local connectivity

Preferred trust model:

```text
same-machine CLI
    → Unix socket / named pipe / loopback-only authenticated IPC

LAN browser/client
    → authenticated local API using pairing/local token

Internet clients
    → Brain only
```

The target architecture does not require users to expose Thoth's local API directly to the public Internet.

### 8.5 Installation

Primary user experience:

```bash
curl -fsSL https://get.thothcraft.com/install.sh | sh
```

Windows equivalent:

```powershell
iwr https://get.thothcraft.com/install.ps1 | iex
```

The bootstrapper may internally use native packages:

```text
Raspberry Pi / Debian / Ubuntu → APT package
macOS                         → pkg/Homebrew
Windows                       → MSI/winget
```

Normal users should not need `pip install thoth`.

Whispy may still be an internal dependency of the packaged Thoth application.

### 8.6 SSH

SSH is not part of the required Thoth runtime contract.

If remote administration is offered, it should be an explicit advanced installation/configuration choice rather than a default side effect of installing Thoth.

---

## 9. Brain Architecture

Brain is the authoritative cloud control plane.

### 9.1 Responsibilities

Brain owns:

- accounts and authentication;
- session handling;
- API keys and scopes;
- device ownership;
- pairing;
- online/offline presence;
- device capabilities;
- remote command routing;
- remote sensor-stream authorization;
- capture metadata;
- cloud storage metadata;
- datasets;
- processing definitions;
- model registry;
- deployment queue and deployment state;
- cloud training orchestration;
- subscriptions/entitlements;
- orders/fulfillment metadata;
- notifications;
- admin controls.

### 9.2 Public API

All clients should converge on a versioned contract, for example:

```text
/v1/auth/...
/v1/devices
/v1/devices/{device_id}
/v1/devices/{device_id}/sensors
/v1/devices/{device_id}/captures
/v1/devices/{device_id}/streams
/v1/captures
/v1/datasets
/v1/models
/v1/deployments
/v1/training/jobs
/v1/account
/v1/billing
/v1/orders
```

### 9.3 Device channel

Thoth initiates an outbound secure connection to Brain.

```text
Thoth
  │
  │ outbound TLS/WebSocket
  ▼
Brain
```

This channel supports:

- heartbeat/presence;
- pending commands;
- deployment notifications;
- stream negotiation;
- capture requests;
- telemetry;
- acknowledgement/result messages.

### 9.4 Remote streaming

Remote stream flow:

```text
Windows Python
     │
   Whispy
     │ authenticated stream request
     ▼
   Brain
     │ authorized route
     ▼
   Thoth
     │
   Whispy local sensor
     │
     ▼
 physical sensor
```

Brain authenticates the user/API key and verifies access to the device/sensor before the stream begins.

### 9.5 API scopes

Initial automation/SDK scopes should include concepts such as:

```text
device:read
sensor:read
sensor:stream
capture:create
capture:read
capture:download
model:read
model:deploy
actuator:control
```

Browser account sessions and long-lived automation/API credentials are separate credential classes.

---

## 10. thothHUB Architecture

`hub.thothcraft.com` is the authenticated browser application.

Target capabilities:

```text
Dashboard
Devices
Live status
Captures
Datasets
Labels
Models
Deployments
Processing
Training
Labs
Billing
Orders
Account/settings
Admin (authorized users only)
```

thothHUB should not own a private interpretation of model/capture/device contracts. It consumes Brain's versioned API.

Binary payloads must be byte-preserving. Large model/data uploads should use a binary-safe Brain endpoint or direct signed storage upload rather than being converted to text by a generic proxy.

---

## 11. Mobile App Architecture

The existing mobile application is Flutter and remains part of the target system.

### 11.1 Target role

The mobile app is a lightweight operational client for Brain.

```text
Flutter mobile app
        │
        ▼
api.thothcraft.com/v1
        │
       Brain
        │
        ▼
      Thoth
```

### 11.2 Initial supported mobile capabilities

The first stable mobile scope should be intentionally smaller than thothHUB:

- account login/session;
- list all authorized devices, including offline devices;
- device online/offline and last-seen state;
- sensor/capability summary;
- latest predictions;
- capture start/stop;
- recent capture status;
- important alerts;
- subscription/account summary;
- product/billing deep links;
- safe, explicit actuator override only where permitted.

### 11.3 Push notifications

Target path:

```text
prediction/device event
        │
        ▼
      Brain
        │
 notification policy
        │
        ▼
 push provider / FCM
        │
        ▼
   Flutter mobile app
```

Push delivery should be treated as a Brain notification feature, not as a direct Thoth-to-mobile connection.

### 11.4 Shared contracts

Mobile must use the same structures as Whispy and thothHUB for:

```text
Device
Prediction
Capture
ModelDeployment status
Account/entitlement
```

The current mobile live-data parsing must be replaced by the versioned Brain contract rather than patched against an undocumented shape.

### 11.5 Complex workflows

Complex research workflows such as model upload, dataset editing, advanced processing graphs, and training configuration may deep-link to `hub.thothcraft.com` rather than being duplicated in Flutter.

---

# Part III — Core Contracts

## 12. Device Contract

```text
Device
├── id
├── stable_uuid
├── name
├── owner/account
├── platform
├── architecture
├── software_version
├── whispy_version
├── online
├── last_seen
├── capabilities
├── sensors[]
└── health
```

Physical LAN IP addresses may be reported for local diagnostics but are not the identity of the device.

---

## 13. Sensor Contract

```text
Sensor
├── id
├── type
├── driver
├── driver_version
├── sample_rate
├── units/schema
├── online
├── capabilities
└── metadata
```

---

## 14. Sensor Sample Contract

```text
SensorSample
├── device_id
├── sensor_id
├── timestamp
├── sequence
├── payload_type
├── payload
├── units/schema
└── metadata
```

A sensor stream contains real measurements. Availability/health is a separate field and must never be silently substituted for a measurement.

---

## 15. Window Contract

```text
SensorWindow
├── start_timestamp
├── end_timestamp
├── sensors[]
├── samples by sensor
├── missing/stale modality markers
├── preprocessing metadata
└── timing metadata
```

Multi-modal processors must receive explicit missing/stale information.

---

## 16. Prediction Contract

```text
Prediction
├── id
├── device_id
├── runtime_model_id
├── timestamp
├── label
├── confidence
├── scores
├── source_window
└── metadata
```

---

## 17. Action Contract

```text
Action
├── type
├── config
├── min_confidence
├── delay_seconds
├── cooldown_seconds
├── timeout_seconds
├── retry_policy
└── result
```

Result:

```text
queued → executing → succeeded | failed | unsupported
```

---

## 18. Model Artifact Contract

Every deployable ML artifact must have a validated manifest.

Example:

```text
artifact/
├── model.pt
└── manifest.json
```

Example manifest:

```json
{
  "format": "thoth-model/v1",
  "name": "radar-occupancy-v2",
  "processor": "torchscript",
  "inputs": [
    {
      "sensor": "radar",
      "window_seconds": 2.0,
      "required_sample_rate": 10
    }
  ],
  "outputs": ["empty", "occupied"],
  "whispy_version": ">=0.9,<1.0",
  "artifact_sha256": "..."
}
```

Brain validates metadata when a model is uploaded. Thoth validates again before installation.

---

## 19. Deployment State Machine

```text
queued
  ↓
received
  ↓
validated
  ↓
installed
  ↓
acknowledged(runtime_model_id)
  ↓
active
  ↓
serving predictions
```

Failure at any stage produces:

```text
failed
├── stage
├── code
└── message
```

`runtime_model_id` must be generated/persisted consistently and survive daemon restart.

---

# Part IV — Data and Control Flows

## 20. Local Sensor Flow

```text
physical sensor
      │
      ▼
Whispy driver
      │
      ▼
SensorSample stream
      │
      ▼
rolling buffer / synchronized window
      │
      ▼
processor
      │
      ▼
prediction
      │
      ├── local dashboard
      ├── local action
      └── optional Brain telemetry
```

This path must operate without Internet access.

---

## 21. Remote Python Sensor Flow

```text
Windows laptop
      │
Python + Whispy
      │
      ▼
Brain authentication/authorization
      │
      ▼
remote stream session
      │
      ▼
Thoth on Raspberry Pi
      │
      ▼
Whispy local sensor
      │
      ▼
physical sensor
```

The Pi does not need an inbound public Internet port.

---

## 22. Capture Flow

```text
CLI / thothHUB / mobile / Whispy
              │
              ▼
             Brain        (for remote request)
              │
              ▼
             Thoth
              │
              ▼
           Whispy
              │
       synchronized capture
              │
              ▼
         local storage
              │
              ├── remain local
              └── upload/sync when requested
```

A capture is treated as a logical lifecycle object rather than unrelated files/chunks.

---

## 23. Model Deployment Flow

```text
model upload
    │
    ▼
Brain validates artifact + manifest
    │
    ▼
deployment queued
    │
    ▼
Thoth receives deployment
    │
    ▼
Whispy processor loader validates/loads
    │
    ▼
runtime_model_id created and persisted
    │
    ▼
Thoth acknowledges Brain
    │
    ▼
activation
    │
    ▼
predictions
```

---

## 24. Cloud Training Flow

Cloud training is a Brain-orchestrated service, not a Whispy server responsibility.

```text
Dataset
  │
  ▼
Brain training job
  │
  ▼
worker
  │
  ▼
training/evaluation
  │
  ▼
model.pt + manifest.json
  │
  ▼
artifact validation
  │
  ▼
Brain model registry
  │
  ▼
normal deployment flow
```

Training jobs must always end in a persistent terminal or recoverable state:

```text
queued
running
completed
failed
cancelled
```

---

## 25. Mobile Notification Flow

```text
Thoth prediction/device event
          │
          ▼
         Brain
          │
 notification policy
          │
          ▼
   push notification service
          │
          ▼
       Flutter app
```

---

# Part V — Security Architecture

## 26. Trust Boundaries

### 26.1 Same-machine Thoth control

Use local IPC or loopback-only authenticated control.

### 26.2 LAN control

LAN clients must authenticate using a local pairing/token mechanism.

A LAN address alone is not authorization.

### 26.3 Internet control

Internet clients communicate through Brain.

```text
Whispy remote client / hub / mobile
              │
              ▼
             Brain
              │
              ▼
             Thoth
```

No requirement exists to expose the Pi's local API directly over the public Internet.

### 26.4 Browser authentication

Browser sessions should use secure server-managed session mechanisms. Browser JavaScript should not rely on unnecessarily long-lived unrestricted bearer credentials.

### 26.5 Python automation

Whispy automation should use scoped API credentials or a browser/device authorization login flow.

### 26.6 Logging

Credentials, passwords, authorization headers, API keys, cookies, and sensitive request bodies must be redacted before logging.

### 26.7 CORS

Production credentialed CORS is restricted to controlled ThothCraft origins, particularly:

```text
https://hub.thothcraft.com
https://www.thothcraft.com      when genuinely necessary
```

Temporary preview deployments must be explicitly authorized rather than accepted by a broad wildcard.

---

# Part VI — Commerce and Product Services

## 27. Commerce Is Separate from the Sensing Runtime

Stripe/payment behavior remains in Brain and thothHUB.

Whispy and Thoth consume entitlements/capabilities, not Stripe implementation details.

```text
Stripe
  │
  ▼
Brain billing/order state
  │
  ▼
entitlements/authorization
  │
  ├── thothHUB
  ├── mobile
  └── API/Whispy decisions
```

Hardware purchase target:

```text
www.thothcraft.com/product
        │
        ▼
hub.thothcraft.com/buy?product=thoth&qty=1
        │
  authentication if needed
        │
        ▼
Brain checkout
        │
        ▼
Stripe
```

Purchase intent must survive authentication.

Brain requires a real `Order`/`OrderItem` lifecycle or an explicitly documented manual fulfillment process.

---

# Part VII — Migration from Current System

## 28. Migration Principle

Do not rewrite everything simultaneously.

Use this rule for each subsystem:

```text
existing implementation
        │
        ▼
target contract defined
        │
        ▼
new implementation added
        │
        ▼
parallel/fixture comparison
        │
        ▼
consumer switched
        │
        ▼
acceptance test
        │
        ▼
old duplicate removed
```

---

## 29. Phase 0 — Freeze Names, Domains and Contracts

Actions:

1. Adopt the five canonical public domains.
2. Adopt the component names Whispy, Thoth, Brain, thothHUB, and Mobile.
3. Mark Vercel/Railway-specific URLs as infrastructure only.
4. Introduce `/v1` API namespace.
5. Define schemas for Device, Sensor, SensorSample, Window, Prediction, Action, ModelArtifact, Deployment, and Capture.
6. Document which contracts are implemented, partial, or target-only.

Exit gate:

- schema tests exist;
- every repository references the same contract definitions/examples.

---

## 30. Phase 1 — Make Whispy the Authoritative Local SDK

Actions:

1. Normalize package/import name to `whispy`.
2. Keep sensor drivers and reusable processors in Whispy.
3. Implement real streaming sample contracts.
4. Implement bounded buffers/windows and explicit timestamps.
5. Implement RuleProcessor and TorchScriptProcessor against real windows.
6. Implement consistent actuator result semantics.
7. Add recorded-fixture tests for radar, CSI, camera, and IMU.

Exit gate:

- the Windows laptop and each Raspberry Pi can run local Whispy tests;
- actual available sensors expose real sample values;
- unavailable physical sensors have deterministic recorded fixtures.

---

## 31. Phase 2 — Move Product Daemon/CLI Responsibility into Thoth

Actions:

1. Move/reimplement useful Whispy CLI-daemon functionality inside `thoth`.
2. Preserve useful old Thoth device/application behavior.
3. Make Thoth import Whispy instead of maintaining duplicate sensor implementations.
4. Implement one persistent Thoth daemon/service.
5. Make `thoth` CLI commands communicate with that service.
6. Persist model/deployment state.
7. Add `thoth doctor` diagnostics.

Do not continue two permanent edge runtimes.

Exit gate:

```text
Pi A: thoth service running
Pi B: thoth service running
```

and from each device:

```bash
thoth status
thoth sensors
thoth capture start
thoth capture stop
```

operate through the persistent daemon.

---

## 32. Phase 3 — Replace Old Thoth Hardware Implementations Incrementally

For each hardware path:

```text
old Thoth driver
      ↓
Whispy driver
      ↓
compare same hardware/fixture
      ↓
Thoth switches to Whispy
      ↓
remove duplicate
```

Suggested order:

1. camera/system sensors;
2. IMU;
3. radar;
4. CSI;
5. environmental sensors;
6. additional hardware.

Exit gate:

- device capture/data shapes remain correct;
- timestamps and units are documented;
- old duplicate code is removed only after equivalence testing.

---

## 33. Phase 4 — Secure Thoth ↔ Brain Device Channel

Actions:

1. Create device-scoped authentication.
2. Make Thoth initiate the outbound connection.
3. Send presence/capabilities through Brain.
4. Implement reconnect/backoff.
5. Implement command acknowledgement.
6. Implement deployment acknowledgement using persistent runtime model IDs.
7. Remove reliance on direct public device exposure.

Exit gate:

- Pi A and Pi B appear independently in Brain;
- disconnect/reconnect is correct;
- commands intended for Pi A never execute on Pi B.

---

## 34. Phase 5 — Build Whispy Remote Client

Actions:

1. Implement `whispy.Client`.
2. Implement user/API-key authentication.
3. Implement remote device discovery.
4. Implement remote sensor inventory.
5. Implement authorized stream negotiation.
6. Implement remote capture APIs.
7. Keep local and remote device interfaces aligned.

Exit gate:

From the Windows laptop:

```python
import whispy

client = whispy.Client()
pi_a = client.device("thoth-pi-a")
pi_b = client.device("thoth-pi-b")
```

both nodes are independently accessible through Brain.

---

## 35. Phase 6 — Standardize Model Deployment

Actions:

1. Enforce `thoth-model/v1` manifest.
2. Make portal/model upload binary-safe.
3. Validate hashes.
4. Make Brain queue deployments.
5. Make Thoth receive and validate them.
6. Make Whispy load the requested processor type.
7. Persist runtime model ID and state.
8. Acknowledge Brain.
9. Activate explicitly.
10. Verify prediction.

Exit gate:

```text
upload
→ queued
→ received
→ validated
→ installed
→ acknowledged(runtime_model_id)
→ active
→ prediction observed
```

survives a Thoth restart.

---

## 36. Phase 7 — Standardize Rules and Actions

Actions:

1. Replace divergent rule schemas with one validated schema.
2. Implement compound AND/OR behavior explicitly.
3. Implement required features such as RMS/SNR only where actually supported.
4. Add confidence, delay, debounce, and cooldown semantics.
5. Make actions asynchronous/bounded.
6. Verify actual GPIO/buzzer callback success.
7. Preserve Home Assistant service parameters.
8. Provide provider-specific webhook payload adapters where advertised.

Exit gate:

- acceptance tests 1–5 pass at software-fixture level;
- unsupported configuration fails before deployment.

---

## 37. Phase 8 — Refactor Brain Around the v1 Contract

Actions:

1. retire unsafe/legacy duplicate routes;
2. enforce tenant ownership everywhere;
3. make delete/update operations real and persistent;
4. make captures logical lifecycle objects;
5. align retention with raw object deletion;
6. provide binary-safe upload routes;
7. implement remote stream authorization;
8. stabilize deployment, prediction, and capture responses.

Exit gate:

- two-user isolation tests pass;
- all current clients can be generated/validated against the same API schema.

---

## 38. Phase 9 — Refactor thothHUB

Actions:

1. deploy canonical `hub.thothcraft.com`;
2. repair clean dependency installation;
3. use Brain v1 contracts;
4. replace unsafe generic binary proxy behavior;
5. update device/capture/model pages to the standardized states;
6. preserve product/plan purchase intent;
7. enforce admin authorization server-side;
8. add Playwright end-to-end tests.

Exit gate:

- hub operations and Whispy remote operations produce the same Brain state transitions.

---

## 39. Phase 10 — Refactor Flutter Mobile App

Actions:

1. generate/use the Brain v1 data contract;
2. replace old live cursor parsing;
3. show online and offline devices;
4. implement safe polling/stream lifecycle;
5. implement capture start/stop;
6. implement current prediction view;
7. implement notification registration when push backend exists;
8. preserve product/plan deep-link intent;
9. deep-link advanced workflows to thothHUB rather than duplicating them.

Exit gate:

- mobile, hub, and Whispy display the same device online state and prediction payload for Pi A and Pi B.

---

## 40. Phase 11 — Repair Cloud Training and Data Lifecycle

Actions:

1. implement a real supported training worker;
2. make every job persist status transitions;
3. emit valid `thoth-model/v1` artifacts;
4. validate deployability before marking a job successful;
5. fix capture/blob retention as one lifecycle;
6. hash uploaded data/model objects;
7. test failure/restart recovery.

Exit gate:

- a small labeled dataset can train, produce an artifact, deploy to a Pi, and make a prediction.

---

## 41. Phase 12 — Repair Commerce Independently

Actions include:

- modern Stripe subscription/invoice parsing;
- idempotent webhook/event handling;
- correct subscription ownership/state reconciliation;
- functional `/buy` route;
- hardware quantity/order handling;
- explicit order/fulfillment workflow;
- consistent price/currency copy;
- return URL verification;
- tax/shipping configuration decisions.

Commerce is a release requirement for selling the product, but it is not allowed to distort the Whispy/Thoth runtime architecture.

---

# Part VIII — Reference Validation Environment

## 42. Available Test Equipment

The reference development environment is:

```text
Windows laptop     → dev-windows
Raspberry Pi A     → thoth-pi-a
Raspberry Pi B     → thoth-pi-b
```

All three devices are initially on the same LAN.

### Windows laptop responsibilities

- main developer machine;
- Whispy remote Python client;
- browser running thothHUB;
- automated test runner;
- optional local Whispy sensing tests if compatible sensors exist;
- API/stream comparison and logging.

### Raspberry Pi A responsibilities

- primary Thoth node;
- primary radar/CSI/camera or available sensor host;
- model deployment target;
- acceptance tests 1–3 where hardware permits.

### Raspberry Pi B responsibilities

- independent second Thoth node;
- IMU/environmental/alternate sensor host where available;
- second-node isolation and reconnect tests;
- acceptance tests 4–5 where hardware permits;
- mock/local actuator target when needed.

---

# Part IX — Six Architecture Acceptance Tests

## 43. Acceptance Test 1 — Radar Occupancy → Adaptive Lighting

### Purpose

Validate a complete real sensor → model → decision → actuator pipeline on one Thoth node.

### Preferred topology

```text
Pi A
 │
Radar
 │
Whispy
 │
window
 │
occupancy processor
 │
Thoth decision policy
 │
Home Assistant or safe test-light actuator

Windows laptop
 │
Whispy/hub monitoring
```

### Required software cases

1. occupied prediction above threshold;
2. empty prediction below minimum delay;
3. empty continuously for 180 seconds;
4. occupancy returns during the empty timer;
5. confidence just below threshold;
6. confidence at/above threshold;
7. actuator timeout/failure;
8. Thoth restart.

### Pass criteria

- real radar windows reach the processor;
- availability booleans are never used as radar samples;
- occupied turns/keeps light on according to policy;
- empty does not turn the light off before 180 seconds;
- returning occupancy cancels the pending off action;
- action result is observable;
- restart restores configured model/action state.

### Physical requirement

If no Home Assistant/light is available, use a deterministic mock actuator first. The software acceptance can pass, but physical actuation remains separately marked pending until real hardware is tested.

---

## 44. Acceptance Test 2 — CSI Fall Detection → Alert

### Topology

```text
Pi A
 │
CSI sensor or recorded CSI fixture
 │
Whispy
 │
fall model
 │
Thoth
 │
Brain / notification action

Windows laptop
 │
Whispy/hub observation
```

Pi B may host a local mock notification receiver if useful.

### Required cases

- normal activity;
- fall event;
- repeated fall during cooldown;
- low-confidence event;
- notification timeout;
- retry;
- Pi A disconnect/reconnect.

### Pass criteria

- real/recorded CSI values, not availability, reach the model;
- one valid alert is produced for a qualifying fall;
- cooldown prevents alert storms;
- alert includes device identity and timestamp;
- network action failure does not block sensor capture/inference;
- reconnect restores service.

---

## 45. Acceptance Test 3 — Vision Intrusion → Local + Remote Actions

### Topology

```text
Pi A camera
   │
 Whispy
   │
people/motion processor
   │
 Thoth
 ┌─┴──────────────┐
 │                │
local action   webhook action
 │                │
Pi B/mock      test endpoint
actuator          │
                  ▼
             Windows observer
```

### Required cases

- person + motion;
- person without motion;
- motion without person;
- neither;
- local actuator failure;
- webhook failure;
- two actions bound to one qualifying prediction.

### Pass criteria

- compound rule/model behavior is evaluated correctly;
- both actions are independently tracked;
- local action cannot return success without confirmed execution;
- webhook uses the configured provider/body contract;
- camera images remain local unless an explicit capture/upload policy says otherwise.

---

## 46. Acceptance Test 4 — IMU Vibration → Emergency Relay/Test Output

### Preferred topology

```text
Pi B
 │
IMU or recorded acceleration fixture
 │
Whispy
 │
RMS / peak feature processor
 │
Thoth
 │
GPIO relay or safe test output
```

### Required cases

- below threshold;
- exactly at threshold;
- above threshold;
- OR condition path;
- missing IMU;
- sensor reconnect;
- action unavailable.

### Measurements

Record:

```text
sensor sample timestamp
window-complete timestamp
prediction timestamp
action-dispatch timestamp
action-confirmed timestamp
```

### Pass criteria

- RMS/peak calculations match reference fixtures;
- no success is returned for nonexistent hardware;
- missing sensor state is explicit;
- measured latency is reported rather than inferred.

Any sub-10ms claim must be based on actual measured end-to-end hardware results.

---

## 47. Acceptance Test 5 — Multi-Sensor Fusion → HVAC/Test Controller

### Preferred topology

```text
Pi A                        Pi B
 │                           │
radar/presence        temp/humidity/CO2
 │                           │
 └───────────┬───────────────┘
             │
        synchronized logical input
             │
        fusion processor
             │
      HVAC/test action
```

### Required cases

- all modalities healthy;
- Pi B offline;
- stale temperature;
- delayed CO2;
- missing radar;
- reconnect;
- invalid synchronization window;
- safe actuator bounds.

### Pass criteria

- missing/stale modalities are explicit;
- the fusion processor never silently treats missing data as valid zeros unless the model manifest explicitly defines that behavior;
- both devices retain their own identity/ownership;
- reconnect updates availability without corrupting synchronized timestamps;
- Home Assistant/test controller receives the intended values.

---

## 48. Acceptance Test 6 — Remote Python Sensor Access

This is a core architectural requirement of Whispy.

### Topology

```text
Windows laptop
      │
 Python + Whispy
      │
      ▼
     Brain
    /     \
   ▼       ▼
 Pi A     Pi B
 Thoth    Thoth
  │        │
Whispy   Whispy
  │        │
sensors  sensors
```

### Example

```python
import whispy

client = whispy.Client()

for device in client.devices():
    print(device.name, device.online)

pi_a = client.device("thoth-pi-a")
radar = pi_a.sensor("radar")

async for sample in radar.stream():
    print(sample.timestamp, sample.payload)
```

### Required cases

1. Windows discovers both Pi devices.
2. Only authorized devices are visible.
3. Each Pi reports its own sensor inventory.
4. Windows receives real sample payloads.
5. Sample timestamps/sequences are monotonic according to the driver contract.
6. Disconnect Pi A network.
7. Pi A stream reports a disconnect/error state.
8. Pi B continues to stream unaffected.
9. Reconnect Pi A.
10. Pi A returns online and can stream again.
11. Restart the Thoth daemon on Pi A.
12. Device re-registers/reconnects without creating a second logical device.
13. Invalid API key is rejected.
14. API key without `sensor:stream` is rejected.
15. Unauthenticated LAN writes/control are rejected.
16. Compare local Pi A Whispy samples with remote Windows Whispy samples for the same capture interval.

### Local/remote equivalence checks

Compare:

```text
sensor type
sample count expectations
payload shape
units/schema
timestamps
sequence numbers
capture identifier
```

Transport delay may differ, but sensing semantics must not.

### Pass condition

This acceptance test passes only when the same Whispy abstraction can access a local sensor and the equivalent remote Thoth sensor through Brain without application-specific code.

---

# Part X — Cross-Device Reliability Tests

## 49. Two-Pi Isolation Matrix

Run all of the following:

```text
Pi A online, Pi B online
Pi A offline, Pi B online
Pi A online, Pi B offline
both offline
both reconnect
restart Pi A Thoth only
restart Pi B Thoth only
deploy model to Pi A only
verify Pi B unchanged
deploy different model to Pi B
capture both simultaneously
cancel capture on Pi A only
verify Pi B continues
attempt command with wrong device ownership
denied
```

This matrix is required because a single-device system can hide routing, ownership, and state-isolation bugs.

---

# Part XI — Release Gates

## 50. Architecture Gates

| Gate | Requirement |
|---|---|
| G1 — Whispy Local | Real/fixture sensor streams operate through Whispy on Pi A and Pi B |
| G2 — Thoth Local | CLI controls a persistent Thoth daemon on both Pis |
| G3 — Processor Runtime | Rule and TorchScript processors consume real Whispy windows |
| G4 — Brain Device Channel | Both Pis maintain authenticated independent connections |
| G5 — Remote Whispy | Windows can access both Pis through Brain |
| G6 — Deployment | Model upload → install → runtime ID → activation → prediction works and persists |
| G7 — Actions | Software-fixture versions of acceptance tests 1–5 pass |
| G8 — Hardware | Available physical sensors/actions pass corresponding hardware validation |
| G9 — thothHUB | Browser performs the same Brain state transitions correctly |
| G10 — Mobile | Flutter app reads the same device/prediction/capture contract correctly |
| G11 — Security | auth/logging/CORS/LAN/API-scope tests pass |
| G12 — Commerce | sandbox purchase/subscription/order matrix passes before unattended sales |
| G13 — Final | all six architecture acceptance tests pass for supported/available hardware |

---

## 51. Repository Migration Matrix

| Current area | Target action |
|---|---|
| Whispy sensor abstractions | Keep and make authoritative |
| Whispy datasets/windows | Keep and normalize |
| Whispy processors | Keep/repair under one contract |
| Whispy actuators | Keep/repair under one contract |
| Whispy CLI | Move product-facing CLI responsibility into Thoth |
| Whispy daemon | Move/rebuild persistent device runtime inside Thoth |
| Old Thoth sensor code | Replace incrementally with Whispy drivers after equivalence tests |
| Old Thoth device/capture logic | Keep useful orchestration; modernize around Whispy |
| Brain auth/device registry | Keep and harden |
| Brain legacy/conflicting endpoints | Remove/migrate to v1 |
| Brain deployment handling | Repair around shared deployment state machine |
| Brain training | Repair after model-artifact contract is stable |
| Brain storage | Make capture/file retention lifecycle consistent |
| ResearchPortal | Keep; product becomes thothHUB |
| ResearchPortal proxy | Make binary-safe or replace with direct/signed upload path |
| Flutter mobile app | Keep; update against Brain v1 |
| Website | Keep as marketing/product only |
| Current installer scripts | Move canonical delivery to get.thothcraft.com |
| Old public provider URLs | Remove from public contract |

---

# Part XII — Testing Strategy

## 52. Test Layers

### 52.1 Unit tests

Whispy:

- driver parsing;
- timestamping;
- windows;
- processors;
- actuator semantics;
- cloud client serialization.

Thoth:

- config;
- daemon state;
- capture orchestration;
- deployment state machine;
- reconnect logic;
- local auth/IPC.

Brain:

- ownership;
- auth/scopes;
- deployment lifecycle;
- stream authorization;
- retention;
- billing state.

### 52.2 Contract tests

The same fixtures must be validated by:

```text
Brain
Whispy remote client
Thoth client
thothHUB
Flutter mobile
```

### 52.3 Integration tests

Use synthetic/recorded sensor data to test full software paths deterministically.

### 52.4 LAN hardware tests

Use the Windows + Pi A + Pi B topology.

### 52.5 Internet-path tests

Even though the devices are on one LAN, remote Whispy acceptance must force the path through Brain rather than shortcutting directly to the LAN address.

### 52.6 Physical hardware-in-loop tests

Where actual radar/CSI/IMU/relay/HA hardware exists, repeat the relevant acceptance test using real hardware.

Software-fixture success is not evidence of physical accuracy or timing.

---

# Part XIII — Product Documentation Structure

## 53. `docs.thothcraft.com`

Recommended structure:

```text
docs.thothcraft.com/
│
├── thoth/
│   ├── install
│   ├── pair
│   ├── cli
│   ├── local-dashboard
│   ├── local-api
│   ├── security
│   ├── updates
│   └── troubleshooting
│
├── whispy/
│   ├── installation
│   ├── local-devices
│   ├── remote-devices
│   ├── sensors
│   ├── streams
│   ├── synchronization
│   ├── datasets
│   ├── processors
│   ├── models
│   └── actuators
│
├── api/
│   ├── authentication
│   ├── api-keys
│   ├── devices
│   ├── captures
│   ├── models
│   ├── deployments
│   ├── streaming
│   └── errors
│
├── hardware/
│   ├── raspberry-pi
│   ├── radar
│   ├── csi
│   ├── camera
│   ├── imu
│   └── environmental
│
└── examples/
    ├── occupancy
    ├── fall-detection
    ├── vision-security
    ├── vibration
    ├── hvac-fusion
    └── remote-python
```

Documentation must label features as:

```text
Implemented
Experimental
Planned
```

A feature should not be documented as implemented until its relevant acceptance gate passes.

---

# Part XIV — Supported Sensing Model

## 54. Sensor Modalities

Initial Whispy targets include:

### Radar

- mmWave/FMCW radar;
- raw frames and/or processed range/Doppler representations;
- timestamped frames;
- occupancy/motion/breathing research.

### CSI

- ESP32 or supported CSI capture sources;
- amplitude/phase/subcarrier representation;
- timestamped packets/windows;
- presence/activity/fall research.

### IMU

- accelerometer;
- gyroscope;
- magnetometer where available;
- explicit units/sample rate.

### Camera

- local OpenCV/UVC/Pi camera sources;
- frame timestamps;
- privacy/upload policy explicit.

### Environmental/System

- temperature;
- humidity;
- CO2 where supported;
- CPU/system telemetry;
- BLE or other plugins where useful.

The driver API must express actual capabilities rather than assuming all devices provide all modalities.

---

# Part XV — Model and SMA Execution

## 55. Edge Execution Loop

Target execution:

```text
sensor drivers
    ↓
bounded timestamped streams
    ↓
synchronized windows
    ↓
processor(s)
    ↓
prediction
    ↓
confidence / hysteresis / delay / cooldown
    ↓
action dispatcher
    ↓
confirmed action result
```

Requirements:

- sensor ingestion must not block on slow webhooks;
- actions execute asynchronously or in bounded workers;
- processor exceptions do not terminate the daemon;
- missing sensors produce explicit state;
- prediction/action state is observable locally and remotely;
- model configuration persists across restart.

---

## 56. Offline-First Behavior

Without Internet:

```text
Thoth + Whispy must still support:
- local sensor discovery;
- local capture;
- local storage;
- local inference;
- local rules/actions;
- local dashboard/API;
- local diagnostics.
```

When connectivity returns:

```text
- presence reconnects;
- permitted metadata sync resumes;
- pending uploads resume according to policy;
- remote clients see the device return online;
```

---

# Part XVI — Mobile and Portal Relationship

## 57. thothHUB vs Mobile

thothHUB is the full research/fleet interface.

Mobile is optimized for quick operational tasks.

| Capability | thothHUB | Mobile |
|---|---:|---:|
| Fleet status | Yes | Yes |
| Device details | Yes | Yes |
| Capture start/stop | Yes | Yes |
| Recent predictions | Yes | Yes |
| Push alerts | Browser optional | Primary mobile feature |
| Dataset labeling | Yes | Limited/deep link |
| Model upload | Yes | Deep link |
| Model deployment | Yes | Optional simplified control |
| Processing pipelines | Yes | No initially |
| Cloud training configuration | Yes | Status/deep link |
| Billing | Yes | Summary/deep link |
| Orders | Yes | Summary |
| Admin | Yes, authorized | No initially |

Both must use the same Brain contract.

---

# Part XVII — Definition of Done

## 58. Architecture Completion Criteria

The migration is complete only when all of the following are true:

1. `whispy` is installable and usable as an independent Python SDK.
2. Thoth is the only supported installed node application/runtime.
3. Thoth uses Whispy for supported sensor/runtime primitives rather than duplicate implementations.
4. Pi A and Pi B run Thoth as managed services.
5. Windows Whispy can access both Pis through Brain.
6. Local and remote Whispy interfaces are semantically aligned.
7. Brain has one documented versioned device/capture/model API.
8. thothHUB consumes that API.
9. Flutter mobile consumes that API.
10. Model deployments have persistent runtime IDs and explicit state transitions.
11. Rule/TorchScript processors consume real sensor windows.
12. Actions use one validated schema and return real execution outcomes.
13. Local LAN control requires authentication.
14. Public Internet device access routes through Brain rather than direct Pi port exposure.
15. Binary uploads are hash-preserving.
16. Current data retention applies coherently to capture storage.
17. All six acceptance tests pass at software-contract level.
18. Available physical hardware scenarios pass hardware-in-loop testing.
19. Canonical domains are live and used consistently.
20. Commerce has its own completed sandbox release gate before unattended purchasing is enabled.

---

## 59. Final Architecture Summary

The target system can be summarized in four statements:

> **Whispy is the programmable sensing layer.** It exposes local and remote sensors, datasets, processors, models, and actions through a reusable Python SDK.

> **Thoth is the managed node application.** It turns a Windows/Linux/Pi/Jetson-class machine into a persistent, secure, remotely manageable sensing and inference node built on Whispy.

> **Brain is the cloud control plane.** It provides identity, ownership, authorization, remote routing, storage, deployments, training orchestration, billing, orders, and notifications.

> **thothHUB and the Flutter mobile app are clients of Brain.** thothHUB provides the complete browser workflow; mobile provides fast fleet, capture, prediction, alert, and account access without redefining backend contracts.

The reference validation topology is:

```text
                         api.thothcraft.com
                              BRAIN
                           /         \
                          /           \
                     Pi A             Pi B
                     Thoth            Thoth
                       │                │
                     Whispy           Whispy
                       │                │
                    sensors           sensors
                          \           /
                           \         /
                           Windows laptop
                        Python + Whispy
                        thothHUB browser
                        integration tests
```

The six acceptance tests — radar occupancy, CSI fall detection, vision intrusion, IMU vibration, multi-sensor HVAC fusion, and remote Python sensor access — are the architecture's end-to-end proof. Repository cleanup or successful builds alone are not sufficient evidence that the architecture is complete.

---

## Appendix A — Recommended User Experiences

### Install Thoth

```bash
curl -fsSL https://get.thothcraft.com/install.sh | sh
```

### Windows install

```powershell
iwr https://get.thothcraft.com/install.ps1 | iex
```

### Install Whispy for Python development

```bash
pip install whispy
```

### Local Python

```python
import whispy

node = whispy.local()
print(node.sensors())
```

### Remote Python

```python
import whispy

client = whispy.Client()
node = client.device("thoth-pi-a")
print(node.sensors())
```

### CLI

```bash
thoth status
thoth sensors
thoth capture start
thoth capture stop
thoth doctor
```

---

## Appendix B — Migration Rule of Thumb

When deciding whether existing code belongs in Whispy or Thoth:

```text
Could a Python researcher reasonably use this without installing the Thoth product?

YES → likely Whispy.
NO, it manages the installed node/product lifecycle → likely Thoth.
```

When deciding whether behavior belongs in Brain or a client:

```text
Does it define ownership, authorization, shared state, routing, persistence,
subscription/order state, or cloud orchestration?

YES → Brain.

Is it presentation or client interaction using those contracts?

YES → thothHUB / mobile / Whispy client.
```

