# ThothCraft: Edge Integrated Intelligence & Sensor-Model-Actuator Platform

## Architecture Documentation

**Version:** 2.1  
**Last Updated:** September 2026  
**Status:** Production

---

## Table of Contents

1. [Executive Summary](#executive-summary)
   - [Quick Start](#quick-start)
2. [System Overview](#system-overview)
3. [Architecture Diagram](#architecture-diagram)
4. [Sensor-Model-Actuator (SMA) Architecture](#sensor-model-actuator-sma-architecture)
   - [Sensors (Input Layer)](#sensors-input-layer)
   - [Models & Edge Inference (Intelligence Layer)](#models--edge-inference-intelligence-layer)
   - [Actuators (Action & Integration Layer)](#actuators-action--integration-layer)
   - [Execution Loop & Trigger Semantics](#execution-loop--trigger-semantics)
5. [Multi-Platform Node Architecture](#multi-platform-node-architecture)
   - [Universal Node Daemon (`thothcraft daemon`)](#universal-node-daemon-thothcraft-daemon)
   - [Host Naming Convention (`thoth-<name>.local`)](#host-naming-convention-thoth-namelocal)
   - [Dual Dashboard Model (Local Edge vs Fleet Portal)](#dual-dashboard-model-local-edge-vs-fleet-portal)
   - [Supported Operating Systems & Shells](#supported-operating-systems--shells)
6. [Real-World SMA Use Cases](#real-world-sma-use-cases)
   - [Use Case 1: Smart Workplace Occupancy & Adaptive Lighting](#use-case-1-smart-workplace-occupancy--adaptive-lighting)
   - [Use Case 2: Privacy-Preserving Fall Detection & Emergency Dispatch](#use-case-2-privacy-preserving-fall-detection--emergency-dispatch)
   - [Use Case 3: Edge Vision Security & Intrusion Warning](#use-case-3-edge-vision-security--intrusion-warning)
   - [Use Case 4: Industrial Vibration Anomaly Detection & Emergency Stop](#use-case-4-industrial-vibration-anomaly-detection--emergency-stop)
   - [Use Case 5: Ambient HVAC Zone Control via Thermal/Presence Fusion](#use-case-5-ambient-hvac-zone-control-via-thermalpresence-fusion)
7. [Component Deep Dives](#component-deep-dives)
   - [Thoth Edge Nodes & Devices](#thoth-edge-nodes--devices)
   - [Brain Backend](#brain-backend)
   - [Research Portal](#research-portal)
   - [Mobile App (Planned)](#mobile-app-planned)
8. [Data Flow](#data-flow)
9. [ML Pipeline](#ml-pipeline)
10. [Security Model](#security-model)
11. [Subscription Model](#subscription-model)
12. [API Reference](#api-reference)
13. [Deployment Architecture](#deployment-architecture)
14. [Future Roadmap](#future-roadmap)

---

## Executive Summary

**ThothCraft** is an end-to-end edge intelligence and **Sensor-Model-Actuator (SMA)** integration platform designed for researchers, enterprise engineers, and developers. ThothCraft unifies the complete edge automation lifecycle:

- **Sensors (Input Layer)**: Discovers and captures multi-modal telemetry across commodity nodes (Windows laptops, macOS workstations, Linux servers) and dedicated hardware (Raspberry Pi, Jetson, ESP32, mmWave radar, IMU, cameras).
- **Models (Intelligence Layer)**: Executes real-time edge intelligence using deterministic rule processors, CV heuristics, and compiled PyTorch TorchScript deep learning models directly on CPU and edge accelerators.
- **Actuators (Action Layer)**: Closes the control loop without cloud latency by driving downstream actions via Home Assistant entities, local device hardware (GPIO, relays, sound), and secure HTTP webhooks.
- **Cloud & Fleet Management**: Synchronizes telemetry and captures with **Brain** and **ResearchPortal**, enabling centralized dataset curation, cloud model training, and one-click fleet deployment back to edge devices.

### Key Value Proposition

| Traditional Approach | ThothCraft Approach |
|---------------------|---------------------|
| Manual, fragmented data collection scripts | Plug-and-play sensor auto-discovery across commodity & edge hardware |
| Disconnected ML models running in notebooks | Closed-loop Sensor-Model-Actuator pipeline on device |
| Isolated home automation / industrial silos | Native actuator plugins (Home Assistant, GPIO/Device, Webhook) |
| Complex, platform-specific edge deployments | Single-command cross-platform installer (`install.ps1`, `install.sh`) |
| High cloud inference latency & privacy leakage | Real-time on-device inference with offline-first local dashboard |
| Siloed research data | Collaborative dataset sharing, labeling, and cloud training |

---

### Quick Start

From zero to a live, cloud-connected sensing node in under five minutes — on any Windows, macOS, or Linux machine, no dedicated hardware required.

#### 1. Install

```powershell
# Windows (PowerShell)
iwr https://thothcraft.com/install.ps1 | iex
```

```bash
# macOS / Linux / Raspberry Pi
curl -fsSL https://thothcraft.com/install.sh | bash
```

Or from source / PyPI-style checkout:

```bash
pip install thothcraft-cli          # CLI + device daemon + local dashboard
pip install "thothcraft-cli[sensors]"  # + OpenCV, pyserial, psutil hardware drivers
```

#### 2. Authenticate & pair the node

```bash
thothcraft login                    # account credentials → JWT stored in ~/.thothcraft
thothcraft pair                     # prints a claim code → approve in the portal
thothcraft daemon                   # starts the node: local API + dashboard on :5000
```

The node immediately advertises itself on the LAN via mDNS:

```bash
thothcraft device info
# Name:      thoth-denver
# Hostname:  thoth-denver.local
# Dashboard: http://thoth-denver.local:5000
```

Open `http://thoth-denver.local:5000` from any machine on the network — no IP hunting, no port forwarding.

#### 3. Verify the fleet

```bash
thothcraft whoami                   # account, backend URL, plan + entitlements
thothcraft devices                  # every paired node, online state, dashboard URL
thothcraft doctor                   # credentials, Brain reachability, mDNS, sensor probe
```

#### 4. Control sensors — CLI, SDK, or dashboard

Built-in drivers (camera, mic, system telemetry, BLE) and third-party hardware (ESP32 CSI receivers, Infineon mmWave radar, Sense HAT IMU, any USB-UART/UVC device) are auto-probed and exposed through one interface:

```bash
thothcraft sensors                  # live inventory of detected hardware
thothcraft sensors stream camera    # tail a sensor stream in the terminal
```

```python
from thothcraft.local import LocalDevice

node = LocalDevice("thoth-denver.local")      # or "192.168.1.50"
print(node.sensors())                          # capability map
frame = node.camera_frame()                    # grab a JPEG frame
occ = node.occupancy()                         # latest occupancy inference
```

Custom hardware plugs in through the `SensorDriver` contract (`metadata / discover / open / stream / close`) — drop the driver in and it appears in the CLI, SDK, and both dashboards automatically.

#### 5. Run models — rules and ML, local or cloud-pushed

**Rule-based models** ship built in: radar SNR / energy-map thresholds for occupancy and localization, CSI variance triggers, OpenCV Haar face/person detection — deterministic, zero-GPU, sub-millisecond:

```python
node.deploy_rule_model({
    "name": "radar-occupancy",
    "sensor": "radar",
    "rule": "snr_db > 12 and energy > 0.4",   # energy-map occupancy rule
    "target_label": "occupied",
    "actuator": "webhook",
    "actuator_config": {"url": "https://hooks.slack.com/..."},
})
```

**ML models** (TorchScript `.pt`, CNN-LSTM, Haar-based CV) deploy two ways:

- **SDK / CLI direct** → `node.deploy("model.pt")` or `thothcraft models install model.pt`
- **Remote portal push** → Research Portal `/models` → *Deploy to device*; Brain queues it and the node's heartbeat pulls + activates it, confirmed end-to-end.

Every prediction lands on the local dashboard instantly and syncs to the portal for labeling, dataset curation, and retraining.

---

## System Overview

ThothCraft operates in a hybrid, edge-first architecture connecting local physical nodes to the central Brain backend and web-based ResearchPortal:

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                                   ThothCraft Platform                                   │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                         │
│  ┌───────────────────────┐       ┌───────────────────────┐     ┌─────────────────────┐  │
│  │   THOTH EDGE NODE     │       │     BRAIN BACKEND     │     │   RESEARCH PORTAL   │  │
│  │  (thoth-<name>.local) │◄─────►│ (FastAPI + PostgreSQL)│◄───►│  (Next.js 15 UI)    │  │
│  ├───────────────────────┤       ├───────────────────────┤     ├─────────────────────┤  │
│  │ • Sensor Layer:       │       │ • Device Registry     │     │ • Fleet Management  │  │
│  │   Camera, Wi-Fi CSI,  │       │ • Model Hub & Storage │     │ • Models & Rules UI │  │
│  │   Radar, IMU, CPU/RAM │       │ • PyTorch ML Training │     │ • Capture Lab & Plot│  │
│  │ • Model Layer:        │       │ • Heartbeat & Sync    │     │ • Minute Labeling   │  │
│  │   RuleProcessor,      │       │ • AI Research Agent   │     │ • Actuator Config   │  │
│  │   TorchScript ML/DL   │       └───────────────────────┘     └─────────────────────┘  │
│  │ • Actuator Layer:     │                   │                            │             │
│  │   Home Assistant,     │                   ▼                            ▼             │
│  │   Device GPIO, Webhook│       ┌───────────────────────┐     ┌─────────────────────┐  │
│  │ • Local Dashboard :5000       │ Cloud Storage (MinIO) │     │ Browser / Client    │  │
│  │ • OpenSSH Server :22  │       │ Subscription Storage  │     │ Direct & Remote LAN │  │
│  └───────────────────────┘       └───────────────────────┘     └─────────────────────┘  │
│                                                                                         │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

### Component Summary

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Thoth Node (`thothcraft`)** | Python 3.10+, Click, OpenCV, PyTorch | Multi-platform edge runtime, local API, SMA pipeline, dashboard |
| **Thoth Device (Dedicated)** | Raspberry Pi OS, Flask, Hardware HATs | Dedicated hardware sensor appliance (Radar, IMU, WiFi CSI) |
| **Brain** | FastAPI, PostgreSQL, PyTorch, OpenAI | Fleet registry, dataset storage, cloud model training, AI agent |
| **Research Portal** | Next.js 15, React 19, TailwindCSS | Cloud web UI for fleet telemetry, model deployment, labeling |
| **Mobile App** | React Native (Planned) | Real-time push alerts, actuator override, and telemetry |

---

## Architecture Diagram

### High-Level System Architecture

```mermaid
flowchart TB
    subgraph SMA[Sensor-Model-Actuator Closed Loop on Edge Node]
        direction LR
        SENS[Sensors: Radar, IMU, Vision, CSI, BLE] --> MODEL[Models: Rules, CV, TorchScript]
        MODEL --> ACT[Actuators: Home Assistant, GPIO/Relay, Webhooks]
    end

    subgraph Nodes[Edge Fleet (thoth-*.local)]
        T1["Commodity PC (thoth-denver.local)"]
        T2["Raspberry Pi (thoth-alex.local)"]
        T3["Jetson / Workstation (thoth-chen.local)"]
    end

    subgraph Cloud[Cloud & Fleet Management Layer]
        B[Brain Backend API]
        DB[(PostgreSQL)]
        S3[(Cloud Storage)]
        TR[PyTorch Training Worker]
    end

    subgraph Client[User Interfaces]
        LD["Local Dashboard (http://thoth-<name>.local:5000)"]
        RP["Research Portal (https://thothcraft.org)"]
        HA[Home Assistant Instance]
    end

    Nodes --- SMA
    T1 -->|Heartbeat & Telemetry| B
    T2 -->|Heartbeat & Telemetry| B
    T3 -->|Heartbeat & Telemetry| B

    B --> DB
    B --> S3
    B --> TR
    RP -->|Fleet REST API| B
    LD -->|Direct Local API| Nodes
    ACT -.->|Smart Home REST| HA
```

### Device Registration & Data Flow

```mermaid
sequenceDiagram
    participant U as User
    participant T as Thoth
    participant B as Brain
    participant RP as Portal
    
    Note over T: Power On
    T->>T: Start Hotspot
    U->>T: Connect
    T->>U: Captive Portal
    U->>T: WiFi Creds
    T->>T: Connect WiFi
    U->>T: Login
    T->>B: Register Device
    B->>B: Create Record
    B-->>T: Success
    
    loop Every 60s
        T->>B: Heartbeat
        B-->>T: Pending
    end
    
    U->>RP: View Devices
    RP->>B: Get List
    B-->>RP: Device List
```

---

## Sensor-Model-Actuator (SMA) Architecture

ThothCraft is engineered as a unified, closed-loop **Sensor-Model-Actuator (SMA)** integration platform. Rather than treating sensor collection, model inference, and device actions as separate pipelines, ThothCraft treats them as a single reactive edge continuum.

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        Closed-Loop Edge Execution Pipeline                             │
├────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                        │
│   [ SENSORS ]                   [ MODELS ]                       [ ACTUATORS ]         │
│   Raw Input Streams             Intelligence & Inference         Action & Control      │
│  ┌─────────────────┐           ┌──────────────────────┐         ┌────────────────────┐ │
│  │ Radar (mmWave)  │           │ RuleProcessor        │         │ Home Assistant     │ │
│  │ Wi-Fi CSI       │──────────►│ • SNR / RMS Logic    │────────►│ • Lights, Switches │ │
│  │ 6-Axis IMU      │  Window   │ • Face Detection CV  │ Trigger │ • Climate, Relays  │ │
│  │ Camera (OpenCV) │  Buffer   ├──────────────────────┤ Criteria├────────────────────┤ │
│  │ Bluetooth LE    │           │ TorchScriptProcessor │         │ Device Hardware    │ │
│  │ System psutil   │           │ • 1D/2D CNN-LSTM     │         │ • GPIO, Buzzer, OS │ │
│  └─────────────────┘           │ • JIT Quantized DL   │         ├────────────────────┤ │
│                                └──────────────────────┘         │ Webhooks           │ │
│                                                                 │ • Slack, Twilio SMS│ │
│                                                                 │ • HTTP REST / HMAC │ │
│                                                                 └────────────────────┘ │
│                                                                                        │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

### Sensors (Input Layer)

The sensor layer provides a normalized streaming interface (`SensorDriver`, `SensorFrame`, `SensorMeta`) across both specialized edge appliances and commodity computing hardware.

#### Sensor Modalities & Hardware Support

1. **mmWave Radar (`radar`)**:
   - Hardware: Infineon BGT60TR13C 60 GHz radar via SPI / MMW-HAT.
   - Modality: Fast Chirp FMCW producing Raw ADC frames (chirps × samples × antennas).
   - Metrics: Range-Doppler maps, range-angle FFTs, target SNR, micro-Doppler signatures.
2. **Wi-Fi Channel State Information (`csi`)**:
   - Hardware: ESP32 or commodity Wi-Fi NICs in monitor mode.
   - Modality: Subcarrier amplitude and phase matrix across OFDM subcarriers (64/128).
   - Metrics: Variance, phase difference, subcarrier correlation for wall-penetrating human presence and respiration.
3. **Inertial Measurement Unit (`imu`)**:
   - Hardware: Sense HAT, MPU-6050, LSM9DS1 (I2C/SPI).
   - Modality: 3-axis accelerometer, 3-axis gyroscope, 3-axis magnetometer sampled at 50–200 Hz.
   - Metrics: Acceleration magnitude, RMS jerk, angular velocity, orientation quaternions.
4. **Computer Vision & Video (`camera`)**:
   - Hardware: Built-in laptop webcams, Raspberry Pi Camera Module 3, USB UVC cameras.
   - Modality: Real-time RGB / grayscale frame grabber via OpenCV.
   - Metrics: Face bounding boxes, human presence counters, optical flow motion vectors.
5. **System Telemetry & Environmental (`env`, `system`)**:
   - Hardware: Built-in system probes (`psutil`), Bluetooth Low Energy (BLE) RSSI scans, battery managers (PiSugar).
   - Metrics: CPU load, RAM usage, active Wi-Fi BSSID, BLE peripheral RSSI proximity.

#### Driver Conformance Interface

Any sensor implements the pluggable `SensorDriver` contract:

```python
class SensorDriver(ABC):
    @abstractmethod
    def metadata(self) -> SensorMeta:
        """Returns name, version, modalities, sample rate."""
        ...
    @abstractmethod
    def discover(self) -> list[dict]:
        """Probes hardware and returns available physical devices."""
        ...
    @abstractmethod
    def open(self, config: dict | None = None) -> None:
        """Initializes and acquires hardware handles."""
        ...
    @abstractmethod
    def stream(self) -> Iterator[SensorFrame]:
        """Yields timestamped sensor frames."""
        ...
    @abstractmethod
    def close(self) -> None:
        """Safely releases hardware resources."""
        ...
```

---

### Models & Edge Inference (Intelligence Layer)

ThothCraft features a dual-engine edge intelligence architecture allowing both deterministic low-latency rules and complex deep learning inference to execute simultaneously:

#### 1. Rule-Based Models (`RuleProcessor`)
- **Deterministic Evaluation**: Evaluates mathematical and logical expressions against rolling sensor windows without GPU or cloud requirements.
- **Statistical Features**: Computes mean, variance, peak-to-peak amplitude, RMS, and SNR over dynamic window lengths.
- **Computer Vision Heuristics**: Embedded OpenCV Face Detection engine with automatic Haar Cascade loader and adaptive fallback heuristic for commodity laptops.
- **Actuator Binding**: Each rule model specifies target actuator criteria (`actuator`, `actuator_config`, `target_label`, `confidence_threshold`).

#### 2. Machine Learning & TorchScript Models (`TorchScriptProcessor`)
- **PyTorch JIT Compilation**: High-performance `.pt` models compiled via `torch.jit.trace` or `torch.jit.script` optimized for CPU and edge TPU execution.
- **Architectures**: Multi-layer CNN-LSTM temporal networks, 2D Radar CNNs, Spatial-Temporal Graph Neural Networks (ST-GNNs).
- **Quantization Support**: 8-bit dynamic quantization (INT8) delivering sub-20ms inference latency on Raspberry Pi 4 and edge nodes.

#### Dual Deployment Pipeline

Models can be installed onto edge nodes through two synchronized pathways:

```mermaid
flowchart TD
    subgraph LocalPath[Pathway A: Local Direct Edge Deployment]
        SDK[Python SDK: local.py] -->|deploy_rule_model| LOCAL_API[Node Local API :5000]
        CLI[thothcraft CLI] -->|models install / rule| LOCAL_API
        LOCAL_API --> REG[Active Model Registry]
    end

    subgraph FleetPath[Pathway B: Central Cloud Fleet Deployment]
        RP[Research Portal: /models] -->|Deploy to Device| BRAIN[Brain Backend API]
        BRAIN -->|Queue Deployment| DB[(Database)]
        NODE[Node Daemon: Heartbeat Poll] -->|GET /heartbeat| BRAIN
        BRAIN -->|Pending Deployment Payload| NODE
        NODE -->|Register & Activate| REG
        NODE -->|Confirm Status: delivered| BRAIN
    end

    REG --> WORKER[Continuous Inference Worker Loop]
```

---

### Actuators (Action & Integration Layer)

When model inference produces a prediction matching specified criteria, ThothCraft immediately dispatches action commands through its modular actuator subsystem (`thothcraft.actuators.base`).

#### Actuator Plugins

1. **`HomeAssistantActuator`**:
   - Integrates with local or remote Home Assistant instances via REST API.
   - Controls entities: `light.turn_on`, `light.turn_off`, `switch.toggle`, `climate.set_temperature`, `cover.open_cover`.
   - Payload:
     ```json
     {
       "type": "home_assistant",
       "base_url": "http://homeassistant.local:8123",
       "token": "eyJhbGciOi...",
       "entity_id": "light.office_desk",
       "action": "turn_on",
       "data": { "brightness": 255, "color_temp": 300 }
     }
     ```

2. **`DeviceActuator`**:
   - Executes local hardware operations directly on the host machine.
   - Actions: GPIO output pin toggling (solenoid, LED, relay), buzzer PWM, audible alerts, and controlled local shell scripts.
   - Payload:
     ```json
     {
       "type": "device",
       "action": "gpio_toggle",
       "pin": 18,
       "state": "HIGH",
       "duration_sec": 5.0
     }
     ```

3. **`WebhookActuator`**:
   - Emits structured HTTP POST requests to enterprise, cloud, and messaging endpoints.
   - Integrates with Slack incoming webhooks, Discord alerts, Twilio SMS gateways, and custom cloud microservices with optional HMAC SHA-256 signature verification.
   - Payload:
     ```json
     {
       "type": "webhook",
       "url": "https://hooks.slack.com/services/T00/B00/XXXX",
       "method": "POST",
       "headers": { "X-Custom-Auth": "secret-token" },
       "body": { "text": "Intrusion detected at thoth-denver.local" }
     }
     ```

---

### Execution Loop & Trigger Semantics

The edge daemon executes a dedicated background inference thread:

1. **Frame Ingestion**: Continuously captures frames from discovered sensors into rolling ring buffers.
2. **Window Feature Extraction**: Computes multi-channel statistical or raw array features.
3. **Inference Execution**: Evaluates active rule models or TorchScript tensors.
4. **Hysteresis & Cooldown**: Evaluates prediction confidence against threshold ($C \ge C_{min}$) and verifies cooldown periods to avoid actuator flapping.
5. **Actuator Dispatch**: Spawns an asynchronous executor thread to invoke the target actuator without blocking sensor ingestion.
6. **Telemetry & Log Stream**: Appends the prediction, confidence, and timestamp to `_RECENT_PREDICTIONS` served to the local dashboard and cloud sync.

---

## Multi-Platform Node Architecture

ThothCraft is not restricted to custom hardware. Any commodity or edge computer (Windows laptop, macOS workstation, Linux server, or Raspberry Pi) can serve as an edge node.

### Universal Node Daemon (`thothcraft daemon`)

The daemon is packaged cleanly inside `thothcraft-cli` and managed via standard OS init systems:

| Platform | Background Management | Service Definition |
|----------|----------------------|--------------------|
| **Windows 10 / 11** | Windows Task Scheduler / Startup Folder | Scheduled Task `Thothcraft` running `thothcraft.exe daemon` |
| **macOS (Sonoma / Sequoia)** | launchd | LaunchAgent `~/Library/LaunchAgents/com.thothcraft.daemon.plist` |
| **Linux / Raspberry Pi** | systemd (User or System) | User service `~/.config/systemd/user/thothcraft.service` |

### Host Naming Convention (`thoth-<name>.local`)

To eliminate IP address ambiguity on local networks, all Thoth nodes adhere to the canonical mDNS hostname standard:

$$\text{Hostname} = \text{thoth-}\langle\text{name}\rangle\text{.local}$$

Where $\langle\text{name}\rangle$ is deterministically mapped from the machine's UUID to a curated list of friendly names (`alex`, `denver`, `chen`, `kyoto`, `oslo`, `tokyo`, etc.).
- Persisted locally in `~/.thothcraft/device.json`.
- Configurable via the `THOTH_HOSTNAME` environment variable.
- Advertised via mDNS/Zeroconf for zero-configuration LAN discovery — the daemon publishes an A record (`thoth-<name>.local` → LAN IP) plus a `_thoth._tcp.local.` service record carrying the device UUID and dashboard URL.
- Reported to Brain on every heartbeat (`device_hostname`), so the portal and `thothcraft devices` can deep-link straight to `http://thoth-<name>.local:5000`.
- Inspectable from the terminal with `thothcraft device info`.

### Dual Dashboard Model (Local Edge vs Fleet Portal)

ThothCraft provides two complementary dashboards designed for different operational needs:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                          Dual Dashboard Architecture                                   │
├────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                        │
│   LOCAL NODE DASHBOARD                           CLOUD RESEARCHPORTAL                  │
│   http://thoth-<name>.local:5000                 https://thothcraft.org                │
│  ┌────────────────────────────────────┐         ┌────────────────────────────────────┐ │
│  │ • Zero-cloud, 100% offline edge UI │         │ • Centralized fleet view           │ │
│  │ • Sensor Lab: live webcam preview  │         │ • Multi-node telemetry & battery   │ │
│  │ • Hardware sensor inventory & state│         │ • Minute capture Start/Stop sync   │ │
│  │ • Real-time prediction event stream│         │ • Multi-modal plot visualizers     │ │
│  │ • Local one-click model evaluation │         │ • Interactive dataset labeling     │ │
│  │ • Direct OpenSSH port 22 access    │         │ • Cloud PyTorch model training     │ │
│  │ • Styled in Thoth design system    │         │ • One-click fleet model deployment │ │
│  └────────────────────────────────────┘         └────────────────────────────────────┘ │
│                                                                                        │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

### Supported Operating Systems & Shells

The installer and CLI tools are fully verified across:
- **Windows**: Windows PowerShell 5.1, PowerShell 7+, Windows Terminal, Git Bash (`bash.exe`), Command Prompt (`cmd.exe`).
- **macOS**: Terminal (`zsh`, `bash`), iTerm2.
- **Linux**: `bash`, `zsh`, `dash` on Debian/Ubuntu, Raspberry Pi OS, Fedora, Arch.

---

## Real-World SMA Use Cases

### Use Case 1: Smart Workplace Occupancy & Adaptive Lighting

```
[ mmWave Radar ] ──► [ TorchScript CNN-LSTM ] ──► [ HomeAssistantActuator ]
BGT60TR13C 60GHz     radar-occupancy-v2            light.turn_on / light.turn_off
```

- **Problem**: Infrared PIR sensors fail when occupants sit still at desks, leading to false darkness and manual overrides.
- **Sensor**: Infineon BGT60TR13C 60 GHz radar capturing range-Doppler micro-motion (chest breathing movements down to 0.5 mm).
- **Model**: Pretrained TorchScript 2D CNN-LSTM model (`radar-occupancy-v2`) deployed from the ResearchPortal Model Hub. Computes occupancy probabilities over 2-second windows.
- **Actuator**: `HomeAssistantActuator` communicating with smart lighting circuits:
  - If `occupied` with confidence $\ge 85\%$: triggers `light.turn_on` with personalized color temperature.
  - If `empty` continuously for 3 minutes: triggers `light.turn_off`.
- **Value**: Completely unobtrusive, zero-camera privacy preservation, elimination of false negatives while saving 38% lighting energy.

---

### Use Case 2: Privacy-Preserving Fall Detection & Emergency Dispatch

```
[ Wi-Fi CSI ] ──► [ Temporal Transformer ] ──► [ WebhookActuator ]
ESP32 / 802.11n     TCN Fall Classifier           Twilio Emergency SMS + Care Dispatch
```

- **Problem**: Elderly individuals often refuse wearable panic pendants and dislike invasive optical cameras in bathrooms and bedrooms.
- **Sensor**: ESP32 Wi-Fi Channel State Information (CSI) receiver monitoring subcarrier distortion of ambient Wi-Fi multipath signals.
- **Model**: Temporal Convolutional Network (TCN) compiled to TorchScript running on an edge node in the living room. Evaluates rapid vertical velocity signatures characteristic of falls versus standard sitting or lying down.
- **Actuator**: `WebhookActuator` configured with Twilio SMS and nurse station dispatch webhooks:
  - When `fall_detected` triggers with confidence $\ge 90\%$: immediately dispatches an SMS with the node's hostname (`thoth-alex.local`), timestamp, and room coordinates to emergency contacts.
- **Value**: Operates through walls and in complete darkness with zero wearables and zero video privacy intrusion.

---

### Use Case 3: Edge Vision Security & Intrusion Warning

```
[ Webcam / UVC ] ──► [ OpenCV RuleProcessor ] ──► [ DeviceActuator & Webhook ]
Built-in / USB Cam    Face Detection CV Engine      Local Siren + Slack Security Alert
```

- **Problem**: Small commercial spaces or server rooms need low-cost perimeter security without transmitting continuous video streams to expensive cloud vision services.
- **Sensor**: Commodity laptop or USB webcam streaming at 5 FPS through ThothCraft's video frame grabber.
- **Model**: Edge `RuleProcessor` with OpenCV face detection heuristic and motion thresholding (`when: "people_count > 0 && motion_energy > 0.45"`).
- **Actuator**: Dual actuation:
  1. `DeviceActuator`: Triggers a local audio buzzer / alert tone on the edge machine.
  2. `WebhookActuator`: Sends an instant incident alert with timestamp and snapshot link to the security Slack channel.
- **Value**: 100% on-device image processing; no video footage leaves the local network, achieving instantaneous alerting and zero cloud bandwidth costs.

---

### Use Case 4: Industrial Vibration Anomaly Detection & Emergency Stop

```
[ 3-Axis IMU ] ──► [ Statistical SNR/RMS Rule ] ──► [ DeviceActuator (GPIO Relay) ]
Sense HAT / MPU     when: "rms_accel > 3.8g"         Emergency Machine Stop Relay
```

- **Problem**: Rotating mechanical machinery (pumps, CNC motors, compressors) suffers catastrophic bearing failure if high-vibration harmonics are not addressed within seconds.
- **Sensor**: 3-axis accelerometer sampled at 200 Hz attached to the bearing housing.
- **Model**: Edge `RuleProcessor` monitoring high-frequency harmonic energy and vibration RMS:
  $$\text{RMS} = \sqrt{\frac{1}{N}\sum_{i=1}^{N} (x_i^2 + y_i^2 + z_i^2)}$$
  Trigger condition: `when: "rms_accel > 3.8 || peak_snr > 24.0"`.
- **Actuator**: `DeviceActuator` toggling a hardware GPIO pin connected to an industrial 24V solid-state relay to trip the machine's emergency stop circuit.
- **Value**: Sub-10ms emergency trip time; operates independently of factory network outages.

---

### Use Case 5: Ambient HVAC Zone Control via Thermal/Presence Fusion

```
[ Radar + Temp + CO2 ] ──► [ Random Forest JIT ] ──► [ HomeAssistantActuator ]
Multi-Sensor Suite         Thermal Zone Classifier       climate.set_temperature
```

- **Problem**: Traditional thermostats measure temperature at a single wall location rather than where occupants are actually situated, leading to hot/cold spots and excessive HVAC cycling.
- **Sensor**: Multi-sensor fusion combining mmWave micro-Doppler presence, ambient temperature, humidity, and CO2 telemetry.
- **Model**: JIT-compiled classifier evaluating occupancy density and metabolic load across room zones.
- **Actuator**: `HomeAssistantActuator` communicating with smart motorized HVAC dampers and smart thermostats (`climate.set_temperature`, `fan.set_percentage`).
- **Value**: Dynamic room balancing based on actual human headcount and occupancy patterns.

---

## Component Deep Dives

### Thoth Edge Nodes & Devices

The Thoth device is a portable Raspberry Pi-based data collection and inference platform shipped pre-flashed in a snap case.

#### Product Variants

| Product | Sensors | Use Cases |
|---------|---------|-----------|
| **Thoth-IMU** | Sense HAT (accelerometer, gyroscope, magnetometer) | Activity recognition, gesture detection, motion analysis |
| **Thoth-Wireless** | ESP32 (WiFi CSI), mmWave radar (MMW-HAT) | Presence detection, breathing monitoring, through-wall sensing |
| **Thoth-Vision** | Camera module | Object detection, pose estimation, scene classification |

#### Hardware Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Thoth Device                              │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │
│  │ Raspberry Pi│  │  PiSugar    │  │  Sense HAT  │          │
│  │   4/Zero 2W │  │  Battery    │  │  (IMU)      │          │
│  │             │  │  Management │  │             │          │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘          │
│         │                │                │                  │
│         └────────────────┼────────────────┘                  │
│                          │                                   │
│  ┌─────────────┐  ┌──────┴──────┐  ┌─────────────┐          │
│  │   ESP32     │  │   GPIO      │  │  MMW-HAT    │          │
│  │  (WiFi CSI) │  │   Header    │  │  (Radar)    │          │
│  │             │  │             │  │             │          │
│  └─────────────┘  └─────────────┘  └─────────────┘          │
│                                                              │
│  ┌─────────────────────────────────────────────────┐        │
│  │              128GB MicroSD Card                  │        │
│  │  • Raspberry Pi OS Lite (64-bit)                │        │
│  │  • Thoth Software Pre-installed                 │        │
│  │  • Data Storage (up to 128GB)                   │        │
│  └─────────────────────────────────────────────────┘        │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

#### Software Stack

```
┌─────────────────────────────────────────┐
│           Application Layer              │
│  ┌─────────────────────────────────────┐│
│  │         Flask Web App               ││
│  │  • REST API (/api/*)                ││
│  │  • WebSocket (real-time streaming)  ││
│  │  • Captive Portal (/setup)          ││
│  │  • Status Dashboard (/status)       ││
│  └─────────────────────────────────────┘│
├─────────────────────────────────────────┤
│           Service Layer                  │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐│
│  │ Device   │ │ Auth     │ │ Sensor   ││
│  │ Manager  │ │ Manager  │ │ Collector││
│  └──────────┘ └──────────┘ └──────────┘│
├─────────────────────────────────────────┤
│           System Services                │
│  • thoth-web.service (Flask app)        │
│  • thoth-hotspot.service (WiFi AP)      │
│  • thoth-collector.service (sensors)    │
│  • thoth-firstboot.service (setup)      │
└─────────────────────────────────────────┘
```

#### Key Files & Directories

| Path | Purpose |
|------|---------|
| `/home/pi/thoth/src/backend/app.py` | Main Flask application |
| `/home/pi/thoth/src/backend/device_manager.py` | Brain server communication |
| `/home/pi/thoth/src/backend/auth_manager.py` | User authentication |
| `/home/pi/thoth/data/` | Collected sensor data |
| `/home/pi/thoth/data/config/` | Device configuration |
| `/home/pi/thoth/WS/collect_csi.py` | WiFi CSI collection script |
| `/home/pi/thoth/setup/` | Installation & image scripts |

#### Data File Naming Convention

```
{type}_{timestamp}.{ext}

Examples:
  imu_2025-10-12.json      # IMU accelerometer/gyroscope data
  csi_2025-10-10.csv       # WiFi Channel State Information
  mfcw_2025-10-15.bin      # mmWave radar data
  img_2025-10-20.jpg       # Camera image
  vid_2025-10-20.mp4       # Camera video
```

#### Device API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/setup` | GET | Captive portal setup page |
| `/status` | GET | Device status dashboard |
| `/health` | GET | System health check |
| `/api/wifi/scan` | GET | Scan available WiFi networks |
| `/api/wifi/connect` | POST | Connect to WiFi network |
| `/api/setup/login` | POST | Authenticate with Brain |
| `/control/start` | POST | Start data collection |
| `/control/stop` | POST | Stop data collection |
| `/data/current` | GET | Latest sensor reading |
| `/data/history` | GET | Historical sensor data |
| `/upload` | POST | Upload data to Brain |

---

### Brain Backend

The Brain is the central backend service handling authentication, device management, data storage, ML training, and AI assistance.

#### Technology Stack

| Layer | Technology |
|-------|------------|
| **Framework** | FastAPI (Python 3.11+) |
| **Database** | PostgreSQL with SQLAlchemy ORM |
| **ML Training** | PyTorch |
| **AI Agent** | OpenAI GPT-4 with function calling |
| **Authentication** | JWT tokens |
| **Task Queue** | APScheduler (background jobs) |

#### Module Structure

```
Brain/
├── server/
│   ├── main.py              # FastAPI app entry point
│   ├── db.py                # SQLAlchemy models
│   ├── auth.py              # JWT authentication
│   ├── config.py            # Environment configuration
│   ├── ml_training.py       # PyTorch training logic
│   ├── endpoints/
│   │   ├── auth_endpoints.py      # Login, register, logout
│   │   ├── device_endpoints.py    # Device registration & status
│   │   ├── file_endpoints.py      # File upload & management
│   │   ├── dataset_endpoints.py   # Dataset CRUD & labeling
│   │   ├── training_endpoints.py  # Training job management
│   │   ├── processing_endpoints.py # Data pipeline config
│   │   ├── sensor_endpoints.py    # Real-time sensor data
│   │   ├── ai_endpoints.py        # AI chatbot queries
│   │   └── webhook_endpoints.py   # Twilio SMS integration
│   └── utils/
│       └── logging_utils.py       # Request/response logging
├── aiagent/
│   ├── handler/
│   │   └── query.py         # OpenAI query processing
│   ├── memory/
│   │   └── memory_manager.py # Long/short-term memory
│   └── context/             # AI context management
└── requirements.txt
```

#### Database Schema

```mermaid
erDiagram
    User ||--o{ Device : owns
    User ||--o{ File : uploads
    User ||--o{ Query : makes
    User ||--o{ Session : has
    User ||--o{ TrainingDataset : creates
    User ||--o{ TrainingJob : runs
    User ||--o{ TrainedModel : owns
    
    Device ||--o{ DeviceFile : contains
    Device ||--o{ FileDeviceUpdate : tracks
    
    File ||--o{ DatasetFile : labeled_in
    File ||--o{ FileDeviceUpdate : synced_via
    
    TrainingDataset ||--o{ DatasetFile : contains
    TrainingDataset ||--o{ TrainingJob : trains
    
    User {
        int userId PK
        string username
        string email
        string password_hash
        string phone_number
        int max_file_size
        datetime created_at
    }
    
    Device {
        int deviceId PK
        int userId FK
        string device_uuid
        string device_name
        string device_type
        boolean online
        int battery_level
        datetime last_seen
        string ip_address
        string mac_address
    }
    
    File {
        int fileId PK
        int userId FK
        string filename
        binary content
        int size
        datetime created_at
    }
    
    DeviceFile {
        int id PK
        int device_id FK
        int user_id FK
        string filename
        int size
        string file_type
        boolean on_device
        boolean on_cloud
        int cloud_file_id FK
        boolean upload_requested
    }
    
    TrainingDataset {
        int id PK
        int user_id FK
        string name
        string description
        datetime created_at
    }
    
    DatasetFile {
        int id PK
        int dataset_id FK
        int file_id FK
        string label
    }
    
    TrainingJob {
        int id PK
        string job_id
        int user_id FK
        int dataset_id FK
        string model_type
        string training_mode
        string status
        int current_epoch
        int total_epochs
        text metrics
        string model_path
    }
    
    TrainedModel {
        int id PK
        int user_id FK
        string job_id
        string name
        string architecture
        float accuracy
        binary model_data
        boolean is_pinned
    }
    
    Query {
        int queryId PK
        int userId FK
        string chatId
        text query_text
        text response
        datetime created_at
    }
```

#### API Endpoint Groups

| Prefix | Purpose | Key Endpoints |
|--------|---------|---------------|
| `/auth` | Authentication | `/login`, `/register`, `/logout`, `/me` |
| `/device` | Device management | `/register`, `/list`, `/status`, `/{id}` |
| `/file` | File operations | `/upload`, `/list`, `/download/{id}` |
| `/datasets` | Dataset management | `/create`, `/list`, `/{id}/files`, `/{id}/train` |
| `/training` | Training jobs | `/jobs`, `/jobs/{id}`, `/jobs/{id}/stop` |
| `/processing` | Data pipelines | `/pipelines`, `/pipelines/{id}` |
| `/query` | AI chatbot | `/query` |
| `/activity` | Dashboard stats | `/stats`, `/recent` |
| `/phone` | Twilio webhooks | `/incoming-message`, `/incoming-call` |

---

### Research Portal

The Research Portal is a modern web application providing the primary user interface for the ThothCraft platform.

#### Technology Stack

| Layer | Technology |
|-------|------------|
| **Framework** | Next.js 14 (App Router) |
| **UI Library** | React 18 |
| **Styling** | TailwindCSS |
| **Icons** | Lucide React |
| **State** | React Context + useState |
| **HTTP Client** | Custom `useApi` hook (fetch-based) |

#### Page Structure

```
ResearchPortal/
├── app/
│   ├── page.tsx                    # Landing page (public)
│   ├── layout.tsx                  # Root layout
│   ├── globals.css                 # Global styles
│   ├── auth/
│   │   ├── login/page.tsx          # Login page
│   │   └── register/page.tsx       # Registration page
│   ├── (protected)/
│   │   ├── layout.tsx              # Authenticated layout + Sidebar
│   │   ├── home/page.tsx           # Dashboard with stats
│   │   ├── devices/page.tsx        # Device management & collection controls
│   │   ├── models/page.tsx         # Sensor-Model-Actuator hub & deployment
│   │   ├── data/page.tsx           # File browser
│   │   ├── processing/page.tsx     # Pipeline builder
│   │   ├── training/page.tsx       # ML training UI
│   │   ├── chatbot/page.tsx        # AI assistant
│   │   └── settings/page.tsx       # User preferences
│   └── api/                        # API routes (if needed)
├── components/
│   ├── Sidebar.tsx                 # Navigation sidebar
│   └── ChatBubble.tsx              # Chat message component
├── contexts/
│   └── AuthContext.tsx             # Authentication state
├── hooks/
│   └── useApi.ts                   # API client hook
└── lib/
    └── utils.ts                    # Utility functions
```

#### Navigation Structure

| Route | Icon | Description |
|-------|------|-------------|
| `/home` | Home | Statistics & overview dashboard |
| `/devices` | Monitor | Online/offline device management & collection controls |
| `/models` | Layers | Sensor-Model-Actuator hub & edge deployment |
| `/data` | Database | File browser with cloud/device status |
| `/processing` | Workflow | Visual data pipeline builder |
| `/training` | Brain | ML model training & deployment |
| `/chatbot` | MessageCircle | AI assistant for queries |
| `/settings` | Settings | User preferences & sharing |

#### Key Features by Page

**Home Dashboard**
- Total devices (online/offline counts)
- Total data files
- Training jobs (active/completed)
- Trained models with best accuracy
- Recent activity feed (48h)

**Devices Page**
- Device list with status indicators
- Real-time collection active status (Start/Stop controls synced to Brain)
- Battery level display
- Last seen timestamps
- IP address and hostname for direct access
- Filter by online/offline

**Models Page (SMA Hub)**
- Dual-mode edge model creation:
  - **Rule-Based Models**: Real-time expressions (`snr_mean > threshold`), statistical windowing, and OpenCV face detection heuristics.
  - **TorchScript / Pretrained Models**: PyTorch `.pt` artifacts for complex computer vision and radar classification.
- Modular Actuator Plugin configuration:
  - **Home Assistant**: Entity action bindings (`light.turn_on`, `switch.toggle`, `climate.set_temperature`).
  - **Device Hardware**: GPIO pin toggles, relays, buzzers, local scripts.
  - **Webhooks**: Configurable HTTP REST notifications for Slack, Discord, Twilio SMS.
- Real-time deployment target selection and instant activation to edge fleet.
- Live deployment status tracking (pending, delivered, active).

**Data Page**
- File browser with type icons (IMU, CSI, MFCW, IMG, VID)
- Cloud vs. device status indicators
- Request upload from device
- File size and timestamps
- Search and filter

**Processing Page**
- Visual drag-and-drop pipeline builder
- Available blocks: Normalize, Filter, Window, FFT, Feature Extract, Augment, Downsample, Standardize
- Block configuration panels
- Pipeline save/load

**Training Page**
- Dataset creation and management
- File labeling interface
- Training configuration:
  - Model architecture (small/medium/large)
  - Training mode (cloud/on-device/federated)
  - Hyperparameters (epochs, batch size, learning rate)
  - Bayesian optimization toggle
- Job monitoring with progress
- Model list with accuracy metrics
- Deploy to device button

**Chatbot Page**
- Conversational AI interface
- System stats context injection
- Query history persistence
- Suggested prompts

**Settings Page**
- Profile management
- Notification preferences (email, push, device alerts, training complete)
- Privacy settings
- Data retention configuration
- *Data sharing permissions (planned)*

---

### Mobile App (Planned)

A React Native mobile application providing:

- **Real-time Notifications**: Alerts when devices go offline, training completes, or inference triggers
- **Remote Device Control**: Start/stop data collection
- **Quick Stats**: Dashboard overview
- **Push Notifications**: Via Firebase Cloud Messaging

---

## Data Flow

### Data Collection Flow

```mermaid
flowchart LR
    subgraph Device["Thoth Device"]
        S[Sensors] --> C[Collector Service]
        C --> LS[Local Storage<br/>128GB SD Card]
    end
    
    subgraph Cloud["Brain Cloud"]
        API[REST API]
        DB[(PostgreSQL)]
        CS[(Cloud Storage)]
    end
    
    subgraph Portal["Research Portal"]
        UI[Web UI]
    end
    
    LS -->|Heartbeat: File List| API
    API -->|Store Metadata| DB
    UI -->|Request Upload| API
    API -->|Set upload_requested| DB
    API -->|Pending Uploads| LS
    LS -->|Upload File| API
    API -->|Store Content| CS
    API -->|Update on_cloud=true| DB
```

### Training Flow

```mermaid
flowchart TB
    subgraph Dataset[Dataset Creation]
        F1[File 1] -->|walking| DS[Dataset]
        F2[File 2] -->|walking| DS
        F3[File 3] -->|running| DS
        F4[File 4] -->|sitting| DS
    end
    
    subgraph Preprocessing[Data Pipeline]
        DS --> N[Normalize]
        N --> W[Window]
        W --> FF[FFT]
        FF --> FE[Feature Extract]
    end
    
    subgraph Training[Model Training]
        FE --> DL[DataLoader]
        DL --> M[CNN-LSTM Model]
        M --> OPT{Bayesian Opt?}
        OPT -->|Yes| BO[Hyperparam Search]
        OPT -->|No| TR[Train]
        BO --> TR
        TR --> EVAL[Evaluate]
    end
    
    subgraph Output[Output]
        EVAL --> TM[Trained Model]
        TM --> DEPLOY[Deploy to Device]
    end
```

### Federated Learning Flow (Planned)

```mermaid
flowchart TB
    subgraph Server[Brain Server]
        GM[Global Model]
        AGG[Aggregator]
    end
    
    subgraph Devices[Edge Devices]
        D1[Thoth 1]
        D2[Thoth 2]
        D3[Thoth 3]
    end
    
    GM -->|Distribute| D1
    GM -->|Distribute| D2
    GM -->|Distribute| D3
    
    D1 -->|Local Train| D1
    D2 -->|Local Train| D2
    D3 -->|Local Train| D3
    
    D1 -->|Gradients| AGG
    D2 -->|Gradients| AGG
    D3 -->|Gradients| AGG
    
    AGG -->|Aggregate| GM
```

### Federated Learning Mathematics

#### FedAvg Algorithm

The Federated Averaging algorithm aggregates model updates from $K$ devices:

$$
w^{t+1} = \sum_{k=1}^{K} \frac{n_k}{n} w_k^{t+1}
$$

Where:
- $w^{t+1}$ = global model weights at round $t+1$
- $w_k^{t+1}$ = local model weights from device $k$
- $n_k$ = number of samples on device $k$
- $n = \sum_{k=1}^{K} n_k$ = total samples across all devices

#### Local Update (per device)

Each device performs $E$ local epochs:

$$
w_k^{t+1} = w_k^t - \eta \nabla \mathcal{L}_k(w_k^t)
$$

Where $\mathcal{L}_k$ is the local loss function on device $k$.

#### Privacy-Preserving Gradient Clipping

To ensure differential privacy:

$$
\tilde{g} = \frac{g}{\max\left(1, \frac{\|g\|_2}{C}\right)} + \mathcal{N}(0, \sigma^2 C^2 I)
$$

Where $C$ = clipping threshold, $\sigma$ = noise multiplier.

### Model Deployment Flow

```mermaid
sequenceDiagram
    participant RP as Research Portal
    participant B as Brain Backend
    participant T as Thoth Node (thothcraft)
    participant A as Actuator (HA / GPIO / Webhook)
    
    RP->>B: POST /api/datasets/models/deploy
    Note over RP,B: Model config + Actuator binding
    B->>B: Persist deployment record (queued)
    B-->>RP: Deployment queued response
    
    Note over T: Periodic Heartbeat (every 30s)
    T->>B: POST /api/device/heartbeat
    B-->>T: Pending deployments (payload with model + actuator)
    
    T->>T: Register in active model registry
    T->>T: Compile / Load rule or TorchScript
    T->>T: Initialize actuator plugin
    T->>B: POST /deployments/{id}/confirm (status: delivered)
    B-->>T: 200 OK
    
    Note over T: Continuous Inference Active
    T->>T: Evaluate sensor window
    opt Criteria Met (confidence >= threshold)
        T->>A: Trigger Actuator Action
        A-->>T: Action Result (OK)
    end
```

### Closed-Loop Inference & Actuation Pipeline on Device

```mermaid
flowchart LR
    subgraph Sensors[Sensor Input Layer]
        RAD[Radar FMCW]
        IMU[IMU 100Hz]
        CSI[Wi-Fi CSI]
        CAM[Camera OpenCV]
    end
    
    subgraph Buffer[Window Ring Buffer]
        BUF[Rolling Buffer & Features]
    end
    
    subgraph Engine[Dual-Engine Edge Inference]
        RULE[RuleProcessor<br/>Heuristics & Thresholds]
        TORCH[TorchScriptProcessor<br/>Compiled PyTorch JIT]
    end
    
    subgraph Logic[Decision & Hysteresis]
        EVAL[Evaluate Target Label & Confidence]
        COOLDOWN[Cooldown / Debounce Gate]
    end
    
    subgraph Actuation[Actuator Integration Layer]
        HA[Home Assistant REST]
        DEV[Device GPIO / Buzzer]
        WEB[HTTP Webhook / SMS]
    end
    
    Sensors --> BUF
    BUF --> RULE
    BUF --> TORCH
    RULE --> EVAL
    TORCH --> EVAL
    EVAL --> COOLDOWN
    COOLDOWN -->|Trigger Validated| HA
    COOLDOWN -->|Trigger Validated| DEV
    COOLDOWN -->|Trigger Validated| WEB
```

---

## ML Pipeline

### Model Architecture

The platform uses a **CNN+LSTM hybrid architecture** optimized for time-series sensor data classification.

```
Input: (batch_size, sequence_length, channels)
       e.g., (32, 128, 6) for 6-axis IMU

┌─────────────────────────────────────────────────────────────┐
│                    CNN Feature Extractor                     │
├─────────────────────────────────────────────────────────────┤
│  Conv1D(in=6, out=64, kernel=3) → BatchNorm → ReLU          │
│  Conv1D(in=64, out=128, kernel=3) → BatchNorm → ReLU        │
│  MaxPool1D(kernel=2)                                         │
│  Dropout(0.3)                                                │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    LSTM Temporal Encoder                     │
├─────────────────────────────────────────────────────────────┤
│  LSTM(input=128, hidden=128, layers=2, bidirectional=True)  │
│  Output: (batch, seq, 256)                                   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Attention Layer                           │
├─────────────────────────────────────────────────────────────┤
│  Self-attention over temporal dimension                      │
│  Weighted sum of LSTM outputs                                │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Classification Head                       │
├─────────────────────────────────────────────────────────────┤
│  Linear(256, 128) → ReLU → Dropout(0.5)                     │
│  Linear(128, num_classes)                                    │
│  Output: (batch, num_classes)                                │
└─────────────────────────────────────────────────────────────┘
```

### Model Types

ThothCraft supports both **Deep Learning** and **Traditional ML** models:

#### Deep Learning Models

**CNN-LSTM Hybrid** - Time-series classification with attention mechanism

| Size | CNN Filters | LSTM Hidden | Total Params | Use Case |
|------|-------------|-------------|--------------|----------|
| **Small** | 32, 64 | 64 | ~50K | Fast inference, limited data |
| **Medium** | 64, 128 | 128 | ~200K | Balanced performance |
| **Large** | 128, 256 | 256 | ~800K | Maximum accuracy |

**Configurable Parameters:**
- `window_size`: Input sequence length (default: 128)
- `input_channels`: Number of sensor channels (default: 6 for IMU)
- `num_classes`: Number of output classes
- `architecture_size`: 'small', 'medium', or 'large'
- `learning_rate`: 0.0001 to 0.01
- `batch_size`: 16, 32, 64, 128
- `epochs`: Training iterations
- `validation_split`: Train/val split ratio

#### Traditional ML Models

**AdaBoost** - Ensemble boosting classifier

**Parameters:**
- `n_estimators`: Number of weak learners (default: 50)
- `learning_rate`: Boosting learning rate (default: 1.0)
- `algorithm`: 'SAMME' or 'SAMME.R' (default: 'SAMME.R')
- `max_depth`: Max depth of decision tree base estimator (default: 1)

**K-Nearest Neighbors (KNN)** - Instance-based learning

**Parameters:**
- `n_neighbors`: Number of neighbors (default: 5)
- `weights`: 'uniform' or 'distance' (default: 'uniform')
- `metric`: Distance metric - 'euclidean', 'manhattan', 'minkowski' (default: 'euclidean')
- `algorithm`: 'auto', 'ball_tree', 'kd_tree', 'brute' (default: 'auto')
- `p`: Power parameter for Minkowski metric (default: 2)

**Support Vector Classifier (SVC)** - Kernel-based classification

**Parameters:**
- `C`: Regularization parameter (default: 1.0)
- `kernel`: 'linear', 'poly', 'rbf', 'sigmoid' (default: 'rbf')
- `gamma`: Kernel coefficient - 'scale', 'auto', or float (default: 'scale')
- `degree`: Degree for polynomial kernel (default: 3)
- `probability`: Enable probability estimates (default: True)
- `max_iter`: Maximum iterations, -1 for no limit (default: -1)

### Mathematical Formulations

#### CNN Layer Operations

The 1D convolution operation for time-series data:

$$
y[n] = \sum_{k=0}^{K-1} x[n+k] \cdot w[k] + b
$$

Where:
- $x$ = input signal of length $N$
- $w$ = learnable kernel weights of size $K$
- $b$ = bias term
- $y$ = output feature map

**Output dimension after Conv1D:**

$$
L_{out} = \left\lfloor \frac{L_{in} + 2P - K}{S} \right\rfloor + 1
$$

Where $P$ = padding, $K$ = kernel size, $S$ = stride.

#### LSTM Cell Equations

The LSTM processes sequential data through gated mechanisms:

$$
\begin{aligned}
f_t &= \sigma(W_f \cdot [h_{t-1}, x_t] + b_f) & \text{(forget gate)} \\
i_t &= \sigma(W_i \cdot [h_{t-1}, x_t] + b_i) & \text{(input gate)} \\
\tilde{C}_t &= \tanh(W_C \cdot [h_{t-1}, x_t] + b_C) & \text{(candidate)} \\
C_t &= f_t \odot C_{t-1} + i_t \odot \tilde{C}_t & \text{(cell state)} \\
o_t &= \sigma(W_o \cdot [h_{t-1}, x_t] + b_o) & \text{(output gate)} \\
h_t &= o_t \odot \tanh(C_t) & \text{(hidden state)}
\end{aligned}
$$

Where $\sigma$ = sigmoid function, $\odot$ = element-wise multiplication.

#### Attention Mechanism

Self-attention computes weighted importance of temporal features:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

For our temporal attention over LSTM outputs $H \in \mathbb{R}^{T \times d}$:

$$
\alpha_t = \frac{\exp(w^T h_t)}{\sum_{i=1}^{T} \exp(w^T h_i)}
$$

$$
c = \sum_{t=1}^{T} \alpha_t h_t
$$

Where $w$ is a learnable attention vector and $c$ is the context vector.

#### Loss Function

Cross-entropy loss for multi-class classification:

$$
\mathcal{L} = -\frac{1}{N} \sum_{i=1}^{N} \sum_{c=1}^{C} y_{i,c} \log(\hat{y}_{i,c})
$$

Where:
- $N$ = batch size
- $C$ = number of classes
- $y_{i,c}$ = ground truth (one-hot)
- $\hat{y}_{i,c}$ = predicted probability

#### Adam Optimizer Update

$$
\begin{aligned}
m_t &= \beta_1 m_{t-1} + (1-\beta_1) g_t \\
v_t &= \beta_2 v_{t-1} + (1-\beta_2) g_t^2 \\
\hat{m}_t &= \frac{m_t}{1-\beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1-\beta_2^t} \\
\theta_t &= \theta_{t-1} - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
\end{aligned}
$$

Default: $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$

#### Traditional ML Model Formulations

**AdaBoost Algorithm:**

For $m = 1, 2, \ldots, M$ boosting iterations:

$$
\alpha_m = \frac{1}{2} \ln\left(\frac{1 - \epsilon_m}{\epsilon_m}\right)
$$

$$
w_i^{(m+1)} = w_i^{(m)} \exp(\alpha_m \mathbb{1}[y_i \neq h_m(x_i)])
$$

Final prediction:

$$
H(x) = \text{sign}\left(\sum_{m=1}^{M} \alpha_m h_m(x)\right)
$$

Where $\epsilon_m$ = weighted error, $h_m$ = weak learner, $\alpha_m$ = learner weight.

**K-Nearest Neighbors:**

Distance-weighted prediction:

$$
\hat{y} = \arg\max_{c} \sum_{i \in \mathcal{N}_k(x)} w_i \cdot \mathbb{1}[y_i = c]
$$

Where $\mathcal{N}_k(x)$ = k-nearest neighbors, $w_i = \frac{1}{d(x, x_i)}$ for distance weighting.

Euclidean distance:

$$
d(x, x') = \sqrt{\sum_{j=1}^{d} (x_j - x'_j)^2}
$$

Manhattan distance:

$$
d(x, x') = \sum_{j=1}^{d} |x_j - x'_j|
$$

**Support Vector Classifier:**

Optimization objective:

$$
\min_{w, b} \frac{1}{2}\|w\|^2 + C \sum_{i=1}^{n} \xi_i
$$

Subject to:

$$
y_i(w^T \phi(x_i) + b) \geq 1 - \xi_i, \quad \xi_i \geq 0
$$

RBF Kernel:

$$
K(x, x') = \exp\left(-\gamma \|x - x'\|^2\right)
$$

Polynomial Kernel:

$$
K(x, x') = (\gamma \langle x, x' \rangle + r)^d
$$

Decision function:

$$
f(x) = \text{sign}\left(\sum_{i=1}^{n_{\text{SV}}} \alpha_i y_i K(x_i, x) + b\right)
$$

#### Model Evaluation Metrics

**Accuracy:**

$$
\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}
$$

**Precision:**

$$
\text{Precision} = \frac{TP}{TP + FP}
$$

**Recall (Sensitivity):**

$$
\text{Recall} = \frac{TP}{TP + FN}
$$

**F1 Score:**

$$
F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}
$$

**Confusion Matrix for Multi-class:**

$$
C_{ij} = \text{count of samples with true label } i \text{ predicted as } j
$$

#### Learning Rate Scheduling

**Step Decay:**

$$
\eta_t = \eta_0 \cdot \gamma^{\lfloor t / s \rfloor}
$$

Where $\gamma$ = decay factor, $s$ = step size.

**Cosine Annealing:**

$$
\eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1 + \cos\left(\frac{t}{T_{max}}\pi\right)\right)
$$

**Exponential Decay:**

$$
\eta_t = \eta_0 \cdot e^{-\lambda t}
$$

#### Early Stopping Criterion

Stop training when validation loss hasn't improved for $p$ epochs:

$$
\text{stop if } \min_{i \in [t-p, t]} \mathcal{L}_{val}^{(i)} \geq \min_{i \in [0, t-p]} \mathcal{L}_{val}^{(i)}
$$

#### Dropout Regularization

During training, each neuron is kept with probability $p$:

$$
\tilde{h} = \frac{1}{p} \cdot h \odot m, \quad m_i \sim \text{Bernoulli}(p)
$$

#### Batch Normalization

$$
\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}
$$

$$
y_i = \gamma \hat{x}_i + \beta
$$

Where $\mu_B$, $\sigma_B^2$ are batch mean/variance, $\gamma$, $\beta$ are learnable.

### Preprocessing Blocks

| Block | Description | Supported Types |
|-------|-------------|-----------------|
| **Normalize** | Scale values to [0,1] or [-1,1] | IMU, CSI, MFCW |
| **Low-Pass Filter** | Remove high-frequency noise | IMU, CSI |
| **High-Pass Filter** | Remove DC offset/drift | IMU, CSI |
| **Window Segmentation** | Split into fixed-size windows with overlap | IMU, CSI, MFCW |
| **FFT Transform** | Convert to frequency domain | IMU, CSI |
| **Feature Extraction** | Statistical features (mean, std, etc.) | IMU, CSI, MFCW |
| **Data Augmentation** | Noise injection, time warping | IMU |
| **Downsample** | Reduce sampling rate | IMU, CSI, MFCW |
| **Standardize** | Zero mean, unit variance | IMU, CSI, MFCW |

### Signal Processing Equations

#### Normalization (Min-Max Scaling)

$$
x_{norm} = \frac{x - x_{min}}{x_{max} - x_{min}}
$$

For range $[-1, 1]$:

$$
x_{norm} = 2 \cdot \frac{x - x_{min}}{x_{max} - x_{min}} - 1
$$

#### Standardization (Z-Score)

$$
x_{std} = \frac{x - \mu}{\sigma}
$$

Where $\mu$ = mean, $\sigma$ = standard deviation.

#### Low-Pass Butterworth Filter

Transfer function:

$$
H(s) = \frac{1}{\sqrt{1 + \left(\frac{s}{\omega_c}\right)^{2n}}}
$$

Where $\omega_c$ = cutoff frequency, $n$ = filter order.

#### Fast Fourier Transform (FFT)

Discrete Fourier Transform:

$$
X[k] = \sum_{n=0}^{N-1} x[n] \cdot e^{-j\frac{2\pi kn}{N}}
$$

Power Spectral Density:

$$
P[k] = \frac{|X[k]|^2}{N}
$$

#### Window Segmentation

For signal $x$ of length $L$, window size $W$, and overlap ratio $r$:

$$
\text{stride} = W \cdot (1 - r)
$$

$$
\text{num\_windows} = \left\lfloor \frac{L - W}{\text{stride}} \right\rfloor + 1
$$

#### Feature Extraction

Statistical features computed per window:

| Feature | Formula |
|---------|---------|
| Mean | $\bar{x} = \frac{1}{N}\sum_{i=1}^{N} x_i$ |
| Variance | $\sigma^2 = \frac{1}{N}\sum_{i=1}^{N} (x_i - \bar{x})^2$ |
| Skewness | $\gamma_1 = \frac{1}{N}\sum_{i=1}^{N} \left(\frac{x_i - \bar{x}}{\sigma}\right)^3$ |
| Kurtosis | $\gamma_2 = \frac{1}{N}\sum_{i=1}^{N} \left(\frac{x_i - \bar{x}}{\sigma}\right)^4 - 3$ |
| RMS | $x_{rms} = \sqrt{\frac{1}{N}\sum_{i=1}^{N} x_i^2}$ |
| Zero Crossing Rate | $ZCR = \frac{1}{N-1}\sum_{i=1}^{N-1} \mathbb{1}[x_i \cdot x_{i+1} < 0]$ |

#### Data Augmentation Techniques

**Gaussian Noise Injection:**

$$
x_{aug} = x + \epsilon, \quad \epsilon \sim \mathcal{N}(0, \sigma^2)
$$

**Time Warping:**

$$
x_{warp}(t) = x(\tau(t)), \quad \tau(t) = t + \alpha \cdot \sin\left(\frac{2\pi t}{T}\right)
$$

**Magnitude Scaling:**

$$
x_{scaled} = x \cdot s, \quad s \sim \mathcal{U}(1-\delta, 1+\delta)
$$

**Random Rotation (for IMU):**

$$
\mathbf{x}_{rot} = \mathbf{R} \cdot \mathbf{x}, \quad \mathbf{R} \in SO(3)
$$

**Jittering:**

$$
x_{jitter}[n] = x[n] + \epsilon_n, \quad \epsilon_n \sim \mathcal{N}(0, \sigma^2)
$$

#### Downsampling

**Decimation by factor $M$:**

$$
x_d[n] = x[nM]
$$

**Anti-aliasing filter before decimation:**

$$
x_{filtered} = x * h_{LP}, \quad f_c = \frac{f_s}{2M}
$$

#### High-Pass Filter

First-order high-pass (DC removal):

$$
y[n] = \alpha \cdot (y[n-1] + x[n] - x[n-1])
$$

Where $\alpha = \frac{RC}{RC + \Delta t} \approx 0.99$ for slow drift removal.

#### Hamming Window

Applied before FFT to reduce spectral leakage:

$$
w[n] = 0.54 - 0.46 \cos\left(\frac{2\pi n}{N-1}\right)
$$

**Windowed signal:**

$$
x_w[n] = x[n] \cdot w[n]
$$

### Data Pipeline Flow Diagram

```mermaid
flowchart LR
    subgraph Input[Raw Data]
        RAW[Time Series]
    end
    
    subgraph Preprocessing[Pipeline]
        N[Normalize] --> F[Filter]
        F --> W[Window]
        W --> FFT[FFT]
        FFT --> FE[Features]
        FE --> STD[Standardize]
    end
    
    subgraph Output[Model Input]
        TENSOR[Tensor]
    end
    
    RAW --> N
    STD --> TENSOR
```

### Training Configuration

```json
{
  "model_type": "medium",
  "training_mode": "cloud",
  "epochs": 100,
  "batch_size": 32,
  "learning_rate": 0.001,
  "validation_split": 0.2,
  "optimizer": "adam",
  "loss_function": "cross_entropy",
  "early_stopping": true,
  "patience": 10,
  "bayesian_optimization": {
    "enabled": true,
    "n_trials": 20,
    "search_space": {
      "learning_rate": [0.0001, 0.01],
      "batch_size": [16, 32, 64],
      "hidden_size": [64, 128, 256]
    }
  }
}
```

---

## Security Model

### Authentication Flow

```mermaid
sequenceDiagram
    participant U as User
    participant RP as Research Portal
    participant B as Brain Backend
    participant T as Thoth Device
    
    U->>RP: Login credentials
    RP->>B: POST /auth/login
    B->>B: Verify credentials
    B->>B: Generate JWT token
    B-->>RP: token and user
    RP->>RP: Store token
    
    U->>T: Login via Captive Portal
    T->>B: POST /auth/login
    B-->>T: token and user
    T->>T: Store token
    T->>B: POST /device/register
    B-->>T: Registration success
```

### Security Measures

| Layer | Measure |
|-------|---------|
| **Transport** | HTTPS/TLS for all API communication |
| **Authentication** | JWT tokens with expiration |
| **Authorization** | User-scoped data access |
| **Device Auth** | Device UUID + user token |
| **Rate Limiting** | Registration attempts throttled |
| **Input Validation** | Pydantic models for all requests |
| **Password Storage** | Bcrypt hashing |

### Data Sharing Permissions (Planned)

| Permission Level | Capabilities |
|------------------|--------------|
| **View Only** | Browse shared data files |
| **Can Train** | Use shared data in training datasets |
| **Can Deploy** | Deploy models trained on shared data |

---

## Subscription Model

### Tier Structure

Plans are enforced server-side through a single entitlement map (`server/entitlements.py`) — every feature gate resolves through `get_entitlements(user)`, never scattered `user.plan ==` checks. Legacy Stripe price names are normalized (`researcher` → `research`).

| Feature | Free | Home | Research |
|---------|------|------|----------|
| **Devices** | 1 | 5 | 10 |
| **Spaces (floor plans)** | 1 | 5 | Unlimited |
| **Cloud Storage** | Rolling 400-minute retention | 10 GB | 100 GB |
| **Raw Data Download/Export** | ❌ | ✅ | ✅ |
| **SDK Access** | Read + device control | Full | Full |
| **Datasets & Model Training** | ❌ | ❌ | ✅ |
| **Custom Model Deployment** | ❌ | ❌ | ✅ |
| **Labs & Notebook Grading** | ❌ | ❌ | ✅ |
| **Billing** | — | Stripe monthly/annual | Stripe monthly/annual |

### Plan Reconciliation

- Stripe webhooks (`customer.subscription.*`, `invoice.*`) keep `user.plan` in sync.
- `POST /api/stripe/sync-subscription` lets an account self-heal on demand — it pulls live subscription state from Stripe and repairs the stored plan if a webhook was missed (e.g. after a domain migration).
- `GET /api/account/entitlements` returns the normalized plan, full entitlement set, account identity, and Stripe linkage — the single endpoint the CLI (`thothcraft whoami`), Hub, and portal use to render plan-gated UI.

### Device Purchase Model

- Thoth devices are sold as **one-time purchases** (Stripe hardware checkout, `kind: "hardware"` metadata).
- Subscription is tied to the **user account** (covers all owned devices).
- Plan device limits are enforced at pairing/registration time (`check_device_limit`).

### Storage Limits

| Location | Limit |
|----------|-------|
| **On-Device** | 128 GB (SD card capacity) |
| **Cloud** | Subscription-based (see tier table) |

---

## API Reference

### Authentication

```http
POST /auth/register
Content-Type: application/json

{
  "username": "researcher1",
  "email": "researcher@university.edu",
  "password": "securepassword123"
}
```

```http
POST /auth/login
Content-Type: application/json

{
  "username": "researcher1",
  "password": "securepassword123"
}

Response:
{
  "success": true,
  "token": "eyJhbGciOiJIUzI1NiIs...",
  "user": {
    "user_id": 1,
    "username": "researcher1",
    "email": "researcher@university.edu"
  }
}
```

### Device Management

```http
POST /device/register
Authorization: Bearer <token>
Content-Type: application/json

{
  "device_id": "550e8400-e29b-41d4-a716-446655440000",
  "device_name": "Thoth-Lab1",
  "device_type": "thoth",
  "ip_address": "192.168.1.100",
  "mac_address": "b8:27:eb:xx:xx:xx",
  "hardware_info": {
    "hostname": "thoth-lab1",
    "os_version": "Raspberry Pi OS 12"
  },
  "files": [
    {"name": "imu_2025-01-08.json", "size": 1048576, "type": "json"}
  ]
}

Response:
{
  "success": true,
  "device_id": "550e8400-e29b-41d4-a716-446655440000",
  "device_name": "Thoth-Lab1",
  "pending_uploads": ["imu_2025-01-07.json"]
}
```

### Device Heartbeat

```http
POST /device/heartbeat
Authorization: Bearer <device_token>
Content-Type: application/json

{
  "device_id": "550e8400-e29b-41d4-a716-446655440000",
  "device_hostname": "thoth-denver.local",
  "capabilities": {"usb_camera": true, "radar": false, "esp32_csi": true},
  "daemon": "thothcraft"
}

Response:
{
  "success": true,
  "pending_deployments": [ {"deployment_id": 12, "name": "radar-occupancy", ...} ],
  "data": {
    "capture_settings": {...},
    "pending_uploads": [],
    "pending_commands": []
  }
}
```

The heartbeat is the node's control plane: it reports the mDNS hostname and probed sensor capabilities (rendered in the portal), and returns queued model deployments, upload requests, and commands.

### Account Entitlements

```http
GET /account/entitlements
Authorization: Bearer <token>

Response:
{
  "plan": "research",
  "entitlements": {"device_limit": 10, "storage_bytes": 107374182400, ...},
  "user": {"user_id": 1, "username": "researcher1", "email": "..."},
  "stripe": {"customer_id": "cus_...", "subscription_id": "sub_...", "plan_expires_at": "..."}
}
```

### Training

```http
POST /datasets/{dataset_id}/train
Authorization: Bearer <token>
Content-Type: application/json

{
  "model_type": "medium",
  "training_mode": "cloud",
  "epochs": 100,
  "batch_size": 32,
  "learning_rate": 0.001,
  "validation_split": 0.2,
  "use_bayesian": true
}

Response:
{
  "success": true,
  "job_id": "job_abc123",
  "status": "pending",
  "message": "Training job queued"
}
```

### AI Query

```http
POST /query
Authorization: Bearer <token>
Content-Type: application/json

{
  "query": "How many devices are online?",
  "chat_id": "chat_123",
  "context": {
    "system_stats": {
      "devices": {"total": 5, "online": 3}
    }
  }
}

Response:
{
  "success": true,
  "query_id": 42,
  "response": "You currently have 3 out of 5 devices online.",
  "chat_id": "chat_123",
  "timestamp": "2025-01-08T11:00:00Z"
}
```

---

## Deployment Architecture

### Production Deployment

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Cloud Infrastructure                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐        │
│  │   CDN / Edge    │     │  Load Balancer  │     │   DNS           │        │
│  │   (Static)      │     │                 │     │   (Route 53)    │        │
│  └────────┬────────┘     └────────┬────────┘     └─────────────────┘        │
│           │                       │                                          │
│           ▼                       ▼                                          │
│  ┌─────────────────┐     ┌─────────────────┐                                │
│  │ Research Portal │     │  Brain Backend  │                                │
│  │   (Next.js)     │     │   (FastAPI)     │                                │
│  │                 │     │                 │                                │
│  │  • Vercel       │     │  • Railway      │                                │
│  │  • Netlify      │     │  • Render       │                                │
│  │  • AWS Amplify  │     │  • AWS ECS      │                                │
│  └─────────────────┘     └────────┬────────┘                                │
│                                   │                                          │
│                    ┌──────────────┼──────────────┐                          │
│                    │              │              │                          │
│                    ▼              ▼              ▼                          │
│           ┌──────────────┐ ┌──────────────┐ ┌──────────────┐               │
│           │  PostgreSQL  │ │ Blob Storage │ │  OpenAI API  │               │
│           │  (Managed)   │ │ (S3/GCS)     │ │              │               │
│           └──────────────┘ └──────────────┘ └──────────────┘               │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Environment Variables

**Brain Backend** (Railway → `api.thothcraft.com`)
```env
DATABASE_URL=postgresql://user:pass@host:5432/thothcraft
OPENAI_API_KEY=sk-...
MODEL_NAME=gpt-4
JWT_SECRET=your-secret-key
# CORS: explicit allow-list lives in server/main.py and already covers
# *.thothcraft.com / *.thothcraft.org / *.vercel.app / *.local / localhost.
# Add one-off origins at deploy time without a redeploy:
CORS_EXTRA_ORIGINS=https://staging.thothcraft.com
STRIPE_SECRET_KEY=sk_live_...
STRIPE_WEBHOOK_SECRET=whsec_...
STRIPE_PRICE_ID_HOME_MONTHLY=price_...
STRIPE_PRICE_ID_RESEARCH_MONTHLY=price_...
TWILIO_ACCOUNT_SID=...
TWILIO_AUTH_TOKEN=...
```

**Research Portal** (Vercel → `portal.thothcraft.com`)
```env
NEXT_PUBLIC_API_URL=https://api.thothcraft.com
```

**Thoth Node** (`thothcraft daemon`)
```env
THOTHCRAFT_API_URL=https://api.thothcraft.com   # Brain endpoint (SDK default)
THOTH_HOSTNAME=denver                            # → thoth-denver.local (optional override)
DEVICE_NAME=Thoth-Lab1
```

---

## Future Roadmap

### Phase 1: Core Platform (Current)
- [x] Thoth device with IMU sensing
- [x] Brain backend with training pipeline
- [x] Research Portal with full UI
- [x] Cloud training with PyTorch
- [x] AI chatbot assistant

### Phase 2: Advanced Sensing (Q2 2026)
- [ ] Thoth-Wireless (WiFi CSI + mmWave)
- [ ] Thoth-Vision (Camera module)
- [ ] Thoth-Audio (Microphone array)
- [ ] Thoth-EEG (Brain-computer interface)

### Phase 3: Federated Learning (Q3 2026)
- [ ] FL orchestration in Brain
- [ ] Gradient aggregation algorithms
- [ ] Privacy-preserving training
- [ ] Multi-device coordination

### Phase 4: Mobile & Notifications (Q4 2026)
- [ ] React Native mobile app
- [ ] Push notifications (FCM)
- [ ] Real-time inference alerts
- [ ] Remote device control

### Phase 5: Enterprise Features (2027)
- [ ] Multi-tenant architecture
- [ ] SSO/SAML integration
- [ ] Audit logging
- [ ] Custom preprocessing blocks
- [ ] On-premise deployment option

---

## Sensing Modalities

### WiFi Channel State Information (CSI)

WiFi CSI captures fine-grained channel characteristics between transmitter and receiver.

#### CSI Mathematical Model

The received signal in OFDM systems:

$$
Y = HX + N
$$

Where:
- $Y$ = received signal vector
- $H$ = channel frequency response (CSI)
- $X$ = transmitted signal
- $N$ = additive noise

CSI for subcarrier $k$:

$$
H_k = |H_k| e^{j\angle H_k}
$$

Where $|H_k|$ = amplitude, $\angle H_k$ = phase.

#### CSI-based Sensing

Human motion affects wireless propagation:

$$
H(t) = H_s + \sum_{i=1}^{M} H_d^{(i)}(t)
$$

Where:
- $H_s$ = static component (walls, furniture)
- $H_d^{(i)}(t)$ = dynamic component from moving object $i$

```mermaid
flowchart LR
    subgraph TX[ESP32 TX]
        T[WiFi TX]
    end
    
    subgraph Environment[Environment]
        STATIC[Static]
        HUMAN[Human]
    end
    
    subgraph RX[ESP32 RX]
        R[WiFi RX]
        CSI[CSI Extract]
    end
    
    subgraph Processing[Thoth Pi]
        SERIAL[Serial]
        PROC[Processing]
    end
    
    T --> R
    T --> STATIC --> R
    T --> HUMAN --> R
    R --> CSI
    CSI --> SERIAL
    SERIAL --> PROC
```

### mmWave Radar (MFCW)

Multi-Frequency Continuous Wave radar for precise motion detection.

#### FMCW Radar Equation

Range calculation from beat frequency:

$$
R = \frac{c \cdot f_b \cdot T_c}{2B}
$$

Where:
- $R$ = target range
- $c$ = speed of light ($3 \times 10^8$ m/s)
- $f_b$ = beat frequency
- $T_c$ = chirp duration
- $B$ = bandwidth

#### Doppler Velocity

$$
v = \frac{\lambda \cdot f_d}{2}
$$

Where:
- $v$ = radial velocity
- $\lambda$ = wavelength
- $f_d$ = Doppler frequency shift

#### Range-Doppler Map

```
         Velocity (m/s)
         -3  -2  -1   0   1   2   3
        ┌───┬───┬───┬───┬───┬───┬───┐
    0.5 │   │   │   │ ░ │   │   │   │
        ├───┼───┼───┼───┼───┼───┼───┤
    1.0 │   │   │ ▓ │ ▓ │ ▓ │   │   │  ← Person walking
        ├───┼───┼───┼───┼───┼───┼───┤
Range   1.5 │   │   │   │ ░ │   │   │   │
(m)     ├───┼───┼───┼───┼───┼───┼───┤
    2.0 │   │   │   │   │   │   │   │
        ├───┼───┼───┼───┼───┼───┼───┤
    2.5 │   │   │   │ ░ │   │   │   │  ← Static object
        └───┴───┴───┴───┴───┴───┴───┘
```

### IMU Sensor Fusion

6-axis IMU combines accelerometer and gyroscope data.

#### Orientation from Gyroscope (Integration)

$$
\theta(t) = \theta(t_0) + \int_{t_0}^{t} \omega(\tau) d\tau
$$

Discrete form:

$$
\theta[n] = \theta[n-1] + \omega[n] \cdot \Delta t
$$

#### Complementary Filter

Fuses accelerometer (low-frequency) and gyroscope (high-frequency):

$$
\theta_{fused} = \alpha \cdot (\theta_{prev} + \omega \cdot \Delta t) + (1-\alpha) \cdot \theta_{accel}
$$

Typical $\alpha = 0.98$ (98% gyro, 2% accelerometer).

#### Accelerometer Tilt Calculation

$$
\text{pitch} = \arctan\left(\frac{a_x}{\sqrt{a_y^2 + a_z^2}}\right)
$$

$$
\text{roll} = \arctan\left(\frac{a_y}{\sqrt{a_x^2 + a_z^2}}\right)
$$

#### Quaternion Representation

Orientation as unit quaternion:

$$
q = q_0 + q_1 i + q_2 j + q_3 k, \quad \|q\| = 1
$$

**Quaternion multiplication:**

$$
q \otimes p = \begin{bmatrix} q_0 p_0 - q_1 p_1 - q_2 p_2 - q_3 p_3 \\ q_0 p_1 + q_1 p_0 + q_2 p_3 - q_3 p_2 \\ q_0 p_2 - q_1 p_3 + q_2 p_0 + q_3 p_1 \\ q_0 p_3 + q_1 p_2 - q_2 p_1 + q_3 p_0 \end{bmatrix}
$$

**Rotation matrix from quaternion:**

$$
R = \begin{bmatrix} 1-2(q_2^2+q_3^2) & 2(q_1 q_2 - q_0 q_3) & 2(q_1 q_3 + q_0 q_2) \\ 2(q_1 q_2 + q_0 q_3) & 1-2(q_1^2+q_3^2) & 2(q_2 q_3 - q_0 q_1) \\ 2(q_1 q_3 - q_0 q_2) & 2(q_2 q_3 + q_0 q_1) & 1-2(q_1^2+q_2^2) \end{bmatrix}
$$

#### Madgwick Filter

Gradient descent orientation filter:

$$
q_{t+1} = q_t - \mu \frac{\nabla f}{\|\nabla f\|}
$$

Where $f$ is the objective function minimizing accelerometer error.

### Sensor Data Rates

| Sensor | Sample Rate | Data Size | Bandwidth |
|--------|-------------|-----------|-----------|
| **IMU (Sense HAT)** | 100 Hz | 24 bytes/sample | 2.4 KB/s |
| **WiFi CSI (ESP32)** | 100 Hz | 384 bytes/sample | 38.4 KB/s |
| **mmWave Radar** | 20 Hz | 2 KB/frame | 40 KB/s |
| **Camera (720p)** | 30 fps | 1.3 MB/frame | 39 MB/s |

---

## Bayesian Hyperparameter Optimization

### Gaussian Process Surrogate Model

The objective function $f(\mathbf{x})$ is modeled as a Gaussian Process:

$$
f(\mathbf{x}) \sim \mathcal{GP}(m(\mathbf{x}), k(\mathbf{x}, \mathbf{x}'))
$$

Where:
- $m(\mathbf{x})$ = mean function (often zero)
- $k(\mathbf{x}, \mathbf{x}')$ = covariance kernel

### Matérn Kernel

$$
k(\mathbf{x}, \mathbf{x}') = \sigma^2 \frac{2^{1-\nu}}{\Gamma(\nu)} \left(\frac{\sqrt{2\nu}d}{\ell}\right)^\nu K_\nu\left(\frac{\sqrt{2\nu}d}{\ell}\right)
$$

Where $d = \|\mathbf{x} - \mathbf{x}'\|$, $\ell$ = length scale, $\nu$ = smoothness.

### Expected Improvement Acquisition Function

$$
\text{EI}(\mathbf{x}) = \mathbb{E}[\max(f(\mathbf{x}) - f(\mathbf{x}^+), 0)]
$$

Closed form:

$$
\text{EI}(\mathbf{x}) = (\mu(\mathbf{x}) - f(\mathbf{x}^+) - \xi) \Phi(Z) + \sigma(\mathbf{x}) \phi(Z)
$$

Where:

$$
Z = \frac{\mu(\mathbf{x}) - f(\mathbf{x}^+) - \xi}{\sigma(\mathbf{x})}
$$

### Optimization Loop

```mermaid
flowchart TB
    subgraph Init[Initialization]
        SAMPLE[Random Sample]
        EVAL1[Evaluate]
    end
    
    subgraph Loop[Bayesian Loop]
        GP[Fit GP]
        ACQ[Max Acquisition]
        NEXT[Select Config]
        EVAL2[Train and Eval]
        UPDATE[Update Obs]
    end
    
    subgraph Output[Output]
        BEST[Best Params]
    end
    
    SAMPLE --> EVAL1
    EVAL1 --> GP
    GP --> ACQ
    ACQ --> NEXT
    NEXT --> EVAL2
    EVAL2 --> UPDATE
    UPDATE --> GP
    UPDATE -->|Done| BEST
```

---

## System State Diagrams

### Device State Machine

```mermaid
stateDiagram-v2
    [*] --> PowerOn
    PowerOn --> HotspotMode
    PowerOn --> WiFiConnect
    HotspotMode --> CaptivePortal
    CaptivePortal --> WiFiConnect
    WiFiConnect --> Online
    WiFiConnect --> HotspotMode
    Online --> Collecting
    Online --> Inferencing
    Collecting --> Online
    Collecting --> Uploading
    Uploading --> Collecting
    Uploading --> Online
    Inferencing --> Online
    Inferencing --> AlertTriggered
    AlertTriggered --> Inferencing
    Online --> Offline
    Offline --> Online
```

### Training Job State Machine

```mermaid
stateDiagram-v2
    [*] --> Pending
    Pending --> Queued
    Queued --> Preparing
    Preparing --> Training
    Preparing --> Failed
    Training --> Training
    Training --> Evaluating
    Training --> Failed
    Training --> Cancelled
    Evaluating --> Completed
    Evaluating --> Failed
    Completed --> Deployed
    Completed --> [*]
    Failed --> [*]
    Cancelled --> [*]
```

### File Sync State Machine

```mermaid
stateDiagram-v2
    [*] --> OnDevice
    OnDevice --> UploadRequested
    OnDevice --> OnDevice
    UploadRequested --> Uploading
    Uploading --> OnCloud
    Uploading --> UploadRequested
    OnCloud --> OnBoth
    OnCloud --> CloudOnly
    OnBoth --> CloudOnly
    CloudOnly --> [*]
    OnBoth --> [*]
```

---

## Appendix

### Glossary

| Term | Definition |
|------|------------|
| **CSI** | Channel State Information - WiFi signal characteristics used for sensing |
| **IMU** | Inertial Measurement Unit - accelerometer + gyroscope + magnetometer |
| **mmWave** | Millimeter wave radar for presence detection |
| **FL** | Federated Learning - distributed training without centralizing data |
| **Captive Portal** | Web page shown when connecting to a WiFi network |
| **Heartbeat** | Periodic status update from device to server |

### References

- [Raspberry Pi Documentation](https://www.raspberrypi.com/documentation/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Next.js Documentation](https://nextjs.org/docs)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [ESP-IDF CSI Guide](https://docs.espressif.com/projects/esp-idf/en/latest/esp32/api-guides/wifi.html#wi-fi-channel-state-information)

---

*This document is maintained by the ThothCraft team. For questions or contributions, contact the development team.*
