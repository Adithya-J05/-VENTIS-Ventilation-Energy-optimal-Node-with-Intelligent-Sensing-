# Closed-Loop Air-Quality Remediation & Energy-Optimal Ventilation Controller
## Product Requirements, System Architecture & Implementation Plan

| Field | Value |
|---|---|
| Document ID | AQR-ARCH-001 |
| Revision | v1.0 (design freeze candidate) |
| Date | 2026-09-02 |
| Status | For review — implementation-ready |
| Codename | **VENTIS** (Ventilation Energy-optimal Node with Intelligent Sensing) |
| Scope | Edge firmware, transport, host ML/optimization, dashboard, data plane |
| Non-scope | Enclosure mechanical design, ducting CFD, regulatory certification |

> **How to read this document.** Sections 1–9 are the required deliverables in the specified order. **Appendix A** is a *flaw register*: every place where the naive reading of the brief is technically unsound, what was changed, and why. Read Appendix A before implementation — several corrections there are load-bearing (index definition, flammable-gas interlock, optimizer/surrogate coupling).

---

## 1. Project Overview

### 1.1 Problem Statement

Ventilation in industrial bays, chemical storehouses, commercial kitchens and chimney-exhaust zones is almost universally **open-loop**: an extractor is either latched on for an entire shift or toggled by an operator who reacts to smell and visible smoke rather than to measured concentration. Both modes are simultaneously wasteful and unsafe. Always-on operation runs the turbine at 100 % duty for the ~85–95 % of the shift in which the air is already within limits, consuming energy monotonically with time rather than with pollutant load, and consuming fan/laser-sensor service life at the same rate. Manual operation, conversely, has a human detection latency of minutes-to-never for colourless hazards (CO₂ accumulation, solvent vapour, sub-10 µm particulate), so the zone spends unmeasured, unrecorded time above exposure limits. Neither mode can answer the question a plant manager actually has — *"will this bay be safe in fifteen minutes, and what is the cheapest way to get it there?"* — because neither mode contains a model of how the air in that specific room decays under forced extraction.

### 1.2 Solution Summary

VENTIS replaces the on/off switch with a **cascaded, forecast-driven closed-loop controller**. An ESP32 edge node samples particulate, CO₂, gas and psychrometric channels at 1 Hz, computes a locally-defined instantaneous air-quality index, and streams it over MQTT to a host machine. The host maintains a **grey-box dynamical model** of the zone — a first-principles mass-balance ODE whose airflow term is a calibrated function of PWM duty, wrapped by a learned residual correction (NARX network primary, XGBoost fallback/ensemble) — and every 10 s solves a receding-horizon constrained optimization for the duty-cycle profile that minimises actuator energy subject to reaching the safe index by a deadline `T_target`. The resulting profile is published back to the edge, which applies it through hardware PWM. Critically, **authority remains at the edge**: the host is an *advisory* optimizer, and a 100 ms edge safety loop with an autonomous degraded-mode controller runs underneath it, so loss of the host degrades performance but never safety. The system is therefore a two-rate cascade — a fast local safety/regulation loop and a slow model-predictive economic loop — rather than a cloud-dependent controller.

### 1.3 Measurable Success Criteria

Every criterion below is measured on the **same set of ≥ 100 held-out remediation events** in the validation chamber (see §7.3), against an always-on 100 %-duty baseline replayed on matched initial conditions.

| # | Criterion | Metric definition | Target | Floor (pass/fail) |
|---|---|---|---|---|
| SC-1 | Energy saving vs. always-on | `1 − (Σ Wh_MPC / Σ Wh_baseline)` over matched events, measured electrically (§7.1, `E_true`) | ≥ 40 % | ≥ 25 % |
| SC-2 | Deadline compliance | Fraction of events with `I_inst(T_target) ≤ I_safe` | ≥ 95 % | ≥ 90 % |
| SC-3 | Recovery-time accuracy | `\|t_reach − T_target\| / T_target`, p90 across events | ≤ 10 % | ≤ 20 % |
| SC-4 | Forecast accuracy — short | RMSE of 60 s-ahead index forecast, held out **by event** | ≤ 3.0 index pts | ≤ 5.0 |
| SC-5 | Forecast accuracy — horizon | RMSE of 300 s-ahead index forecast, held out by event | ≤ 8.0 index pts | ≤ 12.0 |
| SC-6 | False-trigger rate | Actuations ≥ 30 s with no injected/real event, per 72 h of idle monitoring | ≤ 1 | ≤ 3 |
| SC-7 | Missed-event rate | Injected events > 1.5 × `I_safe` not detected within 30 s | 0 % | 0 % |
| SC-8 | Overshoot waste | Energy expended after `I_inst` first crosses below `I_safe`, as % of event energy | ≤ 5 % | ≤ 10 % |
| SC-9 | Undershoot rate | Events terminating with `I_inst > I_safe` at `T_target` | ≤ 5 % | ≤ 10 % |
| SC-10 | Control-loop latency | p95 edge-publish → edge-actuate, excluding sensor frame period | ≤ 1.5 s | ≤ 2.5 s |
| SC-11 | Optimizer solve time | p95 wall-clock for one MPC solve, N = 30 | ≤ 500 ms | ≤ 1200 ms |
| SC-12 | Telemetry completeness | Received samples / expected samples per 24 h (post-backfill) | ≥ 99.5 % | ≥ 99.0 % |
| SC-13 | Fail-safe integrity | Fault-injection suite (Appendix C, FI-01…FI-10) passed with no unsafe state | 10 / 10 | 10 / 10 |
| SC-14 | Actuator chatter | Mean `\|Δduty\|` per control interval during steady remediation | ≤ 0.08 | ≤ 0.15 |

> **SC-1 is reported on true electrical energy `∫V·I dt`, not on the stated objective `∫V² dt`.** The optimizer *minimises* `∫V²dt` because it is smooth and convex-friendly; the *claim* is made on measured watt-hours. See Appendix A, FX-02.

---

## 2. System Architecture Layout

### 2.1 Layer Stack

```
┌──────────────────────────────────────────────────────────────────────────────┐
│ L5  PRESENTATION      Grafana (ops/health)  ·  VENTIS Console (React, control)│
│                       ▲ WebSocket 2 Hz            ▲ SQL / Flux                │
├───────────────────────┼───────────────────────────┼──────────────────────────┤
│ L4  HOST / COMPUTE    FastAPI service ──┬── Ingest & schema validation        │
│     (laptop)                            ├── Feature engineering (Polars)      │
│                                         ├── Grey-box + NARX/XGB forecaster    │
│                                         ├── MPC solver (SLSQP → IPOPT)        │
│                                         ├── TimescaleDB writer                │
│                                         └── Command dispatcher + ack tracker  │
├───────────────────────▲───────────────────────────┬──────────────────────────┤
│ L3  TRANSPORT         │  Mosquitto 2.x broker, TLS 8883, per-node ACL         │
│                       │  telemetry ↑ QoS0 · cmd ↓ QoS1 · state/LWT retained   │
├───────────────────────┼───────────────────────────▼──────────────────────────┤
│ L2  EDGE — LOGIC      Safety FSM (100 ms) · Index compute · Ring buffer       │
│                       Command validator (seq, expiry, bounds) · Watchdog      │
├──────────────────────────────────────────────────────────────────────────────┤
│ L1  EDGE — PHYSICAL   PMS5003 · MH-Z19B · MQ-135 · MQ-2 · DHT22               │
│                       LEDC PWM → gate driver → IRF520 → 2 W turbine           │
│                       ACS712 current sense · bus-voltage divider (feedback)   │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Boundary Contracts

| # | Boundary | Direction | Protocol / transport | Rate | Payload shape |
|---|---|---|---|---|---|
| B1 | Sensor → ESP32 | in | UART1 9600 8N1 (PMS5003), UART2 9600 8N1 (MH-Z19B), 1-Wire (DHT22), ADC1 (MQ×2, I_motor, V_bus) | PM 1 Hz frame, CO₂ 1 Hz poll, DHT 0.5 Hz, ADC 10 Hz ×64 oversample | Binary frames / raw counts |
| B2 | ESP32 → broker | up | MQTT 3.1.1 over TLS 1.2, QoS 0 | 1 Hz active / 0.2 Hz idle | `telemetry` JSON, ~280 B |
| B3 | ESP32 → broker | up | MQTT QoS 1, retained=false | event-driven | `event` JSON (alarm, mode change, fault) |
| B4 | ESP32 → broker | up | MQTT QoS 1, **retained** | on change + 60 s heartbeat | `state` JSON (mode, fw, uptime, sensor health) |
| B5 | broker → host | down | MQTT subscribe, shared topic tree | as B2–B4 | same |
| B6 | host → broker → ESP32 | down | MQTT QoS 1 | every 10 s (control interval) | `cmd` JSON — **duty *profile*, not a scalar** |
| B7 | ESP32 → host | up | MQTT QoS 1 | per command | `cmd/ack` JSON with `seq`, `applied_duty`, `reject_reason` |
| B8 | host → broker → ESP32 | down | MQTT QoS 1, **retained** | on operator change | `cfg` JSON (`I_safe`, `T_target`, thresholds, cadence) |
| B9 | host → TimescaleDB | — | libpq, batched COPY | 1 s micro-batch | hypertable rows |
| B10 | host → console | out | WebSocket JSON | 2 Hz | live state + predicted trajectory + planned profile |
| B11 | broker → host (LWT) | down | MQTT retained | on ungraceful disconnect | `{"online":false,"ts":...}` |

### 2.3 Canonical Payloads

**B2 — `aqr/v1/site/{site}/zone/{zone}/node/{node}/telemetry` (edge → host, QoS 0)**

```json
{
  "seq": 184213,
  "ts": 1788412345.120,
  "up": 92714,
  "pm": {"p1": 6.2, "p25": 148.4, "p10": 191.0, "n03": 4120, "valid": true},
  "co2": {"ppm": 1180, "valid": true},
  "gas": {"mq135_ratio": 2.41, "mq2_ratio": 1.07, "mq135_raw": 1832, "mq2_raw": 901},
  "th": {"t": 31.4, "rh": 58.2, "valid": true},
  "act": {"duty": 0.62, "v_bus": 11.83, "i_mot": 0.149, "rpm": 4180},
  "idx": {"i_inst": 168.2, "driver": "pm25", "i_nowcast": 141.7},
  "mode": "ACTIVE_REMEDIATION",
  "cmd_seq": 9041
}
```

**B6 — `.../node/{node}/cmd` (host → edge, QoS 1)**

```json
{
  "seq": 9042,
  "issued_ts": 1788412345.480,
  "valid_until": 1788412375.480,
  "control_interval_s": 10,
  "profile": [0.62, 0.58, 0.55, 0.51, 0.44, 0.30],
  "slew_max": 0.10,
  "reason": "mpc_ok",
  "model_id": "narx_gb_v7@2026-08-28",
  "predicted_idx": [168.2, 151.0, 133.4, 116.9, 101.2, 88.6]
}
```

The command carries a **profile with an explicit validity deadline**, not a single setpoint. If the host dies at `t`, the edge continues executing the remaining, already-vetted profile until `valid_until`, then transitions to autonomous degraded mode. This converts a host outage from a cliff into a ramp.

**B7 — `.../node/{node}/cmd/ack` (edge → host, QoS 1)**

```json
{"seq": 9042, "rx_ts": 1788412345.502, "accepted": true,
 "applied_duty": 0.62, "clamped": false, "reject_reason": null,
 "edge_mode": "ACTIVE_REMEDIATION", "safety_veto": false}
```

### 2.4 Where the Loop Closes

There are **two nested loops with different periods and different authorities**:

| Loop | Period | Runs on | Authority | Closes at |
|---|---|---|---|---|
| **Inner — safety / regulation** | 100 ms | ESP32 Core 1, `tControl` | **Final.** Can veto, clamp or override any host command. | LEDC duty register |
| **Outer — economic / predictive** | 10 s | Host `mpc_worker` | **Advisory.** Proposes profiles; cannot bypass the inner loop. | Same register, via the inner loop |

The physical control loop therefore closes **entirely at the edge**. The host closes an *economic* loop around it. This is the single most important architectural decision in the document: it makes network quality a performance variable, not a safety variable.

### 2.5 End-to-End Latency Budget

Measured edge-publish timestamp → LEDC register write, LAN-local broker, Wi-Fi 802.11n 2.4 GHz.

| Stage | p50 | p95 | Notes |
|---|---|---|---|
| Index compute + JSON serialize (edge) | 3 ms | 8 ms | cJSON, static buffer |
| MQTT publish + Wi-Fi TX (QoS 0, TLS) | 6 ms | 22 ms | TLS record overhead ~1 ms |
| Broker receive → forward → host recv | 4 ms | 15 ms | Mosquitto, loopback-adjacent LAN |
| Deserialize + Pydantic validate | 2 ms | 6 ms | |
| Feature engineering (1 row, warm cache) | 5 ms | 15 ms | Polars, lag ring in RAM |
| Forecast rollout, N = 30 | 8 ms | 25 ms | XGBoost 30 calls, or NARX 30 fwd passes |
| **MPC solve (warm-started SLSQP)** | **180 ms** | **480 ms** | dominant term; hard-capped at 900 ms |
| Serialize + publish `cmd` (QoS 1) | 8 ms | 30 ms | PUBACK not awaited before local write |
| Edge receive → validate → ack → LEDC write | 5 ms | 16 ms | |
| Actuator mechanical settle (impeller τ) | 250 ms | 600 ms | measured, not assumed — see §9(c) |
| **Total decision latency** | **~221 ms** | **~617 ms** | vs. SC-10 ceiling 1.5 s |
| Total incl. actuator settle + worst-case PMS5003 frame | 1.47 s | 3.52 s | the sensor frame is *not* inside the decision path — see below |

**Budget rationale.** The control interval `T_c = 10 s` is chosen to be ≥ 16× the p95 decision latency (617 ms), ≥ 2.8× the worst-case sensor-to-airflow path (3.52 s) and ≥ 16× the actuator settle time, and ≪ the plant time constant. For a chamber of volume `V` with extraction `Q(u)`, `τ = V/Q`; at the validation scale (`V = 0.06 m³`, `Q ≈ 1.8 m³/h` at full duty) `τ ≈ 120 s`, so `T_c = 10 s ≈ τ/12` — comfortably inside the standard `T_c ∈ [τ/20, τ/5]` band for MPC. Commanding faster than 5 s would chase sensor noise and excite the oscillation mode described in §9(c) without improving the cost function.

---

## 3. Hardware Requirements

### 3.1 Bill of Materials

| # | Component | Key specification | Interface / rail | Role in the system |
|---|---|---|---|---|
| H1 | ESP32 DevKit V1 (WROOM-32) | Dual Xtensa LX6 @ 240 MHz, 520 KB SRAM, 4 MB flash, Wi-Fi 802.11 b/g/n, 16-ch LEDC, 2× SAR ADC | 5 V in (VIN), 3.3 V logic | Edge compute, sensor fusion, index computation, safety FSM, PWM generation |
| H2 | Plantower **PMS5003** *(preferred)* | Laser scattering, PM1.0/2.5/10 + 6 bin counts, 0–500 µg/m³, ≤ 100 mA active, < 200 µA standby, MTTF ≥ 3 yr | UART1 (remapped), 5 V rail, 3.3 V logic | Primary particulate channel — dominant AQI driver |
| H2a | Nova **SDS011** *(alternative)* | Laser, PM2.5/PM10, 70 mA active, < 4 mA sleep, **laser life 8000 h** | UART, 5 V | Alternative particulate channel — **must be duty-cycled**, see §3.2 |
| H3 | Winsen **MH-Z19B** | NDIR CO₂, 0–5000 ppm, ±(50 ppm + 5 %), < 20 mA avg / **150 mA peak**, warm-up 180 s, T90 < 120 s | UART2, 5 V rail, 3.3 V logic | CO₂ / ventilation-adequacy channel and occupancy proxy |
| H4 | **MQ-135** | SnO₂ MOS; NH₃, NOx, benzene, alcohol, smoke; heater ≈ 33 Ω, ~150 mA, ≤ 950 mW | ADC1 via buffer, 5 V heater | **Relative** VOC/NH₃ channel (ratio `Rs/R0`) — *not* a calibrated CO₂ sensor, see Appendix A FX-05 |
| H5 | **MQ-2** | SnO₂ MOS; LPG, propane, H₂, methane, smoke; heater ≈ 31 Ω, ~160 mA | ADC1 via buffer, 5 V heater | Flammable/combustible gas **safety interlock** channel |
| H6 | **DHT22 / AM2302** | −40…80 °C ±0.5 °C, 0–100 % RH ±2–5 %, **min sampling interval 2 s** | 1-Wire, 3.3 V + 4.7 kΩ pull-up | Psychrometry; cross-sensitivity compensation for MQ and PM |
| H7 | 2 W DC motor + centrifugal impeller | 12 V nom., ≈ 167 mA rated, inrush ≈ 0.9 A, brushed | 12 V rail via H8 | Suction actuation — the sole manipulated variable |
| H8 | **IRF520 MOSFET module** *(primary)* | N-ch, 100 V/9.7 A; **V_GS(th) 2.0–4.0 V, R_DS(on) spec'd at V_GS = 10 V** | Low-side switch, 12 V | Low-loss PWM switch — **requires gate driver, see §3.4** |
| H8a | **L298N** *(fallback / bring-up)* | Dual H-bridge, V_IH ≥ 2.3 V (3.3 V-compatible), **~2 V Darlington drop** | 12 V | Bring-up + optional reverse-purge; ~20 % driver loss makes it unsuitable for the final energy claim |
| H9 | TC4420 (or MCP1407) gate driver | 6 A peak, V_DD 4.5–18 V, TTL-compatible input | 12 V rail, input from GPIO18 | Translates 3.3 V logic to a full 12 V gate drive for H8 |
| H10 | ACS712-05B (or INA219 + 0.1 Ω shunt) | ±5 A Hall, 185 mV/A, 2.5 V offset | ADC1 via divider, 5 V | **True energy measurement** `∫V·I dt` for SC-1 |
| H11 | Bus-voltage divider | 10 kΩ / 2.2 kΩ, 1 % | ADC1 (GPIO39) | Measures *actual* applied motor voltage — closes the `duty → V` calibration gap |
| H12 | MCP6002 dual rail-to-rail op-amp | 5 V, unity-gain buffer ×2 | Between MQ AOUT and ADC dividers | Prevents ADC input from loading the MQ load resistor |
| H13 | 1N5822 Schottky + RC snubber | 3 A, 40 V; 100 Ω + 100 nF | Across motor | Inductive flyback + dV/dt snubbing |
| H14 | SSD1306 0.96" OLED *(optional)* | 128×64, I²C 0x3C, ~20 mA | I²C, 3.3 V | Local readout when the host is unreachable |
| H15 | Piezo buzzer + driver transistor | 85 dB @ 10 cm | GPIO33 | Local flammable-gas / interlock alarm |
| H16 | Mains SMPS | **12 V / 2 A (24 W)**, CE/BIS, over-current + over-temp | AC in | Primary supply — see §3.2 justification |
| H17 | Buck converter | 12 V → 5 V, ≥ 3 A, ≥ 88 % efficiency (MP1584 / LM2596 class) | 12 V → 5 V rail | Sensor + MCU rail |
| H18 | UPS module + 3S Li-ion (or 12 V 7 Ah SLA) | ≥ 15 min ride-through at 5 W | 12 V rail | Graceful shutdown, event-log flush, alarm persistence through outage |
| H19 | Bulk + local decoupling | 1000 µF/25 V electrolytic (12 V), 470 µF (5 V), 100 nF per sensor | — | Suppresses PWM-induced rail sag that otherwise corrupts ADC readings |

### 3.2 Power Budget Analysis

**Continuous sensor + logic load (5 V rail):**

| Load | Rail | Avg current | Peak current | Avg power | Duty-cyclable? |
|---|---|---|---|---|---|
| ESP32 DevKit V1 (Wi-Fi active) | 5 V | 80 mA | 240 mA (TX burst) | 0.40 W | No (transport) |
| PMS5003 (continuous mode) | 5 V | 100 mA | 130 mA (fan spin-up) | 0.50 W | **Yes — SET pin** |
| MH-Z19B | 5 V | 20 mA | **150 mA** (IR lamp pulse) | 0.10 W | No (180 s warm-up) |
| MQ-135 heater | 5 V | 150 mA | 150 mA | **0.75 W** | **No** — thermal + baseline drift |
| MQ-2 heater | 5 V | 160 mA | 160 mA | **0.80 W** | **No** — safety channel, must be live |
| DHT22 | 3.3 V | 1.5 mA | 2 mA | 0.005 W | No |
| MCP6002 + dividers | 5 V | 2 mA | 2 mA | 0.01 W | No |
| ACS712 | 5 V | 10 mA | 10 mA | 0.05 W | No |
| OLED (optional) | 3.3 V | 20 mA | 25 mA | 0.07 W | Yes |
| **5 V rail subtotal** | | **≈ 544 mA** | **≈ 870 mA** | **≈ 2.69 W** | |
| Buck loss @ 88 % | | | | +0.37 W | |
| **Reflected to 12 V rail** | 12 V | ≈ 255 mA | ≈ 420 mA | **3.06 W** | |

**Actuator load (12 V rail):**

| Load | Avg (100 % duty) | Inrush / stall | Power |
|---|---|---|---|
| 2 W turbine + IRF520 + driver losses | 175 mA | ~900 mA (≈ 5× rated, ≤ 300 ms) | 2.10 W |

**Totals:** nominal **5.2 W** (≈ 435 mA @ 12 V); worst-case coincident transient **≈ 1.32 A @ 12 V (15.8 W)** for < 300 ms. A **12 V / 2 A (24 W)** SMPS gives 4.6× steady headroom and 1.5× transient headroom — H16 confirmed.

**Why laser particulate sensing forces mains power.** The MQ heater pair alone draws 310 mA continuously and *cannot* be duty-cycled: MOS heaters require 24–48 h burn-in for a stable `R0` and 5–20 min to re-equilibrate from cold, so any power cycling destroys the baseline the ML model depends on (and, for MQ-2, disarms the flammable-gas interlock). Adding the PMS5003's 100 mA fan-and-laser load puts the irreducible continuous draw at ≈ 2.7 W on the 5 V rail *before the motor runs at all.* A 20 000 mAh / 3.7 V power bank stores ≈ 74 Wh, of which ≈ 63 Wh is usable after conversion losses — **≈ 12 h of runtime at nominal load, ≈ 18 h with aggressive PM duty-cycling.** A permanently-installed environmental safety appliance that dies inside one shift is not an appliance. **Decision: mains 12 V/2 A SMPS is the primary source (H16); battery appears only as a 15-minute UPS (H18)** sized to flush the event log, publish a graceful `state` message, sound the local alarm and park the motor safely. This also removes any temptation to duty-cycle the MQ-2 safety channel for power reasons.

**PM sensor duty-cycling policy (life, not power).** PMS5003 is rated MTTF ≥ 3 years continuous, so continuous operation is acceptable. SDS011 (H2a) is rated for **8 000 h of laser life ≈ 11 months continuous** and therefore *must* be duty-cycled if selected. Policy, applied to whichever sensor is fitted:

| Node mode | PM sampling | Effective duty | PMS5003 avg draw |
|---|---|---|---|
| `IDLE_MONITOR` | 30 s on / 150 s off (SET pin low between) | 16.7 % | ≈ 17 mA |
| `ACTIVE_REMEDIATION` | continuous | 100 % | ≈ 100 mA |
| `DEGRADED_AUTONOMOUS` | continuous | 100 % | ≈ 100 mA |

Fan spin-up settling is **≥ 30 s** before a reading is trusted (`valid:false` until then); this is why the idle window is 30 s and not 10 s. Idle-mode PM latency (up to 180 s) is covered by the MQ channels, which stay live and trigger the transition to continuous sampling.

**Motor-driver isolation requirements.** The 12 V motor branch shares no return path with analog ground: motor return goes to the SMPS negative via a dedicated conductor, joining logic ground at a **single star point at the SMPS terminal**. Rationale: at 20 kHz PWM the 0.9 A inrush across even 50 mΩ of shared return produces 45 mV of ground bounce — roughly 56 ADC LSB at 12-bit/3.1 V, which would inject a spurious *correlation between duty cycle and gas readings* directly into the training data and teach the model that running the fan raises pollution. Additional isolation: optional 6N137 optocoupler or Si8610 digital isolator between GPIO18 and the TC4420 input for installations where the motor branch is long or inductive; mandatory 1000 µF bulk capacitance at the driver; H13 flyback + snubber at the motor terminals, not at the board.

### 3.3 GPIO / Pin Allocation Map — ESP32 DevKit V1

Constraints honoured: strapping pins **0, 2, 12, 15** unused; flash pins **6–11** unused; **ADC2 (GPIO 0, 2, 4, 12–15, 25–27) is unusable for analog reads while Wi-Fi is active**, so *every* analog channel is on ADC1 (GPIO 32–39); GPIO 34–39 are input-only and have **no internal pull-ups**.

| GPIO | Peripheral / mode | Connected to | Direction | Notes |
|---|---|---|---|---|
| 4 | GPIO (1-Wire) | DHT22 DATA | bidir | 4.7 kΩ pull-up to 3V3. ADC2 pin used **digitally only** — legal. |
| 16 | UART2 RX | MH-Z19B TX | in | 3.3 V logic out of MH-Z19B — direct connect safe |
| 17 | UART2 TX | MH-Z19B RX | out | 9600 8N1 |
| 25 | UART1 RX (remapped) | PMS5003 TX (pin 5) | in | Default UART1 pins 9/10 are flash — **must remap** |
| 26 | UART1 TX (remapped) | PMS5003 RX (pin 4) | out | Used for passive-mode / sleep commands |
| 27 | GPIO out | PMS5003 SET | out | HIGH = run, LOW = sleep. 10 kΩ pull-up to 3V3 |
| 14 | GPIO out | PMS5003 RESET | out | 10 kΩ pull-up; toggles at boot — pull-up guarantees not-reset |
| 34 | ADC1_CH6 | MQ-135 AOUT → MCP6002 buffer → 10 k/15 k divider | in | Input-only; 100 nF to GND at pin |
| 35 | ADC1_CH7 | MQ-2 AOUT → MCP6002 buffer → 10 k/15 k divider | in | Input-only; 100 nF to GND at pin |
| 36 | ADC1_CH0 (VP) | ACS712 OUT → 10 k/15 k divider | in | Motor current sense |
| 39 | ADC1_CH3 (VN) | 12 V bus → 10 k/2.2 kΩ divider | in | Measured applied voltage for true `V(t)` |
| 32 | ADC1_CH4 | **reserved** | in | Spare analog (3rd gas sensor / NTC on motor body) |
| 18 | LEDC ch0, 11-bit, 20 kHz | TC4420 IN → IRF520 gate | out | **10 kΩ gate-source pull-down mandatory** (boot-float protection) |
| 19 | GPIO in, interrupt | Turbine tachometer (hall / optical) | in | Rising-edge ISR, RPM feedback for actuator-fault detection |
| 21 | I²C SDA | SSD1306 OLED (+ future INA219) | bidir | 4.7 kΩ pull-ups |
| 22 | I²C SCL | as above | out | 400 kHz |
| 23 | GPIO out | Status/heartbeat LED (mode-coded blink) | out | |
| 33 | GPIO out | Buzzer driver (2N3904 + 1 kΩ) | out | Local interlock alarm |
| 13 | GPIO in, `INPUT_PULLUP` | Manual override / acknowledge button | in | Debounced 50 ms in ISR |
| 5 | **unused** | — | — | Outputs PWM at boot; deliberately left free |
| 1 / 3 | UART0 | USB-serial console | — | Reserved for flashing and logs — never repurposed |
| 0, 2, 12, 15 | **unused** | — | — | Strapping pins — per constraint |
| 6–11 | **unused** | — | — | SPI flash — per constraint |

**ADC configuration.** ADC1, attenuation `ADC_ATTEN_DB_11` (≈ 0–3.1 V), 12-bit, **64× hardware oversampling with a 16-sample median pre-filter** to reject PWM switching pickup. Usable linear window restricted to **0.15–2.45 V**; the 10 k/15 k dividers map the 0–5 V sensor swing to 0–3.0 V, so the operating band is set by the sensor scaling to stay inside that window. `esp_adc_cal` two-point + Vref eFuse characterisation applied at boot; without it, the DevKit's ±6 % ADC nonlinearity is indistinguishable from real gas drift.

### 3.4 Signal Isolation & Gate-Drive Notes (3.3 V logic → power stage)

1. **The IRF520 cannot be driven from 3.3 V.** Its `V_GS(th)` is specified as 2.0–4.0 V and `R_DS(on) = 0.27 Ω` is quoted at `V_GS = 10 V`. At 3.3 V a worst-case part is essentially off, and a typical part sits in the linear region dissipating watts. Fix: **TC4420 (H9) powered from the 12 V rail, input driven directly by GPIO18** (TTL-compatible input, `V_IH ≈ 2.4 V`). This yields a 12 V gate swing, `R_DS(on) ≈ 0.2 Ω`, and conduction loss of `I²R = 0.175² × 0.2 ≈ 6 mW` — negligible against the 2 W actuator, which is what makes the energy-saving claim credible.
2. **Boot-state safety.** ESP32 GPIOs are high-impedance from power-on until firmware configures them (tens of ms, longer if the bootloader stalls). A floating TC4420 input is indeterminate. **A 10 kΩ pull-down at the TC4420 input and a 10 kΩ gate-source pull-down at the IRF520 are both mandatory** so the turbine is guaranteed off — never indeterminate — through reset, brown-out and OTA reboot.
3. **Switching frequency.** 20 kHz, above the audible band, chosen so a brushed motor in an occupied bay is silent, and low enough that TC4420 + IRF520 switching losses stay < 1 % (`Q_g ≈ 30 nC → P_sw ≈ Q_g·V_gs·f = 30n × 12 × 20k ≈ 7 mW`). LEDC configured for **11-bit resolution at 20 kHz** (the ESP32 APB-clock limit at that frequency: `log2(80 MHz / 20 kHz) ≈ 11.97`), giving 2048 duty steps — far finer than the actuator or the optimizer can exploit.
4. **Level-shift alternative.** Where TC4420 is unavailable, a BSS138/2N7002 open-drain shifter with a 10 kΩ pull-up to 12 V works but inverts the signal and slows the gate edge to ~2 µs; firmware must then invert the LEDC output (`ledc_channel_config.flags.output_invert = 1`) and the switching-loss figure above must be re-derived. Document whichever is fitted in the build record — a silent inversion here means full duty on boot.
5. **If L298N (H8a) is fitted instead:** ENA ← GPIO18 (PWM), IN1/IN2 ← fixed direction (tie IN1 high, IN2 low via 10 kΩ), and **the `duty → V_motor` map is no longer linear** because of the ~2 V Darlington drop. This is precisely why H11 (bus-voltage sense on GPIO39) exists: the controller must optimise over *measured* applied voltage, never over assumed `V = duty × 12`.
6. **Analog front-end isolation.** MQ AOUT → MCP6002 unity buffer (5 V rail) → 10 k/15 k divider → 100 nF → ADC pin. Buffering *before* the divider keeps the divider from loading the MQ module's on-board load resistor (which would shift `Rs/R0` by ~4 % and be silently absorbed into the model as drift), while the post-buffer divider keeps the ADC source impedance at ≈ 6 kΩ, inside the ESP32 SAR's recommended range.

---

## 4. System Requirements (Software / Non-Functional)

### 4.1 The Index Definition — a prerequisite for every other requirement

Regulatory AQI (both US EPA and India's CPCB NAQI) is defined on **24-hour averaged concentrations**. A 24 h-averaged quantity physically cannot be moved inside a `T_target` of 15 minutes; a controller whose setpoint is a 24 h mean is unstable-by-construction because its measurement lags its actuation by hours. The system therefore uses **two indices with different jobs**:

| Index | Definition | Used for | Not used for |
|---|---|---|---|
| `I_inst` — **Instantaneous Remediation Index** | Sub-indices computed on a **60 s rolling mean** of each channel, aggregated by max operator | **Control.** The MPC setpoint, `I_safe` comparison, all SC-2/3/8/9 metrics | Any regulatory or compliance claim |
| `I_nowcast` — NowCast AQI | EPA NowCast weighting over the trailing 12 h of hourly means | Dashboard reporting, shift logs, comparability with public AQI | Control |

`I_inst = max(SI_PM2.5, SI_PM10, SI_gas, SI_CO2)` with `driver` = the arg-max channel, published in every telemetry frame so that operators and the model can both see *which* pollutant is binding.

**Sub-index breakpoints — CPCB NAQI (default for Indian deployment; piecewise-linear interpolation within each band):**

| AQI band | Category | PM2.5 (µg/m³) | PM10 (µg/m³) | NH₃ (µg/m³) |
|---|---|---|---|---|
| 0–50 | Good | 0–30 | 0–50 | 0–200 |
| 51–100 | Satisfactory | 31–60 | 51–100 | 201–400 |
| 101–200 | Moderate | 61–90 | 101–250 | 401–800 |
| 201–300 | Poor | 91–120 | 251–350 | 801–1200 |
| 301–400 | Very Poor | 121–250 | 351–430 | 1201–1800 |
| 401–500 | Severe | > 250 | > 430 | > 1800 |

**Sub-index breakpoints — US EPA (selectable, for international comparability; PM2.5 per the 2024 revision that lowered the "Good" ceiling from 12.0 to 9.0 µg/m³):**

| AQI band | PM2.5 (µg/m³, 2024) | PM10 (µg/m³, unchanged) |
|---|---|---|
| 0–50 | 0.0–9.0 | 0–54 |
| 51–100 | 9.1–35.4 | 55–154 |
| 101–150 | 35.5–55.4 | 155–254 |
| 151–200 | 55.5–125.4 | 255–354 |
| 201–300 | 125.5–225.4 | 355–424 |
| 301–500 | > 225.4 | > 424 |

**`SI_CO2` and `SI_gas` are project-defined, not regulatory.** Neither CO₂ nor an uncalibrated MOS gas reading is an AQI pollutant. They are mapped onto the same 0–500 scale so the max-operator is meaningful:

- `SI_CO2`: piecewise-linear on ASHRAE 62.1-style ventilation-adequacy anchors — 400 ppm → 0, 800 ppm → 50, 1000 ppm → 100, 1500 ppm → 200, 2500 ppm → 300, 5000 ppm (OSHA PEL) → 500.
- `SI_gas`: computed from the **ratio** `Rs/R0`, not from a fabricated ppm. `SI_gas = 100 · clip((R0/Rs − 1) / (k_alarm − 1), 0, 5)` with `k_alarm` set per-site during commissioning. MQ-135 contributes to `SI_gas`; **MQ-2 does not** — MQ-2 drives the safety interlock (§9) and never the economic objective, because "vent harder" is the wrong response to a flammable atmosphere.

`I_safe` default = **100** (CPCB "Satisfactory" ceiling), site-configurable via the retained `cfg` topic; `T_target` default = **900 s**, range 120–3600 s.

### 4.2 Functional Requirements

| ID | Requirement | Acceptance |
|---|---|---|
| FR-01 | Sample PM at 1 Hz (frame-driven) in active mode, 30 s-on/150 s-off in idle mode | Frame timestamps within ±100 ms of schedule for ≥ 99 % of samples |
| FR-02 | Sample CO₂ at 1 Hz by UART command `0x86`; discard readings for the first 180 s after power-on | `valid:false` asserted throughout warm-up |
| FR-03 | Sample DHT22 at 0.5 Hz (hardware minimum interval is 2 s); on CRC failure, retry once then mark `valid:false` and hold last good for ≤ 30 s | No sample interval < 2.0 s in a 24 h trace |
| FR-04 | Sample both MQ channels at 10 Hz with 64× oversampling, publish the 1 Hz decimated median as both raw counts and `Rs/R0` ratio | Ratio computed with the live `R0`, never a compile-time constant |
| FR-05 | **Disable MH-Z19B ABC self-calibration at every boot** (`FF 01 79 00 00 00 00 00 86`) and verify the acknowledgement | ABC-off confirmed in the boot `state` message |
| FR-06 | Compute `I_inst` at 1 Hz from 60 s rolling means, emitting the binding `driver` channel | Matches an offline reference implementation to within 0.1 index points on a replayed trace |
| FR-07 | Publish `telemetry` at 1 Hz active / 0.2 Hz idle; `state` on change plus a 60 s heartbeat; `event` on every mode transition, interlock, sensor fault and command rejection | Topic/QoS matrix of §2.2 verified by broker-side capture |
| FR-08 | Accept `cmd` only when: `seq` strictly increases, `now < valid_until`, every profile element ∈ [0, 1], `\|Δduty\| ≤ slew_max`, and the payload validates against the schema. Reject otherwise with a reason code | 100 % of a 12-case malformed-command suite rejected, node stays in a defined mode |
| FR-09 | Acknowledge every command on `cmd/ack` within 50 ms of receipt, reporting the duty **actually applied** after clamping and safety veto | p99 ack latency ≤ 50 ms |
| FR-10 | Apply duty through LEDC ch0, 11-bit @ 20 kHz, with a slew limiter of 0.10 duty per 100 ms tick and a 400 ms 100 %-duty kick-start on any 0 → non-zero transition | Oscilloscope capture of the kick-start and of the slew ramp |
| FR-11 | Enforce `d_min = 0.25`: any commanded duty in (0, 0.25) is realised by bang-bang dithering **within** the 10 s control interval, preserving mean airflow | Mean measured RPM tracks the commanded mean to within 10 % |
| FR-12 | Maintain a 600-sample (10 min) RAM ring buffer of telemetry; on broker reconnect, replay gaps to `.../telemetry/backfill` at ≤ 20 Hz until drained | Zero data loss across a 5-minute induced broker outage |
| FR-13 | Auto-recalibrate MQ `R0` during certified clean windows (`I_inst < 40`, RH 30–70 %, stable ≥ 30 min, ≥ 6 h since last update) using an EWMA with `α = 0.02` | `R0` update events logged with before/after values |
| FR-14 | Invalidate PM readings when RH > 90 % (hygroscopic growth) and flag `pm.valid = false`; the index then falls back to remaining channels | Verified in a humidity-chamber test |
| FR-15 | Persist `cfg`, `R0` values, last mode and cumulative energy counters to NVS; restore on boot | Values survive 20 power cycles |
| FR-16 | Support OTA firmware update over HTTPS with rollback on failed self-test, permitted only in `IDLE_MONITOR` | Rollback verified with a deliberately broken image |

### 4.3 Edge Mode State Machine

| Mode | Entry condition | PM duty | Motor behaviour | Exit |
|---|---|---|---|---|
| `BOOT` | power-on / reset | off | **0 % (guaranteed by pull-downs)** | POST passes |
| `WARMUP` | after `BOOT` | continuous | 0 % | 180 s elapsed **and** MH-Z19B valid **and** MQ heaters ≥ 20 min since cold start |
| `IDLE_MONITOR` | `I_inst < I_safe − hysteresis(10)` for 60 s | 16.7 % | 0 % | `I_inst ≥ I_safe` for 10 s → `ACTIVE_REMEDIATION` |
| `ACTIVE_REMEDIATION` | index breach, host reachable | continuous | follows host profile, clamped by inner loop | index clear + host says done → `IDLE_MONITOR`; host silent → `DEGRADED_AUTONOMOUS` |
| `DEGRADED_AUTONOMOUS` | no valid command for `3 × T_c` (30 s) **or** past `valid_until` | continuous | **local rule-based ladder (§9e)** | valid command received → `ACTIVE_REMEDIATION` |
| `INTERLOCK_FLAMMABLE` | MQ-2 `Rs/R0` below flammable threshold | continuous | **motor commanded to 0 %, buzzer on, `event` published** | manual button acknowledgement **and** 10 min below threshold |
| `SENSOR_FAULT` | primary index channel invalid > 120 s | continuous | conservative fixed 60 % | sensor recovers |
| `SAFE_SHUTDOWN` | UPS on battery, < 3 min reserve | off | 0 % after 10 s ramp-down | mains restored |

`INTERLOCK_FLAMMABLE` and `SAFE_SHUTDOWN` **outrank every host command unconditionally.** No `cmd` payload can clear them.

### 4.4 Non-Functional Requirements

| ID | Category | Requirement | Verification |
|---|---|---|---|
| NFR-01 | Latency | p95 decision latency ≤ 1.5 s; p99 ≤ 2.5 s; inner safety loop jitter ≤ 5 ms | Tracing with edge-stamped `seq` correlation |
| NFR-02 | Latency | MPC solve hard-capped at 900 ms; on timeout, publish the best feasible iterate, never nothing | Timeout injection test |
| NFR-03 | Fault tolerance — broker | Exponential-backoff reconnect (1→2→4→8→30 s cap), `clean_session = false`, keep-alive 10 s, retained LWT | 100 broker restarts, no manual intervention |
| NFR-04 | Fault tolerance — host | Host loss ⇒ `DEGRADED_AUTONOMOUS` within 30 s; no unsafe or "stuck-on" state at any point | FI-03 |
| NFR-05 | Fault tolerance — sensor | Any single sensor dropout degrades the index gracefully (max over surviving channels) with a health flag; **PM + CO₂ both down ⇒ `SENSOR_FAULT`** | FI-05, FI-06 |
| NFR-06 | Fault tolerance — Wi-Fi | Ring buffer + backfill covers outages up to 10 min with zero loss; longer outages lose the oldest data first and log a gap event | FI-02 |
| NFR-07 | Data retention | Raw 1 Hz: 90 days · 1-min continuous aggregate: 2 years · 1-hour aggregate: 5 years · commands/events/optimizer runs: 5 years (audit) · model artefacts + training manifests: indefinite | Retention policy applied and verified in TimescaleDB |
| NFR-08 | Security — transport | TLS 1.2 on 8883 with a private CA; anonymous access disabled; per-node username/password minimum, X.509 client certificates preferred; CA pinned in NVS | `mosquitto_sub` without credentials must fail |
| NFR-09 | Security — authorisation | Per-node ACL: a node may publish only under its own `.../node/{id}/#` and subscribe only to its own `cmd` and `cfg` | Cross-node publish attempt rejected |
| NFR-10 | Security — device | Flash encryption + secure boot v2 enabled on production units; UART console disabled in release builds; credentials in NVS, never in source | Firmware build flags audited |
| NFR-11 | Security — host | Broker bound to the LAN interface only, no port-forwarding; TimescaleDB and FastAPI behind local auth; no inbound internet exposure | `nmap` from outside the LAN |
| NFR-12 | Scalability | Topic tree namespaced `site/zone/node`; one optimizer worker per zone in a process pool. **A 4-core laptop sustains ≈ 30 zones at `T_c = 10 s`** (30 × 480 ms p95 ÷ 4 workers ≈ 3.6 s < 10 s). Beyond that: stagger `T_c` per zone, or move to IPOPT (§8.2) for a ~5× solve-time reduction | Load test with simulated nodes |
| NFR-13 | Scalability — transport | Mosquitto handles ≥ 500 msg/s on the target laptop; 30 nodes at 1 Hz = 30 msg/s, i.e. 6 % of budget | Broker metrics under load |
| NFR-14 | Observability | Every command traceable end-to-end via `seq`; every decision stores its `model_id`, solver status, iteration count and predicted trajectory in `optim_runs` | Random audit of 20 events reconstructable |
| NFR-15 | Reproducibility | Every model in the registry pinned to a data snapshot hash, feature-spec version and hyper-parameter set | `mlflow` run reproduces metrics to ±1 % |
| NFR-16 | Maintainability | Payload schemas versioned in the topic path (`aqr/v1/...`); edge rejects a `cmd` whose `schema_v` it does not implement | Cross-version compatibility test |
| NFR-17 | Safety | The flammable-gas interlock path (MQ-2 → motor off + alarm) is implemented **entirely in the inner 100 ms loop** and has no dependency on Wi-Fi, MQTT, the host or the index computation | Code review + FI-08 |

---

## 5. Tech Stack

### 5.1 Firmware

**Choice: ESP-IDF v5.x as the build system, with `arduino-esp32` v3.x compiled in as a component.**

Justification. Pure Arduino gives fast access to mature sensor libraries (`PMS`, `MHZ19`, `DHT`) but hides the FreeRTOS scheduler, offers no first-class task-watchdog control, and its `analogRead` wrapper obscures the ADC1/ADC2 distinction that §3.3 depends on. Pure ESP-IDF gives full control but forces hand-rolled sensor drivers, which is wasted effort on a project whose novelty is the control loop, not UART framing. **Arduino-as-an-IDF-component** is the documented, supported middle path: `idf.py menuconfig` for partition tables, flash encryption, secure boot, `esp_task_wdt` and LEDC configuration; Arduino APIs where a library already solves the problem. This also keeps the door open to strip Arduino out later without restructuring the build.

**RTOS task structure and core pinning.** The Wi-Fi/LwIP stack occupies PRO_CPU (core 0); anything time-critical is therefore pinned to APP_CPU (core 1) so a Wi-Fi retransmission storm cannot delay the safety loop. FreeRTOS priorities: higher number = higher priority.

| Task | Core | Prio | Period | Stack | Responsibility |
|---|---|---|---|---|---|
| `tControl` | 1 | 6 | 100 ms | 4 KB | Safety FSM, interlocks, slew limiter, dither, LEDC write. **Never blocks on I/O.** |
| `tSensePM` | 1 | 4 | frame-driven | 3 KB | UART1 PMS5003 framing, checksum, SET/RESET, spin-up gating |
| `tSenseCO2` | 1 | 4 | 1 s | 3 KB | UART2 MH-Z19B poll, checksum, ABC-disable at boot |
| `tSenseGas` | 1 | 4 | 100 ms | 3 KB | ADC1 oversampling, median filter, `Rs/R0`, `R0` EWMA recalibration |
| `tSenseTH` | 1 | 3 | 2 s | 3 KB | DHT22 bit-bang read with interrupts masked, CRC |
| `tFusion` | 1 | 3 | 1 s | 5 KB | Rolling means, RH/T compensation, `I_inst`, ring-buffer insert |
| `tNet` | 0 | 3 | event | 8 KB | MQTT client, TLS, publish queue drain, backfill replay |
| `tCmd` | 0 | 5 | event | 4 KB | Command deserialize, schema/bounds/seq/expiry validation, ack, hand-off to `tControl` via queue |
| `tHouse` | 0 | 1 | 10 s | 3 KB | SNTP, NVS commits, OTA check, heap/health telemetry |

Inter-task: `xQueueSensor` (depth 16, sensor → fusion), `xQueueCmd` (depth 4, `tCmd` → `tControl`), `xQueueTx` (depth 64, publish backlog). `tControl` reads the latest command from a **double-buffered struct guarded by a critical section**, never from a blocking queue receive, so a stalled network task can never stall actuation. Hardware task watchdog (`esp_task_wdt`) subscribes `tControl` and `tFusion` with a 3 s timeout; a watchdog reset lands in `BOOT` with the motor off by hardware pull-down.

### 5.2 Transport

**Broker: Eclipse Mosquitto 2.x.** Chosen over EMQX for the capstone footprint: single binary, ~5 MB RSS, file-based ACLs, trivially reproducible config, and it saturates far beyond the 30 msg/s the system produces. **EMQX becomes the right answer above ~200 nodes or when clustering / built-in rule-engine / MQTT 5 shared subscriptions are needed** — the topic schema below is deliberately EMQX-compatible so the migration is a config change, not a redesign. Deployment note per §9(d): the broker runs on a **separate always-on host (Raspberry Pi 4 or equivalent), not on the optimizer laptop**, so that closing the laptop lid does not take out the transport as well as the compute.

**Topic schema.**

```
aqr/v1/site/{site_id}/zone/{zone_id}/node/{node_id}/telemetry        ← QoS0, 1 Hz
aqr/v1/site/{site_id}/zone/{zone_id}/node/{node_id}/telemetry/backfill ← QoS1, burst
aqr/v1/site/{site_id}/zone/{zone_id}/node/{node_id}/event            ← QoS1
aqr/v1/site/{site_id}/zone/{zone_id}/node/{node_id}/state            ← QoS1, RETAINED
aqr/v1/site/{site_id}/zone/{zone_id}/node/{node_id}/lwt              ← QoS1, RETAINED
aqr/v1/site/{site_id}/zone/{zone_id}/node/{node_id}/cmd              → QoS1
aqr/v1/site/{site_id}/zone/{zone_id}/node/{node_id}/cmd/ack          ← QoS1
aqr/v1/site/{site_id}/zone/{zone_id}/node/{node_id}/cfg              → QoS1, RETAINED
aqr/v1/site/{site_id}/zone/{zone_id}/host/status                     → QoS1, RETAINED
```

Design rules: version in the path (`v1`) so schema evolution is a subscription change; `zone` above `node` so multi-node rooms aggregate naturally and one optimizer instance can subscribe `.../zone/{z}/node/+/telemetry`; retained topics only for *state*, never for *commands* (a retained command would be re-delivered and re-executed on every reconnect — a classic and dangerous mistake). Payload encoding is **JSON** at ~280 B/frame; at 30 nodes × 1 Hz this is 8.4 kB/s, ~0.07 % of a 100 Mbit LAN, so the debuggability of JSON is worth far more than the bytes CBOR would save. Revisit only if the deployment goes to LoRa/NB-IoT.

### 5.3 Host Backend

| Concern | Choice | Justification |
|---|---|---|
| Language | Python 3.11 | The optimizer, the ML stack and the scientific tooling all live here; nothing in the host path is latency-critical below ~100 ms |
| MQTT client | `aiomqtt` (asyncio wrapper over paho) | Non-blocking; the ingest loop must not stall behind a solve |
| API / service | **FastAPI + Uvicorn** | Async-native, Pydantic schema validation reused for MQTT payload validation, automatic OpenAPI for the console, native WebSocket for B10 |
| Schema validation | Pydantic v2 | One schema definition shared by MQTT ingest, REST and the DB writer |
| Dataframes | **Polars** | Lazy + Arrow-backed; the lag/rolling feature block runs ~5–10× faster than pandas, which matters because it sits inside the 10 s loop |
| Gradient boosting | XGBoost 2.x | `monotone_constraints` support is mandatory (§8.1) |
| Neural dynamics | PyTorch 2.x | NARX is a ~2 k-parameter MLP; PyTorch is chosen over TensorFlow for `torch.func.jacrev`, which gives exact gradients to the optimizer |
| Optimization | **SciPy `minimize(method='SLSQP')` → CasADi + IPOPT** | Phase 1 / Phase 2, see §8.2 |
| Experiment tracking | MLflow | Model registry, data-snapshot hashes, promotion gates |
| Process model | One `uvicorn` app + a `ProcessPoolExecutor` of MPC workers, one per zone | Keeps a slow solve off the ingest loop; NFR-12 |
| Scheduling | APScheduler | Nightly retrain, weekly excitation campaign, drift reports |
| Simulation | A plant simulator implementing the same mass-balance ODE with injected noise and actuator lag | Enables software-in-the-loop CI so controller changes are regression-tested without hardware |

### 5.4 Data Storage

**Choice: TimescaleDB (PostgreSQL 16 + Timescale 2.x), with InfluxDB rejected.** InfluxDB is the more common IoT default and is marginally simpler to point Grafana at, but this system's core workload is *relational joins between telemetry and non-telemetry entities* — "give me every 1 Hz row for the 87 remediation events labelled `solvent_spill` that were controlled by `model_id = narx_gb_v7`, excluding events where any sensor health flag was set." That is a three-way join and a window function; it is one page of SQL in Timescale and an unpleasant Flux exercise in Influx. Since the ML pipeline, not the dashboard, is the primary consumer, full SQL wins.

**Schema.**

```sql
-- 1 Hz raw telemetry, hypertable, 1-day chunks
telemetry(ts TIMESTAMPTZ, node_id TEXT, zone_id TEXT, seq BIGINT,
          pm1 REAL, pm25 REAL, pm10 REAL, n03 INT, pm_valid BOOL,
          co2_ppm REAL, co2_valid BOOL,
          mq135_raw INT, mq135_ratio REAL, mq2_raw INT, mq2_ratio REAL,
          temp_c REAL, rh_pct REAL, th_valid BOOL,
          duty REAL, v_bus REAL, i_motor REAL, rpm INT,
          i_inst REAL, i_driver TEXT, i_nowcast REAL,
          mode TEXT, cmd_seq BIGINT)
  → hypertable(ts), chunk 1 day, compress after 7 days
     (segmentby node_id, orderby ts DESC), retention 90 days

events(ts, node_id, kind, severity, detail JSONB, ack_ts)
commands(seq, node_id, issued_ts, valid_until, profile REAL[],
         predicted_idx REAL[], model_id, solver_status, solve_ms,
         ack_ts, applied_duty, accepted, reject_reason)
episodes(episode_id, node_id, t_start, t_end, label, i_peak, i_start,
         t_reach, energy_wh_true, energy_v2_proxy, controller,
         baseline_ref, notes)          -- the ML unit of observation
model_registry(model_id, family, trained_at, data_snapshot_sha,
               feature_spec_v, hparams JSONB, metrics JSONB, promoted BOOL)
r0_history(ts, node_id, sensor, r0_before, r0_after, trigger)
```

Continuous aggregates: `telemetry_1min` (mean/min/max/stddev per channel, 2 y retention) and `telemetry_1hour` (5 y). Compression achieves roughly 8–15× on this column mix, so 90 days of 1 Hz raw data lands at **≈ 150–250 MB per node** — trivial for a laptop.

### 5.5 Dashboard

Two surfaces, because they answer different questions and one tool does both badly:

1. **Grafana OSS** — infrastructure and operations. Node health, telemetry completeness, broker connection state, `R0` drift trend, heap, RSSI, retention/compression status, alert rules. Direct Postgres data source; zero custom code.
2. **VENTIS Console** — a purpose-built **React + Vite + TypeScript** page served by FastAPI, live over WebSocket at 2 Hz, using **uPlot** for the dense time series (it renders 10 k points at 60 fps where most chart libraries stall) and Recharts for the summary cards. This exists because the control narrative cannot be expressed in Grafana panels: it must overlay the **predicted index trajectory against the realised one**, draw the **planned duty profile as a step-plot into the future**, mark `T_target` and `I_safe` as constraint lines, and run a **live energy-saved counter** comparing measured Wh against the counterfactual always-on baseline for the same event. That overlay of prediction-vs-outcome is the demonstration of the entire thesis, and it is the one view a Grafana panel cannot produce.

---

## 6. Workflow (End-to-End)

### 6.1 Steady-State Control Cycle

```
T+0.000  [EDGE  tSensePM ]  PMS5003 frame arrives on UART1. Checksum verified.
                            Spin-up gate checked (≥30 s since SET high) → pm.valid=true.
T+0.001  [EDGE  tSenseCO2]  Latest MH-Z19B poll response (0x86) decoded, ABC-off asserted at boot.
T+0.002  [EDGE  tSenseGas]  ADC1 ch6/ch7: 64× oversample → 16-sample median → Rs/R0 ratio,
                            using the live R0 (EWMA-updated, FR-13), with T/RH compensation applied.
T+0.003  [EDGE  tFusion  ]  60 s rolling means updated. Sub-indices computed (§4.1).
                            I_inst = max(SI_pm25, SI_pm10, SI_gas, SI_co2); driver channel recorded.
                            Row appended to the 600-sample ring buffer.
T+0.006  [EDGE  tControl ]  100 ms tick — INDEPENDENT of the host:
                              1. Flammable interlock check (MQ-2). If tripped → motor 0 %, buzzer,
                                 event, mode=INTERLOCK_FLAMMABLE. Loop ends here.
                              2. Command freshness: now < valid_until? seq fresh?
                                 If not → mode=DEGRADED_AUTONOMOUS, run the local ladder (§9e).
                              3. Take profile[k] for the current interval; clamp to [0,1];
                                 apply slew limit 0.10/tick; apply d_min dither; kick-start if 0→on.
                              4. Write LEDC duty register.
T+0.009  [EDGE  tNet     ]  telemetry JSON serialized (~280 B), published QoS0 over TLS.
T+0.015  [BROKER         ]  Mosquitto routes to the subscribed host session.
T+0.019  [HOST  ingest   ]  aiomqtt callback → Pydantic validation → reject-and-log on failure.
                            Sequence-gap detector fires a backfill request if seq jumped.
T+0.021  [HOST  features ]  Polars ring: lags {1,5,10,30,60,120 s}, first/second differences,
                            EWMA (τ = 15/60/300 s), cumulative exposure ∫(I−I_safe)+dt,
                            RH/T-compensated gas ratios, actuator state history, time-of-day.
T+0.026  [HOST  gate     ]  Is this a control tick? (every 10 s). If not → persist and stop.
                            If yes and I_inst ≥ I_safe → proceed. Else publish a hold command.
T+0.026  [HOST  forecast ]  Grey-box rollout: for a candidate profile u_{0..N−1},
                              C_{k+1} = C_k + Δt·[ −(Q(u_k)/V + k_dep)·C_k + S_k/V ] + g_θ(x_k, u_k)
                            where Q(u) is the calibrated fan curve and g_θ is the learned residual
                            (NARX primary / XGBoost fallback). N = 30 steps × 10 s = 300 s horizon.
T+0.034  [HOST  mpc      ]  minimize   Σ_k (V_bus·u_k)² · Δt          [ ∝ ∫V² dt ]
                                        + λ_slew Σ_k (u_k − u_{k−1})²
                                        + λ_slack · s²
                            s.t.  0 ≤ u_k ≤ 1
                                  |u_k − u_{k−1}| ≤ 0.10
                                  Î(T_target) ≤ I_safe + s ,  s ≥ 0   [soft terminal]
                                  u_0 within slew of the currently applied duty
                            Warm-started from the previous solution shifted by one step.
                            Hard wall-clock cap 900 ms → return best feasible iterate (NFR-02).
T+0.214  [HOST  dispatch ]  cmd JSON built: seq++, valid_until = now + 30 s, profile = u_{0..5},
                            predicted_idx, model_id, solver status. Published QoS1.
                            Row written to `commands`; solve telemetry written to `optim_runs`.
T+0.222  [EDGE  tCmd     ]  Validate: schema_v, seq strictly increasing, not expired, bounds,
                            slew feasibility. Accept → double-buffered struct. Reject → reason code.
T+0.227  [EDGE  tCmd     ]  cmd/ack published QoS1 with the duty ACTUALLY applied post-clamp.
T+0.232  [EDGE  tControl ]  Next 100 ms tick consumes the new profile. LEDC duty updated.
T+0.482  [PLANT          ]  Impeller reaches the new operating point (τ_mech ≈ 250 ms measured).
T+1.000  [EDGE           ]  Next sensor frame observes the effect. ── LOOP CLOSED ──
T+0.03…  [HOST  persist  ]  1 s micro-batch COPY into TimescaleDB (async, off the control path).
T+0.50   [HOST  ws       ]  Console pushed at 2 Hz: realised vs predicted trajectory, planned
                            profile, live Wh vs. counterfactual always-on baseline.
```

**Episode bookkeeping.** A row is opened in `episodes` when `I_inst` first crosses `I_safe` and closed 120 s after it falls back below with the motor at zero. The episode is the unit of ML observation, of evaluation and of the energy claim — never the individual 1 Hz row (§7.4).

### 6.2 Model Refresh Workflow

Three timescales, deliberately separated:

| Cadence | Mechanism | What updates | Guardrail |
|---|---|---|---|
| **Continuous (online)** | Recursive-least-squares on the grey-box parameters `k_dep` and `S` only | Slow physical drift — filter loading, duct fouling, changed source rate | Parameters clipped to physically plausible ranges; a rejected update raises a health event |
| **Nightly (batch)** | 02:00 local. Rebuild features over a rolling 30-day window; retrain the residual model; evaluate on the last 7 days of held-out **episodes** | The learned residual `g_θ` (NARX weights / XGBoost trees) | **Shadow-first**: the candidate runs alongside production for 24 h, scoring but not commanding. Promotion requires ≥ 3 % RMSE improvement *and* no regression on SC-2/SC-8 |
| **Weekly (excitation)** | Sunday 03:00, or the lowest-occupancy window. A scripted **APRBS campaign**: 45 min of multi-level pseudo-random duty steps under a synthetic or naturally-elevated load, with the safety FSM fully armed | Refreshes the *identification* dataset | Aborted immediately on any interlock or on `I_inst` exceeding 1.5 × the pre-campaign peak |

**Why the weekly excitation campaign is not optional.** Once the controller is closed-loop, the duty cycle becomes a deterministic function of the measured index. The exogenous input loses persistent excitation, the input–output data becomes confounded by feedback, and the identified model degrades toward the closed-loop transfer function rather than the plant — the classic closed-loop identification trap. Two countermeasures run permanently: (i) the scheduled APRBS campaign above, and (ii) a small **ε-exploration dither** — with probability 0.05 per control interval, `u_0` is perturbed by `±U(0.03, 0.08)` before dispatch, clipped to remain feasible, and the perturbation is flagged in `commands.reason` so it can be used as a natural experiment and excluded from performance accounting.

**Drift-triggered retrain.** Any of the following escalates the nightly batch to an immediate retrain and raises an operator event: Page-Hinkley change detection on the one-step forecast residual; PSI > 0.25 on any input feature over 7 days vs. the training window; `R0` moving > 20 % from its commissioning value; or SC-2 deadline compliance dropping below 90 % over a 50-episode rolling window.

---

## 7. Dataset & Required Metrics

### 7.1 Raw Fields (logged at 1 Hz; every field carries a validity flag)

| Field | Unit | Source | Why it is required |
|---|---|---|---|
| `ts` (NTP-disciplined, µs) | s | ESP32 SNTP | Lagged features are meaningless without a stable time base; jitter corrupts `dC/dt` |
| `seq` | — | ESP32 monotonic | Gap detection, backfill, end-to-end tracing |
| `pm1`, `pm25`, `pm10` | µg/m³ | PMS5003 | Primary index drivers and primary model targets |
| `n03` (0.3 µm bin count) | #/0.1 L | PMS5003 | Raw count is less processed than the mass estimate; better early-warning signal |
| `co2_ppm` | ppm | MH-Z19B | Ventilation adequacy, occupancy proxy, **and an independent tracer of the air-exchange rate** — the cleanest observable for identifying `Q(u)` |
| `mq135_raw`, `mq135_ratio` | counts, `Rs/R0` | MQ-135 | VOC/NH₃ relative channel. **Both** stored: `raw` for post-hoc recalibration, `ratio` for modelling |
| `mq2_raw`, `mq2_ratio` | counts, `Rs/R0` | MQ-2 | Flammable-gas interlock; audit trail for every trip |
| `temp_c`, `rh_pct` | °C, % | DHT22 | MOS gas sensors are strongly T/RH-dependent; PM optics are RH-dependent above ~70 % |
| `duty` | 0–1 | Edge, as applied | The manipulated variable. Must be the **applied** value, post-clamp, not the commanded one |
| `v_bus` | V | GPIO39 divider | Actual applied voltage — the driver drop and rail sag make `V = duty × 12` false |
| `i_motor` | A | ACS712 | With `v_bus`, yields **true instantaneous power** — the basis of the SC-1 claim |
| `rpm` | rev/min | Tachometer | Detects a stalled, obstructed or failing impeller that duty alone cannot reveal |
| `i_inst`, `i_driver`, `i_nowcast` | index, enum, index | Edge | Control variable, binding channel, reporting variable |
| `mode` | enum | Edge FSM | Rows from `INTERLOCK`/`SENSOR_FAULT` modes must be excluded from training |
| `cmd_seq` | — | Edge | Joins each telemetry row to the decision that produced it |
| `rssi`, `heap_free`, `up` | dBm, B, s | ESP32 | Distinguishes a network artefact from a physical one during post-mortems |

**Volume.** 24 numeric fields at 1 Hz ≈ 280 B/s on the wire ⇒ **≈ 24 MB/day/node** raw JSON, ≈ 1.7–3 MB/day/node after TimescaleDB compression, ⇒ ≈ 150–250 MB/node for the 90-day raw retention window. Negligible; retention is set by usefulness, not by disk.

### 7.2 Derived Features for the Forecaster

| Group | Features | Rationale |
|---|---|---|
| Lags | `x(t−k)` for `k ∈ {1, 5, 10, 30, 60, 120} s` on `pm25`, `co2`, `i_inst`, `duty` | NARX is by definition a lagged-input model; boosting needs the lags supplied explicitly |
| Rates | First difference `Δx/Δt` and second difference over 5 s and 30 s windows | **The decay rate, not the level, carries the plant dynamics** — the single most informative feature family |
| Smoothing | EWMA at `τ = 15 / 60 / 300 s`; rolling std over 60 s | Separates a genuine trend from sensor noise; the rolling std flags fan-induced turbulence |
| Cumulative exposure | `∫ max(I_inst − I_safe, 0) dt` since episode start | Turns the deadline constraint into a state the model can see; also the compliance-reporting quantity |
| Actuator history | `duty` lags, `Σ duty·Δt` this episode, time since last change, cumulative Wh | Encodes actuator memory and impeller spin-up state |
| Physical terms | `Q(u)/V` from the calibrated fan curve; `τ̂ = V/Q(u)`; `C·Q(u)/V` (the removal term itself) | Feeding the *physics* as a feature is what lets a tree model represent an exponential decay it could never extrapolate |
| Cross-sensitivity corrections | `mq_ratio_corr = mq_ratio / f(T, RH)` using the datasheet correction surface; `pm25_rh_corr = pm25 / (1 + κ·(RH/(100−RH)))` (κ-Köhler hygroscopic growth) | Without these, the model learns "humidity causes pollution" and the controller vents on weather |
| Deltas vs. baseline | Each channel minus its trailing 24 h 5th-percentile | Removes slow baseline drift (`R0` ageing, seasonal ambient) from the fast dynamics |
| Context | Hour-of-day (sin/cos), day-of-week, `mode`, minutes since episode start, `T_target − t_elapsed` | Ambient AQI and source activity are strongly diurnal; remaining-time is a genuine input to the control problem |

**Two features are forbidden.** (1) Anything derived from the *future* of the target (trivial leakage). (2) The **commanded** duty — only the *applied* duty may be used, or the model learns the controller instead of the plant.

### 7.3 Data Collection Plan — how to bootstrap a model with no data

Chicken-and-egg: the optimizer needs a model, and the model needs data generated across a wide range of duty cycles that only an optimizer would produce. Resolved with a **three-phase open-loop-first campaign**, executed in a **sealed 60 L acrylic validation chamber** with a controlled aerosol/CO₂ injection port, a HEPA-filtered make-up-air inlet and a reference-grade co-located sensor where available.

| Phase | Duration | What runs | What it produces |
|---|---|---|---|
| **A — Passive baseline** | 14 days continuous, motor never energised | Sensors only | Noise floor per channel, `R0` commissioning values, ambient diurnal baseline, sensor-health statistics. **Nothing is trainable yet; this phase exists so drift detection has a reference.** |
| **B — Open-loop identification** | ~15 working days (~120 h) automated cycling | Scripted inject → decay under a *prescribed* duty; a rule-based supervisor guards the chamber | The training corpus. Full DoE below. |
| **C — Closed-loop with exploration** | ongoing | MPC in command, with 5 % ε-dither + weekly APRBS | Continuous refresh; keeps the model valid as the plant ages |

**Phase-B design of experiments.**

| Factor | Levels | Values |
|---|---|---|
| Commanded duty | 8 | 0.00, 0.25, 0.35, 0.45, 0.60, 0.75, 0.90, 1.00 |
| Duty pattern | 3 | constant · single step-down · APRBS |
| Initial index band | 3 | 120–180 · 180–280 · 280–400 |
| RH band | 3 | 30–45 % · 45–65 % · 65–80 % |
| Replicates | ≥ 4 | randomised order, to break confounding with time-of-day |

Full factorial is 288 cells, which is unnecessary: a **D-optimal subset of ~120 runs plus 40 randomised replicates ≈ 160 episodes** covers the response surface. At ≈ 10 min decay + ≈ 15 min re-load and re-equilibration, that is ≈ 67 h of chamber time, comfortably inside the 15-day window with a 24/7 automated cycler.

**Trainability thresholds.**

| Model | Minimum viable | Recommended | Note |
|---|---|---|---|
| Grey-box parameters (`Q(u)`, `k_dep`, `S`) | **8 episodes** — one per duty level | 24 | Three physical parameters; fits by least squares |
| XGBoost residual | **60 episodes** (~ 36 000 one-step rows) | 160 | Trees tolerate small data; the binding limit is *episode* count, not row count |
| NARX residual | **100 episodes** | 200+ | ~2 k parameters; below this it memorises episodes |

> **The sample-size number that matters is 160 episodes, not 500 000 rows.** Consecutive 1 Hz samples within one decay are almost perfectly autocorrelated; their effective sample size is close to *one per episode*, not one per row. Any claim of "half a million training samples" is a claim about disk usage, not about statistical power.

### 7.4 Evaluation Metrics

**Validation protocol (non-negotiable).** Splits are made **by episode, grouped and time-ordered** — `GroupKFold` on `episode_id`, plus a final strictly-forward holdout of the most recent 20 % of episodes. Random row-wise splits leak the target through autocorrelation and will report an R² of ~0.99 for a model that is worthless in the loop. Every reported metric names its split.

*Forecasting metrics — evaluated on multi-step rollouts, not one-step-ahead:*

| Metric | Definition | Target | Floor |
|---|---|---|---|
| RMSE @ 60 s | Index-point error, 6-step recursive rollout | ≤ 3.0 | ≤ 5.0 |
| RMSE @ 300 s | Index-point error, 30-step recursive rollout | ≤ 8.0 | ≤ 12.0 |
| MAE @ 300 s | Robust companion to the above | ≤ 6.0 | ≤ 9.0 |
| sMAPE @ 300 s | Scale-free comparison across index bands | ≤ 12 % | ≤ 20 % |
| Time-to-target error | \|predicted t_reach − actual t_reach\| | ≤ 25 s | ≤ 60 s |
| Directional accuracy | Sign of the predicted index change matches reality | ≥ 95 % | ≥ 90 % |
| Skill score | `1 − RMSE_model / RMSE_persistence` at 300 s | ≥ 0.60 | ≥ 0.40 |
| **Monotonicity audit** | Fraction of sampled `(state, u)` pairs where `∂Î/∂u > 0` | **0 %** | **0 %** |
| Calibration | 90 % prediction-interval empirical coverage | 88–92 % | 85–95 % |

*Controller metrics:*

| Metric | Definition | Target | Floor |
|---|---|---|---|
| Energy saving (true) | `1 − Σ∫V·I dt_MPC / Σ∫V·I dt_baseline`, matched episodes | ≥ 40 % | ≥ 25 % |
| Energy saving (proxy) | Same on `∫V² dt`, reported for objective-consistency only | report | — |
| Optimality gap | `E_MPC / E_oracle`, where the oracle re-optimises offline with perfect hindsight | ≤ 1.20 | ≤ 1.40 |
| Deadline compliance | `I_inst(T_target) ≤ I_safe` | ≥ 95 % | ≥ 90 % |
| Time-to-target error | p90 of `\|t_reach − T_target\|/T_target` | ≤ 10 % | ≤ 20 % |
| Overshoot waste | Energy spent after the index clears, as % of episode energy | ≤ 5 % | ≤ 10 % |
| Undershoot rate | Episodes ending above `I_safe` | ≤ 5 % | ≤ 10 % |
| Chatter | Mean `\|Δduty\|` per control interval | ≤ 0.08 | ≤ 0.15 |
| False-trigger rate | Spurious actuations ≥ 30 s per 72 h idle | ≤ 1 | ≤ 3 |
| Detection latency | Injection → `ACTIVE_REMEDIATION` | ≤ 30 s | ≤ 60 s |
| Solver health | Fraction of solves returning a converged, feasible status | ≥ 98 % | ≥ 95 % |

*System metrics:* p95 decision latency ≤ 1.5 s · p95 solve ≤ 500 ms · telemetry completeness ≥ 99.5 %/day · uptime ≥ 99.5 % · fault-injection suite 10/10 · zero commands executed after `valid_until`.

**Baselines the controller must beat.** (1) Always-on 100 % duty for the whole episode. (2) Bang-bang thermostat: 100 % above `I_safe`, 0 % below, with 10-point hysteresis. (3) A tuned PI controller on the index error. (4) The offline hindsight oracle (upper bound). *If the MPC does not beat baseline (2) on energy at equal deadline compliance, the forecasting layer is not earning its complexity* — that comparison is the honest test of the entire thesis and must be reported whichever way it falls.

---

## 8. ML Model Selection & Justification

### 8.1 NARX Neural Network vs. XGBoost Regressor

Both candidates are being asked to serve as the **surrogate plant model inside an optimizer**, which is a far stricter job than "predict AQI accurately." The optimizer will evaluate the model at input trajectories that do not exist in the training data — that is precisely what optimization does — and will exploit any pathology it finds. Accuracy on a held-out set is therefore necessary but nowhere near sufficient.

| Dimension | NARX Neural Network | XGBoost Regressor |
|---|---|---|
| Model class | Recurrent-by-construction dynamical model, `y(t) = f(y(t−1..n_y), u(t−1..n_u))` | Static tabular regressor; dynamics only via hand-supplied lag features |
| Behaviour outside the training envelope | Smooth extrapolation (tanh/linear output); degrades gracefully | **Piecewise-constant — the prediction goes flat.** A tree cannot extrapolate |
| Consequence for the optimizer | Usable gradients everywhere; well-posed NLP | **Pathological.** A flat region means `∂Î/∂u = 0`; SLSQP finds a duty where "more fan changes nothing" and parks there |
| Gradients | Exact and cheap via `torch.func.jacrev`; embeddable in CasADi symbolically | None. Finite differences only, at ~2·N extra rollouts per iteration |
| Data appetite | ~100+ episodes before it stops memorising | Works from ~60 episodes; excellent with small tabular data |
| Handling heterogeneous/missing features | Needs imputation and scaling discipline | Native missing-value handling, no scaling needed |
| Interpretability | Low (black box) | High — SHAP, gain, monotone constraints, partial-dependence plots |
| Inference cost, laptop, N = 30 rollout | ≈ 3–6 ms (2 k-param MLP) | ≈ 8–15 ms (30 sequential `predict` calls, per-call overhead dominates) |
| Training cost | ~2–5 min on CPU | ~10–30 s |
| Robustness to sensor outliers | Moderate | High |
| Ease of enforcing physical monotonicity | Via architecture (e.g. input-convex / sign-constrained weights) — awkward | **Trivial and native: `monotone_constraints`** |

**Recommendation — a grey-box ensemble, with the physics carrying the structure:**

```
Î(t+1) = f_physics(C_t, u_t; Q(u), k_dep, S)          ← mass-balance ODE, always present
         + g_θ(x_t, u_t)                              ← learned RESIDUAL only
```

- **Primary residual model: NARX** (single hidden layer, 24 tanh units, `n_y = 6`, `n_u = 6` lags, ~2 k parameters, weight decay 1e-4, early stopping on episode-grouped validation), because the optimizer needs smooth, differentiable, extrapolatable behaviour and exact gradients, and because a *dynamical* model is the honest representation of a *dynamical* system.
- **Fallback / ensemble member: XGBoost** with `monotone_constraints` set to **−1 on every duty-derived feature** (`duty`, its lags, `Q(u)/V`), which makes "more fan ⇒ never a higher predicted index" a hard structural guarantee rather than something learned. Used (i) as the day-one model while episode count is between 60 and 100, (ii) as an ensemble member — a simple inverse-variance blend, with disagreement between the two members used as an uncertainty signal that widens the MPC's safety margin, and (iii) as the interpretability instrument: SHAP on the XGBoost member is how the sensor-drift and cross-sensitivity stories get told in the report.
- **The `f_physics` term is what makes either model viable.** It supplies the exponential decay, guarantees the correct sign and asymptote everywhere in input space, and reduces the learning problem from "discover the physics of air exchange from 160 episodes" to "correct a known model by 10–20 %". It also means that if `g_θ` fails entirely, the controller degrades to a physics-based MPC that still works.
- **Rejected alternatives, and why:** a plain LSTM/GRU (needs an order of magnitude more data than 160 episodes and is harder to embed in an NLP); Gaussian Process (elegant uncertainty and would be the *right* choice at ~50 episodes, but `O(n³)` scaling breaks down as the corpus grows, and the multi-step rollout compounds awkwardly) — GPs are retained only as an offline uncertainty benchmark; linear ARX (a useful sanity baseline and a fine warm-start, but the `C·Q(u)` removal term is bilinear, so a purely linear model cannot represent it).

**Mandatory guard, whichever model is promoted:** before any model reaches production, the **monotonicity audit** in §7.4 runs over 10 000 sampled `(state, u)` pairs. Any pair with `∂Î/∂u > 0` blocks promotion. This is the single cheapest defence against the failure mode that would otherwise silently destroy this project.

### 8.2 Optimizer Selection

| Option | Verdict | Reasoning |
|---|---|---|
| **SciPy `SLSQP`** | **Phase 1 — ship this first** | Zero extra dependencies; handles the box constraints, the slew constraint and the soft terminal constraint directly; well-understood. With a warm start and `N = 30`, measured p95 ≈ 480 ms — inside budget. Weakness: with finite-difference gradients it needs ~`2N` extra rollouts per iteration, and it is fragile when a constraint is momentarily infeasible |
| SciPy `trust-constr` | Rejected as primary | More robust on ill-conditioned problems but 2–4× slower here; kept as an automatic retry when SLSQP returns a non-converged status |
| **CasADi + IPOPT** | **Phase 2 — the production target** | The NARX residual is a small MLP and transcribes into CasADi symbolics directly, giving **exact analytic gradients and Hessians**. Expected 5–10× solve-time reduction (p95 ≈ 60–100 ms), which is what unlocks NFR-12 scaling to 30+ zones on one laptop |
| `do-mpc` | Recommended **wrapper** for Phase 2 | Built on CasADi/IPOPT; supplies the MPC scaffolding (horizon bookkeeping, state estimation, soft constraints, move-blocking) for free, so the team writes a model and a cost function rather than an MPC framework. Slight loss of control over the solve loop is a fair trade |
| Genetic / PSO / RL | Rejected | Stochastic search gives no convergence guarantee and no reproducibility inside a 900 ms budget. An RL policy would additionally need orders of magnitude more interaction data and provides no constraint guarantee — unacceptable when the constraint is a safety deadline |

**Compute budget check.** The host is a laptop, not the edge; the entire ML and optimization stack has no ESP32 footprint whatsoever. The measured p95 of 480 ms sits inside a 10 s control interval with 20× headroom, so the choice is comfortable rather than marginal — the reason to move to IPOPT is multi-zone scaling and gradient quality, not single-zone feasibility.

**Formulation details that matter more than the solver choice:**

1. **The terminal constraint is soft.** `Î(T_target) ≤ I_safe + s`, with `λ_slack · s²` in the objective and `λ_slack` large (1e4). A live controller must never return "infeasible" — when the deadline is physically unreachable, the correct behaviour is to run at maximum and report the projected miss, not to fail to produce a command.
2. **Move-blocking.** The 30-step horizon is parameterised by 8 decision variables (steps 1,1,1,2,2,4,8,11). This cuts the NLP dimension by ~4× with negligible cost degradation, because the tail of a receding horizon is re-planned before it is ever executed.
3. **`u_0` is constrained to the currently applied duty ± the slew limit**, so the first move is always physically realisable.
4. **Warm start** from the previous solution shifted one step, with the last element repeated — typically halves the iteration count.
5. **`d_min` handling.** The true feasible set `u ∈ {0} ∪ [0.25, 1]` is disjoint and would make this a MINLP. Instead the NLP stays continuous over `[0, 1]` and the *actuator layer* realises sub-`d_min` means by bang-bang dithering inside the 10 s interval (FR-11). This is legitimate because the plant time constant (≈ 120 s) is far longer than the dither period, so the plant integrates the dither and sees only the mean. Avoiding a mixed-integer solve in the inner loop is worth this small piece of firmware.

### 8.3 Retraining Cadence & Drift Strategy

| Signal | Detector | Threshold | Action |
|---|---|---|---|
| Forecast residual drift | Page-Hinkley on one-step residual | `δ = 0.005`, `λ = 12` | Immediate retrain + operator event |
| Feature distribution shift | PSI per feature, 7 days vs. training window | > 0.10 warn · > 0.25 act | Investigate; retrain |
| **MOS sensor ageing** | `R0` tracked in clean-air windows (FR-13) | > 10 % / 30 days warn · > 20 % from commissioning act | Recalibrate; flag pre-drift episodes; consider sensor replacement |
| Seasonal baseline shift | 30-day rolling median of the ambient index | > 25 index points | Retrain with recency weighting (exponential, 30-day half-life) |
| Actuator degradation | RPM at fixed duty, and `Q(u)` implied by the CO₂ tracer decay | RPM > 15 % below commissioning at matched duty | Maintenance event: clean impeller / replace filter, then re-identify `Q(u)` |
| Controller performance | SC-2 over a rolling 50-episode window | < 90 % | Roll back to the last promoted model; investigate |

Base cadence: **nightly shadow retrain, weekly promotion review, monthly full re-identification of `Q(u)` from the CO₂-tracer decay test.** Recency weighting with a 30-day half-life is applied always, so seasonal shift is handled continuously rather than by a discrete event. Every promotion is gated on shadow-mode performance, never on offline metrics alone.

---

## 9. Critical Architectural Risks

Severity × Likelihood on a 1–5 scale; **Sev** is the consequence if unmitigated.

### (a) MQ-series drift and cross-sensitivity → model bias — **Sev 4, Lik 5, the single largest data-quality risk**

An SnO₂ sensor's resistance responds to *reducing gases in general*, not to one analyte. MQ-135 responds to alcohol, benzene, NH₃, NOx and smoke with overlapping curves; MQ-2 responds to LPG, propane, H₂, methane and smoke. Both drift with heater ageing (`R0` typically 10–30 % over a year), both are strongly modulated by temperature and humidity (tens of percent across a normal indoor range), and both require 24–48 h of burn-in before `R0` is even meaningful. The failure this creates is not noise but **bias**: humidity rises in the afternoon, `Rs` falls, the index rises, the controller vents, and the model learns a causal link between weather and pollution that does not exist. Left alone, this produces a controller that reliably wastes energy at 4 p.m.

Mitigations, layered: **(1)** MQ output is *never* converted to a ppm figure — the datasheet ppm curves are drawn for a single analyte in dry air and are fiction in a mixed industrial atmosphere; only the ratio `Rs/R0` enters the system. **(2)** Datasheet T/RH correction surfaces are applied before the ratio is used as a feature (§7.2). **(3)** `R0` is auto-recalibrated by EWMA during certified clean windows (FR-13), with every update logged to `r0_history` for audit. **(4)** MQ contributes to `SI_gas` with a deliberately low weight and *cannot alone* drive a full-power response — PM2.5 and CO₂ are the index backbone because they are the physically-calibrated channels. **(5)** Baseline-differenced features (channel minus its trailing 24 h 5th percentile) let the model see the excursion rather than the absolute level, which is drift-invariant by construction. **(6)** 48 h burn-in is a documented commissioning step, and the commissioning `R0` is stored in NVS. **(7)** Optional but recommended for a graded project: a single co-located reference-grade measurement at commissioning and at 6 months, to bound the absolute error rather than merely tracking relative change.

### (b) PM sensor power draw vs. power source — **Sev 3, Lik 4 — resolved by design, not by mitigation**

Quantified in §3.2: the irreducible continuous sensor load is ≈ 2.7 W on the 5 V rail, of which the two MQ heaters are 1.55 W and cannot be duty-cycled without destroying the `R0` baseline and disarming the safety channel, and the laser PM sensor adds ~0.5 W that can only be duty-cycled in idle mode. Battery operation yields ≈ 12–18 h — less than a working day for a permanently-installed safety appliance. **Design decision: mains 12 V/2 A SMPS is the sole primary source; battery exists only as a 15-minute UPS for graceful shutdown, log flush and alarm persistence.** Residual risks are then power-quality, not power-budget: rail sag during motor inrush corrupting ADC readings (mitigated by 1000 µF bulk + 470 µF on 5 V + star grounding, §3.2) and mains outage (mitigated by the UPS and the `SAFE_SHUTDOWN` mode). Sensor *life* is managed separately from power: PMS5003 at MTTF ≥ 3 years may run continuously; **SDS011 at 8 000 h of laser life must be duty-cycled per §3.2 or it will fail inside a year** — which is the deciding reason to prefer PMS5003.

### (c) Control-loop instability / oscillation — **Sev 4, Lik 3**

The mechanism: if the optimizer re-plans faster than the turbine can mechanically respond, or faster than the sensor can observe the response, the controller reacts to its own un-settled transient. With a PM sensor whose frame period is 1–2.3 s and an impeller with `τ_mech ≈ 250 ms`, an unrestrained controller would hunt — and hunting is doubly punished here because every duty change costs inrush energy, directly attacking the objective the system exists to minimise.

Mitigations: **(1) Timescale separation** — control interval `T_c = 10 s` chosen as ≈ `τ_plant/12` (§2.5), with a hard floor of 5 s enforced in code. **(2) Slew-rate constraint** in the NLP, `|Δu| ≤ 0.10` per interval, and again independently in the edge inner loop (0.10 per 100 ms tick) so a malformed host command cannot slam the actuator. **(3) Move-blocking** (§8.2) which structurally cannot produce a high-frequency plan. **(4) Slew penalty `λ_slew` in the objective**, tuned so chatter costs real objective value. **(5) Index deadband** of ±5 points and 10-point hysteresis on the `IDLE ↔ ACTIVE` transition, preventing limit-cycling around `I_safe`. **(6) Minimum dwell** of 5 s at any non-zero duty before another change. **(7)** The `d_min` dither runs at a fixed sub-interval period, so it is a *known* forced oscillation the model can see, not an emergent one. **(8) Detection**: `SC-14` (mean `|Δduty|`) is monitored live; exceeding 0.15 over 10 intervals raises an event and temporarily doubles `λ_slew`. **(9)** Empirically measure `τ_mech` and the sensor response lag during commissioning; if `τ_mech` exceeds `T_c/10`, increase `T_c` rather than tuning penalties.

### (d) MQTT broker as a single point of failure — **Sev 4, Lik 3**

A single Mosquitto instance is a SPOF for every node in the site. Worse, in the naive deployment it runs *on the optimizer laptop*, which makes broker loss and compute loss the same event and removes any possibility of graceful degradation.

Mitigations: **(1) Deploy the broker on a separate always-on host** (Raspberry Pi 4 or equivalent) so transport survives laptop reboots, lid closures and host crashes — this decorrelation is the highest-value single change in this section. **(2) QoS policy by message class:** telemetry **QoS 0** (loss-tolerant, high rate, gaps repaired by the edge ring buffer and the `telemetry/backfill` topic — paying QoS 1 overhead for 1 Hz data that is superseded a second later is pure waste); commands, acks, events and state **QoS 1** with `clean_session = false` so the broker queues for a briefly-disconnected node. **(3) QoS 2 is deliberately rejected**: its four-way handshake roughly doubles command latency, and exactly-once delivery is achieved more cheaply at the application layer by the strictly-increasing `seq` plus `valid_until`, which makes duplicate commands idempotent *and* stale commands unexecutable — a strictly stronger guarantee than QoS 2, which can still deliver a stale command exactly once. **(4) Retained LWT** on `.../lwt` so the host learns of an ungraceful node death within one keep-alive (10 s). **(5) Exponential-backoff reconnect** with jitter on both edge and host, capped at 30 s. **(6)** The edge ring buffer covers a 10-minute outage with zero data loss. **(7)** Broker loss is *not* a safety event, because of (e). **(8)** Where a site genuinely cannot tolerate transport loss, the migration path is EMQX in a 2-node cluster behind a VIP; the topic schema is already compatible.

### (e) Host disconnect mid-optimization — the fail-safe contract — **Sev 5, Lik 3**

**Last-command-hold is explicitly rejected.** Holding the last command is unsafe in both directions: if the last command was 0 % and pollution then rises, the node sits idle through a hazard; if it was 100 %, the node runs the turbine indefinitely, destroying the energy claim and wearing the motor out. "Hold" is only correct if the world stops when the network does, and it does not.

The contract, in order of precedence:

1. **The command carries its own expiry.** `valid_until = issued + 3 × T_c` (30 s). Within that window, a host outage is invisible: the edge keeps executing the already-vetted profile, which was optimal for the horizon it was computed over.
2. **On expiry → `DEGRADED_AUTONOMOUS`.** The edge runs a local, deterministic, no-network rule ladder on its own locally-computed `I_inst`:

| Local condition (on `I_inst`, 10 s hysteresis) | Duty |
|---|---|
| `< I_safe − 10` | 0 % |
| `I_safe − 10 … I_safe` | 30 % |
| `I_safe … 1.25 × I_safe` | 55 % |
| `1.25 … 1.5 × I_safe` | 80 % |
| `> 1.5 × I_safe` | 100 % |
| MQ-2 flammable threshold breached | **0 % + alarm + `INTERLOCK_FLAMMABLE`** |

This ladder is a *conservative closed-loop controller*, not a fixed default: it still responds to the environment, it simply gives up energy optimality. Expected cost is roughly baseline-thermostat energy — i.e. degraded to the state of the art, never to unsafe.
3. **Watchdogs.** `esp_task_wdt` (3 s) covers `tControl` and `tFusion`; a watchdog reset lands in `BOOT` with the motor held off by the hardware pull-downs of §3.4, so even a firmware hang converges to the safe state.
4. **Sensor-fault fallback.** If the index itself becomes untrustworthy (all primary channels invalid > 120 s), the node enters `SENSOR_FAULT` and holds a fixed conservative 60 % duty with an alarm — because in the absence of measurement, moderate ventilation is the lower-regret action.
5. **The flammable interlock overrides everything** — host, ladder and operator alike — and is implemented entirely inside the 100 ms loop with no dependency on Wi-Fi, MQTT, the host, or even the index computation.
6. **Recovery is explicit.** On the host's return, the first command must carry `seq > last_seen_seq`; the edge publishes a `state` message describing what it did while alone, and the host writes the autonomous interval into `episodes` so it is excluded from controller-performance accounting rather than silently polluting SC-1.

### (f) Flammable atmosphere + brushed DC motor — **Sev 5, Lik 2 — a safety defect in the naive control law**

MQ-2 detects LPG, propane, hydrogen and methane. The intuitive response — "gas detected, extract harder" — places a **brushed, commutating, arcing, non-ATEX-rated motor inside a potentially explosive atmosphere.** Brush arcing is a competent ignition source, and a suction turbine deliberately draws the flammable mixture across it. This is not a hypothetical: it is the standard reason industrial extraction in classified zones uses spark-free or externally-coupled fans.

**Design decision: a flammable-gas detection commands the motor to 0 %, raises the local audible alarm, publishes a high-severity event, and latches `INTERLOCK_FLAMMABLE` until a human acknowledges via the physical button AND the reading has been clear for 10 minutes.** The controller never increases ventilation in response to MQ-2. Where a site genuinely requires extraction of flammable vapour, the correct hardware is an ATEX/IECEx-rated spark-free fan with the motor outside the airstream — outside the fixed BOM of this project, and therefore explicitly out of scope, documented as a deployment precondition. The MQ-2 threshold is set conservatively low (well below any LEL fraction) precisely because a MOS sensor's absolute accuracy does not justify a tight margin.

### (g) Additional risks (compressed)

| # | Risk | Sev | Mitigation |
|---|---|---|---|
| g1 | **ADC2 unusable while Wi-Fi is active** — a classic ESP32 trap that presents as sensors "randomly returning zero" | 4 | All analog on ADC1 (§3.3); a build-time assertion rejects any ADC2 channel assignment |
| g2 | **5 V sensor outputs into 3.3 V pins** | 4 | MQ AOUT buffered and divided (§3.4.6); PMS5003 and MH-Z19B TX are already 3.3 V logic — verified, not assumed |
| g3 | **PM readings invalid at RH > 90 %** (hygroscopic growth inflates the optical diameter) | 3 | κ-Köhler correction as a feature; hard invalidation above 90 % RH (FR-14) with index fallback |
| g4 | **MQ heaters bias the DHT22** — ~1.5 W dissipated centimetres from the temperature sensor | 3 | Physical separation ≥ 50 mm, a thermal baffle, DHT22 upstream in the airflow; verify with a calibrated reference at commissioning |
| g5 | **Timestamp jitter corrupts lag features** | 3 | SNTP-disciplined edge timestamps + monotonic `seq`; the host **never** re-stamps with arrival time; rows with a clock step > 500 ms are flagged and excluded from training |
| g6 | **Closed-loop identifiability loss** — feedback confounds cause and effect | 4 | Weekly APRBS campaign + 5 % ε-dither, flagged in `commands.reason` (§6.2) |
| g7 | **No make-up air path.** Extraction from a sealed volume achieves nothing but a pressure drop; extraction that recirculates simply relocates the particulate | 4 | A filtered make-up-air inlet is a **mechanical precondition** recorded in the commissioning checklist; the mass-balance model's `C_out` term makes the assumption explicit and testable |
| g8 | **Motor inrush corrupts the ADC** via shared ground | 3 | Star ground, bulk capacitance, and ADC sampling **phase-locked to the PWM off-period** |
| g9 | **Optimizer exploits a non-monotone surrogate** | 5 | Grey-box structure + `monotone_constraints` + the mandatory pre-promotion monotonicity audit (§8.1) |
| g10 | **Row-wise validation leakage** inflating reported accuracy | 4 | Episode-grouped, time-ordered splits mandated in §7.4; CI rejects any evaluation that does not name its split |
| g11 | **Scale mismatch** — a 2 W turbine cannot ventilate a manufacturing bay | 4 | Honest scoping: validation is in a 60 L chamber; the architecture is scale-invariant, the actuator is not. See Appendix A FX-08 |
| g12 | **OTA bricking a node** | 3 | Dual OTA partitions, post-flash self-test, automatic rollback; OTA permitted only in `IDLE_MONITOR` |
| g13 | **Credential leakage into version control** | 4 | Credentials in NVS and `.env`, never in source; pre-commit secret scanning; flash encryption on production units |

---

## Appendix A — Flaw Register (design corrections to the original brief)

Each entry states the problem with the naive reading, the correction adopted, and the consequence of *not* adopting it. Entries marked **load-bearing** invalidate the project's central claim if ignored.

| ID | Naive assumption | Why it fails | Correction adopted | If ignored |
|---|---|---|---|---|
| **FX-01** | "Restore safe **AQI** within `T_target`" | Regulatory AQI (EPA and CPCB alike) is defined on **24-hour averaged** concentrations. A 24 h mean cannot be moved in 15 minutes; a controller whose measurement lags its actuation by hours is unstable by construction | Two indices (§4.1): `I_inst` on 60 s rolling means **for control**, `I_nowcast` **for reporting**. Every control claim is stated against `I_inst`, explicitly labelled non-regulatory | **Load-bearing.** The controller would appear to do nothing for hours, then over-react |
| **FX-02** | Minimise `∫V(t)² dt` = minimise energy | `∫V²dt` equals energy only for a fixed resistive load. A fan's electrical power is `V·I`, and its *aerodynamic* power scales roughly with `ω³`. The proxy is monotone in the right direction but is not watt-hours | Keep `∫V²dt` as the smooth optimizer objective; **measure and report true `∫V·I dt`** via ACS712 + bus-voltage sense (H10, H11). SC-1 is claimed on measured Wh | **Load-bearing.** The headline energy-saving number would be unverifiable |
| **FX-03** | Gas detected ⇒ increase extraction | MQ-2 detects **flammable** gases. Drawing an explosive mixture across a brushed, arcing, non-ATEX motor is an ignition path | Flammable detection ⇒ **motor to 0 %**, alarm, latched interlock requiring physical acknowledgement (§9f). ATEX-rated hardware documented as a deployment precondition | **Load-bearing safety defect** |
| **FX-04** | On host loss, hold the last command | Unsafe in both directions: hold-0 through a rising hazard, or hold-100 forever | Command expiry (`valid_until`) + **`DEGRADED_AUTONOMOUS` local rule ladder** (§9e). Degrades to thermostat-grade, never to unsafe | **Load-bearing.** Explicitly required by the brief and the naive reading gets it wrong |
| **FX-05** | "MH-Z19B **or** MQ-135 (CO₂)" — interchangeable | MQ-135 is not a CO₂ sensor. It has no CO₂ selectivity; the widely-copied "CO₂ ppm" library curves are extrapolations with no datasheet basis | **MH-Z19B (NDIR) is mandatory for CO₂.** MQ-135 is retained as a *relative* VOC/NH₃ channel reporting `Rs/R0` only | CO₂ readings would be fabricated, and a fabricated channel would be fed straight into the model |
| **FX-06** | Drive the IRF520 gate from a 3.3 V GPIO | `V_GS(th)` is 2.0–4.0 V and `R_DS(on)` is quoted at `V_GS = 10 V`. At 3.3 V the part sits in its linear region or fails to conduct | TC4420 gate driver on the 12 V rail (H9), plus **mandatory 10 kΩ gate pull-down** for boot safety (§3.4) | MOSFET overheats, or motor never runs, or spins at boot |
| **FX-07** | Any ADC pin will do for the MQ sensors | **ADC2 is disabled whenever Wi-Fi is active.** Reads silently return garbage — one of the most common ESP32 field failures | Every analog channel on ADC1 (GPIO 32–39); build-time assertion against ADC2 (§3.3, g1) | Gas channels intermittently dead exactly when the node is online |
| **FX-08** | A 2 W turbine remediates a manufacturing bay | 2 W ≈ 1.8 m³/h. A 500 m³ bay needs 4–6 air changes/hour ≈ 2500 m³/h — three orders of magnitude more | Validation is scoped to a **60 L instrumented chamber**. The *architecture, control law and ML pipeline are scale-invariant*; the actuator is not. Scale-up requires a sized fan and a re-identified `Q(u)` — a hardware change, not a software one | Unfalsifiable claims; a demo that cannot be reproduced at the stated setting |
| **FX-09** | Any accurate regressor can serve as the MPC's plant model | An optimizer evaluates the model **outside the training envelope by design**, and exploits flat regions (trees) or wrong-sign regions | Grey-box physics + learned residual; `monotone_constraints` on all duty features; **mandatory pre-promotion monotonicity audit** (§8.1) | **Load-bearing.** The optimizer confidently commands nonsense while every offline metric looks excellent |
| **FX-10** | Train/test split at random over rows | 1 Hz rows inside one decay are near-perfectly autocorrelated; random splits leak the target and report R² ≈ 0.99 for a useless model | **Episode-grouped, time-ordered splits** mandated (§7.4); effective sample size counted in episodes | Reported accuracy is fiction; the controller fails on first contact with a new event |
| **FX-11** | Run the broker on the host laptop | Transport and compute then fail as a single event, removing graceful degradation | Broker on a **separate always-on host** (§5.2, §9d) | A closed laptop lid takes out the whole site |
| **FX-12** | `V_motor = duty × 12 V` | Driver drop (≈ 2 V on an L298N), rail sag under inrush and PWM ripple all break this | Measure the applied bus voltage on GPIO39 (H11) and optimise over measured `V` | The energy objective is computed on a fictional voltage |
| **FX-13** | Once closed-loop, the system keeps learning from live data | Feedback removes persistent excitation; the identified model drifts toward the closed-loop transfer function rather than the plant | Weekly APRBS campaign + 5 % ε-dither, both flagged and excluded from performance accounting (§6.2) | Model quality silently decays over weeks with no alarm |
| **FX-14** | Suction alone cleans the air | Extraction from a sealed volume yields a pressure drop; recirculating extraction relocates particulate | Filtered make-up-air inlet is a **commissioning precondition**; the `C_out` term in the mass balance makes it explicit and testable (g7) | The plant model is unidentifiable and the controller under-performs for reasons invisible in the data |
| **FX-15** | MH-Z19B works correctly out of the box | ABC self-calibration is **on by default** and assumes the sensor sees 400 ppm at least once every 24 h — false in a chemical store or a sealed chamber. It will silently re-zero to whatever the local minimum is | ABC disabled at every boot and the acknowledgement verified (FR-05) | CO₂ readings drift toward a fabricated zero over days |
| **FX-16** | Duty is continuous on [0, 1] | A brushed motor will not start below ≈ 25 % duty; the true feasible set is disjoint (`{0} ∪ [0.25, 1]`), which makes the NLP a MINLP | Optimizer stays continuous; the **actuator layer** realises sub-`d_min` means by bang-bang dithering within the 10 s interval (FR-11, §8.2.5) | Commanded low duties produce zero airflow; the model learns a false dead-zone |
| **FX-17** | PMS5003 and SDS011 are interchangeable | SDS011's laser is rated for **8 000 h ≈ 11 months** continuous; PMS5003 is rated MTTF ≥ 3 years | PMS5003 preferred; if SDS011 is fitted, duty-cycling is mandatory (§3.2) | Sensor dies mid-deployment, most likely during the evaluation window |
| **FX-18** | Retained MQTT messages are a convenient default | A **retained command** is re-delivered on every reconnect and re-executed — a stale actuation replayed after an outage | Retain only `state`, `lwt` and `cfg`. **Never retain `cmd`** (§5.2) | The turbine spins up to a stale setpoint every time a node reconnects |

## Appendix B — Implementation Roadmap

| Phase | Duration | Exit criteria |
|---|---|---|
| **P0 — Bench bring-up** | 2 weeks | All sensors reading plausibly; pin map verified against §3.3; gate driver verified on a scope incl. boot state; PWM 20 kHz/11-bit confirmed; ADC1 calibration applied; power budget measured against §3.2 |
| **P1 — Edge firmware + transport** | 3 weeks | FSM and all interlocks implemented and unit-tested; MQTT+TLS with ACLs; ring buffer and backfill proven over a 5-min outage; ack path p99 ≤ 50 ms; fault-injection suite FI-01…FI-10 passing |
| **P2 — Data plane + open-loop campaign** | 4 weeks | TimescaleDB schema live with retention and compression; Grafana ops dashboard; Phase-A baseline complete; Phase-B DoE complete (≥ 160 episodes) |
| **P3 — Models offline** | 3 weeks | Grey-box parameters identified; NARX and XGBoost trained; episode-grouped evaluation meets SC-4/SC-5 floors; **monotonicity audit passes**; MLflow registry with reproducible runs |
| **P4 — MPC in simulation (SIL)** | 2 weeks | Controller beats all four baselines in simulation; SC-11 solve time met; graceful behaviour verified under infeasibility, sensor dropout and host loss |
| **P5 — Closed loop on hardware (HIL)** | 3 weeks | ≥ 100 closed-loop episodes; SC-1…SC-14 measured and reported; failure modes catalogued |
| **P6 — Hardening + write-up** | 2 weeks | Drift detection live; nightly retrain + shadow promotion running; console complete; full documentation and reproducibility package |

## Appendix C — Fault-Injection Suite (SC-13)

| ID | Injected fault | Required behaviour |
|---|---|---|
| FI-01 | Broker killed for 5 min | Node reconnects with backoff; zero telemetry loss after backfill; enters `DEGRADED_AUTONOMOUS` within 30 s and returns cleanly |
| FI-02 | Wi-Fi AP down for 12 min | Ring buffer saturates gracefully, oldest-first, gap event logged; no reboot loop |
| FI-03 | Host process killed mid-solve | Command expiry honoured; `DEGRADED_AUTONOMOUS` within 30 s; ladder tracks the index correctly |
| FI-04 | Malformed / replayed / stale command injected | All rejected with the correct reason code; node mode unchanged; no actuation |
| FI-05 | PMS5003 unplugged | `pm.valid=false`; index falls back to remaining channels; health event raised |
| FI-06 | PMS5003 **and** MH-Z19B unplugged | `SENSOR_FAULT`; fixed conservative 60 % duty; alarm |
| FI-07 | Motor stalled (impeller blocked) | Tachometer mismatch detected within 10 s; fault event; duty capped to protect the motor |
| FI-08 | MQ-2 driven past the flammable threshold | Motor to 0 % within 200 ms; buzzer; latched interlock; **host command to 100 % is refused** |
| FI-09 | Mains removed (UPS engaged) | `SAFE_SHUTDOWN`; 10 s ramp-down; log flush; state published before power loss |
| FI-10 | NTP step of +30 s applied mid-episode | Affected rows flagged; no negative `Δt`; features not corrupted; training excludes the flagged window |

## Appendix D — Open Decisions Requiring Sign-off

1. **Index standard** — CPCB NAQI (recommended, given an Indian deployment context) or US EPA. Affects `I_safe` semantics and all reported numbers. *Default assumed: CPCB.*
2. **PM sensor** — PMS5003 (recommended, 3 yr MTTF, bin counts) vs SDS011 (cheaper, 8 000 h laser). *Default assumed: PMS5003.*
3. **Motor driver** — IRF520 + TC4420 (recommended, ~0.3 % loss, required for a credible energy claim) vs L298N (simpler, ~20 % loss, adds reverse capability). *Default assumed: IRF520 + TC4420.*
4. **Broker host** — separate Raspberry Pi (recommended) vs the optimizer laptop. Cost vs. the SPOF decorrelation in §9d.
5. **Validation vessel volume** — the 60 L chamber assumed throughout sets `τ ≈ 120 s`, which sets `T_c = 10 s` and `N = 30`. A different volume propagates into the entire timing design.
6. **`T_target` default** — 900 s assumed. This is the single most consequential user-facing parameter for SC-1: a longer deadline permits a lower-duty profile and therefore a larger energy saving.

---

### Sources

- [CPCB — National Air Quality Index](https://cpcb.nic.in/National-Air-Quality-Index/) and [About AQI (methodology and sub-index breakpoints)](https://www.cpcb.nic.in/displaypdf.php?id=bmF0aW9uYWwtYWlyLXF1YWxpdHktaW5kZXgvQWJvdXRfQVFJLnBkZg%3D%3D)
- [US EPA — Reconsideration of the NAAQS for Particulate Matter (2024 final rule)](https://www.federalregister.gov/documents/2024/03/06/2024-02637/reconsideration-of-the-national-ambient-air-quality-standards-for-particulate-matter) and [IQAir — 2024 update to the U.S. EPA AQI](https://www.iqair.com/support/knowledge-base/iqair-implements-2024-update-to-u-s-epa-air-quality-index-aqi)
- [Winsen MH-Z19B datasheet](https://www.winsen-sensor.com/d/files/MH-Z19B.pdf) — 4.5–5.5 V, < 20 mA average / 150 mA peak, 3 min warm-up, ABC-disable command `FF 01 79 00 00 00 00 00 86`
- [Plantower PMS5003 series manual](https://www.digikey.lt/htmldatasheets/production/2903006/0/0/1/pms5003-series-manual.html) — ≤ 100 mA active, < 200 µA standby, MTTF ≥ 3 years
