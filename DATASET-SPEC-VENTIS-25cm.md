# VENTIS Dataset Specification & Collection Protocol
## Demonstration Rig: 250 × 250 × 250 mm Sealed Test Chamber

| Field | Value |
|---|---|
| Document ID | AQR-DATA-001 |
| Revision | v1.0 |
| Date | 2026-09-05 |
| Companion document | AQR-ARCH-001 (PRD / Architecture) — **this file supersedes §7 of that document for the demonstration rig** |
| Dataset name | `ventis-chamber-25` |
| Dataset version scheme | `v<MAJOR>.<MINOR>.<PATCH>` — MAJOR = schema change, MINOR = new episodes added, PATCH = QC/label fix only |
| Target model | Grey-box mass balance + NARX / XGBoost residual (AQR-ARCH-001 §8) |

> **Read this first.** Section 1 defines the rig and the physics; **every range, bin edge and duration in this document is derived from those numbers.** If you change the chamber volume, the fan, or the make-up-air path, you must re-derive Sections 2–4 before collecting anything. Section 3 is the parameter registry, Section 6 is the exact file schema, Section 8 is the numbered build-and-run procedure.

---

## 0. Scope, Assumptions and Decisions Taken

| # | Assumption / decision | Value | Change it if… |
|---|---|---|---|
| A-1 | Chamber internal geometry | 250 × 250 × 250 mm (L×B×H), single sealed volume | you use a different vessel — re-derive §1.2 |
| A-2 | Chamber material | 5 mm cast acrylic, silicone-sealed seams, one gasketed access door | — |
| A-3 | Make-up air | HEPA-filtered inlet, `C_in ≈ 0` for PM, `C_in = ambient` for CO₂ | you run sealed (then `Q(u)→0` and the whole experiment fails — see §1.4) |
| A-4 | Primary aerosol source | Incense smoke, two-stage dilution injection | — (secondary sources are mandatory anyway, §4.6) |
| A-5 | CO₂ source | Bicarbonate + acetic acid external generator, syringe injection | you have a regulated CO₂ cylinder — better, use it |
| A-6 | MQ-2 flammable channel | **Never exercised with real flammable gas.** Interlock verified by resistor substitution | never |
| A-7 | Reference instrument | None assumed. Absolute PM accuracy is *not* claimed; the dataset is self-consistent | you have a reference monitor — add it as `ref_*` columns |
| A-8 | Sampling rate | 1 Hz for all channels | — (justified in §1.3; do not reduce) |
| A-9 | Index standard | CPCB NAQI sub-index breakpoints (AQR-ARCH-001 §4.1) | — |
| A-10 | `T_target` grid for this rig | {90, 150, 300, 600} s — **not** the 900 s PRD default | you change the fan |
| A-11 | ESP32 board location | **Outside** the chamber; only sensing elements inside | — (see §1.5) |

---

## 1. Rig Definition and Derived Physics

### 1.1 Physical Configuration

```
                      ┌────────────────────── 250 mm ──────────────────────┐
                      │                                                     │
   P3 injection ──►   │   ○ PMS5003 (mid-height, 125 mm, far wall)          │  ▲
   septum, 60 mm      │                                                     │  │
   from floor         │       ○ MH-Z19B (140 mm, centre)                    │ 250
                      │                                                     │  │
   P5 mixing fan ──►  │   ⌾ 25 mm 5 V fan, 40 mm from floor, aimed          │  │
   ALWAYS ON          │     tangentially — establishes a rotating cell      │  ▼
                      │                                                     │
   P2 HEPA inlet ──►  │   ○ MQ-135  ○ MQ-2   (80 mm, side wall, on a        │
   + needle valve     │      standoff, ≥ 60 mm from DHT22)                  │
   30 mm from floor   │                                                     │
                      │   ○ DHT22 (150 mm, opposite wall from MQ pair)      │
                      │                                                     │
                      └──────────────► P1 exhaust, 40 mm bore, 210 mm ──────┘
                                        from floor → turbine → outside
   P4 cable gland (sealed with neutral-cure silicone)
```

| Port | Purpose | Spec |
|---|---|---|
| P1 | Turbine exhaust | 40 mm bore, top of one face, ducted **out of the room** or to a filter |
| P2 | Make-up air inlet | 40 mm bore, opposite face, bottom, **H13 HEPA cartridge + needle valve** |
| P3 | Injection | 10 mm bore with a silicone septum, Luer-lock adapter for a 100 mL syringe |
| P4 | Cable gland | M16, sealed; sensor harness only |
| P5 | Internal mixing fan | 25 × 25 mm 5 V axial, **constant speed, permanently on during every episode** |

**The internal mixing fan (P5) is not optional.** The entire dataset rests on the well-mixed (CSTR) assumption `dC/dt = −(λ+k_dep)C`. Without forced internal mixing, a 15.6 L box stratifies: the point measurement at the sensor stops representing the volume, the decay curve becomes non-exponential, and the identified `Q(u)` is meaningless. Run it at a fixed voltage, log `mixfan_on`, and never vary it — if you vary it, it becomes a second manipulated variable and the DoE doubles.

### 1.2 Derived Constants

| Quantity | Symbol | Value | Derivation |
|---|---|---|---|
| Geometric volume | `V_geom` | **15.625 L = 0.015625 m³** | 0.25³ |
| Internals displacement | `V_int` | ≈ 0.5–0.8 L (measure it) | water displacement of every internal part |
| **Effective volume** | `V_eff` | **≈ 14.8–15.1 L — measure and record** | `V_geom − V_int`, cross-checked by CO₂ dilution (§8.3 step 12) |
| Surface-to-volume ratio | `S/V` | **24 m⁻¹** | `6L²/L³ = 6/0.25` |
| Turbine flow at full duty | `Q_max` | **0.6–3.0 m³/h — MEASURE, do not assume** | tracer decay, §8.3 |
| Air changes per hour at full duty | `ACH_max` | **38–192 h⁻¹** (nominal design point 115 h⁻¹ at 1.8 m³/h) | `Q/V_eff` |
| Fastest time constant | `τ_min` | **≈ 31 s** at 115 ACH (19 s at 192, 94 s at 38) | `3600/ACH` |
| PM deposition rate | `k_dep` | **0.8–3.0 h⁻¹ — measure** (assume 1.5 h⁻¹ for planning) | Category-B episodes; high because `S/V` = 24 m⁻¹ |
| CO₂ deposition rate | `k_dep,CO₂` | **0 (exactly)** | CO₂ does not deposit — this is what makes it the tracer |
| Chamber leak rate | `k_leak` | **must be ≤ 0.5 h⁻¹** | CO₂ decay with turbine off, §8.3 step 10 |
| Thermal time constant | `τ_th` | **≈ 70 min** | `C_acrylic / (U·A)`, `U ≈ 2.06 W/m²K`, `A = 0.375 m²` |
| Steady thermal rise from MQ heaters | `ΔT` | **≈ +2.0 °C** (≈ +2.6 °C if the ESP32 is inside) | `P/(U·A)`, `P = 1.55 W` |

### 1.3 Why 1 Hz Sampling Is Correct for This Chamber

The fastest dynamic in the system is the decay at full duty, `τ_min ≈ 31 s`. The identification rule of thumb is ≥ 10 samples per time constant. 1 Hz gives **31 samples per `τ_min`** — a 3× margin, and still 19 samples even at the optimistic `Q_max = 3.0 m³/h`. The upper bound is set by the sensor: the PMS5003 emits one frame per 1.0–2.3 s in active mode, so 1 Hz is also the *highest* honest rate. **Do not log faster than 1 Hz and do not decimate below it.** Electrical channels (`imotor_a`, `vbus_v`) are sampled internally at 10 Hz and reported as the 1 s mean, because PWM makes them fast; their 1 s standard deviations are also logged.

### 1.4 Why the Chamber Must Not Be Sealed

A sealed box has `Q = 0` by conservation of mass — the turbine can only pull as much air out as the inlet lets in. With P2 closed, the turbine produces a pressure drop and no flow, `λ(u) ≡ 0` for every `u`, and every duty level produces the same decay curve (deposition only). The dataset would then contain **no information about the manipulated variable at all**, and the model would learn that the fan does nothing. The needle valve on P2 exists so that inlet restriction can be *varied deliberately* as a disturbance factor (Category H), not so the chamber can be run closed.

### 1.5 Thermal and Electrical Hygiene

- **Keep the ESP32 DevKit outside the chamber.** It removes ~0.5 W of heat and a 2.4 GHz radiator from a 15.6 L volume, and it lets you reflash without breaking the seal. Only the sensor heads go inside, on the harness through P4.
- **The MQ pair dissipates 1.55 W inside the chamber.** With `τ_th ≈ 70 min`, the chamber does not reach thermal steady state for **≥ 3.5 hours** from cold. **Rule: power the rig up at least 90 minutes before the first episode of a session, and record `temp_c` drift over the last 15 minutes — it must be < 0.3 °C before the session may start.** A campaign run without this discipline will contain a spurious correlation between temperature and every other channel, because both drift together over the first two hours of every day.
- Mount the DHT22 **≥ 60 mm from the MQ pair** and, ideally, upstream of them in the mixing-fan circulation, so it measures chamber air rather than MQ waste heat.

---

## 2. Relations Between Parameters

Everything in the dataset is connected by the following model. These equations are the *reason* each column exists; a column that does not appear in an equation or a QC rule should not be collected.

### 2.1 Governing Mass Balance (well-mixed chamber)

```
dC/dt = −(λ(u) + k_dep + k_leak)·C  +  S(t)/V_eff  +  λ(u)·C_in
```

| Term | Meaning | Value in this rig |
|---|---|---|
| `C` | Pollutant concentration (µg/m³ for PM, ppm for CO₂) | measured |
| `λ(u) = Q(u)/V_eff` | Forced air-exchange rate, s⁻¹ | **the manipulated variable's physical effect** |
| `k_dep` | Wall/gravitational deposition | ~1.5 h⁻¹ for PM2.5; **0 for CO₂** |
| `k_leak` | Passive leakage with the turbine off | ≤ 0.5 h⁻¹ (acceptance criterion) |
| `S(t)` | Source term | 0 during decay; non-zero only during the injection window |
| `C_in` | Concentration of incoming make-up air | ≈ 0 for PM (HEPA); ambient (400–500 ppm) for CO₂ |

**Decay solution (the shape of every episode after injection stops):**

```
C(t) = C_in + (C₀ − C_in)·exp( −(λ(u) + k_dep + k_leak)·t )
τ_eff = 1 / (λ(u) + k_dep + k_leak)
```

### 2.2 The Three-Step Identification Chain

This is the *only* correct order. Each step uses the previous step's result.

| Step | Experiment | Isolates | Because |
|---|---|---|---|
| 1 | **CO₂ decay, turbine OFF** (Category C, `u = 0`) | `k_leak` | CO₂ has `k_dep = 0` and `λ = 0`, so the observed decay *is* the leak |
| 2 | **PM decay, turbine OFF** (Category B, `u = 0`) | `k_dep` | observed rate − `k_leak` (from step 1) = deposition |
| 3 | **CO₂ decay at each duty level** (Category C, `u > 0`) | **`Q(u)`** | observed rate − `k_leak` = `λ(u)`, therefore `Q(u) = λ(u)·V_eff` |

Fit each by linear regression of `ln((C − C_in)/(C₀ − C_in))` against `t`; the slope is `−(λ + k_dep + k_leak)`. **Subtract `C_in` before taking the log** — forgetting the ambient CO₂ offset is the single most common error in tracer-decay work and produces a curved "exponential" that fits nothing.

### 2.3 Actuator Chain

```
duty_cmd  ──[edge clamp, slew, d_min dither]──►  duty_applied
duty_applied ──►  V_applied = duty_applied · vbus_v − V_drop(driver)
V_applied ──►  rpm  (≈ linear above u_min)
rpm ──►  Q(u)  (∝ rpm for a fixed system curve — fan affinity law)
Q(u) ──►  λ(u) = Q(u)/V_eff  ──►  dC/dt
```

| Relation | Form | Notes |
|---|---|---|
| Dead-zone | `Q(u) = 0` for `u < u_min` | `u_min ≈ 0.25`, **measure it** — brushed-motor stiction |
| Flow curve | `Q(u) = Q_max · φ(u)`, `φ` a monotone increasing spline through the Category-C points, `φ(u_min⁻)=0`, `φ(1)=1` | fit a PCHIP monotone spline over ≥ 8 duty levels; **never a polynomial**, which will wiggle non-monotonically |
| Fan affinity | `Q ∝ N`, `Δp ∝ N²`, `P_shaft ∝ N³` | explains why energy rises far faster than airflow — the reason the optimization has anything to optimize |
| Electrical power | `P_elec = vbus_v · imotor_a` (1 s means) | measured, not modelled |
| Energy | `E = ∫ P_elec dt`, accumulated per episode | the SC-1 quantity |
| Objective proxy | `∫ V_applied² dt` | reported alongside `E`, never instead of it |

### 2.4 Sensor Transfer Relations

| Relation | Form | Purpose |
|---|---|---|
| MQ resistance | `Rs = R_L · (V_cc − V_out)/V_out` | `R_L` is the module's load resistor — **measure it, the pot is factory-random** |
| MQ ratio | `ratio = Rs / R0` | the only MQ quantity that enters the dataset as a feature |
| MQ T/RH correction | `ratio_corr = ratio / f(T, RH)`, `f` from the datasheet correction surface | removes the false "humidity causes pollution" signal |
| PM hygroscopic growth | `C_dry = C_meas / (1 + κ·(RH/(100−RH)))`, `κ ≈ 0.25–0.5` | store **both** `pm25_ugm3` (raw) and `pm25_dry_ugm3` (corrected) |
| Sub-index | `SI = SI_lo + (SI_hi−SI_lo)/(C_hi−C_lo)·(C − C_lo)` | CPCB piecewise-linear interpolation |
| Composite index | `I_inst = max(SI_pm25, SI_pm10, SI_gas, SI_co2)` | `i_driver` records the arg-max |

### 2.5 Feasibility Relation — which `(C₀, T_target, u)` combinations are physically possible

The minimum air-exchange rate needed to bring `C₀` down to `C_safe` within `T_target`:

```
ACH_required = 3600 · ln(C₀/C_safe) / T_target  −  (k_dep + k_leak)[h⁻¹]
```

With `k_dep + k_leak ≈ 2.0 h⁻¹` and `ACH_max ≈ 115 h⁻¹`:

| `C₀/C_safe` | `T_target` = 90 s | 150 s | 300 s | 600 s |
|---|---|---|---|---|
| **2×** (e.g. 120 → 60) | 26 ACH → `u ≈ 0.33` | 15 ACH → `u ≈ 0.28` | 7 ACH → **below `u_min`** | 3 ACH → **deposition alone suffices** |
| **5×** (e.g. 300 → 60) | 63 ACH → `u ≈ 0.58` | 37 ACH → `u ≈ 0.40` | 18 ACH → `u ≈ 0.29` | 8 ACH → below `u_min` |
| **10×** (e.g. 600 → 60) | **91 ACH → `u ≈ 0.82`** | 54 ACH → `u ≈ 0.53` | 26 ACH → `u ≈ 0.33` | 12 ACH → `u ≈ 0.26` |
| **16×** (e.g. 960 → 60) | **>115 ACH — INFEASIBLE** | 68 ACH → `u ≈ 0.62` | 33 ACH → `u ≈ 0.37` | 16 ACH → `u ≈ 0.28` |

*(duty values are illustrative, assuming a roughly linear `φ(u)` above `u_min = 0.25` and `ACH_max = 115`; replace them with your measured `φ(u)` after §8.3.)*

**Design consequence — deliberately include infeasible cells.** Roughly **15 % of Category-D episodes must be drawn from the infeasible or near-infeasible region** (bottom-left of the table). Without them the model never sees an episode where full duty still misses the deadline, and the MPC's soft-constraint slack behaviour (AQR-ARCH-001 §8.2.1) is untrained and untested. Mark these `feasible_flag = false` in the episode manifest.

**Design consequence — avoid the trivial region.** Cells where deposition alone meets the deadline (top-right) contain no information about `u` and must be capped at ≤ 5 % of episodes.

---

## 3. Parameter Registry

Role codes: **C** = controlled (you set it) · **M** = measured (the rig reports it) · **D** = derived (computed from M) · **X** = metadata/context · **N** = nuisance (uncontrolled but recorded).

### 3.1 Controlled Parameters (the experiment design factors)

| # | Parameter | Symbol / column | Unit | Range | Levels used | Resolution | Notes |
|---|---|---|---|---|---|---|---|
| C1 | Commanded duty cycle | `duty_cmd` | – | 0.00–1.00 | **0.00, 0.25, 0.30, 0.40, 0.50, 0.60, 0.75, 0.90, 1.00** (9 levels) | 1/2048 (11-bit LEDC) | Realisable set is `{0} ∪ [u_min, 1]`; values in `(0, u_min)` are dithered |
| C2 | Duty pattern | `pattern` | enum | – | `const`, `step_down`, `step_up`, `ramp`, `aprbs`, `pulse` | – | One per episode |
| C3 | Deadline | `t_target_s` | s | 60–900 | **90, 150, 300, 600** | 1 s | Rig-specific (§0 A-10); 900 s is meaningless here |
| C4 | Initial concentration band | `c0_band` | enum | – | `B1`…`B5` (§4.4) | – | Assigned from the **measured** peak, not the injected dose |
| C5 | Injected dose | `dose_ml` | mL | 0–150 | 0, 5, 15, 40, 100 (pilot-calibrated) | 1 mL | A nuisance knob used to *hit* a `c0_band`; not a modelling feature |
| C6 | Aerosol source | `source_type` | enum | – | `incense`, `nacl`, `cooking_oil`, `paper`, `none` | – | ≥ 3 sources mandatory (§4.6) |
| C7 | Relative-humidity band | `rh_band` | enum | – | `R1` 25–40 %, `R2` 40–60 %, `R3` 60–80 % | – | Set with a saturated-salt tray or a small humidifier before the episode |
| C8 | Temperature band | `t_band` | enum | – | `T1` 20–27 °C, `T2` 27–34 °C, `T3` 34–42 °C | – | Achieved with a small heater mat; **do not** achieve it by shortening warm-up |
| C9 | Inlet restriction | `inlet_pct` | % | 0–100 | 100 (default), 50, 25 | 5 % | Needle-valve opening; a disturbance factor in Category H only |
| C10 | Mixing fan | `mixfan_on` | bool | – | `true` always | – | **Constant. Never varied.** Logged only to prove it |
| C11 | Analyte for gas episodes | `gas_analyte` | enum | – | `none`, `co2`, `ipa` | – | `ipa` = isopropanol headspace for MQ-135. **Never a flammable analyte** |

### 3.2 Measured Parameters

| # | Parameter | Column | Unit | Valid range | Sensor resolution | Sensor / source | Accuracy caveat |
|---|---|---|---|---|---|---|---|
| M1 | PM1.0 mass | `pm1_ugm3` | µg/m³ | 0–999 | 1 | PMS5003 (CF=1 factory) | usable ≤ 500; above that mark `pm_saturated` |
| M2 | PM2.5 mass | `pm25_ugm3` | µg/m³ | 0–999 | 1 | PMS5003 | **primary modelling target** |
| M3 | PM10 mass | `pm10_ugm3` | µg/m³ | 0–999 | 1 | PMS5003 | PMS5003's PM10 is inferred, not sized — treat as a secondary channel |
| M4 | PM2.5 (atm. calib.) | `pm25_atm_ugm3` | µg/m³ | 0–999 | 1 | PMS5003 | second factory calibration; logged for comparison, not modelled |
| M5–M10 | Bin counts ≥0.3/0.5/1.0/2.5/5.0/10 µm | `nc0p3_cnt` … `nc10p0_cnt` | #/0.1 L | 0–65535 | 1 | PMS5003 | **less processed than mass — often the better feature** |
| M11 | PM validity | `pm_valid` | bool | – | – | firmware | false during < 30 s spin-up, on checksum fail, or RH > 90 % |
| M12 | CO₂ | `co2_ppm` | ppm | 0–5000 | 1 | MH-Z19B (0–5000 variant) | ±(50 ppm + 5 % of reading); **ABC disabled** |
| M13 | CO₂ validity | `co2_valid` | bool | – | – | firmware | false for the first 180 s after power-on |
| M14 | MQ-135 raw | `mq135_adc` | counts | 0–4095 | 1 | ADC1_CH6 | 64× oversampled, 16-sample median |
| M15 | MQ-2 raw | `mq2_adc` | counts | 0–4095 | 1 | ADC1_CH7 | as above |
| M16 | MQ-135 resistance | `mq135_rs_ohm` | Ω | 200–200 000 | – | derived from M14 | needs the measured `R_L` |
| M17 | MQ-2 resistance | `mq2_rs_ohm` | Ω | 200–200 000 | – | derived from M15 | |
| M18 | Temperature | `temp_c` | °C | 15–50 | 0.1 | DHT22 | ±0.5 °C; min sampling interval 2 s → held between updates |
| M19 | Relative humidity | `rh_pct` | % | 0–100 | 0.1 | DHT22 | ±2–5 % |
| M20 | T/RH validity | `th_valid` | bool | – | – | firmware | false on CRC failure |
| M21 | Applied duty | `duty_applied` | – | 0.00–1.00 | 1/2048 | firmware | **post-clamp, post-dither.** This — not `duty_cmd` — is the model input |
| M22 | Bus voltage | `vbus_v` | V | 0–15 | 0.01 | GPIO39 divider | 1 s mean of 10 Hz samples |
| M23 | Bus voltage σ | `vbus_v_std` | V | 0–3 | 0.001 | derived | rail-sag diagnostic |
| M24 | Motor current | `imotor_a` | A | 0–1.5 | 0.005 | ACS712 | 1 s mean of 10 Hz samples |
| M25 | Motor current σ | `imotor_a_std` | A | 0–1.0 | 0.001 | derived | commutation/stall diagnostic |
| M26 | Impeller speed | `rpm` | rev/min | 0–8000 | 30 | tachometer | 0 below `u_min` |
| M27 | Ambient temperature | `amb_temp_c` | °C | 10–45 | 0.1 | external DHT22 | outside the chamber; drives the thermal boundary |
| M28 | Ambient humidity | `amb_rh_pct` | % | 0–100 | 0.1 | external DHT22 | |
| M29 | Ambient PM2.5 | `amb_pm25_ugm3` | µg/m³ | 0–999 | 1 | *optional* second PM sensor | if fitted, this is `C_in` — otherwise assume 0 and say so |
| M30 | Wi-Fi RSSI | `rssi_dbm` | dBm | −100–0 | 1 | ESP32 | separates network artefacts from physical ones |
| M31 | Free heap | `heap_free_b` | B | 0–320000 | 1 | ESP32 | leak detection over long sessions |

### 3.3 Derived Parameters

| # | Parameter | Column | Unit | Formula | When computed |
|---|---|---|---|---|---|
| D1 | MQ-135 ratio | `mq135_ratio` | – | `Rs / R0_mq135` | on-edge, live `R0` |
| D2 | MQ-2 ratio | `mq2_ratio` | – | `Rs / R0_mq2` | on-edge |
| D3 | T/RH-corrected MQ-135 ratio | `mq135_ratio_corr` | – | `mq135_ratio / f(T,RH)` | processed layer |
| D4 | Hygroscopically-corrected PM2.5 | `pm25_dry_ugm3` | µg/m³ | `pm25 / (1 + κ·RH/(100−RH))` | processed layer, `κ` in `calibration.yaml` |
| D5–D8 | Sub-indices | `si_pm25`, `si_pm10`, `si_gas`, `si_co2` | index | CPCB piecewise-linear (§2.4) | on-edge |
| D9 | Composite index | `i_inst` | index | `max(D5..D8)` | on-edge |
| D10 | Binding channel | `i_driver` | enum | argmax of D5–D8 | on-edge |
| D11 | PM2.5 rate | `dpm25_dt` | µg/m³/s | centred 5-point Savitzky–Golay derivative | processed layer |
| D12 | CO₂ rate | `dco2_dt` | ppm/s | as above | processed layer |
| D13 | Instantaneous total removal rate | `k_tot_est_hr` | h⁻¹ | `−3600 · (dC/dt)/(C − C_in)` | processed layer; noisy near `C ≈ C_in`, null it below `C − C_in < 20` |
| D14 | Estimated forced exchange | `lambda_est_hr` | h⁻¹ | `k_tot_est_hr − k_dep − k_leak` | processed layer |
| D15 | Electrical power | `pelec_w` | W | `vbus_v · imotor_a` | processed layer |
| D16 | Cumulative energy | `energy_wh_cum` | Wh | `∫ pelec_w dt / 3600`, reset at episode start | processed layer |
| D17 | Objective proxy | `v2_integral_cum` | V²·s | `∫ (duty_applied·vbus_v)² dt` | processed layer |
| D18 | Elapsed time | `t_rel_s` | s | `ts_utc − episode.t0_utc` | interim layer |
| D19 | Time remaining | `t_remaining_s` | s | `t_target_s − t_rel_s` | processed layer |
| D20 | Cumulative exposure | `exposure_idx_s` | index·s | `∫ max(i_inst − i_safe, 0) dt` | processed layer |

### 3.4 Metadata / Context Parameters

| # | Parameter | Column | Type | Notes |
|---|---|---|---|---|
| X1 | Timestamp (UTC) | `ts_utc` | ISO 8601 with `+00:00`, µs precision | NTP-disciplined at the edge; **never** re-stamped by the host |
| X2 | Sequence | `seq` | int64 | Edge monotonic counter; gap detection |
| X3 | Episode ID | `episode_id` | string | §7.2 naming rule |
| X4 | Session ID | `session_id` | string | One session = one continuous rig power-up |
| X5 | Node ID | `node_id` | string | `R1N1` for the single demo node |
| X6 | Firmware version | `fw_version` | string | semver + git short SHA |
| X7 | Rig configuration hash | `rig_cfg_sha` | string(12) | SHA-256 prefix of `rig_config.yaml` — **any physical change must change this** |
| X8 | Calibration set | `calib_id` | string | Points at the `calibration/` file in force |
| X9 | Edge mode | `mode` | enum | `WARMUP`, `IDLE_MONITOR`, `ACTIVE_REMEDIATION`, `SENSOR_FAULT`, … |
| X10 | Experiment category | `category` | enum `A`–`J` | §4.1 |
| X11 | QC flag bitmask | `qc_flags` | int32 | §9.2 |
| X12 | QC status | `qc_status` | enum | `accept` / `quarantine` / `reject` |
| X13 | Split assignment | `split` | enum | `train` / `val` / `test` — **assigned once, at episode level, never per row** |
| X14 | Operator | `operator` | string | initials; catches per-operator injection technique differences |
| X15 | Free-text note | `note` | string | anything unusual; searched during post-mortems |

### 3.5 Parameters Deliberately **Not** Collected

| Excluded | Why |
|---|---|
| `duty_cmd` as a *model feature* | Only `duty_applied` may be a feature. Using the command teaches the model the controller, not the plant |
| MQ "ppm" values | Datasheet ppm curves are single-analyte, dry-air extrapolations. Fabricated numbers must never enter a dataset |
| Any future-derived quantity (`t_reach`, final `C`) as a per-row feature | Direct target leakage |
| Interpolated/imputed sensor values in the **raw** layer | Raw stays raw. Imputation happens in `processed/` and is flagged |
| Manual "corrections" to outliers | Quarantine, never edit (§9.3) |
| Wall-clock local time | UTC only; local time is derived at display time |

---

## 4. Data Categories and How to Categorise

### 4.1 Primary Taxonomy — Experiment Category (`category`)

This is the top-level axis. Every episode belongs to exactly one.

| Code | Category | Turbine | Injection | What it is *for* | Usable for dynamics training? |
|---|---|---|---|---|---|
| **A** | **Background / idle** | off | none | Noise floor, sensor drift, diurnal baseline, `R0` tracking | No — but mandatory for drift detection and for the false-trigger metric |
| **B** | **Natural decay** | **off (`u = 0`)** | PM | Identifies `k_dep`. The `u = 0` anchor of the whole response surface | **Yes — critical** |
| **C** | **Tracer identification** | 9 duty levels | CO₂ | Identifies `k_leak` and `Q(u)` exactly (`k_dep,CO₂ = 0`) | Yes, and it is the source of the `φ(u)` calibration |
| **D** | **Constant-duty PM decay** | constant, 9 levels | PM | **The main corpus.** The response surface over `(u, C₀, RH, T)` | **Yes — the bulk** |
| **E** | **Step-change duty** | one step mid-decay | PM | Transient response, actuator lag, model behaviour under a changing input | **Yes** |
| **F** | **APRBS excitation** | multi-level pseudo-random | PM | Persistent excitation; the frequency content a constant-duty set can never provide | **Yes — highest information per episode** |
| **G** | **Multi-source / mixed** | varied | 2–3 sources, sometimes simultaneous | Prevents overfitting to incense optics; cross-sensitivity realism | Yes |
| **H** | **Disturbance robustness** | varied | PM | Inlet restriction, door-crack leak, external fan, ambient swing | Yes, flagged |
| **I** | **Fault & anomaly (labelled)** | varied | varied | Sensor unplugged, saturated, impeller blocked, clock step, brown-out | **No — excluded from dynamics training.** Trains the validity/health classifier and tests QC |
| **J** | **Closed-loop** | MPC-commanded | PM | Phase-C data once the controller runs; evaluation, not initial training | Yes, but **only with the exploration flag**, and never mixed into the open-loop identification set |

### 4.2 Secondary Axis — Duty Band (`duty_band`)

Assigned from the **episode-mean `duty_applied`** during the decay window.

| Band | Range | Meaning |
|---|---|---|
| `U0` | exactly 0 | passive: deposition + leak only |
| `U1` | 0 < ū ≤ 0.35 | near the stiction threshold — the most nonlinear, most under-sampled region |
| `U2` | 0.35 < ū ≤ 0.55 | low |
| `U3` | 0.55 < ū ≤ 0.75 | mid |
| `U4` | 0.75 < ū ≤ 0.92 | high |
| `U5` | ū > 0.92 | full |

**Balance rule: no duty band may hold more than 25 % or fewer than 8 % of Category-D episodes.** Left to itself, a campaign drifts toward mid duties because they are pleasant to run; the model then extrapolates badly at exactly the two ends the optimizer likes to visit.

### 4.3 Tertiary Axis — Source (`source_type`)

| Code | Source | Preparation | Typical size mode | Why include it |
|---|---|---|---|---|
| `incense` | Incense stick smoke | Burn in a 1 L priming jar, cap, settle 60 s, draw with syringe | ~0.2–0.4 µm | Primary: cheap, repeatable, dense fine fraction |
| `nacl` | Nebulised 0.9 % NaCl, dried | Ultrasonic nebuliser → silica-gel drying tube → priming jar | ~0.3–0.6 µm | Non-combustion control: no VOCs, so it isolates the PM channel from the MQ channels |
| `cooking_oil` | Heated oil aerosol | Hot plate + 2 mL oil in the priming jar | ~0.3–1.0 µm, liquid | Domain realism for the kitchen use case; liquid droplets scatter differently |
| `paper` | Smouldering paper strip | Priming jar, flame extinguished | broad, coarser tail | Loads the PM10 channel, which incense barely touches |
| `none` | – | – | – | Categories A and C |

**Rule: Category D must contain at least 3 distinct sources, and no single source may exceed 60 % of Category-D episodes.** A model trained only on incense learns incense's specific refractive index and size distribution, and its "µg/m³" becomes source-specific. `source_type` is recorded as a categorical feature so the model can account for it explicitly.

### 4.4 Initial-Concentration Bands (`c0_band`) — assignment rule

`C₀` is defined as the **maximum 10 s rolling mean of `pm25_ugm3`** in the window from injection start to injection start + 120 s. It is **measured, not commanded** — the injected dose is only a means of landing in a band.

| Band | `C₀` (µg/m³) | `C₀ / C_safe` at `C_safe = 60` | CPCB category at that level | Target share of Cat. D |
|---|---|---|---|---|
| `B1` | 60 – 120 | 1–2× | Moderate | 15 % |
| `B2` | 120 – 250 | 2–4× | Moderate → Poor | 25 % |
| `B3` | 250 – 450 | 4–7.5× | Very Poor | 30 % |
| `B4` | 450 – 700 | 7.5–12× | Severe | 20 % |
| `B5` | 700 – 950 | 12–16× | Severe (near sensor saturation) | 10 % |

Episodes with `C₀ < 60` are **rejected** (nothing to remediate). Episodes with `C₀ > 950` are **rejected** (`QC_C0_OUT_OF_RANGE`, §9.2) — the PMS5003 is no longer trustworthy there, and a saturated sensor produces a flat top that a model will happily interpret as physics.

### 4.5 Humidity and Temperature Bands

| `rh_band` | Range | | `t_band` | Range |
|---|---|---|---|---|
| `R1` | 25–40 % | | `T1` | 20–27 °C |
| `R2` | 40–60 % | | `T2` | 27–34 °C |
| `R3` | 60–80 % | | `T3` | 34–42 °C |

RH above 80 % is outside the band grid and is **excluded from Category D**: hygroscopic growth dominates the optical measurement and the correction of §2.4 is itself uncertain there. Collect 6–8 episodes above 90 % RH deliberately and label them Category **I** (`QC_PM_RH_INVALID`) — they are what proves the FR-14 invalidation rule works.

### 4.6 Feasibility Label (`feasible_flag`)

Computed after the episode from the measured `C₀`, the measured `φ(u)` and `t_target_s` using §2.5:

- `true` — some `u ≤ 1` reaches `C_safe` by `t_target_s`
- `false` — even full duty misses the deadline (target: **15 %** of Category D)
- `trivial` — deposition alone suffices, `u = 0` would have worked (cap at **5 %**)

### 4.7 The Categorisation Decision Tree (apply in this order)

```
1. Was the turbine ever commanded non-zero?        no ──► u=0 branch (2)   yes ──► (3)
2. Was anything injected?     no ──► A (background)      CO2 ──► C (u=0 leak test)      PM ──► B
3. Was a fault deliberately induced or detected?   yes ──► I  (stop here)
4. Was the analyte CO2?                            yes ──► C
5. Was the controller in command (MPC)?            yes ──► J
6. Duty pattern:  const ──► (7)   step/ramp ──► E   aprbs/pulse ──► F
7. Was a disturbance factor varied (inlet, door, external fan)?   yes ──► H
8. More than one source injected, or a non-primary source?        yes ──► G
9. Otherwise ──► D
```

---

## 5. Dataset Size — Number of Episodes and Records

### 5.1 Episode and Record Budget

Two tiers. **Tier 1 is the minimum that yields a trainable model; Tier 2 is the target for a defensible capstone result.** Row counts are at 1 Hz over the *logged* window (pre-roll + injection + decay + post-roll).

| Cat. | Description | Tier 1 episodes | Tier 2 episodes | Mean logged duration | Tier 1 rows | Tier 2 rows | Tier 1 rig-hours | Tier 2 rig-hours |
|---|---|---|---|---|---|---|---|---|
| A | Background / idle | 3 × 8 h | 7 × 24 h | continuous | 86 400 | 604 800 | 24.0 *(unattended)* | 168.0 *(unattended)* |
| B | Natural decay (`u=0`) | 12 | 20 | 1260 s | 15 120 | 25 200 | 5.2 | 8.7 |
| C | CO₂ tracer identification | 20 | 30 | 540 s | 10 800 | 16 200 | 4.0 | 6.0 |
| D | Constant-duty PM decay | 72 | 180 | 420 s | 30 240 | 75 600 | 12.0 | 30.0 |
| E | Step-change duty | 20 | 60 | 600 s | 12 000 | 36 000 | 4.3 | 13.0 |
| F | APRBS excitation | 12 | 40 | 780 s | 9 360 | 31 200 | 3.2 | 10.7 |
| G | Multi-source / mixed | 18 | 45 | 420 s | 7 560 | 18 900 | 3.0 | 7.5 |
| H | Disturbance robustness | 10 | 30 | 480 s | 4 800 | 14 400 | 1.8 | 5.5 |
| I | Fault & anomaly | 8 | 25 | 300 s | 2 400 | 7 500 | 1.1 | 3.3 |
| **Total — attended (B–I)** | | **172** | **430** | | **92 280** | **225 000** | **34.6 h** | **84.7 h** |
| **Total incl. Category A** | | 172 + A | 430 + A | | **178 680** | **829 800** | 34.6 h attended + 24 h unattended | 84.7 h attended + 168 h unattended |

**Storage:** Tier 1 ≈ **47 MB** as CSV, ≈ **6.5 MB** as Parquet (Zstd). Tier 2 ≈ **216 MB** CSV, ≈ **30 MB** Parquet. Raw JSONL capture is roughly 2.2× the CSV size. All of it fits on a laptop; retention is set by usefulness, not by disk.

**Calendar:** Tier 1 ≈ **6 working days** of attended chamber time (6 episodes/hour average, 6 h/day) plus 3 unattended overnight background runs. Tier 2 ≈ **15 working days** plus a week of background. Category-A background runs go overnight and cost no attended time.

### 5.2 The Sample-Size Number That Actually Matters

> **Effective sample size = 172 (Tier 1) or 430 (Tier 2) episodes — not 180 000 or 830 000 rows.**

Consecutive 1 Hz samples inside one exponential decay are almost perfectly autocorrelated; they carry roughly one independent observation *per episode*, not per row. Every model-capacity decision, every train/test split and every confidence interval must be computed on **episode count**. Reporting "830 000 training samples" is a statement about disk usage.

Minimum episode counts before a model class is permitted to train (from AQR-ARCH-001 §7.3, restated for this rig):

| Model | Minimum | Comfortable | Blocking category |
|---|---|---|---|
| Grey-box parameters `k_leak`, `k_dep`, `φ(u)` | **B ≥ 8, C ≥ 18** | B ≥ 20, C ≥ 30 | C must cover all 9 duty levels ≥ 2× |
| XGBoost residual | **60 dynamics episodes** (B+D+E+F+G+H) | 160 | needs `U1` and `U5` bands populated |
| NARX residual | **100 dynamics episodes** | 200+ | needs Categories E and F for excitation |

Tier 1 provides 144 dynamics episodes → XGBoost trainable, NARX marginal. Tier 2 provides 375 → both comfortable.

### 5.3 Balance Constraints (checked automatically at packaging, §8.6)

| Constraint | Rule |
|---|---|
| Duty band coverage | Every band `U0`–`U5` ≥ 8 % and ≤ 25 % of Category-D episodes |
| `c0_band` coverage | Within ±5 pp of the shares in §4.4 |
| Source coverage | ≥ 3 sources; none > 60 % of Category D |
| RH band coverage | Each of `R1`–`R3` ≥ 20 % of Category D |
| Feasibility mix | `feasible_flag = false` between 10 % and 20 %; `trivial` ≤ 5 % |
| Temporal spread | No calendar day contributes > 20 % of episodes in any single category |
| Replication | Every `(duty level × c0_band)` cell that is used at all has ≥ 2 replicates on **different days** |
| Drift controls | ≥ 2 fixed-reference control episodes (`u = 0.60`, `B3`, `incense`, `R2`, `T1`) every session |

---

## 6. Dataset Format

### 6.1 Three-Layer Architecture

| Layer | Directory | Format | Mutability | Contents |
|---|---|---|---|---|
| **Raw** | `data/raw/` | **JSONL**, gzip, one line per MQTT telemetry frame, byte-for-byte as received | **IMMUTABLE. Write once, never edit, never delete.** | The permanent evidence. If a bug is found in parsing three months from now, everything downstream is rebuilt from here |
| **Interim** | `data/interim/` | **Parquet** (Zstd level 3), one file per episode, plus a CSV mirror | Rebuildable | Parsed, type-cast, gap-detected, time-aligned to a strict 1 Hz grid, episode-sliced. **No feature engineering, no imputation** |
| **Processed** | `data/processed/` | **Parquet** (Zstd level 3), partitioned by `category`, plus a single CSV export | Rebuildable | Derived columns, corrections, denormalised episode metadata, split assignment. This is what the model reads |

**Rule: every layer is reproducible from the one above it by a versioned script.** If you cannot regenerate `processed/` from `raw/` with one command, the dataset is not a dataset — it is a folder.

### 6.2 Encoding Conventions (apply everywhere, no exceptions)

| Aspect | Rule |
|---|---|
| Text encoding | UTF-8, no BOM |
| Line endings | LF (`\n`) only |
| Timestamps | ISO 8601 with explicit offset, microsecond precision, **UTC always**: `2026-09-10T04:15:30.120000+00:00` |
| Decimal separator | `.` — never a comma; no thousands separators |
| Booleans | `true` / `false` lowercase in CSV; native boolean in Parquet |
| Missing values | **Empty field in CSV; `null` in Parquet.** Never `-999`, never `NA`, never `NaN`, never `0` |
| Enumerations | lowercase snake_case string literals from the fixed vocabularies in §3 |
| Units | Encoded in the column name suffix (`_ugm3`, `_ppm`, `_c`, `_pct`, `_a`, `_v`, `_w`, `_s`, `_hr`, `_cnt`, `_ohm`, `_b`, `_dbm`) — the schema is self-documenting |
| Column order | **Fixed, as listed in §6.3.** Do not reorder; downstream positional reads exist |
| Numeric types | `float32` for sensor channels, `float64` for cumulative integrals and time, `int32`/`int64` for counts and sequence, `bool` for flags |
| Compression | Parquet + Zstd level 3; JSONL + gzip level 6 |
| Sort order | Every file strictly ascending by `ts_utc`, no duplicate timestamps |

### 6.3 Time-Series Schema — exact column order

**`interim/` — 49 columns.** CSV header line, verbatim:

```
ts_utc,t_rel_s,seq,episode_id,session_id,node_id,pm1_ugm3,pm25_ugm3,pm10_ugm3,pm25_atm_ugm3,nc0p3_cnt,nc0p5_cnt,nc1p0_cnt,nc2p5_cnt,nc5p0_cnt,nc10p0_cnt,pm_valid,co2_ppm,co2_valid,mq135_adc,mq135_rs_ohm,mq135_ratio,mq2_adc,mq2_rs_ohm,mq2_ratio,temp_c,rh_pct,th_valid,duty_cmd,duty_applied,vbus_v,vbus_v_std,imotor_a,imotor_a_std,rpm,mixfan_on,si_pm25,si_pm10,si_gas,si_co2,i_inst,i_driver,mode,amb_temp_c,amb_rh_pct,amb_pm25_ugm3,rssi_dbm,heap_free_b,qc_flags
```

**`processed/` — the 49 above, then 13 derived, then 15 denormalised episode attributes = 77 columns.** Additional header segment, verbatim:

```
pm25_dry_ugm3,mq135_ratio_corr,mq2_ratio_corr,dpm25_dt,dco2_dt,k_tot_est_hr,lambda_est_hr,pelec_w,energy_wh_cum,v2_integral_cum,t_remaining_s,exposure_idx_s,phase,category,source_type,pattern,c0_band,rh_band,t_band,duty_band,inlet_pct,t_target_s,feasible_flag,gas_analyte,split,qc_status,fw_version,rig_cfg_sha,calib_id
```

**Type and constraint table (abbreviated to the fields with non-obvious rules):**

| Column | Type | Nullable | Constraint | Violation → |
|---|---|---|---|---|
| `ts_utc` | timestamp[us, UTC] | no | strictly increasing, exactly 1.000 s spacing after gridding | `QC_TIME_IRREGULAR` |
| `t_rel_s` | float64 | no | ≥ −60 (pre-roll is negative), monotone | reject |
| `seq` | int64 | no | strictly increasing; gaps allowed but counted | `QC_SEQ_GAP` |
| `episode_id` | string | no | matches the regex in §7.2 | reject |
| `pm25_ugm3` | float32 | **yes** | 0 ≤ x ≤ 999; null when `pm_valid = false` | `QC_PM_INVALID` |
| `pm_valid` | bool | no | false during spin-up, bad checksum, RH > 90 % | — |
| `co2_ppm` | float32 | yes | 350 ≤ x ≤ 5000 | `QC_CO2_RANGE` |
| `mq135_ratio` | float32 | yes | 0.02 ≤ x ≤ 5.0 | `QC_MQ_RANGE` |
| `temp_c` | float32 | yes | 10 ≤ x ≤ 55 | `QC_TH_RANGE` |
| `rh_pct` | float32 | yes | 0 ≤ x ≤ 100 | `QC_TH_RANGE` |
| `duty_applied` | float32 | no | 0 ≤ x ≤ 1; equals 0 or ≥ `u_min` after dither averaging | `QC_DUTY_INVALID` |
| `vbus_v` | float32 | no | 10.5 ≤ x ≤ 13.5 | `QC_RAIL_SAG` |
| `imotor_a` | float32 | no | 0 ≤ x ≤ 1.5 | `QC_CURRENT_RANGE` |
| `rpm` | int32 | yes | 0, or 1500 ≤ x ≤ 8000 | `QC_RPM_IMPLAUSIBLE` |
| `i_inst` | float32 | no | 0 ≤ x ≤ 500 | reject |
| `i_driver` | dictionary<string> | no | one of `pm25`,`pm10`,`gas`,`co2` | reject |
| `phase` | dictionary<string> | no | one of `preroll`,`inject`,`decay`,`postroll` | reject |
| `k_tot_est_hr` | float32 | **yes** | null when `(C − C_in) < 20 µg/m³` — the derivative is meaningless there | — |
| `qc_flags` | int32 | no | bitmask, §9.2 | — |
| `split` | dictionary<string> | no | `train`/`val`/`test`, **constant within an episode** | reject |

### 6.4 Episode Manifest Schema — `data/processed/episodes.csv` (and `.parquet`)

One row per episode. This is the file a human reads and the file the DoE balance checks run against.

```
episode_id,session_id,node_id,category,pattern,source_type,gas_analyte,dose_ml,
t0_utc,t_inject_utc,t_decay_start_utc,t_end_utc,duration_s,n_rows,
duty_cmd_nominal,duty_mean_applied,duty_band,duty_levels_json,
c0_pm25_ugm3,c0_band,c_end_pm25_ugm3,c_safe_ugm3,i0_inst,i_end_inst,
t_target_s,t_reach_s,deadline_met,feasible_flag,
rh_mean_pct,rh_band,temp_mean_c,t_band,amb_temp_mean_c,amb_rh_mean_pct,
inlet_pct,mixfan_on,
k_tot_fit_hr,lambda_fit_hr,r2_fit,tau_eff_s,
energy_wh,v2_integral,peak_current_a,
r0_mq135_ohm,r0_mq2_ohm,v_eff_l,u_min,
fw_version,rig_cfg_sha,calib_id,operator,
qc_status,qc_flags,exclusion_reason,split,note,
raw_sha256,interim_sha256
```

| Field group | Purpose |
|---|---|
| Identity + timing | Joins to the time series; defines the four phase boundaries |
| Design factors | Everything the DoE balance checker in §5.3 reads |
| Outcomes | `c0`, `t_reach_s`, `deadline_met`, `energy_wh` — the evaluation quantities |
| Fitted physics | `k_tot_fit_hr`, `lambda_fit_hr`, `r2_fit` — **`r2_fit < 0.97` is a QC failure**: it means the decay was not exponential, which means the chamber was not well mixed |
| Provenance | `fw_version`, `rig_cfg_sha`, `calib_id`, `operator`, checksums |
| Governance | `qc_status`, `exclusion_reason`, `split` |

### 6.5 Raw JSONL Line Format

One JSON object per line, exactly as published on `.../telemetry`, wrapped with reception metadata. Nothing is removed, reformatted or rounded.

```json
{"rx_utc":"2026-09-10T04:15:30.128411+00:00","topic":"aqr/v1/site/lab/zone/chamber/node/R1N1/telemetry","qos":0,"payload":{"seq":184213,"ts":1788412530.120,"pm":{"p1":6.2,"p25":148.4,"p10":191.0,"n03":4120,"valid":true},"co2":{"ppm":1180,"valid":true},"gas":{"mq135_ratio":2.41,"mq2_ratio":1.07,"mq135_raw":1832,"mq2_raw":901},"th":{"t":31.4,"rh":58.2,"valid":true},"act":{"duty":0.62,"v_bus":11.83,"i_mot":0.149,"rpm":4180},"idx":{"i_inst":168.2,"driver":"pm25"},"mode":"ACTIVE_REMEDIATION"}}
```

---

## 7. Folder Structure and Naming

### 7.1 Directory Tree

```
ventis-chamber-25/
├── README.md                          ← 5-line orientation, points at DATASET_CARD.md
├── DATASET_CARD.md                    ← the dataset card (§11.1)
├── CHANGELOG.md                       ← every version bump, what changed, why
├── VERSION                            ← single line, e.g. "1.2.0"
├── LICENSE
├── .gitignore                         ← ignores data/raw/, data/interim/, data/processed/
├── .gitattributes                     ← LFS/DVC pointers if used
├── checksums/
│   ├── raw.sha256
│   ├── interim.sha256
│   └── processed.sha256
│
├── config/
│   ├── rig_config.yaml                ← physical rig definition; its SHA is rig_cfg_sha
│   ├── schema_timeseries.yaml         ← the 49/77-column contract, machine-readable
│   ├── schema_episodes.yaml
│   ├── qc_rules.yaml                  ← thresholds for every rule in §9
│   ├── doe_plan.csv                   ← the randomised run order, generated once, seed recorded
│   └── split_policy.yaml              ← how splits are assigned; seed recorded
│
├── calibration/
│   ├── calib_2026-09-08.yaml          ← R0, R_L, V_eff, u_min, k_leak, k_dep, phi_u, kappa
│   ├── phi_u_2026-09-08.csv           ← the measured duty → Q(u) curve, 9 points + spline knots
│   └── adc_cal_2026-09-08.yaml        ← esp_adc_cal two-point values per channel
│
├── data/
│   ├── raw/                           ← IMMUTABLE
│   │   └── 2026-09-10/
│   │       ├── S20260910_R1_01.session.jsonl.gz          ← whole-session capture
│   │       └── E20260910T041530Z_R1_D_INC_012.raw.jsonl.gz
│   ├── interim/
│   │   └── 2026-09-10/
│   │       ├── E20260910T041530Z_R1_D_INC_012.parquet
│   │       └── E20260910T041530Z_R1_D_INC_012.csv        ← mirror, for eyeballing
│   └── processed/
│       ├── episodes.parquet
│       ├── episodes.csv
│       ├── splits.parquet
│       ├── timeseries/
│       │   ├── category=A/part-0000.parquet
│       │   ├── category=B/part-0000.parquet
│       │   ├── category=C/part-0000.parquet
│       │   ├── category=D/part-0000.parquet   (… E, F, G, H, I, J)
│       └── export/
│           └── ventis-chamber-25_v1.2.0_full.csv.gz      ← single flat CSV, for portability
│
├── logs/
│   ├── sessions/
│   │   └── S20260910_R1_01.session.yaml     ← operator log, one per session
│   ├── incidents/
│   │   └── 2026-09-11_impeller_jam.md
│   └── qc/
│       └── qc_report_v1.2.0.html
│
├── notebooks/                          ← exploration only; nothing here is a pipeline step
└── docs/
    ├── DATASET-SPEC-VENTIS-25cm.md     ← this document
    └── rig_photos/
```

### 7.2 Naming Rules

**Episode ID — the primary key of the entire dataset.**

```
E<YYYYMMDD>T<HHMMSS>Z_<RIG>_<CAT>_<SRC>_<NNN>

regex:  ^E\d{8}T\d{6}Z_R\d_[A-J]_(INC|NAC|OIL|PAP|CO2|NON)_\d{3}$
example: E20260910T041530Z_R1_D_INC_012
```

| Segment | Meaning | Vocabulary |
|---|---|---|
| `E…Z` | UTC start instant of the episode (`t0`), to the second | — |
| `R1` | Rig identifier | `R1` (add `R2` only for a second physical chamber) |
| `D` | Experiment category | `A`–`J` per §4.1 |
| `INC` | Source | `INC` incense · `NAC` NaCl · `OIL` cooking oil · `PAP` paper · `CO2` carbon dioxide · `NON` none |
| `012` | Zero-padded sequence within the session | `001`–`999` |

**Session ID:** `S<YYYYMMDD>_<RIG>_<NN>` → `S20260910_R1_01`. One session = one continuous rig power-up. A power cycle starts a new session, because it resets thermal state and `R0` conditioning.

**File names:**

| Artefact | Pattern | Example |
|---|---|---|
| Raw episode capture | `<episode_id>.raw.jsonl.gz` | `E20260910T041530Z_R1_D_INC_012.raw.jsonl.gz` |
| Raw session capture | `<session_id>.session.jsonl.gz` | `S20260910_R1_01.session.jsonl.gz` |
| Interim time series | `<episode_id>.parquet` / `.csv` | `E20260910T041530Z_R1_D_INC_012.parquet` |
| Session operator log | `<session_id>.session.yaml` | `S20260910_R1_01.session.yaml` |
| Calibration set | `calib_<YYYY-MM-DD>.yaml` | `calib_2026-09-08.yaml` |
| Dataset export | `ventis-chamber-25_v<VER>_full.csv.gz` | `ventis-chamber-25_v1.2.0_full.csv.gz` |
| QC report | `qc_report_v<VER>.html` | `qc_report_v1.2.0.html` |

**Hard naming rules:** lowercase for directories, the given case for IDs · **no spaces, ever** · no characters outside `[A-Za-z0-9._-]` · date directories as `YYYY-MM-DD` (sorts correctly) · IDs inside filenames as `YYYYMMDD` (compact) · no version number inside a per-episode filename — versions apply to the dataset, not to episodes · **never rename an episode after it is written**; if the ID is wrong, quarantine it and record the correction in `CHANGELOG.md`.

---

## 8. Step-by-Step Creation Procedure

### 8.1 Phase 0 — Create the Dataset Folder (do this first, before any hardware)

Run once, from wherever you keep projects. These are setup commands, not project code.

```bash
# 1. Skeleton
mkdir -p ventis-chamber-25/{config,calibration,checksums,notebooks,docs/rig_photos}
mkdir -p ventis-chamber-25/data/{raw,interim}
mkdir -p ventis-chamber-25/data/processed/{timeseries,export}
mkdir -p ventis-chamber-25/logs/{sessions,incidents,qc}
cd ventis-chamber-25

# 2. Category partitions for the processed layer
for c in A B C D E F G H I J; do mkdir -p "data/processed/timeseries/category=$c"; done

# 3. Version and governance stubs
echo "0.1.0" > VERSION
printf '# Changelog\n\n## 0.1.0 — %s\n- Skeleton created.\n' "$(date -u +%F)" > CHANGELOG.md
: > checksums/raw.sha256 ; : > checksums/interim.sha256 ; : > checksums/processed.sha256

# 4. Keep raw data out of git; track only text and config
cat > .gitignore <<'G'
data/raw/
data/interim/
data/processed/
!data/processed/episodes.csv
notebooks/.ipynb_checkpoints/
G

# 5. Verify
find . -type d | sort
```

**Expected result: 26 directories, no data files.** Commit this skeleton before collecting anything — the folder structure is part of the specification, not an afterthought.

Then populate `config/` with the five files from §11 (`rig_config.yaml`, `schema_timeseries.yaml`, `schema_episodes.yaml`, `qc_rules.yaml`, `split_policy.yaml`) and commit again. **`rig_config.yaml` must exist before the first episode**, because its SHA-256 prefix becomes `rig_cfg_sha` on every row.

### 8.2 Phase 1 — Build and Instrument the Rig (checklist)

| # | Task | Acceptance |
|---|---|---|
| 1.1 | Cut and bond the 250 mm cube from 5 mm acrylic; silicone every seam internally | No light visible through any seam |
| 1.2 | Fit the gasketed access door with over-centre latches | Compresses the foam gasket visibly along its whole length |
| 1.3 | Drill and fit P1 (40 mm exhaust), P2 (40 mm inlet + HEPA + needle valve), P3 (10 mm septum), P4 (M16 gland), P5 (mixing-fan mount) | Ports at the heights in §1.1 |
| 1.4 | Mount sensors at the specified positions; DHT22 ≥ 60 mm from the MQ pair | Photograph and store in `docs/rig_photos/` |
| 1.5 | Route the harness through P4; seal the gland with neutral-cure silicone | 24 h cure before any leak test |
| 1.6 | Mount the ESP32, driver, TC4420 and PSU **outside** the chamber on a DIN rail or plate | Nothing dissipating > 0.2 W inside except the sensors |
| 1.7 | Fit the tachometer on the impeller; verify a clean pulse train on a scope | Stable count at fixed duty, ±2 % |
| 1.8 | Fit ACS712 in the motor return and the 10 k/2.2 k divider on the 12 V bus | Readings track a bench meter within 5 % |
| 1.9 | Measure each MQ module's load resistor `R_L` with the module unpowered | Record both values in `calibration/` — factory pots vary by ±30 % |
| 1.10 | Wire the mixing fan to a fixed 5 V, independent of the ESP32 | It must run even if the ESP32 resets |
| 1.11 | Star-ground the motor return at the PSU terminal | Ground-bounce check: `imotor_a` step of 0.5 A shifts `mq135_adc` by < 3 counts |
| 1.12 | Write `config/rig_config.yaml`, commit, record its SHA | `rig_cfg_sha` populated |

### 8.3 Phase 2 — Commissioning and Calibration (the numbered sequence — order matters)

> **Nothing collected before step 16 completes belongs in the dataset.**

1. **MQ burn-in.** Power the MQ pair continuously for **48 hours** with the chamber door open. Do not interrupt. This is not optional; `R0` before 48 h is not a number, it is a transient.
2. **ADC characterisation.** Apply `esp_adc_cal` two-point + eFuse Vref; verify each ADC1 channel against a bench reference at 0.5 / 1.5 / 2.5 V. Record in `calibration/adc_cal_<date>.yaml`. Acceptance: ≤ 1 % error across 0.15–2.45 V.
3. **Sensor sanity.** Confirm the PMS5003 reads < 5 µg/m³ in filtered air, the MH-Z19B reads 400–500 ppm in outdoor air with ABC **disabled**, and the DHT22 tracks a reference hygrometer within 5 % RH.
4. **Measure `V_int`.** Remove every internal part, measure its volume by water displacement, sum. Record.
5. **Measure `V_eff` geometrically.** `V_eff = 15.625 L − V_int`. Expected 14.8–15.1 L.
6. **Cross-check `V_eff` by CO₂ dilution.** Inject a known volume of pure CO₂ (`v_inj`) into the sealed, mixed chamber; the steady rise satisfies `ΔC[ppm] ≈ 10⁶ · v_inj / V_eff`. Agreement with step 5 within 5 % confirms both the volume and the injection technique. If it disagrees, you have a leak — go to step 10.
7. **Leak test, mechanical.** Pressurise gently (a few mbar, a manometer or a water column) with P1 and P2 plugged; the pressure must hold for 60 s. Fix leaks before proceeding.
8. **Mixing test.** Inject a dose at P3; with the mixing fan on, `pm25_ugm3` must reach a plateau within **60 s** and the plateau must be flat within ±5 % for 30 s. If it is not flat, the chamber is not well mixed — reposition or speed up the mixing fan. **Repeat until it passes. Everything downstream assumes this.**
9. **Thermal equilibration test.** From cold, log `temp_c` for 4 hours with the MQ pair powered. Record the curve; confirm `τ_th` and the steady rise. Set the session warm-up rule from the result (expect ≥ 90 min).
10. **`k_leak` — CO₂ decay, turbine OFF.** Inject CO₂ to ≈ 3000 ppm, mixing fan on, turbine off, inlet valve **fully open**. Log 20 min. Fit `ln((C−C_in)/(C₀−C_in))` vs `t`. **Acceptance: `k_leak ≤ 0.5 h⁻¹`.** Repeat 3× and average. Record.
11. **`u_min` — stiction threshold.** Ramp `duty_cmd` from 0 in 0.01 steps, 5 s dwell each, until `rpm > 0`. Then ramp down until it stops. Record both; set `u_min` = the *rising* threshold + 0.03 margin. Expect 0.22–0.30. Repeat 5× at each of `T1` and `T3` — stiction is temperature-dependent.
12. **`φ(u)` and `Q_max` — CO₂ tracer at every duty level (Category C).** For each of the 9 duty levels: inject CO₂ to ≈ 3000 ppm, mix 60 s, apply the duty, log until CO₂ < 700 ppm or 8 min elapses. Fit the decay slope; `λ(u) = slope − k_leak`; `Q(u) = λ(u)·V_eff`. **3 replicates per level, on ≥ 2 different days.** Acceptance: `r² ≥ 0.98` on every fit, and `Q(u)` monotone increasing. Fit a PCHIP monotone spline; save to `calibration/phi_u_<date>.csv`.
13. **`k_dep` — PM decay, turbine OFF (Category B).** Inject to band `B3`, turbine off, inlet open, log 20 min. Fit; `k_dep = slope − k_leak`. **5 replicates.** Expect 0.8–3.0 h⁻¹. Record per-source (`k_dep` differs between incense and NaCl — record both).
14. **Injection dose calibration.** For each source, inject 5 / 15 / 40 / 100 mL from the priming jar, 3 replicates each, and record the resulting `C₀`. Build the dose → `c0_band` lookup table and store it in `calibration/`. This table is a *convenience*; the band is still assigned from the measured `C₀`.
15. **`R0` commissioning.** With the chamber purged, filtered and thermally settled, record 30 min of MQ output and set `R0` = the median `Rs`. Store per sensor with the timestamp. This is the reference against which all future drift is measured (AQR-ARCH-001 §8.3).
16. **Write `calibration/calib_<date>.yaml`** containing `V_eff`, `u_min`, `k_leak`, `k_dep` per source, `φ(u)` knots, `Q_max`, `R_L` per MQ, `R0` per MQ, `κ` (start at 0.3), ADC coefficients. Commit. **`calib_id` on every subsequent row points here.**

### 8.4 Phase 3 — Pilot (20 episodes, discard the data, keep the lessons)

Run 20 Category-D episodes spanning the extremes: `{u = 0.25, 0.60, 1.00} × {B1, B3, B5}`, plus 2 at `u = 0` and 2 deliberately infeasible. Then:

| Check | Pass condition | If it fails |
|---|---|---|
| Exponential fit quality | `r²_fit ≥ 0.97` on every decay | mixing is inadequate → return to §8.3 step 8 |
| Monotonicity | Mean `k_tot_fit_hr` increases with duty band | `φ(u)` is wrong or the inlet is restricted → re-run step 12 |
| Band reachability | Every `c0_band` hit at least twice with the dose table | recalibrate doses (step 14) |
| Episode cycle time | ≤ 12 min including purge | shorten the purge with full duty, not by cutting the pre-roll |
| Row completeness | ≥ 99 % of expected 1 Hz rows present | fix Wi-Fi/logging before the campaign |
| Thermal stability | `temp_c` drift < 0.3 °C over 15 min pre-session | extend warm-up |

**Delete the pilot episodes from `processed/`; keep them in `raw/` under `data/raw/pilot/`.** They were collected under a moving calibration and must not enter the corpus.

### 8.5 Phase 4 — The Episode Procedure (the atomic recipe)

Every episode, without exception, follows this timeline. `t_rel = 0` is the injection instant.

| Phase | `t_rel` | Duty | Action |
|---|---|---|---|
| *(purge — not logged as part of the episode)* | — | 1.00 | Run until `pm25 < 10 µg/m³` and `co2 < 550 ppm`, then 30 s more. Typically 120 s |
| *(settle)* | — | 0 | 60 s at zero duty; confirm no rise (proves the purge worked and the inlet is clean) |
| **`preroll`** | −60 → 0 | 0 | **Start logging.** Establishes the pre-injection baseline for every channel |
| **`inject`** | 0 → 40 | 0 | Inject the dose through P3 over ≤ 10 s; mixing fan disperses it. **Do not touch the turbine** |
| **`decay`** | 40 → 40+D | **the episode's pattern** | Apply the duty pattern at exactly `t_rel = 40`. `D = clamp(3·τ_eff_expected, 180, 1200)` and always ≥ `t_target_s + 60` |
| **`postroll`** | 40+D → 100+D | unchanged | 60 s tail; captures the approach to the floor and any overshoot |
| *(reset)* | — | 1.00 | Purge for the next episode |

**Definitions fixed by this timeline:**

- `C₀` = maximum 10 s rolling mean of `pm25_ugm3` over `t_rel ∈ [0, 50]`
- `t_target` and `t_reach` are both measured **from `t_rel = 40`**, not from injection
- `deadline_met` = `i_inst ≤ i_safe` at `t_rel = 40 + t_target_s`
- `energy_wh` integrates over `t_rel ∈ [40, 40+D]` only

**Operator rules during an episode:** do not open the door · do not walk past the inlet · do not change the room's ventilation · log anything unusual in `note` immediately, not from memory afterwards.

### 8.6 Phase 5 — Run the Campaign

1. **Generate the run order once.** Build `config/doe_plan.csv` from the DoE in §5, then **shuffle it with a recorded seed**. Never run in the order the design was written — that confounds every factor with time-of-day and sensor drift.
2. **Block by day.** Assign roughly equal numbers of each category to each session, so no calendar day is a single category (balance rule §5.3).
3. **Open each session** with: power on → 90 min warm-up → thermal-stability check (< 0.3 °C over 15 min) → 2 fixed-reference control episodes (`u = 0.60`, `B3`, `incense`, `R2`, `T1`) → write `logs/sessions/<session_id>.session.yaml`.
4. **Run episodes** from the shuffled plan, following §8.5 exactly.
5. **Close each session** with 2 more fixed-reference control episodes and a 30-min clean-air `R0` window.
6. **Plot the control episodes across the campaign every evening.** Their fitted `k_tot_fit_hr` and `C₀`-for-fixed-dose should be flat. A trend means the rig is drifting — a fouled impeller, a loading filter, or MQ ageing — and you must find it *now*, not at analysis time.
7. **Ingest the same day.** `raw → interim` while the session is fresh; a parsing failure discovered a week later costs a week of episodes.
8. **Run the QC report daily** (§9) and quarantine failures immediately.
9. **Re-run `R0` commissioning weekly** and after any power interruption > 10 min.
10. **Re-run `φ(u)` identification (step 12) every 2 weeks** and after any impeller or filter service. If `Q_max` has moved > 10 %, bump `rig_cfg_sha` and treat pre-change and post-change episodes as different regimes.

### 8.7 Phase 6 — Package, Validate, Freeze

1. Rebuild `interim/` from `raw/` end-to-end with the versioned parser. Confirm it reproduces byte-identically.
2. Build `processed/` from `interim/` + `calibration/` + `episodes.csv`.
3. Run every QC rule in §9; write `logs/qc/qc_report_v<VER>.html`.
4. Run the balance checks in §5.3. **Any failure blocks the release** — collect the missing cells rather than relaxing the rule.
5. Assign splits per §10, write `splits.parquet`, and verify no `episode_id` appears in two splits.
6. Compute SHA-256 for every file into `checksums/`.
7. Write the dataset card (§11.1) and update `CHANGELOG.md` and `VERSION`.
8. Export the flat CSV to `data/processed/export/`.
9. Tag the repository. **From this point the version is immutable**: new episodes create a new MINOR version; a QC or label correction creates a new PATCH; a schema change creates a new MAJOR.

---

## 9. Data Quality — Getting a Dataset With No Bad Data In It

### 9.1 The Governing Principle

**Nothing is deleted and nothing is silently edited.** Bad data is *labelled* and *excluded*, and the exclusion is recorded with a reason. A dataset whose bad rows were quietly removed cannot be audited, and its exclusion criteria become invisible — which is itself a source of bias. Raw stays raw forever; `qc_status` and `qc_flags` do the work.

| `qc_status` | Meaning | Enters model training? | Enters evaluation? |
|---|---|---|---|
| `accept` | Passed every rule | Yes | Yes |
| `quarantine` | Failed a soft rule; recoverable or useful elsewhere | No | Only if the analysis explicitly opts in |
| `reject` | Failed a hard rule; physically meaningless | No | No |

### 9.2 QC Flag Bitmask (`qc_flags`, int32)

| Bit | Flag | Level | Trigger |
|---|---|---|---|
| 0 | `QC_SEQ_GAP` | soft | ≥ 1 missing `seq` in the episode |
| 1 | `QC_SEQ_GAP_MAJOR` | **hard** | > 2 % of expected rows missing, or any gap > 10 s |
| 2 | `QC_TIME_IRREGULAR` | soft | sample spacing outside 1.0 ± 0.2 s before gridding |
| 3 | `QC_CLOCK_STEP` | **hard** | NTP step > 500 ms inside the episode |
| 4 | `QC_PM_INVALID` | soft | `pm_valid = false` for > 5 % of decay rows |
| 5 | `QC_PM_SATURATED` | **hard** | `pm25_ugm3 ≥ 950` for > 3 consecutive seconds |
| 6 | `QC_PM_RH_INVALID` | **hard** | `rh_pct > 90` at any point in the decay window |
| 7 | `QC_CO2_RANGE` | soft | `co2_ppm` outside 350–5000 |
| 8 | `QC_CO2_WARMUP` | **hard** | episode started < 180 s after a CO₂-sensor power-on |
| 9 | `QC_MQ_RANGE` | soft | `mq*_ratio` outside 0.02–5.0 |
| 10 | `QC_MQ_R0_STALE` | soft | `R0` older than 7 days |
| 11 | `QC_TH_RANGE` | soft | `temp_c` or `rh_pct` outside the valid range |
| 12 | `QC_TH_DROPOUT` | soft | `th_valid = false` for > 10 % of rows |
| 13 | `QC_DUTY_INVALID` | **hard** | `duty_applied` in `(0, u_min)` for > 5 s without dither, or outside [0,1] |
| 14 | `QC_DUTY_MISMATCH` | **hard** | `\|duty_applied − duty_cmd\|` > 0.05 without a logged clamp reason |
| 15 | `QC_RAIL_SAG` | soft | `vbus_v < 11.0` for > 2 s |
| 16 | `QC_CURRENT_RANGE` | **hard** | `imotor_a > 1.2 A` outside the 300 ms kick-start window |
| 17 | `QC_RPM_IMPLAUSIBLE` | **hard** | `rpm = 0` while `duty_applied > u_min` for > 3 s (stall / blocked impeller) |
| 18 | `QC_FIT_POOR` | **hard** | `r2_fit < 0.97` on the decay fit — **the chamber was not well mixed** |
| 19 | `QC_C0_OUT_OF_RANGE` | **hard** | `C₀ < 60` (nothing to remediate) or `C₀ > 950` (sensor saturated) |
| 20 | `QC_THERMAL_UNSETTLED` | **hard** | session-opening thermal check not passed, or `temp_c` drift > 1.0 °C during the episode |
| 21 | `QC_DOOR_EVENT` | **hard** | door opened during the episode (operator-logged or inferred from a step in `co2_ppm`) |
| 22 | `QC_OPERATOR_ABORT` | **hard** | operator marked the episode bad |
| 23 | `QC_CALIB_MISMATCH` | **hard** | `calib_id` or `rig_cfg_sha` differs from the session's declared value |

**Rule: any hard flag ⇒ `qc_status = reject`. Any soft flag ⇒ `quarantine` for review; an operator may promote it to `accept` only with a written justification in `note`.** Category-I episodes carry hard flags *by design* and keep `qc_status = accept` with `category = I` — they are the labelled fault set, and they are excluded from dynamics training by category, not by QC.

### 9.3 Handling Specific Situations

| Situation | Do this | Never do this |
|---|---|---|
| A few missing 1 Hz samples | Grid to 1 Hz, leave the gap as `null`, set `QC_SEQ_GAP` | Forward-fill or interpolate in `interim/` |
| Interpolation genuinely needed for a model | Do it in `processed/`, add an `<col>_imputed` boolean, cap at ≤ 2 % of rows | Impute without a flag |
| A single wild spike | Leave it. Flag the episode. Let the model's robust loss deal with it | Hand-edit the value |
| A sensor drifted mid-campaign | Recalibrate, bump `calib_id`, and treat before/after as different `calib_id` groups in analysis | Retro-scale old data to match new |
| An episode was botched | `qc_status = reject`, `exclusion_reason` written in plain language | Delete the file |
| A label was wrong | Fix `episodes.csv`, bump the PATCH version, note it in `CHANGELOG.md` | Fix it silently |
| Duplicate timestamps | Keep the first, flag `QC_TIME_IRREGULAR` | Average them |
| PMS5003 saturated | Reject the episode; reduce the dose next time | Clip to 999 and carry on |

### 9.4 Pre-Release Acceptance Gate

The dataset may not be tagged until **all** of the following hold:

1. ≥ 95 % of collected episodes are `accept` (a lower rate means the *procedure* is broken, not the data)
2. Every balance constraint in §5.3 passes
3. Every Category-C fit has `r² ≥ 0.98`; every Category-B and -D fit has `r² ≥ 0.97`
4. `φ(u)` is strictly monotone increasing over `[u_min, 1]`
5. `k_leak ≤ 0.5 h⁻¹` on the most recent measurement
6. Fixed-reference control episodes show no trend in `k_tot_fit_hr` exceeding ±10 % across the campaign
7. No `episode_id` appears in more than one split
8. Every row's `rig_cfg_sha` and `calib_id` resolve to a file present in the repository
9. `processed/` regenerates from `raw/` reproducibly
10. Every checksum in `checksums/` verifies

---

## 10. Splitting Policy

| Rule | Specification |
|---|---|
| **Unit of splitting** | **`episode_id`. Never a row.** A row-wise split leaks the target through autocorrelation and will report an inflated R² for a useless model |
| Method | `GroupShuffleSplit` on `episode_id`, **stratified** on the tuple `(category, duty_band, c0_band, source_type)` |
| Proportions | 70 % train / 15 % validation / 15 % test, by episode count |
| Additional temporal holdout | The **most recent 15 % of episodes by `t0_utc`** form a second, disjoint `test_forward` set. Report on both: the stratified test set measures interpolation, the forward set measures whether the model survives drift |
| Category placement | Categories B and C (identification) go **entirely into train** — they are calibration, not evaluation. Category I never enters any dynamics split. Category J is evaluation-only |
| Balance guard | Every `(duty_band, c0_band)` cell present in test must also be present in train |
| Seed | Recorded in `config/split_policy.yaml`; the same seed must reproduce the same split exactly |
| Immutability | Splits are assigned **once per dataset version** and written to `splits.parquet`. Re-splitting to improve a metric is data dredging; if you must re-split, that is a new MINOR version with a note in `CHANGELOG.md` |
| Leakage audit | Before release, assert: no `episode_id` in two splits · no `session_id` spanning train and test *for the same fixed-reference control episodes* · no derived feature references a future timestamp |

---

## 11. File Templates

### 11.1 `config/rig_config.yaml`

```yaml
rig_id: R1
description: "250 x 250 x 250 mm sealed acrylic chamber, VENTIS demonstration rig"
schema_version: 1
geometry:
  internal_mm: {l: 250, b: 250, h: 250}
  v_geom_l: 15.625
  v_internals_l: 0.62          # measured by displacement, §8.3 step 4
  v_eff_l: 15.005              # v_geom - v_internals, cross-checked §8.3 step 6
  surface_to_volume_m_inv: 24.0
  material: "cast acrylic 5 mm, silicone-sealed"
ports:
  p1_exhaust:   {bore_mm: 40, height_mm: 210, duct: "to room exterior"}
  p2_inlet:     {bore_mm: 40, height_mm: 30, filter: "H13 HEPA", valve: "needle, 0-100%"}
  p3_injection: {bore_mm: 10, height_mm: 60, fitting: "silicone septum + luer"}
  p4_gland:     {size: "M16", sealed: true}
  p5_mixfan:    {size_mm: 25, voltage_v: 5.0, height_mm: 40, always_on: true}
sensors:
  pms5003:  {position_mm: [125, 125, 125], interface: "uart1", pins: {rx: 25, tx: 26, set: 27, reset: 14}}
  mhz19b:   {position_mm: [125, 125, 140], interface: "uart2", pins: {rx: 16, tx: 17}, range_ppm: 5000, abc: disabled}
  mq135:    {position_mm: [40, 125, 80], adc_pin: 34, r_load_ohm: 985}
  mq2:      {position_mm: [40, 165, 80], adc_pin: 35, r_load_ohm: 1012}
  dht22:    {position_mm: [210, 125, 150], pin: 4}
actuator:
  motor: "2 W brushed DC, 12 V nominal"
  driver: "IRF520 low-side + TC4420 gate driver"
  pwm: {pin: 18, freq_hz: 20000, resolution_bits: 11}
  tach_pin: 19
electrical:
  vbus_divider: {r_top_k: 10.0, r_bot_k: 2.2, adc_pin: 39}
  current_sense: {device: "ACS712-05B", sensitivity_mv_per_a: 185, adc_pin: 36}
  supply: "12 V / 2 A SMPS + 12->5 V buck"
firmware:
  version: "0.4.2+g1a2b3c4"
  sample_rate_hz: 1
  electrical_internal_rate_hz: 10
```

### 11.2 `calibration/calib_<date>.yaml`

```yaml
calib_id: calib_2026-09-08
valid_from: "2026-09-08T00:00:00+00:00"
rig_cfg_sha: "9f2c1a7b4e05"
volume:
  v_eff_l: 15.005
  method: "displacement + CO2 dilution cross-check (agreement 2.1%)"
actuator:
  u_min_rising: 0.26
  u_min_falling: 0.21
  u_min_used: 0.29            # rising + 0.03 margin
  q_max_m3_h: 1.74
  ach_max_h: 115.9
  phi_u_file: "phi_u_2026-09-08.csv"
  phi_u_fit: "pchip monotone, r2 = 0.994"
transport:
  k_leak_h: 0.31              # CO2 decay, turbine off, 3 replicates
  k_leak_sd_h: 0.04
deposition:
  k_dep_h: {incense: 1.62, nacl: 1.18, cooking_oil: 2.05, paper: 2.41}
  k_dep_sd_h: {incense: 0.11, nacl: 0.09, cooking_oil: 0.17, paper: 0.22}
gas_sensors:
  mq135: {r_load_ohm: 985, r0_ohm: 41200, r0_measured_utc: "2026-09-08T02:10:00+00:00"}
  mq2:   {r_load_ohm: 1012, r0_ohm: 8740,  r0_measured_utc: "2026-09-08T02:10:00+00:00"}
pm_correction:
  kappa_hygroscopic: 0.30
adc:
  file: "adc_cal_2026-09-08.yaml"
index:
  standard: "CPCB-NAQI"
  i_safe: 100
  c_safe_pm25_ugm3: 60
dose_lookup:                   # §8.3 step 14, mL from priming jar -> observed C0 band
  incense:      {5: B1, 15: B2, 40: B3, 100: B4, 150: B5}
  nacl:         {5: B1, 20: B2, 55: B3, 120: B4}
  cooking_oil:  {10: B1, 30: B2, 70: B3, 140: B4}
```

### 11.3 `logs/sessions/<session_id>.session.yaml`

```yaml
session_id: S20260910_R1_01
rig_id: R1
operator: "AK"
calib_id: calib_2026-09-08
rig_cfg_sha: "9f2c1a7b4e05"
fw_version: "0.4.2+g1a2b3c4"
power_on_utc:  "2026-09-10T01:30:00+00:00"
warmup_ok_utc: "2026-09-10T03:05:00+00:00"
thermal_check: {drift_c_15min: 0.18, pass: true}
ambient_start: {temp_c: 29.4, rh_pct: 61.0, pm25_ugm3: 38}
control_episodes_open: [E20260910T031200Z_R1_D_INC_001, E20260910T032400Z_R1_D_INC_002]
episodes: 24
control_episodes_close: [E20260910T093000Z_R1_D_INC_025, E20260910T094200Z_R1_D_INC_026]
r0_window_utc: ["2026-09-10T09:55:00+00:00", "2026-09-10T10:25:00+00:00"]
incidents: []
notes: "Room AC cycled at 06:10; ambient RH dropped 8 pp. Episodes 014-016 flagged."
```

### 11.4 `config/doe_plan.csv` — header and two example rows

```
run_order,planned_episode_seq,category,pattern,source_type,gas_analyte,duty_plan_json,dose_ml,c0_band_target,rh_band,t_band,inlet_pct,t_target_s,replicate,day_block,seed
1,001,D,const,incense,none,"[0.60]",40,B3,R2,T1,100,150,1,1,20260908
2,002,F,aprbs,incense,none,"[0.25,0.90,0.40,1.00,0.30,0.75]",100,B4,R2,T1,100,300,1,1,20260908
```

### 11.5 `DATASET_CARD.md` — required headings

```
# ventis-chamber-25 — Dataset Card
1. Summary               — what it is, one paragraph
2. Motivation            — the control problem it exists to solve
3. Rig & provenance      — chamber, sensors, calibration set, rig_cfg_sha
4. Composition           — episode and row counts per category (§5.1 table)
5. Schema                — link to §6.3; column count and version
6. Collection protocol   — link to §8; dates, operators, sessions
7. Preprocessing         — what raw -> interim -> processed does, exactly
8. Splits                — policy, seed, counts per split
9. Known limitations     — §12
10. Uses and misuses     — what this dataset does NOT support
11. Licence & citation
12. Maintenance          — versioning rules, contact, changelog pointer
```

### 11.6 Example `processed/` rows (first 6 columns + key channels, for orientation)

```
ts_utc,t_rel_s,seq,episode_id,...,pm25_ugm3,co2_ppm,rh_pct,duty_applied,i_inst,phase,category,c0_band,split
2026-09-10T04:15:30.000000+00:00,-60.0,184153,E20260910T041530Z_R1_D_INC_012,...,4.1,462,58.1,0.00,3.4,preroll,D,B3,train
2026-09-10T04:16:30.000000+00:00,0.0,184213,E20260910T041530Z_R1_D_INC_012,...,12.7,468,58.0,0.00,10.6,inject,D,B3,train
2026-09-10T04:16:52.000000+00:00,22.0,184235,E20260910T041530Z_R1_D_INC_012,...,331.6,471,57.6,0.00,331.9,inject,D,B3,train
2026-09-10T04:17:10.000000+00:00,40.0,184253,E20260910T041530Z_R1_D_INC_012,...,326.9,470,57.5,0.60,329.2,decay,D,B3,train
2026-09-10T04:18:40.000000+00:00,130.0,184343,E20260910T041530Z_R1_D_INC_012,...,74.3,466,57.9,0.60,123.9,decay,D,B3,train
2026-09-10T04:19:40.000000+00:00,190.0,184403,E20260910T041530Z_R1_D_INC_012,...,31.2,464,58.2,0.60,52.0,decay,D,B3,train
```

---

## 12. Known Limitations (state these in the dataset card — do not let a reader discover them)

1. **No reference instrument.** PM values are PMS5003 factory-calibrated units. The dataset is internally self-consistent but carries **no absolute accuracy claim**, and mass concentrations are source-dependent because the sensor infers mass from scattering.
2. **Single chamber, single node.** Nothing here characterises inter-unit variation. A model trained on `R1` should not be assumed to transfer to a second rig without re-identification.
3. **Scale.** 15 L with 115 ACH is a fast, well-mixed, idealised plant. A real room is slower, stratified and has multiple sources. The *architecture and method* transfer; the *fitted parameters* do not (AQR-ARCH-001, FX-08).
4. **MQ channels are relative.** `Rs/R0` only. No ppm figure is claimed, and cross-sensitivity between the injected sources and the MQ response is real and uncorrected beyond the T/RH surface.
5. **The flammable-gas channel is never exercised with real flammable gas.** MQ-2 interlock behaviour in the dataset comes from resistor-substitution tests, labelled Category I.
6. **CO₂ is a tracer here, not a pollutant.** Its dynamics identify `Q(u)`; its `SI_co2` values are project-defined and non-regulatory.
7. **Operator-induced variance.** Injection technique varies between people. `operator` is recorded precisely so this can be tested for; if it turns out to matter, that is a finding to report, not to hide.
8. **Category J (closed-loop) data is confounded by the controller** and must never be pooled with open-loop identification data.

---

## Appendix — One-Page Quick Reference

| Thing | Value |
|---|---|
| Chamber | 250³ mm, `V_geom` 15.625 L, `V_eff` ≈ 15.0 L, `S/V` 24 m⁻¹ |
| `Q_max` / `ACH_max` | measure; expect 0.6–3.0 m³/h → 38–192 h⁻¹ (design point 1.8 / 115) |
| `τ_min` | ≈ 31 s at full duty |
| `k_dep` / `k_leak` | measure; expect 0.8–3.0 h⁻¹ / ≤ 0.5 h⁻¹ |
| `u_min` | measure; expect 0.22–0.30 |
| Thermal warm-up | **≥ 90 min**, `τ_th` ≈ 70 min, MQ rise ≈ +2 °C |
| Sampling | 1 Hz (electrical 10 Hz internally → 1 s mean + σ) |
| Duty levels | 0, 0.25, 0.30, 0.40, 0.50, 0.60, 0.75, 0.90, 1.00 |
| `T_target` grid | 90, 150, 300, 600 s |
| `c0_band` | B1 60–120 · B2 120–250 · B3 250–450 · B4 450–700 · B5 700–950 µg/m³ |
| `I_safe` / `C_safe` | 100 index / 60 µg/m³ PM2.5 (CPCB) |
| Categories | A background · B natural decay · C CO₂ tracer · D constant duty · E step · F APRBS · G multi-source · H disturbance · I fault · J closed-loop |
| Episode timeline | preroll −60→0 · inject 0→40 · decay 40→40+D · postroll +60 |
| Episodes | **Tier 1 = 172** (≈ 6 days) · **Tier 2 = 430** (≈ 15 days) |
| Rows | Tier 1 ≈ 179 k · Tier 2 ≈ 830 k — **but effective N = episode count** |
| Size | Tier 1 ≈ 6.5 MB Parquet · Tier 2 ≈ 30 MB Parquet |
| Identification order | `k_leak` (CO₂, u=0) → `k_dep` (PM, u=0) → `Q(u)` (CO₂, each u) |
| Split unit | **episode**, stratified, 70/15/15 + a forward 15 % |
| Golden rule | Raw is immutable · label bad data, never delete it · split by episode, never by row |
