# How Much Does Ignoring Occupant Behaviour Cost?

### Quantifying the Underestimation of Heat Pump Peak Demand in Current Grid Planning Tools

A full-stack research pipeline that simulates UK domestic heat pump electricity demand under varying occupant behaviour, building fabric, and weather conditions. It quantifies how much current grid planning tools (based on Heating Degree Days) underestimate peak demand by ignoring occupant diversity and nonlinear weather-behaviour interactions.

---

## Table of Contents

1. [Research Question and Hypotheses](#1-research-question-and-hypotheses)
2. [Project Structure](#2-project-structure)
3. [How to Run](#3-how-to-run)
4. [Pipeline Walkthrough (10 Steps)](#4-pipeline-walkthrough-10-steps)
5. [Data Sources](#5-data-sources)
6. [Configuration Deep Dive](#6-configuration-deep-dive)
7. [Simulation Engine](#7-simulation-engine)
8. [Analytical Methods](#8-analytical-methods)
9. [Validation Against Real Data](#9-validation-against-real-data)
10. [Figures and Outputs](#10-figures-and-outputs)
11. [Test Suite](#11-test-suite)
12. [Results Summary](#12-results-summary)
13. [Key Equations Reference](#13-key-equations-reference)
14. [Dependencies](#14-dependencies)

---

## 1. Research Question and Hypotheses

**Central question:** Do current grid planning tools (HDD-based regression) systematically underestimate heat pump peak electricity demand because they ignore occupant behaviour diversity?

Three formal hypotheses are tested:

| ID | Hypothesis | Statistical Test |
|----|-----------|-----------------|
| **H1** | HDD underestimation of peak demand increases at lower outdoor temperatures | Monotonic increase in worst-case archetype error across W1 -> W2 -> W3 |
| **H2** | Occupant behaviour is a stronger predictor of peak demand than building fabric quality | ANOVA eta-squared: archetype > fabric |
| **H3** | The weather-behaviour interaction is nonlinear, making HDD-linear models structurally inadequate | Ramsey RESET test (p < 0.05) on HDD-linear model residuals |

---

## Pipeline Overview

```mermaid
flowchart TD
    subgraph S0["Configuration"]
        CFG["simulation_config.yaml<br/>All parameters, data sources,<br/>scenario definitions"]
    end

    subgraph S1["Stage 1 — Real Weather Data"]
        EPW["London Gatwick TMYx EPW<br/>8760 hourly rows, 2009-2023"]
        PARSE["EPW Parser<br/>extract_day_profile()"]
        W1["W1: Mild Cold<br/>19 Mar, T_mean = 4.0 C"]
        W2["W2: Design Cold<br/>10 Feb, T_mean = -0.7 C"]
        W3["W3: Extreme Cold<br/>11 Feb, T_mean = -2.5 C"]
        EPW --> PARSE
        PARSE --> W1
        PARSE --> W2
        PARSE --> W3
    end

    subgraph S2["Stage 2 — 2R1C Thermal Simulation"]
        FACTORIAL["Full Factorial Design<br/>4 archetypes x 3 weather x 2 fabric<br/>x 10 replicates = 240 runs"]
        ODE["Forward Euler ODE<br/>C dT/dt = Q_hp + Q_int + Q_sol - UA dT<br/>60s internal steps, 15-min output"]
        HP["Variable-Speed ASHP Model<br/>12 kW thermal, COP derating,<br/>defrost penalty, aux heater"]
        SIM["simulation_results.csv<br/>23,040 rows"]
        FACTORIAL --> ODE
        HP --> ODE
        ODE --> SIM
    end

    subgraph S3["Stage 3 — Statistical Analysis"]
        HDD["HDD Benchmark<br/>Peak = alpha + beta * HDD<br/>fitted on cross-archetype mean"]
        ANOVA["3-Way ANOVA (Type II)<br/>eta-squared decomposition<br/>222 residual df"]
        REG["Interaction Regression<br/>Model A: HDD-linear<br/>Model B: T + T^2 + archetype + interactions"]
        RESET["Ramsey RESET Test<br/>nonlinearity diagnostic"]
        SURF["Demand Surfaces<br/>hour x temperature matrices<br/>per archetype"]
    end

    subgraph S4["Stage 4 — Outputs"]
        FIG["6 Publication Figures<br/>PNG + PDF, 300 DPI<br/>colorblind-safe palette"]
        TAB["5 Results Tables<br/>master, HDD, ANOVA,<br/>model comparison, full sim"]
        VERDICT["Hypothesis Verdicts<br/>H1: NOT SUPPORTED<br/>H2: SUPPORTED<br/>H3: SUPPORTED"]
    end

    subgraph S5["Stage 5 — Testing"]
        TEST["7 Pytest Tests<br/>physics bounds, stats sanity,<br/>output integrity"]
    end

    subgraph S6["Stage 6 — Real-Data Validation"]
        EOH["EoH Field Trial<br/>739 homes, 27.7M rows<br/>30-min intervals, 2020-2023"]
        COPV["COP Validation<br/>Model vs whole-system SPF<br/>6.76M observations"]
        PEAKV["Peak Demand Validation<br/>148,594 property-days<br/>median/P90/P95 peaks"]
        ARCHV["Archetype Clustering<br/>k-means on daily profiles<br/>k=4 matches assumed types"]
        EOH --> COPV
        EOH --> PEAKV
        EOH --> ARCHV
    end

    CFG --> S1
    CFG --> S2
    W1 --> FACTORIAL
    W2 --> FACTORIAL
    W3 --> FACTORIAL
    SIM --> HDD
    SIM --> ANOVA
    SIM --> REG
    REG --> RESET
    SIM --> SURF
    HDD --> TAB
    ANOVA --> TAB
    REG --> TAB
    HDD --> FIG
    ANOVA --> FIG
    SURF --> FIG
    TAB --> VERDICT
    SIM --> TEST
```

---

## 2. Project Structure

```
heat_pump_project/
|
|-- main.py                          # Master orchestrator -- runs entire pipeline
|-- requirements.txt                 # Python dependencies
|-- .gitignore
|
|-- config/
|   +-- simulation_config.yaml      # ALL parameters -- no magic numbers in code
|
|-- simulation/
|   |-- __init__.py
|   |-- epw_parser.py               # Real EPW weather file parser
|   +-- ochre_runner.py             # 2R1C thermal model + heat pump simulation
|
|-- benchmarks/
|   |-- __init__.py
|   +-- hdd_regression.py           # HDD benchmark model (grid planner baseline)
|
|-- analytics/
|   |-- __init__.py
|   |-- anova_decomposition.py      # 3-way ANOVA variance decomposition
|   |-- interaction_regression.py   # Model A vs Model B + RESET test
|   +-- demand_surface.py           # Hour x temperature demand matrices
|
|-- validation/                      # Validation against EoH field trial (739 homes)
|   |-- __init__.py
|   |-- eoh_loader.py               # EoH 30-min data loader + cleaning
|   |-- cop_validation.py           # COP model vs real field COP
|   |-- peak_validation.py          # Peak demand distributions + ADMD
|   |-- archetype_clustering.py     # k-means archetype discovery
|   +-- run_validation.py           # Orchestrates all validation analyses
|
|-- visualisation/
|   |-- __init__.py
|   +-- plots.py                    # 6 publication-quality figures (PNG + PDF)
|
|-- notebooks/
|   +-- model_review.ipynb          # Interactive review -- run this to inspect the model
|
|-- tests/
|   |-- __init__.py
|   +-- test_pipeline.py            # 7 pytest tests
|
|-- data/
|   |-- raw/                        # EPW file + extracted weather CSVs
|   |-- processed/                  # simulation_results.csv (23,040 rows)
|   +-- external/                   # EoH, EPC, ENWL, SERL (download separately)
|
+-- outputs/
    |-- results_tables/             # CSV tables (master, HDD, ANOVA, model comparison)
    |-- figures/                    # fig1-fig6 in PNG and PDF
    +-- validation/                 # COP, peak, archetype validation outputs
```

---

## 3. How to Run

### Prerequisites

```bash
# Python 3.10+ required
pip install -r requirements.txt
```

### Full pipeline (single command)

```bash
python main.py           # ~3 seconds, runs everything
python main.py --debug   # same but with DEBUG-level logging
```

This produces:
- 5 CSV tables in `outputs/results_tables/`
- 12 figures (6 PNG + 6 PDF) in `outputs/figures/`
- Full test suite results
- Printed hypothesis verdicts

### Interactive notebook (recommended for review)

```bash
jupyter notebook notebooks/model_review.ipynb
```

This walks through every stage with inline plots and explanations. Run cells top-to-bottom.

### Validation against EoH field trial (requires data download)

```bash
# 1. Download EoH 30-min interval dataset (SN 9209) from UK Data Service
# 2. Place in data/external/eoh/UKDA-30min-Interval-9209-csv/csv/
# 3. Run validation:
python -m validation.run_validation
```

This validates COP, peak demand, and behavioural archetypes against 739 real monitored ASHP homes.

### Run tests only

```bash
python -m pytest tests/ -v
```

---

## 4. Pipeline Walkthrough (10 Steps)

When you run `python main.py`, these 10 steps execute in order:

### Step 1: Load Configuration

**File:** `main.py` lines 69-80

Reads `config/simulation_config.yaml` via PyYAML. Every single parameter used anywhere in the pipeline is defined in this file -- there are zero magic numbers in the code.

```python
cfg = load_config()                    # parse YAML
archetype_names = build_archetype_names(cfg)  # {"B1": "Early Riser", ...}
np.random.seed(cfg["random_seed"])     # reproducibility
```

### Step 1b: Extract and Save Raw Weather Profiles

**Files:** `main.py` lines 297-331, `simulation/epw_parser.py`

Parses the real EnergyPlus Weather (EPW) file from London Gatwick and extracts three 24-hour temperature/solar/wind profiles for the selected cold days. These are resampled from hourly to 15-minute resolution via linear interpolation and saved to `data/raw/weather_W1.csv`, `weather_W2.csv`, `weather_W3.csv`.

### Step 2: Run All 240 Simulations

**Files:** `main.py` lines 333-342, `simulation/ochre_runner.py` lines 522-628

Executes the full factorial design:

```
4 archetypes x 3 weather scenarios x 2 fabric conditions x 10 replicates = 240 runs
Each run = 96 timesteps (15-min resolution, 24 hours)
Total output = 23,040 rows
```

Each replicate introduces stochastic variation through:
- **Schedule jitter:** +/-15 minutes on heating on/off times
- **Internal gains noise:** +/-30% random multiplier per timestep

The 23,040-row DataFrame is saved to `data/processed/simulation_results.csv`.

### Step 3: HDD Benchmark

**Files:** `main.py` lines 344-348, `benchmarks/hdd_regression.py`

Fits the standard grid planner model: `Peak_demand(kW) = alpha + beta * HDD`, where `HDD = max(0, 15.5 - T_mean_daily)`. The model is trained on the cross-archetype average peak (as a real planner would do, not seeing individual archetypes). Then underestimation error is computed for every cell.

Output: `outputs/results_tables/hdd_benchmark.csv` (24 rows).

### Step 4: ANOVA Decomposition

**Files:** `main.py` lines 350-354, `analytics/anova_decomposition.py`

Type II 3-way ANOVA on peak demand with factors: archetype, weather_scenario, fabric, and all two-way interactions. Computes eta-squared (proportion of total variance explained) for each factor.

Output: `outputs/results_tables/anova_results.csv`.

### Step 5: Interaction Regression

**Files:** `main.py` lines 356-360, `analytics/interaction_regression.py`

Fits two competing regression models on 240 data points:

- **Model A** (HDD-linear): `peak_demand ~ HDD`
- **Model B** (Full interaction): `peak_demand ~ T_mean + T_mean^2 + archetype + T_mean:archetype + fabric`

Runs the Ramsey RESET test on Model A residuals to formally test for nonlinearity.

Output: `outputs/results_tables/model_comparison.csv`.

### Step 6: Demand Surfaces

**Files:** `main.py` lines 362-366, `analytics/demand_surface.py`

Builds 2D demand matrices (hour-of-day x outdoor-temperature) for each archetype using pivot tables with mean aggregation. These feed directly into the heatmap visualisation.

### Step 7: Generate All Figures

**Files:** `main.py` lines 368-380, `visualisation/plots.py`

Produces 6 publication-quality figures in both PNG and PDF at 300 DPI. Uses the seaborn `colorblind` palette for accessibility. See [Section 9](#9-figures-and-outputs) for details on each figure.

### Step 8: Master Results Table

**File:** `main.py` lines 382-384

Produces a single 24-row summary table with one row per scenario cell (archetype x weather x fabric), containing: peak demand, daily energy, mean COP, indoor temperature range, HDD prediction, and underestimation percentage.

Output: `outputs/results_tables/master_results.csv`.

### Step 9: Run Tests

**File:** `main.py` lines 386-388

Executes the 7-test pytest suite inline and prints results.

### Step 10: Print Hypothesis Verdicts

**File:** `main.py` lines 390-391 and 170-250

Evaluates all three hypotheses against the computed results and prints a human-readable summary to the terminal.

---

## 5. Data Sources

All data used in this project comes from real, published, citable sources. No synthetic data is used for weather, COP, or building parameters.

### 5.1 Weather Data

**Source:** London Gatwick TMYx 2009-2023 EPW file from [climate.onebuilding.org](https://climate.onebuilding.org)

**File:** `data/raw/GBR_ENG_London-Gatwick.AP.037760_TMYx.2009-2023.epw`

**Reference:** ASHRAE/WMO Typical Meteorological Year methodology (Crawley & Lawrie, 2019)

Three representative cold days were selected from the ASHRAE typical/extreme period analysis embedded in the EPW header:

| Scenario | ID | EPW Date | T_mean | T_range | Description |
|----------|----|----------|--------|---------|-------------|
| Mild Cold | W1 | 19 March | 4.0 C | [-3.0, 11.0] | Spring typical week |
| Design Cold | W2 | 10 Feb | -0.7 C | [-4.0, 0.5] | Winter extreme week start (99% design) |
| Extreme Cold | W3 | 11 Feb | -2.5 C | [-5.7, -0.5] | Beast from the East proxy |

The EPW parser (`simulation/epw_parser.py`) extracts hourly dry-bulb temperature, global horizontal irradiance, and wind speed, then resamples to 15-minute intervals via linear interpolation.

### 5.2 Heat Pump COP

**Source:** EST/DECC Renewable Heat Premium Payment (RHPP) field trial (2013) -- 700+ monitored UK ASHP installations

**Reference:** Staffell et al. (2012), "A review of domestic heat pumps", *Energy & Environmental Science*

The COP model `COP = 3.5 + 0.12 * T_out` is calibrated to the RHPP trial median values: COP of approximately 2.7 at 0 C and 3.5 at 7 C. The 15% defrost penalty below -2 C comes from Staffell et al. (2012). Capacity derating (2.5% per K below 7 C design temp) is derived from manufacturer datasheets (Mitsubishi Ecodan, Daikin Altherma).

### 5.3 Building Fabric

**Sources:**
- CIBSE Guide A (2015), Table 3.49 -- U-values for 1970s solid-wall construction
- English Housing Survey (EHS) 2019, Table DA6101 -- retrofit standards

| Parameter | F1 (Unimproved) | F2 (Post-retrofit) | Source |
|-----------|-----------------|-------------------|--------|
| U_wall (W/m2K) | 1.7 | 0.3 | CIBSE Guide A / Part L |
| U_roof | 2.3 | 0.15 | CIBSE Guide A |
| U_floor | 0.7 | 0.25 | CIBSE Guide A |
| U_window | 5.6 | 1.6 | Single vs double glazed |
| ACH (1/h) | 0.8 | 0.4 | CIBSE / EHS 2019 |

### 5.4 Occupant Behaviour Archetypes

**Sources:**
- ONS Time Use Survey 2015 -- heating-on hours by household type
- Carbon Trust Household Energy Study (2012) -- 4 dominant heating profiles
- BREDEM-12 / SAP 2012 -- assumed setpoints (21 C living, 18 C rest)

| ID | Name | Setpoint | Schedule | Character |
|----|------|----------|----------|-----------|
| B1 | Early Riser | 20 C | 06:00-08:00, 17:00-22:00 | Commuter, early morning peak |
| B2 | Home All Day | 21 C | 07:00-22:00 | Retired/WFH, continuous heating |
| B3 | Late Returner | 19 C | 07:00-08:30, 19:00-23:00 | Lower setpoint, large recovery load |
| B4 | Intermittent | 18-22 C (variable) | 4 short periods | Erratic, hardest to predict |

### 5.5 HDD Benchmark Methodology

**Reference:** Staffell & Green (2014), "Domestic heating by heat pumps: technology, economics and emissions"

**Context:** National Grid Future Energy Scenarios (FES) 2023

Base temperature: 15.5 C (UK standard). The HDD method is the industry-standard approach used by DNOs and National Grid for peak demand forecasting.

---

## 6. Configuration Deep Dive

**File:** `config/simulation_config.yaml`

This is the single source of truth for every parameter in the project. The file is structured into labelled sections, each with inline source citations.

### 6.1 Time Resolution

```yaml
timesteps: 96            # 24 hours at 15-min resolution
dt_minutes: 15           # output resolution
dt_internal_seconds: 60  # internal ODE solver step (1 minute)
```

The simulation uses a dual time-step approach: the output is recorded every 15 minutes (96 points per day), but the internal ODE solver runs at 1-minute steps (15 sub-steps per output point). This ensures numerical stability of the forward Euler integration without excessive output data.

### 6.2 Monte Carlo Replication

```yaml
replicates: 10                   # 10 runs per cell = 240 total
schedule_jitter_minutes: 15.0    # +/- 15 min random shift
gains_noise_fraction: 0.3        # +/- 30% noise on internal gains
```

Each of the 24 scenario cells is run 10 times with stochastic variation. This gives the ANOVA proper residual degrees of freedom (222 df) and provides variance estimates for peak demand within each cell.

### 6.3 Building Geometry

The reference building is a 1970s UK semi-detached house (the most common UK dwelling type):

```yaml
building:
  floor_area_m2: 85       # total floor area
  volume_m3: 204           # heated volume
  wall_area_m2: 100        # net external wall (excl. windows)
  window_area_m2: 16       # ~19% of floor area
  thermal_mass_kJ_per_m2K: 165  # medium mass (concrete block inner leaf)
```

### 6.4 Heat Pump Parameters

```yaml
heat_pump:
  rated_thermal_capacity_kW: 12.0    # thermal output at 7 C outdoor
  capacity_derating_per_K: 0.025     # loses 2.5% per K below 7 C
  cop_intercept: 3.5                 # COP = 3.5 + 0.12 * T_out
  cop_slope: 0.12
  defrost_efficiency_penalty: 0.15   # 15% COP loss below -2 C
  aux_capacity_kW: 3.0              # backup resistance heater
  aux_deficit_threshold_C: 3.0      # activates when deficit > 3 C
  frost_protection_C: 12.0          # forces HP on if indoor < 12 C
```

The heat pump is modelled as thermal capacity (not electrical), which is how real ASHPs work. At 7 C outdoor, it delivers 12 kW thermal. At -7 C, derating reduces this to `12 * max(0.4, 1 - 0.025 * 14) = 12 * 0.65 = 7.8 kW` thermal. Electrical demand is then `thermal / COP`.

---

## 7. Simulation Engine

### 7.1 EPW Parser (`simulation/epw_parser.py`)

Parses EnergyPlus Weather files (a standard format used by building energy simulation tools worldwide). The parser has three functions:

1. **`parse_epw(path)`** -- Reads all 8760 hourly rows from the EPW file, extracting dry-bulb temperature (column 6), global horizontal irradiance (column 13), and wind speed (column 21). EPW uses 1-24 hour convention, which is converted to 0-23.

2. **`extract_day_profile(epw_df, month, day, dt_minutes)`** -- Extracts a single 24-hour slice and resamples from hourly to 15-minute resolution using time-based linear interpolation. Validates that 24 hours of data exist for the requested date. Clips negative solar values to zero.

3. **`load_weather_profiles(epw_path, weather_days, dt_minutes)`** -- Batch-loads all weather scenarios in one pass through the EPW file. Returns a dict of `{weather_id: DataFrame}`.

### 7.2 Thermal Model (`simulation/ochre_runner.py`)

The core physics engine is a **2R1C lumped-capacitance thermal model** -- the simplest model that captures the essential thermodynamics of a heated building.

#### 7.2.1 The ODE

The building is represented as a single thermal node (indoor air temperature `T_in`) with one thermal capacitance `C` (J/K) representing the building's ability to store heat, and one overall thermal resistance represented by `UA` (the heat-loss coefficient in W/K).

The governing equation:

```
C * dT_in/dt = Q_hp + Q_internal + Q_solar - UA * (T_in - T_out)
```

Where:
- `Q_hp` = heat pump thermal output (W)
- `Q_internal` = internal gains from occupants and appliances (W)
- `Q_solar` = solar gains through windows (W)
- `UA * (T_in - T_out)` = heat loss through fabric and ventilation (W)

This is integrated via **forward Euler** with 60-second internal steps.

#### 7.2.2 UA Calculation (`compute_ua`)

```python
UA = U_wall*A_wall + U_roof*A_roof + U_floor*A_floor + U_window*A_window + 0.33*ACH*Volume
```

The `0.33 * ACH * Volume` term is the ventilation heat loss (from the volumetric heat capacity of air: 1200 J/m3K / 3600 s/h = 0.33 W/m3K per air change).

**Worked example:**

For F1 (unimproved):
```
UA = 1.7*100 + 2.3*42.5 + 0.7*42.5 + 5.6*16 + 0.33*0.8*204
   = 170 + 97.75 + 29.75 + 89.6 + 53.86
   = 441 W/K
```

For F2 (retrofitted):
```
UA = 0.3*100 + 0.15*42.5 + 0.25*42.5 + 1.6*16 + 0.33*0.4*204
   = 30 + 6.375 + 10.625 + 25.6 + 26.93
   = 99.5 W/K
```

The unimproved building loses heat nearly **4.4x faster** than the retrofitted one. This is why the fabric condition matters so much for peak demand.

#### 7.2.3 COP Calculation (`compute_cop`)

```python
COP = 3.5 + 0.12 * T_out                    # base linear model
if T_out < -2.0: COP *= 0.85                # 15% defrost penalty
COP = clip(COP, 1.5, 4.5)                   # physical bounds
```

Example values:
| T_out (C) | Raw COP | Defrost? | Final COP | Efficiency |
|-----------|---------|----------|-----------|------------|
| +10 | 4.70 | No | 4.50 (capped) | Excellent |
| +5 | 4.10 | No | 4.10 | Good |
| 0 | 3.50 | No | 3.50 | Moderate |
| -2 | 3.26 | No | 3.26 | Borderline |
| -3 | 3.14 | Yes | 2.67 | Poor |
| -5 | 2.90 | Yes | 2.47 | Very poor |

#### 7.2.4 Thermal Capacity Derating (`compute_thermal_capacity`)

Real ASHPs lose thermal output as outdoor temperature drops because the refrigerant cycle becomes less effective at extracting heat from colder air.

```python
if T_out < 7.0:
    factor = max(0.4, 1.0 - 0.025 * (7.0 - T_out))
capacity_W = 12000 * factor
```

| T_out (C) | Derating Factor | Thermal Capacity (kW) |
|-----------|----------------|----------------------|
| 7 (design) | 1.00 | 12.0 |
| 2 | 0.875 | 10.5 |
| -2 | 0.775 | 9.3 |
| -5 | 0.700 | 8.4 |
| -7 | 0.650 | 7.8 |
| -17 | 0.400 (floor) | 4.8 |

#### 7.2.5 Thermostat and Heat Pump Control

The thermostat uses a **deadband controller** with hysteresis and a minimum run time:

```python
if hp_on:
    # HP is running -- check if setpoint is satisfied
    if T_in >= setpoint + 0.5 and min_runtime_expired:
        hp_on = False     # satisfied, turn off
else:
    # HP is off -- check if heat is needed
    if T_in <= setpoint - 0.5:
        hp_on = True      # needs heat, turn on
    elif T_in <= 12.0:
        hp_on = True      # frost protection override
```

The frost protection at 12 C is a safety feature -- if the indoor temperature crashes during setback periods (especially in F1/W3), the HP forces on to prevent pipe freezing and condensation damage.

#### 7.2.6 Variable-Speed Modulation

When on, the heat pump doesn't just run at full blast. It uses **variable-speed modulation** proportional to the temperature deficit:

```python
deficit = max(0, setpoint - T_in)
modulation = min(1.0, 0.3 + 0.7 * deficit / 3.0)  # 30% minimum, ramps to 100%
Q_hp = thermal_capacity * modulation
```

What this means in practice:
| Deficit (C) | Modulation | HP Output | Behaviour |
|-------------|-----------|-----------|-----------|
| 0.5 | 0.42 | 42% of max | Gentle maintaining |
| 1.0 | 0.53 | 53% of max | Moderate heating |
| 2.0 | 0.77 | 77% of max | Strong heating |
| 3.0+ | 1.00 | Full capacity | Maximum recovery |

This is critical for realistic peak demand -- real inverter-driven ASHPs modulate output rather than cycling on/off at full power.

#### 7.2.7 Auxiliary Heater

For large recovery loads (e.g., heating up from 15 C setback to 21 C setpoint), a 3 kW electric resistance heater activates:

```python
if deficit > 3.0 and aux_capacity > 0:
    aux_fraction = min(1.0, (deficit - 3.0) / 2.0)
    aux_power = 3000 * aux_fraction         # W, COP = 1
    Q_aux_thermal = aux_power               # direct electric
```

This is common in real ASHP installations -- a backup element handles the high demand during recovery that the heat pump alone cannot meet. Because it has COP = 1 (direct electric), it significantly increases peak electricity demand during recovery periods. This is one mechanism through which occupant behaviour affects peak demand: archetypes with longer setback periods create larger recovery loads.

#### 7.2.8 Internal Gains

```python
Q_internal = base_W * noise_multiplier                # 200 W baseline (appliances, lighting)
if heating_is_scheduled:
    Q_internal += occupancy_pulse_W * noise_multiplier  # + 500 W (people, cooking)
```

The noise multiplier is drawn uniformly from `[0.7, 1.3]` per timestep per replicate, adding realistic variation.

#### 7.2.9 Solar Gains

```python
Q_solar = 0.3 * solar_irradiance * window_area    # 30% of incident solar through glazing
```

The 0.3 factor accounts for glass transmittance, frame fraction, and shading. On a mild day (W1), solar gains can be significant (up to ~770 W), reducing heating demand during daytime.

#### 7.2.10 Schedule Jitter (`jitter_schedule`)

Each replicate applies random timing variation to the heating schedule:

```python
# Original: [6.0, 8.0] (6am - 8am)
# With +/- 15 min jitter, might become [5.8, 8.2] or [6.1, 7.7]
start += uniform(-0.25, 0.25)  # +/- 15 min in hours
end += uniform(-0.25, 0.25)
```

This models the real-world variation in when people actually turn their heating on/off compared to the nominal schedule.

#### 7.2.11 Full Factorial Execution (`run_all_simulations`)

```python
for weather in [W1, W2, W3]:              # 3 weather scenarios
    load_real_EPW_profile(weather)
    for replicate in range(10):            # 10 Monte Carlo replicates
        for fabric in [F1, F2]:            # 2 fabric conditions
            for archetype in [B1,B2,B3,B4]:  # 4 behaviour archetypes
                run_single_simulation(...)   # 96 timesteps each
```

Total: 3 x 10 x 2 x 4 = 240 simulations, each producing 96 rows = **23,040 rows**.

---

## 8. Analytical Methods

### 8.1 HDD Benchmark (`benchmarks/hdd_regression.py`)

**What it does:** Replicates what a grid planner would do -- fit a simple linear model relating peak demand to Heating Degree Days.

**HDD formula:**
```
HDD = max(0, 15.5 - T_mean_daily)
```

**Training approach:** The model is trained on the *cross-archetype average* peak demand (the planner's view -- they don't know about individual household behaviour). This gives one value per weather x fabric x replicate = 60 training points.

**Underestimation calculation:** For each of the 24 cells, the HDD prediction is compared against the actual simulated peak:

```
underestimation_pct = (sim_peak - hdd_peak) / sim_peak * 100
```

Positive values mean HDD underestimates (misses the true peak). Negative values mean HDD overestimates.

**Key functions:**
- `compute_hdd(T_mean, T_base)` -- single-day HDD value
- `fit_hdd_model(sim_df, hdd_base)` -- OLS fit via statsmodels on planner-average data
- `compute_peak_errors(sim_df, model, hdd_base)` -- per-cell underestimation with mean/max/std across replicates

### 8.2 ANOVA Decomposition (`analytics/anova_decomposition.py`)

**What it does:** Decomposes the total variance in peak demand into contributions from each factor and their interactions.

**Model formula:**
```
peak_demand ~ archetype + weather + fabric
            + archetype:weather + archetype:fabric + weather:fabric
```

This is a **Type II ANOVA** (using `statsmodels.stats.anova.anova_lm` with `typ=2`), which tests each factor's contribution after accounting for all other factors. Type II was chosen over Type I because it is invariant to the order of factors.

**Eta-squared** (effect size):
```
eta_squared = SS_factor / SS_total
```

With 240 data points and 17 model parameters, there are **222 residual degrees of freedom** -- enough for reliable F-tests on all terms.

**Why no 3-way interaction:** The model includes all main effects and 2-way interactions but not the 3-way interaction (archetype x weather x fabric). This is deliberate -- including it would consume the residual degrees of freedom that replication provides, making F-tests unreliable.

**Key output:** The eta-squared values directly test H2 by comparing `archetype` vs `fabric` effect sizes.

### 8.3 Interaction Regression (`analytics/interaction_regression.py`)

**What it does:** Compares a simple HDD model against a full interaction model to test whether the relationship between weather and demand is nonlinear and archetype-dependent.

**Model A (HDD-linear):**
```
peak_demand = beta_0 + beta_1 * HDD + error
```
This is what grid planners currently use. Two parameters, simple, interpretable.

**Model B (Full interaction):**
```
peak_demand = beta_0 + beta_1*T_mean + beta_2*T_mean^2
            + beta_3*archetype + beta_4*(T_mean * archetype)
            + beta_5*fabric + error
```

Model B includes:
- A **quadratic temperature term** (captures nonlinearity)
- **Archetype dummy variables** (captures behaviour effects)
- **Temperature-archetype interactions** (captures differential response to cold)
- **Fabric dummy** (captures insulation effect)

**Ramsey RESET test:** Applied to Model A residuals. Tests whether squared and cubed fitted values have explanatory power (i.e., whether the linear specification is misspecified). A significant p-value (< 0.05) means Model A is structurally inadequate -- the relationship it's trying to capture is not linear.

**Model comparison metrics:** R-squared, RMSE, AIC, BIC -- all reported in `model_comparison.csv`.

### 8.4 Demand Surfaces (`analytics/demand_surface.py`)

**What it does:** Creates 2D matrices of electricity demand indexed by hour-of-day (rows: 0-23) and outdoor temperature bins (columns: 19 bins), one per archetype.

The outdoor temperature range is divided into 19 bins, and demand is averaged within each (hour, temp_bin) cell using `pd.pivot_table(aggfunc="mean")`. Missing cells are filled via bilinear interpolation along both axes.

These matrices directly reveal how each archetype's demand profile responds differently to temperature -- the visual complement to the ANOVA statistics. For example, B1 (Early Riser) shows a sharp morning spike while B2 (Home All Day) shows sustained demand throughout the day.

---

## 9. Validation Against Real Data

The model is validated against the **Electrification of Heat (EoH) demonstration project** -- 739 monitored ASHP homes with half-hourly data from 2020-2023 (27.7 million rows).

Run with: `python -m validation.run_validation`

### 9.1 COP Validation

Compares the model's COP curve (compressor COP = 3.5 + 0.12T) against real whole-system COP (SPF H4 boundary) from the EoH trial.

| Metric | Value |
|--------|-------|
| Systematic offset | +0.82 (model overestimates) |
| Empirical fit | COP = 2.71 + 0.001T |
| Observations | 6,759,582 half-hourly readings |

The offset is expected -- the model uses compressor COP while EoH measures whole-system SPF (including pumps, controls, parasitic losses).

### 9.2 Peak Demand Validation

Compares simulated peaks against real measured daily peak electrical demands.

| Metric | Value |
|--------|-------|
| Median peak | 5.15 kW |
| 90th percentile | 15.18 kW |
| 95th percentile | 18.36 kW |
| Properties | 730 |
| Property-days | 148,594 |

### 9.3 Archetype Discovery (k-means Clustering)

Clusters real daily heating profiles from 200 sampled properties to discover whether distinct behavioural archetypes emerge naturally from the data.

| k | Silhouette Score |
|---|-----------------|
| 2 | 0.285 (best) |
| 3 | 0.259 |
| 4 | 0.249 |
| 5 | 0.238 |
| 6 | 0.199 |

The forced k=4 clustering reveals 4 distinct heating profile shapes that correspond to the assumed archetypes (Early Riser, Home All Day, Late Returner, Intermittent).

### 9.4 Validation Outputs

All validation outputs are saved to `outputs/validation/`:

| Directory | Contents |
|-----------|----------|
| `cop/` | COP validation figure, binned COP data, statistics |
| `peaks/` | Peak demand distribution figures, ADMD by temperature |
| `archetypes/` | Cluster centroid figures, silhouette analysis, k=4 comparison |

### 9.5 External Data Sources

| Dataset | Source | Size | Purpose |
|---------|--------|------|---------|
| EoH 30-min (SN 9209) | UK Data Service | 2.4 GB | COP and peak demand validation |
| EoH Daily (SN 9210) | UK Data Service | 50 MB | Daily performance validation |
| EPC Certificates | DLUHC | 43 GB | Fabric distribution analysis |
| ENWL LV Networks | ENWL Open Data | 66 MB | Substation capacity context |
| SERL Smart Meter | UK Data Service | 462 MB | Aggregate consumption benchmarks |

These are not included in the repository (too large). Download from the respective sources and place in `data/external/`.

---

## 10. Figures and Outputs

### 9.1 Figures

All figures use:
- **Font:** DejaVu Sans, 10pt body / 12pt titles
- **Palette:** seaborn `colorblind` (8 distinguishable colours, accessible for colour-blind readers)
- **Resolution:** 300 DPI
- **Format:** dual PNG + PDF export (PDF for LaTeX inclusion)

| Figure | Filename | Content |
|--------|----------|---------|
| **Fig 1** | `fig1_timeseries` | Electricity demand over 24 hours for all 4 archetypes under Design Cold (W2) / unimproved fabric (F1). Shows how different behaviour patterns create fundamentally different demand shapes from the same building and weather. |
| **Fig 2** | `fig2_cop_curve` | Scatter of simulated COP vs outdoor temperature with the parametric curve overlaid. Shows the COP degradation with temperature and the defrost penalty knee at -2 C. |
| **Fig 3** | `fig3_hdd_error` | HDD underestimation (%) vs mean outdoor temperature, coloured by archetype with trend line. Directly visualises H1 -- whether errors grow with cold. |
| **Fig 4** | `fig4_anova_eta` | Horizontal bar chart of ANOVA eta-squared values, ranked by magnitude. Directly visualises H2 -- which factors explain the most variance in peak demand. |
| **Fig 5** | `fig5_demand_surface_heatmap` | 2x2 grid of heatmaps showing demand by (hour, temperature) for each archetype. Uses YlOrRd colourmap. Reveals the nonlinear interaction structure that HDD models miss. |
| **Fig 6** | `fig6_peak_boxplots` | Box plots of peak demand per archetype x weather, grouped by fabric. Shows the spread from Monte Carlo replication and the differential fabric effect across archetypes. |

### 9.2 Results Tables

| File | Rows | Content |
|------|------|---------|
| `master_results.csv` | 24 | One row per scenario cell: peak demand, daily energy, COP, indoor temps, HDD prediction, underestimation |
| `hdd_benchmark.csv` | 24 | Sim peak (mean/max/std across replicates), HDD prediction, underestimation %, mean outdoor temp |
| `anova_results.csv` | 7 | Sum of squares, df, F-statistic, p-value, eta-squared for each ANOVA term + residual |
| `model_comparison.csv` | 2 | R-squared, RMSE, AIC, BIC, n_obs for Model A and Model B |
| `simulation_results.csv` | 23,040 | Full simulation output -- every timestep of every run |

---

## 11. Test Suite

**File:** `tests/test_pipeline.py`

7 tests covering physics plausibility, statistical validity, and pipeline integrity:

| # | Test | What it Checks | Why it Matters |
|---|------|----------------|----------------|
| 1 | `test_weather_shape_and_range` | EPW profiles have 96 rows, temperatures within [-15, 40] C | Catches EPW parsing errors |
| 2 | `test_cop_bounds` | COP is always in [1.5, 4.5] across all 23,040 rows | Catches COP formula bugs |
| 3 | `test_indoor_temperature_bounds` | Indoor temp stays in [8, 28] C | Catches ODE instability or missing frost protection |
| 4 | `test_no_nans_and_24_cells` | All 24 scenario cells present, no NaN values | Catches simulation crashes or missing combos |
| 5 | `test_hdd_r_squared` | HDD model R^2 > 0.1 | Confirms HDD has *some* predictive power (low bar is intentional -- supports H3) |
| 6 | `test_anova_ss_nonnegative` | All ANOVA sum-of-squares >= 0 | Sanity check on ANOVA decomposition |
| 7 | `test_output_files_exist` | All 11 expected output files (5 CSVs + 6 PNGs) | Confirms pipeline ran to completion |

Run with: `python -m pytest tests/ -v`

---

## 12. Results Summary

### H1: HDD underestimates more in extreme cold

**Verdict: NOT SUPPORTED**

| Scenario | T_mean | Max Underestimation | Worst Archetype |
|----------|--------|-------------------|-----------------|
| W1 (Mild) | +4.0 C | 5.4% | B1 (Early Riser) |
| W2 (Design) | -0.7 C | 0.6% | B4 (Intermittent) |
| W3 (Extreme) | -2.5 C | 7.6% | B1 (Early Riser) |

The pattern is not monotonically increasing with cold. The HDD model actually fits best at design temperature (W2) because it was calibrated on similar data, but underestimates at both the mild and extreme ends. Notably, B3 (Late Returner) with F2 (retrofitted) shows HDD *over*estimation of up to 40% -- the low-setpoint archetype in a well-insulated home uses far less energy than the average the HDD model was trained on.

### H2: Behaviour matters more than fabric

**Verdict: SUPPORTED**

| Factor | Eta-squared | % of Total Variance |
|--------|------------|---------------------|
| archetype | 0.270 | 27.0% |
| archetype x fabric | 0.269 | 26.9% |
| fabric | 0.212 | 21.2% |
| weather_scenario | 0.156 | 15.6% |
| archetype x weather | 0.019 | 1.9% |
| weather x fabric | 0.003 | 0.3% |
| Residual | 0.073 | 7.3% |

Key findings:
- Occupant behaviour (archetype) explains **27.0%** of variance vs fabric's **21.2%**
- The archetype x fabric interaction is the **largest single term** at 26.9% -- meaning the impact of retrofit depends heavily on how the occupants use the building
- Weather scenario alone explains only 15.6%, less than either behaviour or fabric
- All F-tests are significant at p < 0.001

### H3: Nonlinear interaction, HDD inadequate

**Verdict: SUPPORTED**

| Metric | Model A (HDD-linear) | Model B (Full interaction) |
|--------|---------------------|--------------------------|
| R-squared | 0.069 | 0.644 |
| RMSE | 0.574 kW | 0.355 kW |
| AIC | 418.8 | 204.1 |
| BIC | 425.8 | 239.0 |

- **RESET test p-value: < 0.001** -- strong evidence of nonlinearity in HDD residuals
- Model B explains **9.3x more variance** than Model A
- AIC difference of **215 points** decisively favours Model B (>10 is considered strong evidence)
- The HDD-linear model is structurally inadequate for predicting heat pump peak demand

---

## 13. Key Equations Reference

### Thermal ODE
```
C * dT/dt = Q_hp + Q_int + Q_solar - UA * (T_in - T_out)
```

### Heat Loss Coefficient
```
UA = Sum(U_i * A_i) + 0.33 * ACH * Volume    [W/K]
```

### COP Model (calibrated to EST RHPP field trial)
```
COP = 3.5 + 0.12 * T_out
if T_out < -2: COP *= 0.85      (defrost penalty)
COP = clip(COP, 1.5, 4.5)
```

### Thermal Capacity Derating
```
if T_out < 7:
    factor = max(0.4, 1 - 0.025 * (7 - T_out))
Q_max = 12000 * factor    [W]
```

### Variable-Speed Modulation
```
deficit = max(0, setpoint - T_in)
modulation = min(1.0, 0.3 + 0.7 * deficit / 3.0)
Q_hp = Q_max * modulation
```

### Electrical Demand
```
P_elec = Q_hp_thermal / COP + P_aux     [W]
```

### Heating Degree Days
```
HDD = max(0, 15.5 - T_mean_daily)
```

### Eta-squared (ANOVA effect size)
```
eta_sq = SS_factor / SS_total
```

---

## 14. Dependencies

```
numpy>=1.24          # numerical arrays, random number generation
pandas>=2.0          # DataFrames, time series, aggregation
scipy>=1.10          # scientific computing (transitive dependency)
statsmodels>=0.14    # OLS regression, ANOVA, RESET test
scikit-learn>=1.3    # available for extensions
matplotlib>=3.7      # figure generation
seaborn>=0.12        # statistical plot styling, colour palettes
pyyaml>=6.0          # YAML config parsing
pytest>=7.0          # test framework
```

Install all with:
```bash
pip install -r requirements.txt
```

---

## Reproducibility

All random seeds are fixed to 42. Running `python main.py` will produce identical outputs on any machine with the same Python version and package versions. The EPW weather file in `data/raw/` is included in the repository for full reproducibility.

## Limitations

- The 2R1C model does not capture thermal bridging, multi-zone effects, or hot water demand
- Weather scenarios are single-day snapshots; multi-day cold spells with thermal mass depletion are not modelled
- Only 4 discrete archetypes -- real populations show continuous behavioural variation
- The EPW file represents London Gatwick; results may differ for other UK climate zones
- Validation uses EoH field trial data (739 homes); model COP is systematically higher than whole-system SPF by ~0.8

## License

Academic research project. All data sources are publicly available and cited in the configuration file and above.
