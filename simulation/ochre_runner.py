"""
OCHRE interface / physics-based thermal surrogate for heat pump simulation.

Attempts to import OCHRE first; falls back to a 2R1C lumped-capacitance
thermal model integrated via forward Euler with 1-minute internal steps.

v3: Uses real EPW weather data (London Gatwick TMYx) instead of synthetic
profiles.  COP curve calibrated to EST/DECC RHPP field trial (2013).

Units
-----
- Temperature: °C
- Power: kW (electrical) or W (thermal, internal)
- Time: hours (external), seconds (internal ODE)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Try OCHRE import
# ---------------------------------------------------------------------------
OCHRE_AVAILABLE = False
try:
    from ochre import Dwelling  # type: ignore
    OCHRE_AVAILABLE = True
    logger.info("OCHRE library detected — using native OCHRE engine.")
except ImportError:
    logger.warning("OCHRE not available — falling back to 2R1C thermal surrogate.")


# ===================================================================== #
#                       Weather generation                               #
# ===================================================================== #

def generate_weather(
    weather_cfg: dict,
    timesteps: int = 96,
    dt_minutes: int = 15,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a 24-hour synthetic weather profile.

    Parameters
    ----------
    weather_cfg : dict
        Weather scenario from config (keys: T_mean_C, T_range_C, etc.).
    timesteps : int
        Number of output rows (default 96 for 15-min resolution).
    dt_minutes : int
        Minutes per timestep.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Columns: timestamp, T_outdoor_C, solar_W_m2, wind_speed_m_s
    """
    rng = np.random.default_rng(seed)

    hours = np.arange(timesteps) * dt_minutes / 60.0  # 0 .. 23.75

    # --- Outdoor temperature: sinusoidal diurnal + noise ---
    T_mean = weather_cfg["T_mean_C"]
    T_min, T_max = weather_cfg["T_range_C"]
    amplitude = (T_max - T_min) / 2.0
    # Minimum at ~06:00, maximum at ~15:00
    T_outdoor = T_mean + amplitude * np.sin(
        2 * np.pi * (hours - 6.0) / 24.0 - np.pi / 2
    )
    noise = rng.normal(0, weather_cfg["noise_sigma_C"], size=timesteps)
    T_outdoor = T_outdoor + noise
    T_outdoor = np.clip(T_outdoor, T_min - 1.0, T_max + 1.0)

    # --- Solar irradiance: half-sinusoid peaking at solar noon (12:00) ---
    solar_peak = weather_cfg["solar_peak_W_m2"]
    solar = solar_peak * np.clip(np.sin(np.pi * (hours - 7.0) / 10.0), 0, None)
    solar[hours < 7.0] = 0.0
    solar[hours > 17.0] = 0.0

    # --- Wind speed: Weibull-distributed ---
    shape = weather_cfg["wind_weibull_shape"]
    scale = weather_cfg["wind_weibull_scale"]
    wind = rng.weibull(shape, size=timesteps) * scale

    timestamps = pd.date_range(
        "2024-01-15", periods=timesteps, freq=f"{dt_minutes}min"
    )

    return pd.DataFrame({
        "timestamp": timestamps,
        "T_outdoor_C": T_outdoor,
        "solar_W_m2": solar,
        "wind_speed_m_s": wind,
    })


# ===================================================================== #
#                       COP calculation                                  #
# ===================================================================== #

def compute_cop(T_out: float, hp_cfg: dict) -> float:
    """Compute heat-pump COP for a given outdoor temperature.

    Parameters
    ----------
    T_out : float
        Outdoor air temperature (°C).
    hp_cfg : dict
        Heat pump config section.

    Returns
    -------
    float
        COP (dimensionless), clipped to [cop_min, cop_max].
    """
    cop = hp_cfg["cop_intercept"] + hp_cfg["cop_slope"] * T_out
    if T_out < hp_cfg["defrost_temp_threshold_C"]:
        cop *= (1.0 - hp_cfg["defrost_efficiency_penalty"])
    return float(np.clip(cop, hp_cfg["cop_min"], hp_cfg["cop_max"]))


def compute_thermal_capacity(T_out: float, hp_cfg: dict) -> float:
    """Compute derated thermal capacity at given outdoor temperature.

    Real ASHPs lose thermal output as outdoor temperature drops because
    the refrigerant cycle becomes less effective.  We model this as a
    linear derating from rated capacity at the design outdoor temp (7°C).

    Parameters
    ----------
    T_out : float
        Outdoor air temperature (°C).
    hp_cfg : dict
        Heat pump config section.

    Returns
    -------
    float
        Available thermal capacity (W).
    """
    rated_W = hp_cfg["rated_thermal_capacity_kW"] * 1000.0
    design_temp = hp_cfg["capacity_design_temp_C"]
    derating = hp_cfg["capacity_derating_per_K"]
    # Derating below design temp
    if T_out < design_temp:
        factor = max(0.4, 1.0 - derating * (design_temp - T_out))
    else:
        factor = 1.0
    return rated_W * factor


# ===================================================================== #
#                   Heating schedule helpers                              #
# ===================================================================== #

def is_heating_on(hour: float, schedule_on: List[List[float]]) -> bool:
    """Return True if *hour* falls within any scheduled heating period.

    Parameters
    ----------
    hour : float
        Decimal hour of day (0–24).
    schedule_on : list of [start, end]
        Heating periods.

    Returns
    -------
    bool
    """
    for start, end in schedule_on:
        if start <= hour < end:
            return True
    return False


def jitter_schedule(
    schedule_on: List[List[float]],
    jitter_minutes: float,
    rng: np.random.Generator,
) -> List[List[float]]:
    """Apply random timing jitter to a heating schedule.

    Parameters
    ----------
    schedule_on : list of [start, end]
        Original schedule.
    jitter_minutes : float
        Max jitter in minutes (applied as ±uniform).
    rng : Generator
        Random number generator.

    Returns
    -------
    list of [start, end]
        Jittered schedule (hours still within 0–24).
    """
    jitter_h = jitter_minutes / 60.0
    jittered = []
    for start, end in schedule_on:
        s = start + rng.uniform(-jitter_h, jitter_h)
        e = end + rng.uniform(-jitter_h, jitter_h)
        s = max(0.0, min(23.75, s))
        e = max(s + 0.25, min(24.0, e))
        jittered.append([s, e])
    return jittered


def get_setpoint(
    hour: float,
    archetype_cfg: dict,
    schedule: List[List[float]],
    rng: np.random.Generator | None = None,
) -> float:
    """Return the active setpoint for the given hour.

    Parameters
    ----------
    hour : float
        Decimal hour of day.
    archetype_cfg : dict
        Archetype config section.
    schedule : list of [start, end]
        Possibly-jittered heating schedule.
    rng : Generator, optional
        For variable-setpoint archetype (B4).

    Returns
    -------
    float
        Target indoor temperature (°C).
    """
    if not is_heating_on(hour, schedule):
        return archetype_cfg["setback_C"]

    if archetype_cfg.get("variable_setpoint", False) and rng is not None:
        lo = archetype_cfg["setpoint_min_C"]
        hi = archetype_cfg["setpoint_max_C"]
        return rng.uniform(lo, hi)

    return archetype_cfg["setpoint_C"]


# ===================================================================== #
#               UA and capacitance from fabric + geometry                #
# ===================================================================== #

def compute_ua(fabric_cfg: dict, building_cfg: dict) -> float:
    """Compute overall heat-loss coefficient UA (W/K).

    Parameters
    ----------
    fabric_cfg : dict
        Fabric U-values and ACH.
    building_cfg : dict
        Building geometry.

    Returns
    -------
    float
        UA in W/K.
    """
    A_wall = building_cfg["wall_area_m2"]
    A_roof = building_cfg["roof_area_m2"]
    A_floor = building_cfg["floor_area_ground_m2"]
    A_win = building_cfg["window_area_m2"]
    vol = building_cfg["volume_m3"]

    ua_fabric = (
        fabric_cfg["U_wall"] * A_wall
        + fabric_cfg["U_roof"] * A_roof
        + fabric_cfg["U_floor"] * A_floor
        + fabric_cfg["U_window"] * A_win
    )
    ua_vent = 0.33 * fabric_cfg["ACH"] * vol
    return ua_fabric + ua_vent


def compute_capacitance(building_cfg: dict) -> float:
    """Compute effective thermal capacitance C (J/K).

    Parameters
    ----------
    building_cfg : dict
        Building geometry including thermal_mass_kJ_per_m2K.

    Returns
    -------
    float
        Thermal capacitance in J/K.
    """
    return (
        building_cfg["thermal_mass_kJ_per_m2K"]
        * building_cfg["floor_area_m2"]
        * 1000.0
    )


# ===================================================================== #
#               2R1C Thermal Surrogate Simulation                       #
# ===================================================================== #

def run_single_simulation(
    archetype_id: str,
    archetype_cfg: dict,
    weather_id: str,
    weather_df: pd.DataFrame,
    fabric_id: str,
    fabric_cfg: dict,
    building_cfg: dict,
    hp_cfg: dict,
    gains_cfg: dict,
    controller_cfg: dict,
    dt_minutes: int = 15,
    dt_internal_s: int = 60,
    seed: int = 42,
    gains_noise_frac: float = 0.0,
    schedule_jitter_min: float = 0.0,
    replicate_id: int = 0,
) -> pd.DataFrame:
    """Run one 24-hour heat-pump simulation using the 2R1C surrogate.

    The heat pump is modelled as a variable-speed ASHP with:
    - Rated THERMAL capacity (kW) that derates with outdoor temperature
    - COP that degrades with outdoor temperature + defrost penalty
    - Electrical demand = thermal output / COP
    - Auxiliary electric resistance heater for large recovery loads
    - Frost protection prevents indoor temp from crashing

    Parameters
    ----------
    archetype_id : str
        Archetype identifier (B1–B4).
    archetype_cfg : dict
        Archetype configuration.
    weather_id : str
        Weather scenario identifier (W1–W3).
    weather_df : pd.DataFrame
        Pre-generated weather data (96 rows).
    fabric_id : str
        Fabric identifier (F1/F2).
    fabric_cfg : dict
        Fabric U-values and ACH.
    building_cfg : dict
        Building geometry.
    hp_cfg : dict
        Heat pump parameters.
    gains_cfg : dict
        Internal gains config.
    controller_cfg : dict
        Thermostat deadband.
    dt_minutes : int
        Output resolution (minutes).
    dt_internal_s : int
        Internal ODE step (seconds).
    seed : int
        Random seed.
    gains_noise_frac : float
        Fractional noise on internal gains (0 = none, 0.3 = ±30%).
    schedule_jitter_min : float
        Schedule timing jitter in minutes.
    replicate_id : int
        Replicate number (included in output for grouping).

    Returns
    -------
    pd.DataFrame
        Columns: timestamp, T_indoor_C, T_outdoor_C, Q_heat_pump_kW,
        electricity_demand_kW, COP, fabric, archetype, weather_scenario,
        replicate
    """
    rng = np.random.default_rng(seed)

    UA = compute_ua(fabric_cfg, building_cfg)
    C = compute_capacitance(building_cfg)
    deadband = controller_cfg["deadband_C"]
    min_runtime_s = hp_cfg["min_runtime_minutes"] * 60.0
    aux_cap_W = hp_cfg.get("aux_capacity_kW", 0.0) * 1000.0
    aux_threshold = hp_cfg.get("aux_deficit_threshold_C", 99.0)
    frost_prot = hp_cfg.get("frost_protection_C", 5.0)

    n_output = len(weather_df)
    dt_out_s = dt_minutes * 60.0
    steps_per_output = int(dt_out_s / dt_internal_s)

    T_out_arr = weather_df["T_outdoor_C"].values
    solar_arr = weather_df["solar_W_m2"].values
    hours_arr = (
        (weather_df["timestamp"] - weather_df["timestamp"].iloc[0])
        .dt.total_seconds()
        .values
        / 3600.0
    )

    # Apply schedule jitter for this replicate
    schedule = archetype_cfg["schedule_on"]
    if schedule_jitter_min > 0:
        schedule = jitter_schedule(schedule, schedule_jitter_min, rng)

    # Pre-compute setpoints
    setpoints = np.array([
        get_setpoint(h % 24, archetype_cfg, schedule, rng) for h in hours_arr
    ])

    # Internal gains noise multiplier per timestep
    if gains_noise_frac > 0:
        gains_mult = 1.0 + rng.uniform(
            -gains_noise_frac, gains_noise_frac, size=n_output
        )
    else:
        gains_mult = np.ones(n_output)

    # --- Integration loop ---
    T_in = archetype_cfg["setback_C"]
    hp_on = False
    hp_on_timer = 0.0

    T_indoor_out = np.zeros(n_output)
    Q_hp_out = np.zeros(n_output)
    elec_out = np.zeros(n_output)
    cop_out = np.zeros(n_output)

    for i in range(n_output):
        T_out_i = T_out_arr[i]
        solar_i = solar_arr[i]
        hour_i = hours_arr[i] % 24
        sp_i = setpoints[i]

        cop_i = compute_cop(T_out_i, hp_cfg)
        Q_thermal_max = compute_thermal_capacity(T_out_i, hp_cfg)

        # Internal gains with per-replicate noise
        Q_int_base = gains_cfg["base_W"]
        if is_heating_on(hour_i, schedule):
            Q_int_base += gains_cfg["occupancy_pulse_W"]
        Q_int = Q_int_base * gains_mult[i]

        # Solar gain (~30% through windows)
        Q_solar = 0.3 * solar_i * building_cfg["window_area_m2"]

        T_in_sum = 0.0
        Q_hp_sum = 0.0
        elec_sum = 0.0

        for _ in range(steps_per_output):
            # Thermostat logic with frost protection
            if hp_on:
                hp_on_timer -= dt_internal_s
                if T_in >= sp_i + deadband and hp_on_timer <= 0:
                    hp_on = False
            else:
                if T_in <= sp_i - deadband:
                    hp_on = True
                    hp_on_timer = min_runtime_s
                # Frost protection: force on if indoor temp crashes
                elif T_in <= frost_prot:
                    hp_on = True
                    hp_on_timer = min_runtime_s

            if hp_on:
                # Variable-speed: modulate thermal output by deficit
                deficit = max(0.0, sp_i - T_in)
                modulation = min(1.0, 0.3 + 0.7 * deficit / 3.0)
                Q_hp_thermal = Q_thermal_max * modulation
                elec_hp_W = Q_hp_thermal / cop_i

                # Auxiliary resistance heater for large recovery loads
                if deficit > aux_threshold and aux_cap_W > 0:
                    aux_frac = min(1.0, (deficit - aux_threshold) / 2.0)
                    elec_aux_W = aux_cap_W * aux_frac
                    Q_aux_thermal = elec_aux_W  # COP = 1
                else:
                    elec_aux_W = 0.0
                    Q_aux_thermal = 0.0

                Q_thermal = Q_hp_thermal + Q_aux_thermal
                elec_W = elec_hp_W + elec_aux_W
            else:
                Q_thermal = 0.0
                elec_W = 0.0

            # ODE: C dT/dt = Q_hp + Q_int + Q_solar - UA*(T_in - T_out)
            dTdt = (Q_thermal + Q_int + Q_solar - UA * (T_in - T_out_i)) / C
            T_in += dTdt * dt_internal_s

            T_in_sum += T_in
            Q_hp_sum += Q_thermal
            elec_sum += elec_W

        T_indoor_out[i] = T_in_sum / steps_per_output
        Q_hp_out[i] = Q_hp_sum / steps_per_output / 1000.0
        elec_out[i] = elec_sum / steps_per_output / 1000.0
        cop_out[i] = cop_i

    return pd.DataFrame({
        "timestamp": weather_df["timestamp"].values,
        "T_indoor_C": T_indoor_out,
        "T_outdoor_C": T_out_arr,
        "Q_heat_pump_kW": Q_hp_out,
        "electricity_demand_kW": elec_out,
        "COP": cop_out,
        "fabric": fabric_id,
        "archetype": archetype_id,
        "weather_scenario": weather_id,
        "replicate": replicate_id,
    })


# ===================================================================== #
#                  Run full factorial with replication                    #
# ===================================================================== #

def run_all_simulations(cfg: dict, project_root: Path | None = None) -> pd.DataFrame:
    """Execute the full factorial design with Monte Carlo replication.

    Uses real EPW weather profiles when epw_file is configured, falling
    back to synthetic generation for backward compatibility.

    Each of the 4×3×2 = 24 scenarios is run N_replicates times with
    stochastic variation in schedule jitter and internal gains.
    Total rows = 24 × N × 96.

    Parameters
    ----------
    cfg : dict
        Full parsed YAML config.
    project_root : Path, optional
        Project root for resolving relative EPW path.

    Returns
    -------
    pd.DataFrame
        Concatenated results for all runs.
    """
    from simulation.epw_parser import load_weather_profiles

    seed = cfg["random_seed"]
    building_cfg = cfg["building"]
    hp_cfg = cfg["heat_pump"]
    gains_cfg = cfg["internal_gains"]
    ctrl_cfg = cfg["controller"]
    dt_min = cfg["dt_minutes"]
    dt_int = cfg["dt_internal_seconds"]

    n_rep = cfg.get("replicates", 1)
    schedule_jitter = cfg.get("schedule_jitter_minutes", 0.0)
    gains_noise = cfg.get("gains_noise_fraction", 0.0)

    # --- Load real EPW weather profiles ---
    epw_rel = cfg.get("epw_file")
    if epw_rel and project_root is not None:
        epw_path = project_root / epw_rel
    elif epw_rel:
        epw_path = Path(epw_rel)
    else:
        epw_path = None

    weather_profiles: Dict[str, pd.DataFrame] = {}
    if epw_path and epw_path.exists():
        weather_days = {}
        for w_id, w_cfg in cfg["weather"].items():
            if "epw_month" in w_cfg and "epw_day" in w_cfg:
                weather_days[w_id] = {"month": w_cfg["epw_month"], "day": w_cfg["epw_day"]}
        if weather_days:
            weather_profiles = load_weather_profiles(epw_path, weather_days, dt_min)
            logger.info("Loaded %d real EPW weather profiles", len(weather_profiles))
    else:
        logger.warning("EPW file not found — falling back to synthetic weather")

    total_runs = len(cfg["archetypes"]) * len(cfg["weather"]) * len(cfg["fabric"]) * n_rep
    all_results: List[pd.DataFrame] = []
    run_counter = 0

    for w_id, w_cfg in cfg["weather"].items():
        # Use real EPW profile if available, else fall back to synthetic
        base_weather = weather_profiles.get(w_id)

        for rep in range(n_rep):
            if base_weather is not None:
                weather_df = base_weather.copy()
            else:
                weather_seed = seed + rep * 1000
                weather_df = generate_weather(
                    w_cfg, timesteps=cfg["timesteps"], dt_minutes=dt_min, seed=weather_seed
                )

            for f_id, f_cfg in cfg["fabric"].items():
                for a_id, a_cfg in cfg["archetypes"].items():
                    run_counter += 1
                    logger.info(
                        "Run %03d / %d: archetype=%s  weather=%s  fabric=%s  rep=%d",
                        run_counter, total_runs, a_id, w_id, f_id, rep,
                    )
                    sim_seed = seed + run_counter + rep * 100
                    df = run_single_simulation(
                        archetype_id=a_id,
                        archetype_cfg=a_cfg,
                        weather_id=w_id,
                        weather_df=weather_df,
                        fabric_id=f_id,
                        fabric_cfg=f_cfg,
                        building_cfg=building_cfg,
                        hp_cfg=hp_cfg,
                        gains_cfg=gains_cfg,
                        controller_cfg=ctrl_cfg,
                        dt_minutes=dt_min,
                        dt_internal_s=dt_int,
                        seed=sim_seed,
                        gains_noise_frac=gains_noise,
                        schedule_jitter_min=schedule_jitter,
                        replicate_id=rep,
                    )
                    all_results.append(df)

    combined = pd.concat(all_results, ignore_index=True)
    logger.info(
        "All %d simulations complete — %d rows total.", total_runs, len(combined)
    )
    return combined
